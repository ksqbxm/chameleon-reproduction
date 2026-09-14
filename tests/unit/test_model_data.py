from dataclasses import replace

import pytest

from chameleon import ClusterState, ModelConfig, WorkerIdentity


@pytest.fixture
def torch_module():
    import torch
    return torch


@pytest.fixture
def model_config():
    return ModelConfig(vocab_size=17, hidden_size=8, num_layers=2, num_heads=2,
                       sequence_length=4, global_batch_size=5, micro_batch_size=2)


@pytest.fixture
def model(torch_module, model_config, device):
    from chameleon.model import build_initial_model
    return build_initial_model(model_config, device=device)


def test_complete_inventory(model, torch_module):
    from chameleon.model import parameter_inventory
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    inventory = parameter_inventory(model)
    assert {p.name for p in inventory} == set(expected)
    assert len(inventory) == len(expected)
    assert {p.module_id for p in inventory} == {
        "embedding", "blocks.0", "blocks.1", "final_norm", "lm_head",
    }
    for entry in inventory:
        parameter = expected[entry.name]
        assert entry.shape == tuple(parameter.shape)
        assert entry.numel == parameter.numel()
        assert entry.nbytes == parameter.numel() * parameter.element_size()

    model.extra = torch_module.nn.Linear(8, 8)
    model.blocks[0].norm1.bias.requires_grad_(False)
    updated = parameter_inventory(model)
    assert {p.name for p in updated} == {
        name for name, p in model.named_parameters() if p.requires_grad
    }
    assert {p.module_id for p in updated if p.name.startswith("extra.")} == {"extra"}


def test_training_forward_is_deterministic_and_causal(model, torch_module, device):
    torch = torch_module
    model.train()
    dropouts = [module for module in model.modules() if isinstance(module, torch.nn.Dropout)]
    assert dropouts and all(module.p == 0 for module in dropouts)
    assert all(block.self_attn.dropout == 0 for block in model.blocks)
    assert model.embedding.weight.data_ptr() != model.lm_head.weight.data_ptr()
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], device=device)
    before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state() if device == "cuda" else None
    first = model(tokens)
    second = model(tokens)
    assert first.shape == (2, 4, 17) and first.dtype == torch.float64
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(before, torch.get_rng_state())
    if device == "cuda":
        assert torch.equal(cuda_before, torch.cuda.get_rng_state())
    changed = tokens.clone()
    changed[:, -1] = 6
    tolerance = dict(rtol=1e-7, atol=1e-9) if device == "cuda" else dict(rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(first[:, :-1], model(changed)[:, :-1], **tolerance)


def test_initial_seed_is_repeatable_and_preserves_rng(model_config, torch_module, device):
    from chameleon.model import build_initial_model, parameter_inventory
    torch = torch_module
    before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state() if device == "cuda" else None
    first = build_initial_model(model_config, device=device)
    second = build_initial_model(model_config, device=device)
    assert torch.equal(before, torch.get_rng_state())
    if device == "cuda":
        assert torch.equal(cuda_before, torch.cuda.get_rng_state())
    assert parameter_inventory(first) == parameter_inventory(second)
    for name, parameter in first.named_parameters():
        torch.testing.assert_close(parameter, dict(second.named_parameters())[name], rtol=0, atol=0)
    different = build_initial_model(replace(model_config, seed=model_config.seed + 1), device=device)
    assert not torch.equal(first.embedding.weight, different.embedding.weight)


@pytest.mark.parametrize("partitioned", [False, True])
def test_fp32_initializers_enforce_full_cuda_matmul_precision(torch_module, device, model_config, monkeypatch, partitioned):
    from chameleon.model import build_initial_model, build_initial_stage
    torch = torch_module
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    model = (build_initial_stage(model_config, ("embedding", "blocks.0"), device=device, dtype=torch.float32)
             if partitioned else build_initial_model(model_config, device=device, dtype=torch.float32))
    assert all(p.dtype == torch.float32 and p.device.type == device for p in model.parameters())
    assert torch.backends.cuda.matmul.allow_tf32 == (device != "cuda")


def test_fp32_full_batch_matches_partitioned_micro_batch_gradients(torch_module, device, monkeypatch):
    from chameleon.data import make_batch
    from chameleon.model import build_initial_model, build_initial_stage
    from chameleon.reference import sample_losses
    torch = torch_module
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=8, micro_batch_size=2)
    full = build_initial_model(config, device=device, dtype=torch.float32)
    first = build_initial_stage(config, ("embedding", "blocks.0"), device=device, dtype=torch.float32)
    last = build_initial_stage(config, ("blocks.1", "final_norm", "lm_head"), device=device, dtype=torch.float32)
    batch = make_batch(tuple(range(8)), config, device=device)
    sample_losses(full(batch.inputs), batch.targets).sum().backward()
    for left in range(0, 8, 2):
        activation = first(batch.inputs[left:left + 2])
        received = activation.detach().requires_grad_()
        sample_losses(last(received), batch.targets[left:left + 2]).sum().backward()
        activation.backward(received.grad)
    expected = dict(full.named_parameters())
    for stage in (first, last):
        for name, parameter in stage.named_parameters():
            torch.testing.assert_close(parameter.grad / 8, expected[name].grad / 8,
                                       rtol=1e-4, atol=1e-6, msg=lambda message: f"{name}: {message}")


@pytest.mark.parametrize("shape", [(4,), (2, 3), (1, 2, 4)])
def test_forward_rejects_wrong_sequence_shape(model, torch_module, device, shape):
    with pytest.raises(ValueError, match="sequence length"):
        model(torch_module.zeros(shape, dtype=torch_module.long, device=device))


def test_ids_use_only_committed_step_and_fixed_global_batch(model_config):
    from chameleon.data import next_sample_ids
    state = ClusterState((WorkerIdentity("w0", 0, 0),), model_config.global_batch_size)
    assert next_sample_ids(state) == (0, 1, 2, 3, 4)
    all_ids = []
    for step in range(4):
        committed = replace(state, committed_global_step=step)
        assert next_sample_ids(committed) == tuple(range(step * 5, (step + 1) * 5))
        all_ids.extend(next_sample_ids(committed))
        rebuilt = replace(committed, generation=1, workers=(WorkerIdentity("w0", 0, 1),))
        assert next_sample_ids(rebuilt) == next_sample_ids(committed)
    assert all_ids == list(range(20))


def test_data_is_an_id_function_without_rng_or_cursor(model_config, torch_module, device):
    from chameleon.data import make_batch
    torch = torch_module
    before = torch.get_rng_state().clone()
    cuda_before = torch.cuda.get_rng_state() if device == "cuda" else None
    first = make_batch((0, 1, 2, 3, 4), model_config, device=device)
    repeated = make_batch(first.sample_ids, model_config, device=device)
    shuffled = make_batch((4, 1, 0), model_config, device=device)
    torch.testing.assert_close(first.inputs, repeated.inputs, rtol=0, atol=0)
    torch.testing.assert_close(first.targets, repeated.targets, rtol=0, atol=0)
    torch.testing.assert_close(shuffled.inputs, first.inputs[[4, 1, 0]], rtol=0, atol=0)
    torch.testing.assert_close(shuffled.targets, first.targets[[4, 1, 0]], rtol=0, atol=0)
    assert first.inputs.shape == first.targets.shape == (5, 4)
    assert first.inputs.dtype == first.targets.dtype == torch.long
    assert first.inputs.device.type == device
    assert first.inputs.min().item() >= 0 and first.inputs.max().item() < model_config.vocab_size
    assert first.targets.min().item() >= 0 and first.targets.max().item() < model_config.vocab_size
    for row, sample_id in enumerate(first.sample_ids):
        expected = [(sample_id * (position + 1) + position * position + 3 * position + 1)
                    % model_config.vocab_size for position in range(5)]
        assert first.inputs[row].tolist() == expected[:-1]
        assert first.targets[row].tolist() == expected[1:]
    assert torch.equal(before, torch.get_rng_state())
    if device == "cuda":
        assert torch.equal(cuda_before, torch.cuda.get_rng_state())


@pytest.mark.parametrize("ids", [(), (-1,), (True,), (1.5,)])
def test_invalid_sample_ids(ids, model_config):
    from chameleon.data import make_batch
    with pytest.raises(ValueError, match="sample_id"):
        make_batch(ids, model_config)
