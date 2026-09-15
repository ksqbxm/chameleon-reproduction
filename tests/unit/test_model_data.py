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


def test_cli_and_all_topologies_share_the_model_stage_layout_contract(monkeypatch):
    from chameleon import cli, model
    from chameleon.runtime import DynamicTopology, ReroutingTopology, SymmetricTopology

    config = ModelConfig(num_layers=2, global_batch_size=4, micro_batch_size=2)
    stages = (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head"))
    workers = (WorkerIdentity("layout-0", 0, 0), WorkerIdentity("layout-1", 1, 0))
    state = ClusterState(workers, config.global_batch_size)
    assert cli.validate_stage_layout is model.validate_stage_layout
    calls = []
    validate = model.validate_stage_layout

    def record(config_value, stages_value):
        calls.append(stages_value)
        return validate(config_value, stages_value)

    monkeypatch.setattr(model, "validate_stage_layout", record)
    SymmetricTopology(state, config, stages)
    DynamicTopology(state, config, (stages,), (2,))
    ReroutingTopology(state, config, stages, (2,), ((0, 1),))
    assert calls == [stages, stages, stages]


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


def test_attention_trains_only_query_value_bias_and_preserves_softmax(model, torch_module, device):
    torch = torch_module
    hidden_size = model.config.hidden_size
    hidden = torch.arange(2 * 3 * hidden_size, device=device, dtype=torch.float64).reshape(2, 3, hidden_size) / 10
    mask = torch.ones(3, 3, device=device, dtype=torch.bool).triu(1)
    tolerance = dict(rtol=1e-7, atol=1e-9) if device == "cuda" else dict(rtol=1e-8, atol=1e-10)
    for block in model.blocks:
        attention = block.self_attn
        biases = list(attention.parametrizations.in_proj_bias.parameters())
        assert len(biases) == 2 and all(p.shape == (hidden_size,) and p.requires_grad for p in biases)
        assert "in_proj_bias" not in dict(attention.named_parameters())
        assert biases[0].untyped_storage().data_ptr() != biases[1].untyped_storage().data_ptr()
        with torch.no_grad():
            biases[0].copy_(torch.linspace(-0.2, 0.3, hidden_size, device=device))
            biases[1].copy_(torch.linspace(0.1, 0.4, hidden_size, device=device))
        query, key, value = attention.in_proj_bias.chunk(3)
        torch.testing.assert_close(query, biases[0], rtol=0, atol=0)
        torch.testing.assert_close(value, biases[1], rtol=0, atol=0)
        assert torch.count_nonzero(key).item() == 0
        # An independent, unrestricted MHA has the same outputs even with a nonzero key bias.
        unrestricted = torch.nn.MultiheadAttention(hidden_size, model.config.num_heads, batch_first=True,
                                                    device=device, dtype=torch.float64)
        unrestricted.load_state_dict({
            "in_proj_weight": attention.in_proj_weight,
            "in_proj_bias": attention.in_proj_bias,
            "out_proj.weight": attention.out_proj.weight,
            "out_proj.bias": attention.out_proj.bias,
        })
        with torch.no_grad():
            unrestricted.in_proj_bias[hidden_size:2 * hidden_size].fill_(1.25)
        actual, _ = attention(hidden, hidden, hidden, attn_mask=mask, need_weights=False)
        expected, _ = unrestricted(hidden, hidden, hidden, attn_mask=mask, need_weights=False)
        torch.testing.assert_close(actual, expected, **tolerance)


def test_current_attention_bias_state_roundtrips_without_legacy_parameters(model, torch_module, device):
    from chameleon.model import build_initial_model
    torch = torch_module
    with torch.no_grad():
        for block in model.blocks:
            block.self_attn.parametrizations.in_proj_bias.original0.fill_(0.2)
            block.self_attn.parametrizations.in_proj_bias.original1.fill_(-0.3)
    state = model.state_dict()
    assert all(not name.endswith("self_attn.in_proj_bias") for name in state)
    restored = build_initial_model(model.config, device=device)
    restored.load_state_dict(state, strict=True)
    assert state.keys() == restored.state_dict().keys()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, dict(restored.named_parameters())[name], rtol=0, atol=0)


@pytest.mark.parametrize("partitioned", [False, True])
def test_fp32_initializers_enforce_full_cuda_matmul_precision(torch_module, device, model_config, monkeypatch, partitioned):
    from chameleon.model import build_initial_model, build_initial_stage
    torch = torch_module
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    model = (build_initial_stage(model_config, ("embedding", "blocks.0"), device=device, dtype=torch.float32)
             if partitioned else build_initial_model(model_config, device=device, dtype=torch.float32))
    assert all(p.dtype == torch.float32 and p.device.type == device for p in model.parameters())
    assert torch.backends.cuda.matmul.allow_tf32 == (device != "cuda")


@pytest.mark.parametrize("sample_counts", [(4, 4), (6, 5), (10, 6, 3)])
def test_fp32_full_batch_matches_three_partitioned_adamw_steps(torch_module, device, monkeypatch, sample_counts):
    from chameleon.data import make_batch
    from chameleon.model import build_initial_model, build_initial_stage
    from chameleon.reference import ReferenceTrainer, sample_losses
    torch = torch_module
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", True)
    batch_size = sum(sample_counts)
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=batch_size, micro_batch_size=2)
    reference = ReferenceTrainer(build_initial_model(config, device=device, dtype=torch.float32),
                                 ClusterState((WorkerIdentity("reference", 0, 0),), batch_size))
    pipelines = [tuple(build_initial_stage(config, modules, device=device, dtype=torch.float32)
                       for modules in (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head")))
                 for _ in sample_counts]
    stages = [stage for pipeline in pipelines for stage in pipeline]
    optimizers = [torch.optim.AdamW(stage.parameters()) for stage in stages]
    owners = {}
    for stage in stages:
        for name, parameter in stage.named_parameters():
            owners.setdefault(name, []).append(parameter)
    assert set(owners) == dict(reference.model.named_parameters()).keys()
    assert all(len(replicas) == len(pipelines) for replicas in owners.values())
    for step in range(3):
        expected = reference.train_step()
        batch = make_batch(range(step * batch_size, (step + 1) * batch_size), config, device=device)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        start = 0
        for (first, last), count in zip(pipelines, sample_counts):
            for left in range(start, start + count, config.micro_batch_size):
                right = min(left + config.micro_batch_size, start + count)
                activation = first(batch.inputs[left:right])
                received = activation.detach().requires_grad_()
                sample_losses(last(received), batch.targets[left:right]).sum().backward()
                activation.backward(received.grad)
            start += count
        for replicas in owners.values():
            gradient = torch.stack([p.grad for p in replicas]).sum(dim=0) / batch_size
            for parameter in replicas:
                parameter.grad.copy_(gradient)
        for stage, optimizer in zip(stages, optimizers):
            optimizer.step()
            for name, parameter in stage.named_parameters():
                torch.testing.assert_close(parameter.detach().cpu(), expected.parameters[name],
                                           rtol=1e-4, atol=1e-6, msg=lambda message: f"step {step + 1}, {name}, parameter: {message}")
                torch.testing.assert_close(parameter.grad.cpu(), expected.gradients[name],
                                           rtol=1e-4, atol=1e-6, msg=lambda message: f"step {step + 1}, {name}, gradient: {message}")
                state = optimizer.state[parameter]
                assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
                for field, value in state.items():
                    torch.testing.assert_close(value.cpu(), expected.optimizer_state[name][field],
                                               rtol=1e-4, atol=1e-6, msg=lambda message: f"step {step + 1}, {name}, AdamW.{field}: {message}")


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
