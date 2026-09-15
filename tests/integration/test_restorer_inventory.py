"""Additional live PyTorch inventory checks, independent of transport planning."""

import pytest

from chameleon.contracts import ModelConfig, UnrecoverableStateError
from chameleon.state_sources import ADAMW_FIELDS, adamw_inventory


@pytest.fixture
def live_model(device):
    import torch
    from chameleon.model import build_initial_model

    model = build_initial_model(ModelConfig(vocab_size=7, hidden_size=4, num_heads=1, num_layers=1),
                                device=device)
    model.extra = torch.nn.Linear(4, 4, device=device, dtype=torch.float64)
    model.frozen = torch.nn.Parameter(torch.ones(1, device=device), requires_grad=False)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), amsgrad=False)
    for _ in range(3):
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    for parameter in optimizer.param_groups[0]["params"]:
        state = optimizer.state[parameter]
        assert set(state) == set(ADAMW_FIELDS[1:])
        assert state["step"].item() == 3
    return torch, model, optimizer


def test_live_inventory_exact_bytes_complete_endpoints_and_added_parameters(live_model):
    _, model, optimizer = live_model
    before = {p: {k: t.clone() for k, t in state.items()} for p, state in optimizer.state.items()}
    inventory = adamw_inventory(model, optimizer, committed_global_step=3)
    expected = {(name, kind) for name, p in model.named_parameters() if p.requires_grad for kind in ADAMW_FIELDS}
    assert {t.key for t in inventory} == expected
    assert len(inventory) == len(expected)
    assert {t.module_id for t in inventory} == {"embedding", "blocks.0", "final_norm", "lm_head", "extra"}
    for row in inventory:
        parameter = dict(model.named_parameters())[row.parameter_name]
        value = parameter if row.kind == "parameter" else optimizer.state[parameter][row.kind]
        assert (row.nbytes, row.shape, row.dtype) == (value.numel() * value.element_size(), tuple(value.shape), str(value.dtype))
    for parameter, state in before.items():
        for key, tensor in state.items():
            assert tensor.equal(optimizer.state[parameter][key])


def test_rebuild_stage_preserves_live_adamw_group_and_state(live_model):
    from copy import deepcopy
    from chameleon.model import PipelineStage
    from chameleon.recovery import live_tensors, rebuild_stage, tensor_digest

    torch, model, optimizer = live_model
    modules = ("embedding", "blocks.0", "final_norm", "lm_head", "extra")
    structure = deepcopy(model).to(device="meta")
    old_model = PipelineStage(model, modules)
    old_group = optimizer.param_groups[0]
    old_group["lr"] = .004
    assert old_group["lr"] != optimizer.defaults["lr"]
    assert old_group["decoupled_weight_decay"] is True

    baseline_model = deepcopy(old_model)
    baseline_optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=.004,
                                           weight_decay=old_group["weight_decay"], amsgrad=False)
    baseline_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))

    source_tensors = live_tensors(old_model, optimizer)
    source_hashes = {key: tensor_digest(value) for key, value in source_tensors.items()}
    targets = dict(source_tensors)
    migrated_name = "lm_head.weight"
    for kind in ADAMW_FIELDS:
        targets[migrated_name, kind] = source_tensors[migrated_name, kind].detach().clone()

    recovered_model, recovered_optimizer = rebuild_stage(
        structure, modules, old_model, optimizer, targets)
    recovered_parameters = dict(recovered_model.named_parameters())
    old_parameters = dict(old_model.named_parameters())
    owned = recovered_optimizer.param_groups[0]["params"]
    expected_parameters = list(recovered_parameters.values())
    assert len(owned) == len(expected_parameters)
    assert all(actual is expected for actual, expected in zip(owned, expected_parameters))
    assert len(owned) == len(set(owned))
    assert {key: value for key, value in recovered_optimizer.param_groups[0].items() if key != "params"} == {
        key: value for key, value in old_group.items() if key != "params"
    }

    for name, parameter in recovered_parameters.items():
        if name == migrated_name:
            assert parameter is not old_parameters[name]
            assert parameter.data_ptr() == targets[name, "parameter"].data_ptr()
        else:
            assert parameter is targets[name, "parameter"]
        for kind in ADAMW_FIELDS[1:]:
            assert recovered_optimizer.state[parameter][kind] is targets[name, kind]
    assert {key: tensor_digest(value) for key, value in source_tensors.items()} == source_hashes
    adamw_inventory(recovered_model, recovered_optimizer, committed_global_step=3)

    for current in (recovered_model, baseline_model):
        for parameter in current.parameters():
            parameter.grad = torch.ones_like(parameter)
    recovered_optimizer.step()
    baseline_optimizer.step()
    tolerance = (dict(rtol=1e-7, atol=1e-9) if next(recovered_model.parameters()).is_cuda
                 else dict(rtol=1e-8, atol=1e-10))
    for (name, actual), (expected_name, expected) in zip(
            recovered_model.named_parameters(), baseline_model.named_parameters()):
        assert name == expected_name
        torch.testing.assert_close(actual, expected, **tolerance)
        for kind in ADAMW_FIELDS[1:]:
            torch.testing.assert_close(recovered_optimizer.state[actual][kind],
                                       baseline_optimizer.state[expected][kind], **tolerance)
    adamw_inventory(recovered_model, recovered_optimizer, committed_global_step=4)


@pytest.mark.parametrize("module", ("embedding", "final_norm", "lm_head", "extra"))
@pytest.mark.parametrize("kind", ADAMW_FIELDS[1:])
def test_live_missing_optimizer_tensor_fails(live_model, module, kind):
    _, model, optimizer = live_model
    parameter = next(p for name, p in model.named_parameters() if name.startswith(f"{module}."))
    del optimizer.state[parameter][kind]
    with pytest.raises(UnrecoverableStateError, match=module):
        adamw_inventory(model, optimizer, committed_global_step=3)


@pytest.mark.parametrize("change", ("amsgrad", "missing_owner", "shape", "step", "stale_step", "dtype"))
def test_live_inventory_rejects_invalid_state(live_model, change):
    torch, model, optimizer = live_model
    parameter = next(p for p in model.parameters() if p.requires_grad)
    if change == "amsgrad":
        optimizer.param_groups[0]["amsgrad"] = True
    elif change == "missing_owner":
        group = optimizer.param_groups[0]
        group["params"] = [p for p in group["params"] if p is not parameter]
    elif change == "shape":
        optimizer.state[parameter]["exp_avg"] = parameter.new_zeros(1)
    elif change == "stale_step":
        optimizer.state[parameter]["step"].fill_(2)
    elif change == "dtype":
        optimizer.state[parameter]["exp_avg"] = optimizer.state[parameter]["exp_avg"].to(dtype=torch.float32)
    else:
        optimizer.state[parameter]["step"].fill_(0)
    with pytest.raises((ValueError, UnrecoverableStateError)):
        adamw_inventory(model, optimizer, committed_global_step=3)
