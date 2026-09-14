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
