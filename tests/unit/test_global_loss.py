import math

import pytest

from chameleon import ClusterState, WorkerIdentity
from chameleon.global_loss import GlobalBatchAccounting


def accounting(size=10, step=0, *, ownership=None):
    state = ClusterState((WorkerIdentity("single-device", 0, 0),), size,
                        committed_global_step=step)
    return GlobalBatchAccounting(state, expected_owners=ownership if ownership is not None
                                 else {"owner0": {"weight"}})


@pytest.fixture
def torch_module():
    import torch
    return torch


def test_five_three_two_uses_global_sum_and_rejects_mean_of_means():
    ledger = accounting()
    pipeline_losses = ((1., 2., 3., 4., 5.), (11., 12., 13.), (21., 22.))
    sample_id = 0
    for losses in pipeline_losses:
        for loss in losses:
            ledger.add_micro_batch((sample_id,), loss, sample_count=1)
            sample_id += 1
    report = ledger.report()
    assert report.global_sample_count == 10
    assert report.loss_global_sum == 94
    assert report.loss_global_mean == 9.4
    wrong = sum(sum(losses) / len(losses) for losses in pipeline_losses) / 3
    assert wrong != pytest.approx(report.loss_global_mean)


def test_unequal_and_partial_micro_batches_count_samples():
    ledger = accounting(21, step=2)
    # Ten micro-batches, distributed [5, 3, 2], contain 21 samples.
    sizes = ((2, 1, 3, 2, 1), (4, 1, 2), (3, 2))
    sample_id = 42
    for pipeline in sizes:
        for size in pipeline:
            ids = tuple(range(sample_id, sample_id + size))
            ledger.add_micro_batch(ids, float(sum(ids)), sample_count=size)
            sample_id += size
    report = ledger.report()
    assert report.global_sample_count == 21
    assert report.loss_global_sum == sum(range(42, 63))
    assert report.loss_global_mean == 52


@pytest.mark.parametrize("ids,count,error", [
    ((0,), 2, "sample_count"), ((), 0, "sample_count"), ((0,), True, "sample_count"),
    ((0,), 1.0, "sample_count"), ((-1,), 1, "sample_id"), ((True,), 1, "sample_id"),
    ((0.5,), 1, "sample_id"), ((0, 0), 2, "duplicate"), ((10,), 1, "current global step"),
])
def test_invalid_batch_is_rejected_without_consuming_ids(ids, count, error):
    ledger = accounting()
    with pytest.raises(ValueError, match=error):
        ledger.add_micro_batch(ids, 1., sample_count=count)
    ledger.add_micro_batch(range(10), 10., sample_count=10)
    assert ledger.report().global_sample_count == 10


@pytest.mark.parametrize("loss", [-1., math.inf, math.nan, True])
def test_invalid_loss_sum_does_not_consume_samples(loss):
    ledger = accounting(1)
    with pytest.raises(ValueError, match="loss_sum"):
        ledger.add_micro_batch((0,), loss, sample_count=1)
    ledger.add_micro_batch((0,), 2., sample_count=1)
    assert ledger.report().loss_global_sum == 2


def test_missing_duplicate_and_wrong_step_ids_fail():
    ledger = accounting(3, step=1)
    ledger.add_micro_batch((3, 4), 2., sample_count=2)
    with pytest.raises(ValueError, match="missing"):
        ledger.report()
    with pytest.raises(ValueError, match="duplicate"):
        ledger.add_micro_batch((4, 5), 2., sample_count=2)
    for stale_or_future in (0, 6):
        with pytest.raises(ValueError, match="current global step"):
            ledger.add_micro_batch((stale_or_future,), 1., sample_count=1)
    ledger.add_micro_batch((5,), 1., sample_count=1)
    assert ledger.report().global_sample_count == 3


def test_normalization_requires_complete_sample_partition():
    ledger = accounting()
    ledger.add_micro_batch((0,), 1., sample_count=1)
    with pytest.raises(ValueError, match="missing"):
        ledger.sum_and_normalize_gradients_({})


def test_micro_batch_loss_is_sample_sum_with_masked_tokens(torch_module, device):
    from chameleon.global_loss import micro_batch_loss_sum
    torch = torch_module
    logits = torch.tensor([[[0., 0.], [0., math.log(3)], [0., 0.]],
                           [[0., math.log(3)], [0., 0.], [0., 0.]]],
                          dtype=torch.float64, device=device, requires_grad=True)
    targets = torch.tensor([[0, 1, -100], [0, -100, -100]], device=device)
    loss_sum = micro_batch_loss_sum(logits, targets)
    expected = (math.log(2) + math.log(4 / 3)) / 2 + math.log(4)
    tolerance = dict(rtol=1e-7, atol=1e-9) if device == "cuda" else dict(rtol=1e-8, atol=1e-10)
    assert loss_sum.item() == pytest.approx(expected, rel=tolerance["rtol"], abs=tolerance["atol"])
    nll = -logits.log_softmax(-1).gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    oracle = nll[0, :2].mean() + nll[1, 0]
    expected_gradient, = torch.autograd.grad(oracle, logits)
    loss_sum.backward()
    torch.testing.assert_close(logits.grad, expected_gradient, **tolerance)
    assert torch.count_nonzero(logits.grad[targets == -100]).item() == 0
    wrong_token_mean = (nll[0, :2].sum() + nll[1, 0]) / 3
    assert wrong_token_mean.item() != pytest.approx(loss_sum.item() / 2)
    with pytest.raises(ValueError, match="valid token"):
        micro_batch_loss_sum(logits, torch.full_like(targets, -100))


@pytest.mark.parametrize("logits_shape,targets_shape", [
    ((2, 3), (2, 3)), ((2, 3, 4), (2, 2)), ((0, 3, 4), (0, 3)),
    ((2, 0, 4), (2, 0)),
])
def test_loss_rejects_mismatched_or_empty_samples(torch_module, device, logits_shape, targets_shape):
    from chameleon.global_loss import micro_batch_loss_sum
    torch = torch_module
    with pytest.raises(ValueError, match="sample/token dimensions"):
        micro_batch_loss_sum(torch.zeros(logits_shape, device=device),
                             torch.zeros(targets_shape, dtype=torch.long, device=device))


def test_owner_sum_divides_once_and_includes_endpoint_names(torch_module, device):
    torch = torch_module
    tolerance = dict(rtol=1e-7, atol=1e-9) if device == "cuda" else dict(rtol=1e-8, atol=1e-10)
    ledger = accounting(ownership={
        "owner0": {"embedding.weight", "blocks.0.weight"},
        "owner1": {"embedding.weight", "final_norm.bias"},
        "owner2": {"embedding.weight", "lm_head.weight"},
    })
    ledger.add_micro_batch(range(10), 94., sample_count=10)
    owners = {
        "owner0": {"embedding.weight": torch.tensor([15., 30.], dtype=torch.float64, device=device),
                   "blocks.0.weight": torch.tensor([20.], dtype=torch.float64, device=device)},
        "owner1": {"embedding.weight": torch.tensor([36., 72.], dtype=torch.float64, device=device),
                   "final_norm.bias": torch.tensor([5.], dtype=torch.float64, device=device)},
        "owner2": {"embedding.weight": torch.tensor([43., 86.], dtype=torch.float64, device=device),
                   "lm_head.weight": torch.tensor([7.], dtype=torch.float64, device=device)},
    }
    report = ledger.sum_and_normalize_gradients_(owners)
    assert report == ledger.report()
    for owner in owners.values():
        torch.testing.assert_close(owner["embedding.weight"],
                                   torch.tensor([9.4, 18.8], dtype=torch.float64, device=device),
                                   **tolerance)
    assert owners["owner0"]["blocks.0.weight"].item() == pytest.approx(
        2., rel=tolerance["rtol"], abs=tolerance["atol"])
    assert owners["owner1"]["final_norm.bias"].item() == pytest.approx(
        0.5, rel=tolerance["rtol"], abs=tolerance["atol"])
    assert owners["owner2"]["lm_head.weight"].item() == pytest.approx(
        0.7, rel=tolerance["rtol"], abs=tolerance["atol"])
    before = owners["owner0"]["embedding.weight"].clone()
    with pytest.raises(ValueError, match="already normalized"):
        ledger.sum_and_normalize_gradients_(owners)
    with pytest.raises(ValueError, match="already normalized"):
        ledger.add_micro_batch((0,), 1., sample_count=1)
    torch.testing.assert_close(owners["owner0"]["embedding.weight"], before, rtol=0, atol=0)


@pytest.mark.parametrize("error_case,error", [
    ("empty", "owner IDs"), ("missing", "missing gradient"),
    ("duplicate", "overlapping"), ("shape", "mismatch"), ("dtype", "mismatch"),
    ("device", "mismatch"), ("integer", "real floating"),
])
def test_bad_owner_buffers_fail_before_mutating_gradients(torch_module, device, error_case, error):
    torch = torch_module
    ledger = accounting(1, ownership={"owner0": {"weight"}, "owner1": {"weight"}})
    ledger.add_micro_batch((0,), 1., sample_count=1)
    gradient = torch.tensor([2.], dtype=torch.float64, device=device)
    other = torch.tensor([3.], dtype=torch.float64, device=device)
    owners = {"owner0": {"weight": gradient}, "owner1": {"weight": other}}
    if error_case == "empty":
        owners = {}
    elif error_case == "missing":
        owners["owner1"]["weight"] = None
    elif error_case == "duplicate":
        owners["owner1"]["weight"] = gradient
    elif error_case == "shape":
        owners["owner1"]["weight"] = torch.zeros(2, dtype=torch.float64, device=device)
    elif error_case == "dtype":
        owners["owner1"]["weight"] = other.float()
    elif error_case == "device":
        owners["owner1"]["weight"] = torch.empty(1, dtype=torch.float64,
                                                 device="cpu" if device == "cuda" else "meta")
    elif error_case == "integer":
        owners["owner1"]["weight"] = other.long()
    with pytest.raises(ValueError, match=error):
        ledger.sum_and_normalize_gradients_(owners)
    assert gradient.item() == 2
    assert other.item() == 3
    # A rejected finalize can be retried with the original, valid SUM buffers.
    ledger.sum_and_normalize_gradients_({"owner0": {"weight": gradient}, "owner1": {"weight": other}})
    assert gradient.item() == other.item() == 5


@pytest.mark.parametrize("ownership,error", [
    ({}, "expected_owners"), ({"": {"weight"}}, "owner IDs"),
    ({0: {"weight"}}, "owner IDs"), ({"owner0": "weight"}, "parameter names"),
    ({"owner0": {""}}, "parameter names"), ({"owner0": {0}}, "parameter names"),
    ({"owner0": set()}, "trainable parameter"),
])
def test_invalid_expected_ownership_is_rejected(ownership, error):
    with pytest.raises(ValueError, match=error):
        accounting(ownership=ownership)


@pytest.mark.parametrize("owners,error", [
    ({}, "owner IDs"),
    ({"owner0": {"weight": None, "lm_head.weight": None}}, "owner IDs"),
    ({"owner0": {"weight": None, "lm_head.weight": None},
      "extra": {"weight": None, "lm_head.weight": None}}, "owner IDs"),
    ({"owner0": {"weight": None}, "owner1": {"weight": None, "lm_head.weight": None}}, "inventory"),
    ({"owner0": {}, "owner1": {"weight": None, "lm_head.weight": None}}, "inventory"),
    ({"owner0": {"weight": None, "lm_head.weight": None, "extra": None},
      "owner1": {"weight": None, "lm_head.weight": None}}, "inventory"),
    ({"owner0": {"weight": None, "lm_head.weight": None},
      "owner1": {"weight": None, "misspelled_head": None}}, "inventory"),
])
def test_omitted_extra_or_wrong_owner_contributions_fail_before_tensor_access(owners, error):
    ledger = accounting(1, ownership={"owner0": {"weight", "lm_head.weight"},
                                      "owner1": {"weight", "lm_head.weight"}})
    ledger.add_micro_batch((0,), 1., sample_count=1)
    # None values are never reached: the inventory must be validated first.
    with pytest.raises(ValueError, match=error):
        ledger.sum_and_normalize_gradients_(owners)
    assert ledger.report().global_sample_count == 1


def test_expected_ownership_is_an_immutable_step_snapshot():
    ownership = {"owner0": {"weight"}, "owner1": {"weight"}}
    ledger = accounting(1, ownership=ownership)
    ledger.add_micro_batch((0,), 1., sample_count=1)
    ownership["owner0"].add("extra")
    del ownership["owner1"]
    with pytest.raises(ValueError, match="owner IDs"):
        ledger.sum_and_normalize_gradients_({"owner0": {"weight": None, "extra": None}})
    with pytest.raises(ValueError, match="inventory"):
        ledger.sum_and_normalize_gradients_({"owner0": {"weight": None, "extra": None},
                                            "owner1": {"weight": None}})


@pytest.mark.parametrize("omission,error", [
    ("owner", "owner IDs"), ("one-head", "inventory"), ("all-heads", "inventory"),
    ("extra-parameter", "inventory"), ("wrong-name", "inventory"),
])
def test_incomplete_inventory_preserves_gradients_and_can_be_corrected(torch_module, device, omission, error):
    torch = torch_module
    ownership = {"owner0": {"embedding.weight", "lm_head.weight"},
                 "owner1": {"embedding.weight", "lm_head.weight"}}
    ledger = accounting(2, ownership=ownership)
    ledger.add_micro_batch((0, 1), 2., sample_count=2)
    complete = {
        "owner0": {"embedding.weight": torch.tensor([2.], dtype=torch.float64, device=device),
                   "lm_head.weight": torch.tensor([4.], dtype=torch.float64, device=device)},
        "owner1": {"embedding.weight": torch.tensor([6.], dtype=torch.float64, device=device),
                   "lm_head.weight": torch.tensor([8.], dtype=torch.float64, device=device)},
    }
    supplied = {owner: dict(gradients) for owner, gradients in complete.items()}
    before = {owner: {name: gradient.clone() for name, gradient in gradients.items()}
              for owner, gradients in complete.items()}
    if omission == "owner":
        del supplied["owner1"]
    elif omission == "one-head":
        del supplied["owner1"]["lm_head.weight"]
    elif omission == "all-heads":
        for gradients in supplied.values():
            del gradients["lm_head.weight"]
    elif omission == "extra-parameter":
        supplied["owner1"]["extra"] = torch.zeros(1, dtype=torch.float64, device=device)
    else:
        supplied["owner1"]["misspelled_head"] = supplied["owner1"].pop("lm_head.weight")
    with pytest.raises(ValueError, match=error):
        ledger.sum_and_normalize_gradients_(supplied)
    for owner, gradients in complete.items():
        for name, gradient in gradients.items():
            torch.testing.assert_close(gradient, before[owner][name], rtol=0, atol=0)
    ledger.sum_and_normalize_gradients_(complete)
    for gradients in complete.values():
        assert gradients["embedding.weight"].item() == 4
        assert gradients["lm_head.weight"].item() == 6


@pytest.mark.parametrize("alias", ["same-object", "detach", "view", "partial", "transpose",
                                  "different-name", "different-dtype"])
def test_storage_aliases_are_rejected_without_mutation(torch_module, device, alias):
    torch = torch_module
    base = torch.arange(8, dtype=torch.float64, device=device)
    first = base[:4].reshape(2, 2)
    second = first
    second_name = "weight"
    if alias == "detach":
        second = first.detach()
    elif alias == "view":
        second = first.view_as(first)
    elif alias == "partial":
        second = base[2:6].reshape(2, 2)
    elif alias == "transpose":
        second = first.t()
    elif alias == "different-name":
        second, second_name = first.detach(), "lm_head.weight"
    elif alias == "different-dtype":
        second, second_name = first.view(torch.float32), "lm_head.weight"
    if alias != "same-object":
        assert first is not second
    ledger = accounting(2, ownership={"owner0": {"weight"}, "owner1": {second_name}})
    ledger.add_micro_batch((0, 1), 2., sample_count=2)
    owners = {"owner0": {"weight": first}, "owner1": {second_name: second}}
    before = base.clone()
    with pytest.raises(ValueError, match="overlapping"):
        ledger.sum_and_normalize_gradients_(owners)
    torch.testing.assert_close(base, before, rtol=0, atol=0)
    # A rejected finalize has not marked the step normalized.
    with pytest.raises(ValueError, match="overlapping"):
        ledger.sum_and_normalize_gradients_(owners)


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "permute", "singleton", "scalar", "empty"])
def test_disjoint_dense_views_are_valid_owner_buffers(torch_module, device, layout):
    torch = torch_module
    base = torch.arange(16, dtype=torch.float64, device=device)
    first, second = base[:8].reshape(2, 4), base[8:].reshape(2, 4)
    if layout == "transpose":
        first, second = first.t(), second.t()
    elif layout == "permute":
        first = first.reshape(2, 2, 2).permute(2, 0, 1)
        second = second.reshape(2, 2, 2).permute(2, 0, 1)
    elif layout == "singleton":
        first = first.as_strided((1, 2, 4), (1000, 4, 1))
        second = second.as_strided((1, 2, 4), (1000, 4, 1))
    elif layout == "scalar":
        first, second = base[0], base[1]
    elif layout == "empty":
        first, second = base[:0], base[:0]
    expected = (first.clone() + second.clone()) / 2
    ledger = accounting(2, ownership={"owner0": {"weight"}, "owner1": {"weight"}})
    ledger.add_micro_batch((0, 1), 2., sample_count=2)
    ledger.sum_and_normalize_gradients_({"owner0": {"weight": first}, "owner1": {"weight": second}})
    tolerance = dict(rtol=1e-7, atol=1e-9) if device == "cuda" else dict(rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(first, expected, **tolerance)
    torch.testing.assert_close(second, expected, **tolerance)


@pytest.mark.parametrize("layout", ["holes", "internal-overlap", "sparse"])
def test_unsupported_layout_fails_before_any_parameter_is_normalized(torch_module, device, layout):
    torch = torch_module
    good = torch.tensor([8.], dtype=torch.float64, device=device)
    base = torch.arange(12, dtype=torch.float64, device=device).reshape(3, 4)
    if layout == "holes":
        bad = base[:, ::2]
    elif layout == "internal-overlap":
        bad = base[:1].expand(3, 4)
    else:
        bad = base.to_sparse()
    ledger = accounting(2, ownership={"owner0": {"embedding.weight", "lm_head.weight"}})
    ledger.add_micro_batch((0, 1), 2., sample_count=2)
    with pytest.raises(ValueError, match="dense"):
        ledger.sum_and_normalize_gradients_({"owner0": {"embedding.weight": good, "lm_head.weight": bad}})
    assert good.item() == 8
    torch.testing.assert_close(base, torch.arange(12, dtype=torch.float64, device=device).reshape(3, 4),
                               rtol=0, atol=0)


def test_separate_parameters_in_shared_storage_and_empty_owner_are_supported(torch_module, device):
    torch = torch_module
    base = torch.arange(8, dtype=torch.float64, device=device)
    first, second = base[:4], base[4:]
    before = base.clone()
    ledger = accounting(2, ownership={"owner0": {"embedding.weight", "lm_head.weight"}, "owner1": set()})
    ledger.add_micro_batch((0, 1), 2., sample_count=2)
    ledger.sum_and_normalize_gradients_({"owner0": {"embedding.weight": first, "lm_head.weight": second},
                                        "owner1": {}})
    torch.testing.assert_close(base, before / 2, rtol=0, atol=0)
