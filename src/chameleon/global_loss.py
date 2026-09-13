"""Shared sample accounting and single-device owner SUM for both policies."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from .contracts import ClusterState, _finite, _integer
from .data import next_sample_ids

if TYPE_CHECKING:
    import torch


def micro_batch_loss_sum(logits: "torch.Tensor", targets: "torch.Tensor") -> "torch.Tensor":
    """Return a differentiable SUM of per-sample valid-token means."""
    import torch
    from torch.nn import functional as F

    if (logits.ndim != 3 or targets.ndim != 2 or logits.shape[:2] != targets.shape
            or not targets.shape[0] or not targets.shape[1]):
        raise ValueError("logits and targets must have matching nonempty sample/token dimensions")
    counts = targets.ne(-100).sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("each sample must have at least one valid token")
    token_losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none",
                                  ignore_index=-100)
    return (token_losses.sum(dim=1) / counts).sum()


@dataclass(frozen=True)
class GlobalLossReport:
    loss_global_sum: float
    global_sample_count: int

    @property
    def loss_global_mean(self) -> float:
        return self.loss_global_sum / self.global_sample_count


class GlobalBatchAccounting:
    """One global step; record each logical sample once, including rerouted work.

    Owner buffers already contain local gradient SUMs from all their logical
    tasks. The single-device SUM below is the Task 03 mathematical gate;
    distributed owner AllReduce belongs to the later runtime tasks.
    """

    def __init__(self, state: ClusterState, *,
                 expected_owners: Mapping[str, set[str] | frozenset[str]]):
        if not expected_owners:
            raise ValueError("expected_owners must contain physical owner IDs and parameter names")
        for owner_id, names in expected_owners.items():
            if not isinstance(owner_id, str) or not owner_id.strip():
                raise ValueError("expected owner IDs must be nonempty strings")
            if (not isinstance(names, (set, frozenset))
                    or any(not isinstance(name, str) or not name.strip() for name in names)):
                raise ValueError("expected parameter names must be a set of nonempty strings")
        self._expected_owners = {owner: frozenset(names) for owner, names in expected_owners.items()}
        if not any(self._expected_owners.values()):
            raise ValueError("expected_owners must include at least one trainable parameter")
        self._expected_ids = set(next_sample_ids(state))
        self._seen_ids: set[int] = set()
        self._loss_sums: list[float] = []
        self._normalized = False

    def add_micro_batch(self, sample_ids: Iterable[int], loss_sum: float, *,
                       sample_count: int) -> None:
        if self._normalized:
            raise ValueError("global step gradients are already normalized")
        ids = tuple(sample_ids)
        _integer("sample_count", sample_count)
        if sample_count != len(ids):
            raise ValueError("sample_count must match the actual sample IDs")
        for sample_id in ids:
            _integer("sample_id", sample_id, 0)
        unique = set(ids)
        if len(unique) != len(ids) or unique & self._seen_ids:
            raise ValueError("duplicate sample ID in the global partition")
        if not unique <= self._expected_ids:
            raise ValueError("sample IDs must belong to the current global step")
        _finite("loss_sum", loss_sum)
        self._seen_ids.update(unique)
        self._loss_sums.append(loss_sum)

    def report(self) -> GlobalLossReport:
        if self._seen_ids != self._expected_ids:
            raise ValueError("missing sample IDs from the global partition")
        return GlobalLossReport(math.fsum(self._loss_sums), len(self._seen_ids))

    def sum_and_normalize_gradients_(self, owners: Mapping[str, Mapping[str, "torch.Tensor"]]
                                     ) -> GlobalLossReport:
        """SUM same-name owner buffers, divide once by samples, copy to all owners.

        Owner IDs and parameter names must exactly match the declared inventory.
        A rerouting peer passes its already accumulated buffer once. Buffers use
        dense, non-overlapping strided layouts, including transposes/permutes;
        disjoint dense views of a shared allocation are allowed.
        """
        if self._normalized:
            raise ValueError("global step gradients are already normalized")
        report = self.report()
        if owners.keys() != self._expected_owners.keys():
            raise ValueError("owner IDs must exactly match the expected physical owners")
        for owner_id, names in self._expected_owners.items():
            if owners[owner_id].keys() != names:
                raise ValueError(f"gradient names must match the expected inventory for {owner_id}")
        import torch

        by_name: dict[str, list[torch.Tensor]] = {}
        spans: dict[torch.device, list[tuple[int, int]]] = {}
        for owner in owners.values():
            for name, gradient in owner.items():
                if not isinstance(gradient, torch.Tensor):
                    raise ValueError(f"missing gradient tensor for {name}")
                if gradient.layout != torch.strided or not gradient.is_floating_point():
                    raise ValueError("owner gradients must be dense real floating tensors")
                group = by_name.setdefault(name, [])
                if group and (gradient.shape != group[0].shape or gradient.dtype != group[0].dtype
                              or gradient.device != group[0].device):
                    raise ValueError(f"owner gradient shape/dtype/device mismatch for {name}")
                group.append(gradient)
                if gradient.numel():
                    stride_expected = 1
                    for stride, size in sorted((stride, size) for size, stride
                                               in zip(gradient.shape, gradient.stride()) if size > 1):
                        if stride != stride_expected:
                            raise ValueError("owner gradient buffers must have non-overlapping dense layouts")
                        stride_expected *= size
                    start = gradient.data_ptr()
                    spans.setdefault(gradient.device, []).append(
                        (start, start + gradient.numel() * gradient.element_size()))
        for ranges in spans.values():
            ranges.sort()
            if any(left[1] > right[0] for left, right in zip(ranges, ranges[1:])):
                raise ValueError("overlapping owner gradient buffers")
        with torch.no_grad():
            for group in by_name.values():
                total = group[0].detach().clone()
                for gradient in group[1:]:
                    total.add_(gradient.detach())
                total.div_(report.global_sample_count)
                for gradient in group:
                    gradient.copy_(total)
        self._normalized = True
        return report
