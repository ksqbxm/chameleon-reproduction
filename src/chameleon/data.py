"""Sample IDs and fixed-length next-token data without an RNG or data cursor."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .contracts import ClusterState, ModelConfig, _integer

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class SampleBatch:
    sample_ids: tuple[int, ...]
    inputs: "torch.Tensor"
    targets: "torch.Tensor"


def next_sample_ids(state: ClusterState) -> tuple[int, ...]:
    start = state.committed_global_step * state.global_batch_size
    return tuple(range(start, start + state.global_batch_size))


def make_batch(sample_ids: Iterable[int], config: ModelConfig, *,
               device: str = "cpu") -> SampleBatch:
    ids = tuple(sample_ids)
    if not ids:
        raise ValueError("sample_ids must not be empty")
    for sample_id in ids:
        _integer("sample_id", sample_id, 0)
    import torch

    tokens = torch.tensor([
        [(sample_id * (position + 1) + position * position + 3 * position + 1)
         % config.vocab_size for position in range(config.sequence_length + 1)]
        for sample_id in ids
    ], dtype=torch.long, device=device)
    return SampleBatch(ids, tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous())
