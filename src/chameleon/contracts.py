"""Immutable input contracts; no planning or recovery algorithms live here."""

import math
from dataclasses import dataclass
from typing import Literal


def _integer(name: str, value: int, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _finite(name: str, value: float, *, positive: bool = False) -> None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {bound}")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 128
    hidden_size: int = 32
    num_layers: int = 4
    num_heads: int = 4
    sequence_length: int = 16
    global_batch_size: int = 8
    micro_batch_size: int = 1
    seed: int = 42
    dropout: float = 0.0
    amsgrad: bool = False
    paper_path: str | None = None

    def __post_init__(self) -> None:
        for name in ("vocab_size", "hidden_size", "num_layers", "num_heads",
                     "sequence_length", "global_batch_size", "micro_batch_size"):
            _integer(name, getattr(self, name))
        _integer("seed", self.seed, 0)
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.micro_batch_size > self.global_batch_size:
            raise ValueError("micro_batch_size must not exceed global_batch_size")
        _finite("dropout", self.dropout)
        if self.dropout != 0:
            raise ValueError("dropout must be 0 for deterministic training")
        if self.amsgrad is not False:
            raise ValueError("AdamW amsgrad must be False")


@dataclass(frozen=True)
class WorkerIdentity:
    worker_id: str
    rank: int
    generation: int

    def __post_init__(self) -> None:
        if not isinstance(self.worker_id, str) or not self.worker_id.strip():
            raise ValueError("worker_id must be a nonempty string")
        _integer("rank", self.rank, 0)
        _integer("generation", self.generation, 0)


@dataclass(frozen=True)
class ClusterState:
    workers: tuple[WorkerIdentity, ...]
    global_batch_size: int
    generation: int = 0
    committed_global_step: int = 0

    def __post_init__(self) -> None:
        _integer("global_batch_size", self.global_batch_size)
        _integer("generation", self.generation, 0)
        _integer("committed_global_step", self.committed_global_step, 0)
        if not isinstance(self.workers, tuple) or not self.workers:
            raise ValueError("workers must be a nonempty tuple")
        if not all(isinstance(w, WorkerIdentity) for w in self.workers):
            raise ValueError("workers must contain WorkerIdentity values")
        if len({w.worker_id for w in self.workers}) != len(self.workers):
            raise ValueError("worker IDs must be unique")
        if len({w.rank for w in self.workers}) != len(self.workers):
            raise ValueError("worker ranks must be unique")
        if any(w.generation != self.generation for w in self.workers):
            raise ValueError("worker generation must match cluster generation")


@dataclass(frozen=True)
class ExecutionPlan:
    plan_id: str
    policy: Literal["rerouting", "dynamic"]
    global_batch_size: int
    generation: int
    estimated_step_time_s: float
    estimated_transition_time_s: float

    def __post_init__(self) -> None:
        if not isinstance(self.plan_id, str) or not self.plan_id.strip():
            raise ValueError("plan_id must be a nonempty string")
        if self.policy not in ("rerouting", "dynamic"):
            raise ValueError("policy must be rerouting or dynamic")
        _integer("global_batch_size", self.global_batch_size)
        _integer("generation", self.generation, 0)
        _finite("estimated_step_time_s", self.estimated_step_time_s, positive=True)
        _finite("estimated_transition_time_s", self.estimated_transition_time_s)


@dataclass(frozen=True)
class FailureEvent:
    failed_worker_ids: tuple[str, ...]
    generation: int
    committed_global_step: int

    def __post_init__(self) -> None:
        _integer("generation", self.generation, 0)
        _integer("committed_global_step", self.committed_global_step, 0)
        ids = self.failed_worker_ids
        if (not isinstance(ids, tuple) or not ids
                or any(not isinstance(i, str) or not i.strip() for i in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("failed_worker_ids must be nonempty and unique")


@dataclass(frozen=True)
class DecisionResult:
    plan: ExecutionPlan
    inter_fault_duration_s: float
    score: float

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ExecutionPlan):
            raise ValueError("plan must be an ExecutionPlan")
        _finite("inter_fault_duration_s", self.inter_fault_duration_s, positive=True)
        _finite("score", self.score, positive=True)
        if self.inter_fault_duration_s <= self.plan.estimated_transition_time_s:
            raise ValueError("plan is unavailable: D must exceed transition time")


class UnrecoverableStateError(RuntimeError):
    """No complete parameter/optimizer replica survives for a required module."""
