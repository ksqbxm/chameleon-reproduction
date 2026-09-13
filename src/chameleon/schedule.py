"""Canonical FIFO 1F1B operations shared by measurement, estimation and runtime."""

from dataclasses import dataclass

from .contracts import _integer


OperationKey = tuple[int, int, int, str]


@dataclass(frozen=True)
class Operation:
    pipeline: int
    stage: int
    micro_batch: int
    kind: str
    phase: str
    dependencies: tuple[OperationKey, ...]

    @property
    def key(self) -> OperationKey:
        return self.pipeline, self.stage, self.micro_batch, self.kind


def build_1f1b_schedule(num_stages: int, pipeline_micro_batches: int, *,
                        pipeline: int = 0) -> tuple[tuple[Operation, ...], ...]:
    """One ordered queue per stage; Nm here belongs to this pipeline, not global DP."""
    _integer("num_stages", num_stages)
    _integer("pipeline_micro_batches", pipeline_micro_batches)
    _integer("pipeline", pipeline, 0)
    queues = []
    for stage in range(num_stages):
        warmup = min(num_stages - stage - 1, pipeline_micro_batches)
        order = [("forward", mb, "warmup") for mb in range(warmup)]
        for mb in range(pipeline_micro_batches - warmup):
            order.extend((("forward", mb + warmup, "steady"), ("backward", mb, "steady")))
        order.extend(("backward", mb, "cooldown")
                     for mb in range(pipeline_micro_batches - warmup, pipeline_micro_batches))
        operations = []
        for kind, mb, phase in order:
            dependencies = [operations[-1].key] if operations else []
            if kind == "forward" and stage > 0:
                dependencies.append((pipeline, stage - 1, mb, "forward"))
            if kind == "backward":
                dependencies.append((pipeline, stage, mb, "forward"))
                if stage + 1 < num_stages:
                    dependencies.append((pipeline, stage + 1, mb, "backward"))
            operations.append(Operation(pipeline, stage, mb, kind, phase,
                                        tuple(dict.fromkeys(dependencies))))
        queues.append(tuple(operations))
    return tuple(queues)
