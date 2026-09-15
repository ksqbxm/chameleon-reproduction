"""Paper Eq. 9--14 estimates, with explicit batch scope and derivations.

Time estimates cover pipeline computation. Unmeasured P2P, gradient collectives,
optimizer updates and control overhead are not invented or equated to zero.
"""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass

from .contracts import _finite, _integer, stable_hash
from .profiler import validate_snapshot
from .schedule import OperationKey, build_1f1b_schedule


@dataclass(frozen=True)
class TimeEstimate:
    step_time_s: float | None
    derivation: dict
    trace: tuple[dict, ...] = ()
    reasons: tuple[str, ...] = ()

    @property
    def feasible(self) -> bool:
        return not self.reasons


@dataclass(frozen=True)
class MemoryEstimate:
    stages: tuple[dict, ...]
    derivation: dict
    reasons: tuple[str, ...]

    @property
    def feasible(self) -> bool:
        return not self.reasons


def _symmetric_batch(global_micro_batches, dp_size):
    _integer("global_micro_batches", global_micro_batches)
    _integer("dp_size", dp_size)
    if global_micro_batches % dp_size:
        raise ValueError("symmetric estimation requires equal integer micro-batches per pipeline")
    return global_micro_batches // dp_size


def estimate_symmetric_time(*, num_stages: int, global_micro_batches: int,
                            dp_size: int, forward_s: float, backward_s: float) -> TimeEstimate:
    """Eq. 9: uniform stage durations and equal data partitions."""
    _integer("num_stages", num_stages)
    nm = _symmetric_batch(global_micro_batches, dp_size)
    _finite("forward_s", forward_s, positive=True)
    _finite("backward_s", backward_s, positive=True)
    slots = num_stages + nm - 1
    time_s = slots * (forward_s + backward_s)
    _finite("estimated time", time_s, positive=True)
    return TimeEstimate(time_s, {"equation": 9, "num_stages": num_stages,
                                "global_micro_batches": global_micro_batches,
                                "pipeline_micro_batches": (nm,) * dp_size,
                                "forward_s": forward_s, "backward_s": backward_s,
                                "slots": slots, "model": "uniform-stage paper approximation"})


def estimate_operation_time(durations_s: dict[OperationKey, float], *, num_stages: int,
                            pipeline_micro_batches: int, pipeline: int = 0) -> TimeEstimate:
    """Eq. 11 with canonical dependencies and independently supplied operation durations."""
    queues = build_1f1b_schedule(num_stages, pipeline_micro_batches, pipeline=pipeline)
    return _estimate_operation_time(queues, durations_s)


def _estimate_operation_time(queues, durations_s):
    operations = {op.key: op for queue in queues for op in queue}
    if durations_s.keys() != operations.keys():
        raise ValueError("durations must cover exactly every scheduled operation")
    children = {key: [] for key in operations}
    pending = {}
    for key, op in operations.items():
        _finite("operation duration", durations_s[key], positive=True)
        pending[key] = len(op.dependencies)
        for dependency in op.dependencies:
            children[dependency].append(key)
    ready = deque(key for key in operations if pending[key] == 0)
    ends, trace = {}, []
    while ready:
        key = ready.popleft()
        op = operations[key]
        start = max((ends[parent] for parent in op.dependencies), default=0.)
        end = start + durations_s[key]
        _finite("operation end", end, positive=True)
        ends[key] = end
        trace.append({"pipeline": op.pipeline, "stage": op.stage, "micro_batch": op.micro_batch,
                      "kind": op.kind, "phase": op.phase, "dependencies": op.dependencies,
                      "start_s": start, "end_s": end, "duration_s": durations_s[key]})
        for child in children[key]:
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    return TimeEstimate(max(ends.values()), {"equation": 11,
                        "pipeline": queues[0][0].pipeline, "num_stages": len(queues),
                        "pipeline_micro_batches": len(queues[0]) // 2,
                        "recurrence": "start = max(predecessor ends); end = start + duration",
                        "model": "1F1B computation dependency estimate"}, tuple(trace))


def estimate_pipeline_time(stage_forward_s: tuple[float, ...], stage_backward_s: tuple[float, ...],
                           pipeline_micro_batches: int, *, pipeline: int = 0) -> TimeEstimate:
    if not stage_forward_s or len(stage_forward_s) != len(stage_backward_s):
        raise ValueError("forward/backward durations must cover the same nonempty stages")
    queues = build_1f1b_schedule(len(stage_forward_s), pipeline_micro_batches, pipeline=pipeline)
    durations = {op.key: (stage_forward_s if op.kind == "forward" else stage_backward_s)[op.stage]
                 for queue in queues for op in queue}
    estimate = _estimate_operation_time(queues, durations)
    return TimeEstimate(estimate.step_time_s, dict(estimate.derivation,
                        stage_forward_s=stage_forward_s, stage_backward_s=stage_backward_s), estimate.trace)


def estimate_rerouting_time(*, global_micro_batches: int, dp_size: int,
                            failures_per_stage: tuple[int, ...], forward_s: float,
                            backward_s: float) -> TimeEstimate:
    """Eq. 12 for one failure, Eq. 13 otherwise; layout and DP degree stay unchanged."""
    if not failures_per_stage:
        raise ValueError("failures_per_stage must cover a nonempty pipeline")
    nm = _symmetric_batch(global_micro_batches, dp_size)
    _finite("forward_s", forward_s, positive=True)
    _finite("backward_s", backward_s, positive=True)
    for value in failures_per_stage:
        _integer("stage failures", value, 0)
    inputs = {"equation": 12 if sum(failures_per_stage) == 1 else 13,
              "global_micro_batches": global_micro_batches,
              "pipeline_micro_batches": (nm,) * dp_size, "dp_size": dp_size,
              "failures_per_stage": failures_per_stage, "forward_s": forward_s,
              "backward_s": backward_s, "model": "uniform-stage, evenly rerouted paper approximation"}
    reasons = tuple(f"stage {i}: Fi={fi} >= Ndp={dp_size}; no healthy stage peer"
                    for i, fi in enumerate(failures_per_stage) if fi >= dp_size)
    if reasons:
        return TimeEstimate(None, inputs, reasons=reasons)
    extra_slots = tuple(nm * fi / (dp_size - fi) for fi in failures_per_stage)
    slots = len(failures_per_stage) + nm - 1 + sum(extra_slots)
    time_s = slots * (forward_s + backward_s)
    _finite("estimated time", time_s, positive=True)
    return TimeEstimate(time_s, dict(inputs, extra_slots_per_stage=extra_slots, slots=slots))


def estimate_stage_memory(layer_counts: tuple[int, ...], *, average_parameter_bytes: float,
                          average_optimizer_bytes: float,
                          average_activation_bytes: float, extra_static_bytes: tuple[float, ...],
                          extra_activation_bytes: tuple[float, ...], capacities_bytes: tuple[int, ...],
                          pipeline: int = 0) -> MemoryEstimate:
    """Eq. 14 plus measured non-block costs. The paper's Npp-i factor is not capped by Nm."""
    _integer("pipeline", pipeline, 0)
    pp = len(layer_counts)
    if not pp or any(len(values) != pp for values in
                     (extra_static_bytes, extra_activation_bytes, capacities_bytes)):
        raise ValueError("memory inputs must cover the same nonempty stages")
    for name, value in (("parameter", average_parameter_bytes), ("optimizer", average_optimizer_bytes),
                        ("activation", average_activation_bytes)):
        _finite(f"average {name} bytes", value)
    rows, reasons = [], []
    for stage, count in enumerate(layer_counts):
        _integer("layer count", count, 0)
        _finite("extra static bytes", extra_static_bytes[stage])
        _finite("extra activation bytes", extra_activation_bytes[stage])
        _integer("capacity bytes", capacities_bytes[stage], 0)
        static = count * (2 * average_parameter_bytes + average_optimizer_bytes)
        activation_slots = pp - stage
        dynamic = activation_slots * count * average_activation_bytes
        extra = extra_static_bytes[stage] + activation_slots * extra_activation_bytes[stage]
        peak = static + dynamic + extra
        _finite("estimated peak bytes", peak)
        reason = (f"pipeline {pipeline} stage {stage}: estimated {peak:g} bytes > "
                  f"capacity {capacities_bytes[stage]} bytes (layers={count}, static={static:g}, "
                  f"dynamic={dynamic:g}, extra={extra:g})") if peak > capacities_bytes[stage] else None
        if reason:
            reasons.append(reason)
        rows.append({"pipeline": pipeline, "stage": stage, "layer_count": count,
                     "static_layer_bytes": static, "dynamic_layer_bytes": dynamic,
                     "activation_slots": activation_slots, "extra_bytes": extra,
                     "peak_bytes": peak, "capacity_bytes": capacities_bytes[stage], "oom_reason": reason})
    return MemoryEstimate(tuple(rows), {"equation": 14, "average_parameter_bytes": average_parameter_bytes,
                          "average_optimizer_bytes": average_optimizer_bytes,
                          "average_gradient_bytes": average_parameter_bytes,
                          "average_activation_bytes": average_activation_bytes,
                          "model": "average-layer approximation plus measured endpoint tensor bytes"}, tuple(reasons))


class Estimator:
    """Consume the sole versioned Profiler format, never training or recovery tensors."""

    def __init__(self, profile: dict, *, expected_identity: dict, layer_modules: tuple[str, ...]):
        validate_snapshot(profile, expected_identity)
        if not profile["steps"]:
            raise ValueError("estimation requires measured profile steps")
        modules = profile["identity"]["module_parameter_bytes"]
        if (not layer_modules or any(name not in modules for name in layer_modules)
                or len(set(layer_modules)) != len(layer_modules)):
            raise ValueError("layer_modules must name unique measured model layers")
        self.profile = deepcopy(profile)
        self.layer_modules = tuple(layer_modules)

    def _layout(self, stage_modules):
        if not stage_modules or any(not stage for stage in stage_modules):
            raise ValueError("stage layout must contain nonempty stages")
        flattened = tuple(name for stage in stage_modules for name in stage)
        if flattened != tuple(self.profile["identity"]["module_order"]):
            raise ValueError("each pipeline layout must contain every model module exactly once in model order")

    def stage_durations(self, stage_modules) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Sum measured module EMAs, including all endpoint modules, per micro-batch."""
        self._layout(stage_modules)
        metrics = self.profile["metrics"]
        return tuple(tuple(sum(metrics[f"modules.{module}.{kind}_s"]["ema"] for module in stage)
                           for stage in stage_modules) for kind in ("forward", "backward"))

    def dynamic_time(self, layouts, pipeline_micro_batches: tuple[int, ...], *,
                     global_micro_batches: int) -> TimeEstimate:
        """Eq. 10 selects the slowest pipeline; Eq. 11 computes each pipeline."""
        _integer("global_micro_batches", global_micro_batches)
        if not layouts or len(layouts) != len(pipeline_micro_batches):
            raise ValueError("layouts and micro-batch partitions must cover the same pipelines")
        layouts = tuple(tuple(tuple(stage) for stage in layout) for layout in layouts)
        pipeline_micro_batches = tuple(pipeline_micro_batches)
        for count in pipeline_micro_batches:
            _integer("pipeline micro-batches", count)
        if sum(pipeline_micro_batches) != global_micro_batches:
            raise ValueError("pipeline micro-batches must sum to global_micro_batches")
        estimates = []
        for pipeline, (layout, count) in enumerate(zip(layouts, pipeline_micro_batches)):
            forward, backward = self.stage_durations(layout)
            estimates.append(estimate_pipeline_time(forward, backward, count, pipeline=pipeline))
        return TimeEstimate(max(e.step_time_s for e in estimates), {
            "equation": 10, "pipeline_equation": 11, "global_micro_batches": global_micro_batches,
            "pipeline_micro_batches": pipeline_micro_batches,
            "layouts": layouts,
            "profile_hash": stable_hash(self.profile),
            "pipeline_times_s": tuple(e.step_time_s for e in estimates),
            "pipelines": tuple(e.derivation for e in estimates),
            "profile_identity": deepcopy(self.profile["identity"]),
            "duration_source": "measured module forward/backward EMA per micro-batch",
            "boundary": "computation estimate; excludes unmeasured communication, optimizer and control overhead",
        }, tuple(row for estimate in estimates for row in estimate.trace))

    def memory(self, stage_modules, capacities_bytes: tuple[int, ...], *, pipeline: int = 0) -> MemoryEstimate:
        self._layout(stage_modules)
        step = self.profile["steps"][-1]
        memory = step["memory"]
        count = sum(row["kind"] == "forward" for row in step["trace"])
        layers = set(self.layer_modules)
        sizes = memory["modules"]
        averages = {"average_parameter_bytes": sum(sizes[name]["parameter_bytes"] for name in layers) / len(layers)}
        averages["average_optimizer_bytes"] = sum(sizes[name]["adamw_bytes"] for name in layers) / len(layers)
        activation = {name: max(self.profile["metrics"][f"modules.{name}.saved_activation_bytes"]["samples"])
                      for name in (*sizes, "loss_and_runtime")}
        averages["average_activation_bytes"] = sum(activation[name] for name in layers) / len(layers)
        extras_static = tuple(sum(2 * sizes[name]["parameter_bytes"] + sizes[name]["adamw_bytes"]
                                  for name in stage if name not in layers)
                              for stage in stage_modules)
        extras_activation = [sum(activation[name] for name in stage if name not in layers) for stage in stage_modules]
        # Loss/runtime saved tensors belong to the output stage and cannot disappear from the estimate.
        extras_activation[-1] += activation["loss_and_runtime"]
        estimate = estimate_stage_memory(tuple(sum(name in layers for name in stage) for stage in stage_modules),
                                        **averages, extra_static_bytes=extras_static,
                                        extra_activation_bytes=tuple(extras_activation),
                                        capacities_bytes=capacities_bytes, pipeline=pipeline)
        return MemoryEstimate(estimate.stages, dict(estimate.derivation, profile_step_id=step["step_id"],
                              source_kind=memory["kind"], measured_micro_batches=count,
                              extra_static_bytes=extras_static, extra_activation_bytes=tuple(extras_activation),
                              activation_source="maximum measured per-forward saved-tensor bytes",
                              boundary="logical saved-tensor bytes per occurrence; not measured physical peak HBM"),
                              estimate.reasons)
