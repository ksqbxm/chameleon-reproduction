"""Algorithm 1 dynamic search for one survivor state, without policy selection.

Plans describe logical target slots; physical worker matching, state-source checks
and measured transition costs belong to the Restorer. Memory capacity is uniform
across target slots. No transition estimate is fabricated here.
"""

from dataclasses import asdict, dataclass, field
from itertools import accumulate, combinations, product

from .contracts import ClusterState, ModelConfig, WorkerIdentity, _finite, _integer, stable_hash
from .estimators import Estimator, MemoryEstimate, TimeEstimate


def _range_values(name, values):
    values = tuple(values)
    if not values:
        raise ValueError(f"{name} must be an explicit nonempty range")
    for value in values:
        _integer(name, value)
    return tuple(sorted(set(values)))


def integer_partitions(survivors: int, dp: int, r_pp) -> tuple[tuple[int, ...], ...]:
    """Canonical nondecreasing pipeline lengths, using every survivor."""
    _integer("survivors", survivors)
    _integer("dp", dp)
    allowed = _range_values("Rpp", r_pp)

    def visit(remaining, slots, minimum, prefix):
        if slots == 0:
            if remaining == 0:
                yield prefix
            return
        for length in allowed:
            if length >= minimum and length * slots <= remaining <= allowed[-1] * slots:
                yield from visit(remaining - length, slots - 1, length, (*prefix, length))

    return tuple(visit(survivors, dp, allowed[0], ()))


def repair_zero_partitions(partitions: tuple[int, ...]) -> tuple[int, ...] | None:
    """Fill zero partitions from the current largest donor; IDs are tuple indices."""
    if not partitions:
        raise ValueError("partitions must be nonempty")
    for count in partitions:
        _integer("partition", count, 0)
    if sum(partitions) < len(partitions):
        return None
    counts = list(partitions)
    for pipeline in range(len(counts)):
        if counts[pipeline] == 0:
            donor = max(range(len(counts)), key=lambda i: (counts[i], -i))
            counts[donor] -= 1
            counts[pipeline] = 1
    return tuple(counts)


def batch_distributions(global_micro_batches: int, pipeline_lengths: tuple[int, ...]
                        ) -> tuple[tuple[int, ...], ...]:
    """Proportional floor allocation, then every remainder distribution and repair."""
    _integer("global_micro_batches", global_micro_batches)
    lengths = tuple(pipeline_lengths)
    if not lengths:
        raise ValueError("pipeline_lengths must be nonempty")
    for length in lengths:
        _integer("pipeline length", length)
    if global_micro_batches < len(lengths):
        return ()
    total = sum(lengths)
    base = tuple(global_micro_batches * length // total for length in lengths)

    def visit(pipeline, remaining, prefix):
        if pipeline == len(base) - 1:
            yield repair_zero_partitions((*prefix, base[pipeline] + remaining))
            return
        for count in range(remaining + 1):
            yield from visit(pipeline + 1, remaining - count, (*prefix, base[pipeline] + count))

    return tuple(sorted(set(visit(0, global_micro_batches - sum(base), ()))))


def layer_distributions(num_layers: int, num_stages: int) -> tuple[tuple[int, ...], ...]:
    """Balanced block counts: each remainder stage gets exactly one extra block."""
    _integer("num_layers", num_layers)
    _integer("num_stages", num_stages)
    base, remainder = divmod(num_layers, num_stages)
    return tuple(sorted(tuple(base + (stage in extra) for stage in range(num_stages))
                        for extra in combinations(range(num_stages), remainder)))


def layer_layouts(module_order: tuple[str, ...], layer_modules: tuple[str, ...], num_stages: int
                  ) -> tuple[tuple[tuple[str, ...], ...], ...]:
    """Keep model order, prefix on the first stage and suffix on the last stage.

    Non-block modules between blocks stay with the preceding block. Zero-block
    endpoint stages are valid; stages with no model module are excluded.
    """
    if (not layer_modules or len(set(module_order)) != len(module_order)
            or tuple(name for name in module_order if name in layer_modules) != tuple(layer_modules)):
        raise ValueError("layer_modules must be unique and follow module_order")
    positions = tuple(module_order.index(name) for name in layer_modules)
    layouts = []
    for counts in layer_distributions(len(layer_modules), num_stages):
        boundaries = (0, *(positions[count] if count < len(positions) else positions[-1] + 1
                           for count in accumulate(counts[:-1])), len(module_order))
        layout = tuple(tuple(module_order[start:end]) for start, end in zip(boundaries, boundaries[1:]))
        if all(layout):
            layouts.append(layout)
    return tuple(layouts)


@dataclass(frozen=True)
class DynamicPlan:
    global_batch_size: int
    generation: int
    survivors: tuple[WorkerIdentity, ...]
    profile_hash: str
    pipeline_lengths: tuple[int, ...]
    pipeline_micro_batches: tuple[int, ...]
    layouts: tuple[tuple[tuple[str, ...], ...], ...]
    time: TimeEstimate | None
    memory: tuple[MemoryEstimate, ...]
    plan_id: str = field(init=False)

    def __post_init__(self):
        ClusterState(self.survivors, self.global_batch_size, self.generation)
        if (not all(isinstance(values, tuple) for values in
                    (self.pipeline_lengths, self.pipeline_micro_batches, self.layouts))
                or any(not isinstance(layout, tuple) or any(not isinstance(stage, tuple) for stage in layout)
                       for layout in self.layouts)):
            raise ValueError("plan lengths, micro-batches and nested layouts must be tuples")
        for count in self.pipeline_lengths:
            _integer("pipeline length", count)
        for count in self.pipeline_micro_batches:
            _integer("pipeline micro-batch count", count)
        if (not self.pipeline_lengths or len(self.pipeline_lengths) != len(self.pipeline_micro_batches)
                or tuple(len(layout) for layout in self.layouts) != self.pipeline_lengths
                or sum(self.pipeline_lengths) != len(self.survivors)):
            raise ValueError("pipeline lengths, layouts and micro-batches must match survivors")
        if self.time is not None:
            if self.layouts != self.time.derivation["layouts"]:
                raise ValueError("plan layout must match its time estimate")
            if self.profile_hash != self.time.derivation["profile_hash"]:
                raise ValueError("plan profile must match its time estimate")
            if (self.pipeline_micro_batches != self.time.derivation["pipeline_micro_batches"]
                    or sum(self.pipeline_micro_batches) != self.time.derivation["global_micro_batches"]):
                raise ValueError("plan micro-batches must match the estimated distribution")
            if self.time.feasible:
                _finite("estimated step time", self.time.step_time_s, positive=True)
        object.__setattr__(self, "plan_id", "dynamic-" + stable_hash({
            "workers": tuple(asdict(w) for w in sorted(self.survivors, key=lambda w: w.worker_id)),
            "generation": self.generation, "global_batch_size": self.global_batch_size,
            "lengths": self.pipeline_lengths, "batch": self.pipeline_micro_batches,
            "layouts": self.layouts, "profile_hash": self.profile_hash,
        }))

    @property
    def policy(self) -> str:
        return "dynamic"

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(reason for estimate in self.memory for reason in estimate.reasons)

    @property
    def feasible(self) -> bool:
        return not self.reasons and self.time is not None and self.time.feasible

    @property
    def estimated_step_time_s(self) -> float | None:
        return self.time.step_time_s if self.time is not None else None


class NoFeasibleDynamicPlanError(RuntimeError):
    def __init__(self, reasons: tuple[str, ...], rejected_plan: DynamicPlan | None):
        self.reasons = reasons
        self.rejected_plan = rejected_plan
        super().__init__("; ".join(reasons))


class Planner:
    def __init__(self, estimator: Estimator, *, config: ModelConfig, r_dp, r_pp,
                 memory_capacity_bytes: int):
        if estimator.profile["identity"]["config_hash"] != stable_hash(asdict(config)):
            raise ValueError("planner config must match profile config identity")
        if estimator.layer_modules != tuple(f"blocks.{i}" for i in range(config.num_layers)):
            raise ValueError("planner layers must match the configured blocks in model order")
        _integer("memory_capacity_bytes", memory_capacity_bytes, 0)
        self.estimator = estimator
        self.config = config
        self.r_dp = _range_values("Rdp", r_dp)
        self.r_pp = _range_values("Rpp", r_pp)
        self.memory_capacity_bytes = memory_capacity_bytes
        self.global_micro_batches = (config.global_batch_size + config.micro_batch_size - 1) // config.micro_batch_size

    def candidates(self, state: ClusterState):
        """Enumerate only this state's dynamic candidates, retaining OOM diagnostics."""
        if state.global_batch_size != self.config.global_batch_size:
            raise ValueError("state must preserve the configured global batch size")
        workers = tuple(sorted(state.workers, key=lambda worker: worker.worker_id))
        order = tuple(self.estimator.profile["identity"]["module_order"])
        profile_hash = stable_hash(self.estimator.profile)
        for dp in self.r_dp:
            for lengths in integer_partitions(len(workers), dp, self.r_pp):
                batches = batch_distributions(self.global_micro_batches, lengths)
                if not batches:
                    continue
                options = []
                for i, pp in enumerate(lengths):
                    options.append(tuple((layout, self.estimator.memory(layout, (self.memory_capacity_bytes,) * pp, pipeline=i))
                                         for layout in layer_layouts(order, self.estimator.layer_modules, pp)))
                for splits in product(*options):
                    layouts, memory = zip(*splits)
                    memory_feasible = all(estimate.feasible for estimate in memory)
                    for batch in batches:
                        time = (self.estimator.dynamic_time(layouts, batch, global_micro_batches=self.global_micro_batches)
                                if memory_feasible else None)
                        yield DynamicPlan(state.global_batch_size, state.generation, workers, profile_hash,
                                          lengths, batch, layouts, time, memory)

    def best_dynamic_plan(self, state: ClusterState) -> DynamicPlan:
        """Minimize estimated post-recovery step time; break exact ties by plan ID."""
        best, rejected, reasons = None, None, set()
        for plan in self.candidates(state):
            if not plan.feasible:
                if rejected is None:
                    rejected = plan
                reasons.update(plan.reasons)
            elif best is None or (plan.estimated_step_time_s, plan.plan_id) < (best.estimated_step_time_s, best.plan_id):
                best = plan
        if best is None:
            details = tuple(sorted(reasons)) or (
                f"no legal dynamic plan: {len(state.workers)} survivors, Rdp={self.r_dp}, Rpp={self.r_pp}, "
                f"{self.global_micro_batches} global micro-batches, {self.config.num_layers} layers; "
                "pipelines require data and nonempty stages",)
            raise NoFeasibleDynamicPlanError(details, rejected)
        return best
