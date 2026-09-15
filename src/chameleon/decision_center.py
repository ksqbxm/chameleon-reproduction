"""Candidate evaluation and Equation 8 selection, without runtime recovery."""

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from time import perf_counter

from .contracts import (
    ClusterState, DecisionResult, ExecutionPlan, FailureEvent, ModelConfig,
    UnrecoverableStateError, WorkerIdentity, _finite, _integer,
)
from .estimators import Estimator, MemoryEstimate, TimeEstimate, estimate_rerouting_time
from .planner import DynamicPlan, NoFeasibleDynamicPlanError, Planner
from .plan_cache import PlanCache
from .profiler import _hash
from .restorer import MigrationManifest, MissingTransitionCalibrationError, Restorer, TransitionEstimate
from .state_sources import StateTensor, WorkerInventory, build_state_source_map


@dataclass(frozen=True)
class RecoveryState:
    cluster: ClusterState
    failure: FailureEvent
    layouts: tuple[tuple[tuple[str, ...], ...], ...]
    pipeline_workers: tuple[tuple[WorkerIdentity | None, ...], ...]
    pipeline_micro_batches: tuple[int, ...]
    required: tuple[StateTensor, ...]
    inventories: tuple[WorkerInventory, ...]
    recovery_id: str = field(init=False)

    def __post_init__(self):
        if (self.failure.generation != self.cluster.generation
                or self.failure.committed_global_step != self.cluster.committed_global_step
                or not set(self.failure.failed_worker_ids) <= {w.worker_id for w in self.cluster.workers}):
            raise ValueError("failure must match the cluster identities and committed safe point")
        if (not isinstance(self.layouts, tuple) or not self.layouts
                or any(not isinstance(layout, tuple) or not layout
                       or any(not isinstance(stage, tuple) or not stage
                              or any(not isinstance(module, str) or not module.strip() for module in stage)
                              for stage in layout) for layout in self.layouts)
                or not isinstance(self.pipeline_workers, tuple) or not self.pipeline_workers
                or len(self.pipeline_workers) != len(self.layouts)
                or any(not isinstance(pipeline, tuple) or len(pipeline) != len(layout)
                       for pipeline, layout in zip(self.pipeline_workers, self.layouts))):
            raise ValueError("original topology requires nonempty layouts matching every pipeline's workers")
        if (not isinstance(self.pipeline_micro_batches, tuple)
                or len(self.pipeline_micro_batches) != len(self.layouts)):
            raise ValueError("original micro-batch partitions must cover every pipeline")
        for count in self.pipeline_micro_batches:
            _integer("original pipeline micro-batches", count)
        workers = tuple(w for pipeline in self.pipeline_workers for w in pipeline if w is not None)
        if len(workers) != len(self.cluster.workers) or set(workers) != set(self.cluster.workers):
            raise ValueError("original topology must cover each full worker identity exactly once")
        if (not isinstance(self.required, tuple) or any(not isinstance(t, StateTensor) for t in self.required)
                or not isinstance(self.inventories, tuple)
                or any(not isinstance(i, WorkerInventory) for i in self.inventories)):
            raise ValueError("recovery inventories must be immutable metadata tuples")
        failure = asdict(self.failure)
        failure["failed_worker_ids"] = tuple(sorted(self.failure.failed_worker_ids))
        object.__setattr__(self, "recovery_id", _hash({"failure": failure, "B": self.cluster.global_batch_size,
            "layouts": self.layouts, "micro_batches": self.pipeline_micro_batches,
            "workers": tuple(tuple(asdict(w) if w is not None else None for w in pipeline)
                             for pipeline in self.pipeline_workers)}))

    @property
    def survivor_state(self):
        workers = tuple(w for w in self.cluster.workers if w.worker_id not in self.failure.failed_worker_ids)
        if not workers:
            raise UnrecoverableStateError("no surviving workers")
        return ClusterState(workers, self.cluster.global_batch_size, self.cluster.generation,
                            self.cluster.committed_global_step)


@dataclass(frozen=True)
class StageTaskRoute:
    pipeline: int
    stage: int
    micro_batch: int
    worker: WorkerIdentity


@dataclass(frozen=True)
class ReroutingPlan:
    state: RecoveryState
    profile_hash: str
    time: TimeEstimate
    memory: tuple[MemoryEstimate, ...]
    routes: tuple[StageTaskRoute, ...]
    plan_id: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.routes, tuple):
            raise ValueError("routes must be an immutable tuple")
        object.__setattr__(self, "plan_id", "rerouting-" + _hash({"recovery_id": self.state.recovery_id,
            "profile": self.profile_hash, "routes": tuple(asdict(route) for route in self.routes)}))

    @property
    def policy(self):
        return "rerouting"

    @property
    def global_batch_size(self):
        return self.state.cluster.global_batch_size

    @property
    def generation(self):
        return self.state.cluster.generation

    @property
    def estimated_step_time_s(self):
        return self.time.step_time_s


@dataclass(frozen=True)
class PolicyCandidate:
    plan_id: str
    policy: str
    global_batch_size: int
    generation: int
    estimated_step_time_s: float | None
    transition: TransitionEstimate | None
    common_control_time_s: float | None
    memory: tuple[MemoryEstimate, ...]
    execution: ReroutingPlan | DynamicPlan | MigrationManifest | None
    derivation: dict
    reasons: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.plan_id, str) or not self.plan_id.strip():
            raise ValueError("plan_id must be a nonempty string")
        if self.policy not in ("rerouting", "dynamic"):
            raise ValueError("policy must be rerouting or dynamic")
        _integer("global_batch_size", self.global_batch_size)
        _integer("generation", self.generation, 0)
        for name, positive in (("estimated_step_time_s", True), ("common_control_time_s", False)):
            value = getattr(self, name)
            if value is not None:
                _finite(name, value, positive=positive)
        if self.transition is not None and self.transition.common_control_time_s != self.common_control_time_s:
            raise ValueError("candidate common control differs from its transition estimate")
        if isinstance(self.execution, MigrationManifest) and self.transition is not None:
            if self.transition.manifest_id != self.execution.manifest_id:
                raise ValueError("candidate transition must match its migration manifest")
        if isinstance(self.execution, ReroutingPlan) and self.transition is not None:
            if self.transition.manifest_id is not None or self.transition.estimated_transition_time_s != 0:
                raise ValueError("rerouting paper transition must be zero and have no migration manifest")
        if self.execution is not None:
            plan = self.execution.plan if isinstance(self.execution, MigrationManifest) else self.execution
            if (not isinstance(plan, (ReroutingPlan, DynamicPlan))
                    or (self.plan_id, self.policy, self.global_batch_size, self.generation,
                        self.estimated_step_time_s, self.memory)
                    != (plan.plan_id, plan.policy, plan.global_batch_size, plan.generation,
                        plan.estimated_step_time_s, plan.memory)):
                raise ValueError("candidate identity and estimates must match the actual execution")
        reasons = self.reasons + tuple(reason for memory in self.memory for reason in memory.reasons)
        object.__setattr__(self, "reasons", tuple(dict.fromkeys(reasons)))
        if not self.reasons and (self.estimated_step_time_s is None or self.transition is None):
            raise ValueError("feasible candidates require step and transition estimates")

    @property
    def feasible(self):
        return not self.reasons

    @property
    def estimated_transition_time_s(self):
        return self.transition.estimated_transition_time_s if self.transition is not None else None

    @property
    def execution_plan(self):
        if not self.feasible:
            raise ValueError("candidate is infeasible")
        return ExecutionPlan(self.plan_id, self.policy, self.global_batch_size, self.generation,
                             self.estimated_step_time_s, self.estimated_transition_time_s)


@dataclass(frozen=True)
class PolicyDecision(DecisionResult):
    candidate: PolicyCandidate
    derivation: dict

    def __post_init__(self):
        super().__post_init__()
        if not self.candidate.feasible or self.plan != self.candidate.execution_plan:
            raise ValueError("decision plan must match its feasible candidate")
        expected = (self.plan.global_batch_size / self.plan.estimated_step_time_s) * (
            (self.inter_fault_duration_s - self.plan.estimated_transition_time_s) / self.inter_fault_duration_s)
        if self.score != expected:
            raise ValueError("decision score must match Equation 8")


class NoUsablePolicyError(RuntimeError):
    def __init__(self, derivation: dict):
        self.derivation = derivation
        details = tuple(reason for row in derivation["candidates"] for reason in row["rejection_reasons"])
        super().__init__("no usable policy: " + ("; ".join(details) or "no candidates"))


def select_policy(candidates: tuple[PolicyCandidate, ...], inter_fault_duration_s=None) -> PolicyDecision:
    """D is supplied by the caller; no MTBF prediction or step-only fallback."""
    _finite("inter_fault_duration_s", inter_fault_duration_s, positive=True)
    candidates = tuple(candidates)
    if (len({c.policy for c in candidates}) != len(candidates)
            or len({c.plan_id for c in candidates}) != len(candidates)
            or len({(c.global_batch_size, c.generation) for c in candidates}) > 1):
        raise ValueError("candidates must have unique policies/plan IDs and the same batch/generation")
    duration = inter_fault_duration_s
    record = {"equation": 8, "B": candidates[0].global_batch_size if candidates else None,
              "D": duration, "duration_source": "caller supplied inter-fault duration; not predicted",
              "formula": "(B / t_step) * ((D - t_transition) / D)",
              "transition_model": "policy-specific paper model; measured common control is reported separately",
              "tie_break": "smaller transition, then stable plan ID (project rule)", "candidates": []}
    usable = []
    for candidate in candidates:
        row = asdict(candidate)
        row.update(estimated_transition_time_s=candidate.estimated_transition_time_s,
                   feasible=candidate.feasible, score=None, throughput_samples_s=None, useful_window_fraction=None,
                   selected=False, rejection_reasons=candidate.reasons)
        record["candidates"].append(row)
        if not candidate.feasible:
            continue
        step, transition = candidate.estimated_step_time_s, candidate.estimated_transition_time_s
        if duration <= transition:
            row["rejection_reasons"] = ("D <= t_transition: candidate unavailable in this window",)
            continue
        throughput = candidate.global_batch_size / step
        fraction = (duration - transition) / duration
        score = throughput * fraction
        _finite("Equation 8 score", score, positive=True)
        row.update(score=score, throughput_samples_s=throughput, useful_window_fraction=fraction)
        usable.append((candidate, score, row))
    if not usable:
        raise NoUsablePolicyError(record)
    winner, score, selected_row = min(usable, key=lambda item: (
        -item[1], item[0].estimated_transition_time_s, item[0].plan_id))
    for _, _, row in usable:
        if row is not selected_row:
            row["rejection_reasons"] = ("lower Equation 8 score" if row["score"] < score
                                       else "lost deterministic tie-break",)
    selected_row["selected"] = True
    record.update(selected_policy=winner.policy, selected_plan_id=winner.plan_id)
    return PolicyDecision(winner.execution_plan, duration, score, winner, record)


class DecisionCenter:
    def __init__(self, config: ModelConfig, *, expected_identity: dict, r_dp, r_pp,
                 memory_capacity_bytes: int, plan_cache: PlanCache | None = None):
        self.config = config
        self.expected_identity = deepcopy(expected_identity)
        self.r_dp, self.r_pp = tuple(r_dp), tuple(r_pp)
        self.memory_capacity_bytes = memory_capacity_bytes
        if plan_cache is not None and not isinstance(plan_cache, PlanCache):
            raise ValueError("plan_cache must be a PlanCache")
        self.plan_cache = plan_cache

    def _planner(self, profile):
        estimator = Estimator(profile, expected_identity=self.expected_identity,
                              layer_modules=tuple(f"blocks.{i}" for i in range(self.config.num_layers)))
        planner = Planner(estimator, config=self.config, r_dp=self.r_dp, r_pp=self.r_pp,
                          memory_capacity_bytes=self.memory_capacity_bytes)
        return estimator, planner

    def precompute_dynamic(self, states, profile: dict, *, max_failures: int):
        """Populate D-independent searches for explicit scenarios with 1..k failures."""
        _integer("max_failures", max_failures)
        states = tuple(states)
        if not states:
            raise ValueError("precompute states must be nonempty")
        if self.plan_cache is None:
            raise ValueError("precompute requires a PlanCache")
        _, planner = self._planner(profile)
        records = []
        for state in states:
            if not isinstance(state, RecoveryState):
                raise ValueError("precompute states must contain RecoveryState values")
            failed = len(state.failure.failed_worker_ids)
            if not 1 <= failed <= max_failures:
                raise ValueError("precompute scenarios must contain 1..max_failures failed workers")
            lookup = self.plan_cache.search(planner, state)
            records.append({"recovery_id": state.recovery_id, "failed_workers": failed,
                            "cache_key": lookup.key, "cache_hit": lookup.hit,
                            "feasible": lookup.feasible,
                            "plan_id": lookup.plan.plan_id if lookup.plan is not None else None,
                            "reasons": lookup.reasons})
        return tuple(records)

    def _rerouting(self, state, estimator, common_control, source_reasons):
        survivors = state.survivor_state
        dp, layout = len(state.layouts), state.layouts[0]
        profile_hash = _hash(estimator.profile)
        transition = TransitionEstimate(0., 0., common_control,
                                       {"source": "rerouting paper-model approximation of zero"}, None)
        derivation = {"policy_transition_source": "rerouting paper-model approximation of zero",
                      "common_control_source": "measured bootstrap proxy" if common_control is not None else "unmeasured",
                      "original_layouts": state.layouts, "original_micro_batches": state.pipeline_micro_batches}
        if any(other != layout for other in state.layouts) or len(set(state.pipeline_micro_batches)) != 1:
            return PolicyCandidate("rerouting-unavailable-" + _hash({"recovery": state.recovery_id,
                "profile": profile_hash}), "rerouting", survivors.global_batch_size, survivors.generation,
                None, transition, common_control, (), None, derivation, source_reasons + (
                "Eq.12/13 requires identical original layouts and equal integer micro-batches per pipeline",))
        pp, nm = len(layout), state.pipeline_micro_batches[0]
        global_nm = sum(state.pipeline_micro_batches)
        forward, backward = estimator.stage_durations(layout)
        alive = set(survivors.workers)
        peers = tuple(tuple(sorted((pipeline[stage] for pipeline in state.pipeline_workers
                                    if pipeline[stage] in alive), key=lambda w: w.worker_id)) for stage in range(pp))
        failures = tuple(dp - len(row) for row in peers)
        derivation.update(stage_forward_s=forward, stage_backward_s=backward,
                          duration_source="measured module EMAs, including endpoints; largest forward/backward stage",
                          boundary="uniform-stage Eq.12/13 approximation for the original symmetric topology")
        time = estimate_rerouting_time(global_micro_batches=global_nm, dp_size=dp,
                                      failures_per_stage=failures, forward_s=max(forward), backward_s=max(backward))
        derivation["time"] = time.derivation
        routes, offsets = [], [0] * pp
        served_pipelines = {worker: set() for worker in alive}
        for pipeline, workers in enumerate(state.pipeline_workers):
            for stage, worker in enumerate(workers):
                for micro_batch in range(nm):
                    if worker in alive:
                        owner = worker
                    elif peers[stage]:
                        owner = peers[stage][offsets[stage] % len(peers[stage])]
                        offsets[stage] += 1
                    else:
                        continue
                    routes.append(StageTaskRoute(pipeline, stage, micro_batch, owner))
                    served_pipelines[owner].add(pipeline)
        baseline = estimator.memory(layout, (self.memory_capacity_bytes,) * pp)
        local = {inventory.worker: set(inventory.tensors) for inventory in state.inventories}
        rows, memory_reasons, ownership_reasons = [], [], []
        for pipeline, workers in enumerate(state.pipeline_workers):
            for stage, worker in enumerate(workers):
                if worker not in alive:
                    continue
                needed = {t for t in state.required if t.module_id in layout[stage]}
                if not needed <= local[worker]:
                    ownership_reasons.append(f"rerouting peer {worker.worker_id} lacks complete stage {stage} parameter/AdamW state")
                served = tuple(sorted(served_pipelines[worker]))
                base = baseline.stages[stage]
                static = base["static_layer_bytes"] + baseline.derivation["extra_static_bytes"][stage]
                activation = (base["dynamic_layer_bytes"]
                              + base["activation_slots"] * baseline.derivation["extra_activation_bytes"][stage])
                peak = static + len(served) * activation
                _finite("rerouting peak bytes", peak)
                reason = (f"rerouting worker {worker.worker_id}: estimated {peak:g} bytes > "
                          f"capacity {self.memory_capacity_bytes} bytes") if peak > self.memory_capacity_bytes else None
                if reason:
                    memory_reasons.append(reason)
                rows.append(dict(base, pipeline=pipeline, worker_id=worker.worker_id, served_pipelines=served,
                                 static_bytes=static, activation_bytes_per_stream=activation,
                                 activation_streams=len(served), peak_bytes=peak, oom_reason=reason))
        memory = MemoryEstimate(tuple(rows), {"equation": 14, "baseline": baseline.derivation,
            "boundary": "conservative per-logical-pipeline activation slots; parameters/AdamW retained once per owner"},
            tuple(memory_reasons))
        execution = ReroutingPlan(state, profile_hash, time, (memory,), tuple(routes))
        return PolicyCandidate(execution.plan_id, "rerouting", survivors.global_batch_size, survivors.generation,
                               time.step_time_s, transition, common_control, (memory,), execution, derivation,
                               source_reasons + time.reasons + tuple(ownership_reasons))

    def evaluate_candidates(self, state: RecoveryState, profile: dict) -> tuple[PolicyCandidate, PolicyCandidate]:
        """Evaluate rerouting and Algorithm 1's best dynamic plan independently of D."""
        estimator, planner = self._planner(profile)
        restorer = Restorer(profile, expected_identity=self.expected_identity)
        survivors = state.survivor_state
        if survivors.global_batch_size != self.config.global_batch_size:
            raise ValueError("recovery must preserve the configured global batch size")
        if sum(state.pipeline_micro_batches) != planner.global_micro_batches:
            raise ValueError("original micro-batch partitions must conserve global_micro_batches")
        for layout in dict.fromkeys(state.layouts):
            estimator.stage_durations(layout)
        module_bytes = {module: sum(t.nbytes for t in state.required
                                   if t.module_id == module and t.kind == "parameter")
                        for module in {t.module_id for t in state.required}}
        if module_bytes != profile["identity"]["module_parameter_bytes"]:
            raise ValueError("required inventory must match the complete profiled model")
        sources, source_reasons = None, ()
        try:
            sources = build_state_source_map(survivors, state.required, state.inventories)
        except UnrecoverableStateError as error:
            source_reasons = (str(error),)
        common = restorer.common_control_time_s
        rerouting = self._rerouting(state, estimator, common, source_reasons)
        if self.plan_cache is None:
            start = perf_counter()
            try:
                dynamic = planner.best_dynamic_plan(survivors)
            except NoFeasibleDynamicPlanError as error:
                search_time = perf_counter() - start
                dynamic = None
                rejected, search_reasons = error.rejected_plan, error.reasons
                cache_record = {"enabled": False, "key": None, "hit": False}
            else:
                search_time = perf_counter() - start
                rejected, search_reasons = None, ()
                cache_record = {"enabled": False, "key": None, "hit": False}
        else:
            start = perf_counter()
            lookup = self.plan_cache.search(planner, state)
            dynamic = lookup.plan
            rejected, search_reasons = lookup.rejected_plan, lookup.reasons
            search_time = 0. if lookup.hit else perf_counter() - start
            cache_record = {"enabled": True, "key": lookup.key, "hit": lookup.hit}
        if dynamic is None:
            plan_id = rejected.plan_id if rejected else "dynamic-unavailable-" + _hash({
                "recovery": state.recovery_id, "profile": _hash(profile), "Rdp": planner.r_dp,
                "Rpp": planner.r_pp, "capacity": planner.memory_capacity_bytes})
            candidate = PolicyCandidate(plan_id, "dynamic", survivors.global_batch_size,
                survivors.generation, None, None, common, rejected.memory if rejected else (), rejected,
                {"unoverlapped_search_time_s": search_time, "search_rejection_reasons": search_reasons,
                 "diagnostic_layouts": rejected.layouts if rejected else (),
                 "dynamic_search_cache": cache_record}, source_reasons + search_reasons)
            return rerouting, candidate
        derivation = {"time": dynamic.time.derivation, "unoverlapped_search_time_s": search_time,
                      "search_objective": "Algorithm 1 minimum post-recovery step time; independent of D",
                      "dynamic_search_cache": cache_record}
        execution, transition, reasons = dynamic, None, source_reasons
        if sources is not None:
            execution = restorer.plan(dynamic, sources)
            try:
                transition = restorer.estimate_transition(execution, unoverlapped_search_time_s=search_time)
            except MissingTransitionCalibrationError as error:
                reasons = (str(error),)
        candidate = PolicyCandidate(dynamic.plan_id, "dynamic", survivors.global_batch_size, survivors.generation,
                                    dynamic.estimated_step_time_s, transition, common, dynamic.memory,
                                    execution, derivation, reasons)
        return rerouting, candidate

    def select(self, state: RecoveryState, profile: dict, inter_fault_duration_s=None) -> PolicyDecision:
        _finite("inter_fault_duration_s", inter_fault_duration_s, positive=True)
        return select_policy(self.evaluate_candidates(state, profile), inter_fault_duration_s)
