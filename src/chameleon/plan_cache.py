"""In-memory Algorithm 1 cache; policy scoring and survivor state stay uncached."""

from copy import deepcopy
from dataclasses import asdict, dataclass

from .planner import DynamicPlan, NoFeasibleDynamicPlanError, Planner
from .profiler import _hash


@dataclass(frozen=True)
class DynamicSearchLookup:
    key: str
    hit: bool
    plan: DynamicPlan | None
    reasons: tuple[str, ...]
    rejected_plan: DynamicPlan | None

    @property
    def feasible(self):
        return self.plan is not None


@dataclass(frozen=True)
class _Entry:
    plan: DynamicPlan | None
    reasons: tuple[str, ...]
    rejected_plan: DynamicPlan | None


class PlanCache:
    """Cache only D-independent dynamic searches for exact recovery inputs."""

    def __init__(self):
        self._entries: dict[str, _Entry] = {}
        self._hits = 0
        self._misses = 0

    @property
    def stats(self):
        return {"entries": len(self._entries), "hits": self._hits, "misses": self._misses}

    def clear(self):
        self._entries.clear()
        self._hits = 0
        self._misses = 0

    def key(self, planner: Planner, recovery_state) -> str:
        """Bind topology/failure, model/config, full profile and search limits; omit D."""
        failure = asdict(recovery_state.failure)
        failure["failed_worker_ids"] = tuple(sorted(failure["failed_worker_ids"]))
        return _hash({
            "schema_version": 1,
            "state": {
                "recovery_id": recovery_state.recovery_id,
                "cluster": asdict(recovery_state.cluster),
                "failure": failure,
                "survivor": asdict(recovery_state.survivor_state),
                "layouts": recovery_state.layouts,
                "pipeline_micro_batches": recovery_state.pipeline_micro_batches,
            },
            "model_identity": planner.estimator.profile["identity"]["model_hash"],
            "config_identity": planner.estimator.profile["identity"]["config_hash"],
            "config": asdict(planner.config),
            "profile_hash": _hash(planner.estimator.profile),
            "search": {
                "Rdp": planner.r_dp,
                "Rpp": planner.r_pp,
                "memory_capacity_bytes": planner.memory_capacity_bytes,
            },
        })

    def search(self, planner: Planner, recovery_state) -> DynamicSearchLookup:
        key = self.key(planner, recovery_state)
        if key in self._entries:
            self._hits += 1
            entry = deepcopy(self._entries[key])
            return DynamicSearchLookup(key, True, entry.plan, entry.reasons, entry.rejected_plan)
        self._misses += 1
        try:
            entry = _Entry(planner.best_dynamic_plan(recovery_state.survivor_state), (), None)
        except NoFeasibleDynamicPlanError as error:
            entry = _Entry(None, error.reasons, error.rejected_plan)
        self._entries[key] = deepcopy(entry)
        return DynamicSearchLookup(key, False, entry.plan, entry.reasons, entry.rejected_plan)
