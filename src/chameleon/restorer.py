"""Migration and communication planning; no tensor transfer or topology commit."""

from copy import deepcopy
from dataclasses import asdict, dataclass

from .coloring import conflict_graph, dsatur
from .contracts import WorkerIdentity, _finite, _integer
from .hungarian import hungarian
from .planner import DynamicPlan
from .profiler import _hash, validate_snapshot
from .state_sources import StateSourceMap, StateTensor
from .transfer_calibration import calibration_times_s


def synchronization_rounds(unit_devices):
    """Shared deterministic module rounds for manifests and real owner SUMs."""
    return dsatur(conflict_graph(unit_devices))


@dataclass(frozen=True)
class TargetSlot:
    pipeline: int
    stage: int
    modules: tuple[str, ...]

    def __post_init__(self):
        _integer("pipeline", self.pipeline, 0)
        _integer("stage", self.stage, 0)
        if (not isinstance(self.modules, tuple) or not self.modules
                or any(not isinstance(m, str) or not m.strip() for m in self.modules)
                or len(set(self.modules)) != len(self.modules)):
            raise ValueError("target slot requires unique nonempty module IDs")


@dataclass(frozen=True)
class TensorAction:
    tensor: StateTensor
    source: WorkerIdentity
    destination: WorkerIdentity
    slot: TargetSlot

    @property
    def retained(self):
        return self.source == self.destination


@dataclass(frozen=True)
class MigrationManifest:
    manifest_id: str
    plan: DynamicPlan
    assignments: tuple[tuple[TargetSlot, WorkerIdentity], ...]
    cost_matrix: tuple[tuple[int, ...], ...]
    migration_bytes: int
    actions: tuple[TensorAction, ...]
    migration_rounds: tuple[tuple[TensorAction, ...], ...]
    synchronization_rounds: tuple[tuple[str, ...], ...]
    held_sources: tuple[tuple[WorkerIdentity, StateTensor], ...]
    release_after_ack: tuple[tuple[WorkerIdentity, StateTensor], ...]


class MigrationAcknowledgements:
    """Release permission only after every target validates its complete state."""

    def __init__(self, manifest: MigrationManifest):
        self._manifest = manifest
        self._pending = set(manifest.actions)

    def acknowledge(self, action: TensorAction, *, manifest_id: str):
        if manifest_id != self._manifest.manifest_id:
            raise ValueError("ACK manifest differs from the current recovery")
        if action not in self._pending:
            raise ValueError("unknown or duplicate target tensor ACK")
        self._pending.remove(action)

    @property
    def complete(self):
        return not self._pending

    def releasable_sources(self):
        if not self.complete:
            raise RuntimeError("source state must remain alive until all target ACKs")
        return self._manifest.release_after_ack


def migration_cost_matrix(sources: StateSourceMap, slots: tuple[TargetSlot, ...]):
    """Rows are stable survivor IDs; columns are logical target slots."""
    modules = {t.module_id for t in sources.required}
    if (len(slots) != len(sources.inventories)
            or len({(s.pipeline, s.stage) for s in slots}) != len(slots)
            or any(not set(s.modules) <= modules for s in slots)):
        raise ValueError("target slots must be unique, nonempty and match survivor count")
    return tuple(tuple(sum(t.nbytes for t in sources.required
                           if t.module_id in slot.modules and t not in inventory.tensors)
                       for slot in slots) for inventory in sources.inventories)


@dataclass(frozen=True)
class TransitionEstimate:
    unoverlapped_search_time_s: float
    migration_time_s: float
    common_control_time_s: float | None
    derivation: dict
    manifest_id: str | None

    def __post_init__(self):
        _finite("unoverlapped_search_time_s", self.unoverlapped_search_time_s)
        _finite("migration_time_s", self.migration_time_s)
        _finite("policy transition", self.estimated_transition_time_s)
        if self.common_control_time_s is not None:
            _finite("common_control_time_s", self.common_control_time_s)
            _finite("total transition", self.estimated_total_time_s)

    @property
    def estimated_transition_time_s(self):
        """Policy-specific paper-model time for Equation 8."""
        return self.unoverlapped_search_time_s + self.migration_time_s

    @property
    def estimated_total_time_s(self):
        if self.common_control_time_s is None:
            return None
        return self.estimated_transition_time_s + self.common_control_time_s


class MissingTransitionCalibrationError(ValueError):
    """Policy transition cannot be estimated from the measured calibration."""


class Restorer:
    def __init__(self, profile: dict, *, expected_identity: dict):
        validate_snapshot(profile, expected_identity)
        self._profile = deepcopy(profile)
        self._profile_hash = _hash(self._profile)
        self._transfer_times = {}
        self._bootstrap_time = None
        for report in self._profile["calibrations"]:
            transfers, self._bootstrap_time = calibration_times_s(report)
            for size in report["tensor_bytes"]:
                self._transfer_times[size] = max(transfers[0, 1, size], transfers[1, 0, size])

    @property
    def common_control_time_s(self):
        return self._bootstrap_time

    def plan(self, dynamic: DynamicPlan, sources: StateSourceMap) -> MigrationManifest:
        state = sources.state
        if (not dynamic.feasible or dynamic.generation != state.generation
                or dynamic.global_batch_size != state.global_batch_size
                or set(dynamic.survivors) != set(state.workers)
                or dynamic.profile_hash != self._profile_hash):
            raise ValueError("dynamic plan must match survivor state and profile identity")
        modules = {t.module_id for t in sources.required}
        module_bytes = {module: sum(t.nbytes for t in sources.required
                                   if t.module_id == module and t.kind == "parameter") for module in modules}
        if module_bytes != self._profile["identity"]["module_parameter_bytes"]:
            raise ValueError("state inventory must match the complete profiled model")
        for layout in dynamic.layouts:
            flattened = tuple(module for stage in layout for module in stage)
            if flattened != tuple(self._profile["identity"]["module_order"]):
                raise ValueError("each target pipeline must contain the complete ordered model")
        slots = tuple(TargetSlot(pipeline, stage, stage_modules)
                      for pipeline, layout in enumerate(dynamic.layouts)
                      for stage, stage_modules in enumerate(layout))
        costs = migration_cost_matrix(sources, slots)
        matching = hungarian(costs)
        assignments = tuple(sorted(((slots[column], inventory.worker)
                                    for inventory, column in zip(sources.inventories, matching.columns)),
                                   key=lambda pair: (pair[0].pipeline, pair[0].stage)))
        local = {i.worker: set(i.tensors) for i in sources.inventories}
        actions = []
        for slot, worker in assignments:
            for tensor in sources.required:
                if tensor.module_id in slot.modules:
                    source = worker if tensor in local[worker] else sources.sources_for(tensor)[0]
                    actions.append(TensorAction(tensor, source, worker, slot))
        migrations = {f"transfer-{i:08d}": action for i, action in enumerate(actions) if not action.retained}
        transfer_graph = conflict_graph({key: (a.source.worker_id, a.destination.worker_id)
                                         for key, a in migrations.items()})
        migration_rounds = tuple(tuple(migrations[key] for key in row) for row in dsatur(transfer_graph))
        owners = {module: tuple(worker.worker_id for slot, worker in assignments if module in slot.modules)
                  for module in modules}
        sync_rounds = synchronization_rounds(owners)
        held = tuple((i.worker, t) for i in sources.inventories for t in sorted(i.tensors, key=lambda t: t.key))
        target_state = {(a.destination, a.tensor) for a in actions}
        release = tuple(pair for pair in held if pair not in target_state)
        assert matching.total_cost == sum(a.tensor.nbytes for a in actions if not a.retained)
        manifest_id = "migration-" + _hash({"plan_id": dynamic.plan_id,
                                          "committed_global_step": state.committed_global_step,
                                          "actions": tuple(asdict(action) for action in actions)})
        return MigrationManifest(manifest_id, dynamic, assignments, costs, matching.total_cost, tuple(actions),
                                 migration_rounds, sync_rounds, held, release)

    def estimate_transition(self, manifest: MigrationManifest, *,
                            unoverlapped_search_time_s: float) -> TransitionEstimate:
        """Measured two-rank costs as an explicit proxy; never extrapolate tensor sizes."""
        _finite("unoverlapped search time", unoverlapped_search_time_s)
        if manifest.plan.profile_hash != self._profile_hash:
            raise ValueError("manifest profile identity differs from transition profile")
        if self._bootstrap_time is None:
            raise MissingTransitionCalibrationError("missing transfer/bootstrap calibration for dynamic transition")
        round_times = []
        for row in manifest.migration_rounds:
            times = []
            for action in row:
                if action.tensor.nbytes not in self._transfer_times:
                    raise MissingTransitionCalibrationError(f"missing P2P calibration for tensor size {action.tensor.nbytes}")
                times.append(self._transfer_times[action.tensor.nbytes])
            round_times.append(max(times))
        migration = sum(round_times)
        return TransitionEstimate(unoverlapped_search_time_s, migration, self._bootstrap_time, {
            "migration_bytes": manifest.migration_bytes, "migration_round_times_s": tuple(round_times),
            "search_source": "caller measured search that did not overlap training",
            "transfer_source": "slower measured direction and endpoint mean, exact tensor size",
            "common_control_source": "measured two-rank bootstrap; account equally for both policies",
            "boundary": "two-rank homogeneous calibration proxy; target-scale bootstrap, edge bandwidth, "
                        "ACK and module/optimizer reconstruction overhead require runtime measurement",
        }, manifest.manifest_id)
