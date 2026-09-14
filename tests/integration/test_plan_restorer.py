from copy import deepcopy
from dataclasses import replace
from itertools import permutations

import pytest

from chameleon.contracts import UnrecoverableStateError, WorkerIdentity
from chameleon.restorer import MigrationAcknowledgements, Restorer, TargetSlot, migration_cost_matrix
from chameleon.state_sources import ADAMW_FIELDS, StateTensor, WorkerInventory, build_state_source_map
from test_dynamic_planner_oracle import make_planner, survivor_state


def required_metadata(profile):
    """Controlled parameter and optimizer bytes; not real trained state."""
    return tuple(StateTensor(f"{module}.weight", module, kind, () if kind == "step" else (1,),
                             "controlled", 4 if kind == "step" else size)
                 for module, size in profile["identity"]["module_parameter_bytes"].items()
                 for kind in ADAMW_FIELDS)


def calibrated_profile(profile, sizes):
    """Exact controlled aggregation oracle, using the existing calibration schema."""
    profile = deepcopy(profile)
    device = profile["identity"]["device"]
    records = []
    for rank in range(2):
        transfers = [{"source": src, "destination": dst, "tensor_bytes": size, "iteration": iteration,
                      "execution_time_s": size / 100 + src + rank + iteration,
                      "wall_time_s": size / 100 + src + rank + iteration + .5}
                     for src, dst in ((0, 1), (1, 0)) for size in sizes for iteration in range(2)]
        records.append({"rank": rank, "worker_id": f"calibration-{rank}", "generation": 0,
                        "pid": 101 + rank, "device_identity": deepcopy(device), "group_bootstrap_s": [2., 4.],
                        "transfers": transfers, "verified": True})
    profile["calibrations"] = [{"schema_version": 1, "device": "cpu", "backend": "gloo", "world_size": 2,
        "tensor_bytes": list(sizes), "iterations": 2, "bootstrap_rounds": 2, "records": records,
        "audit": {"port": 12345, "rendezvous_dir": "artifacts/controlled-calibration",
                  "workers": [{"pid": 101 + rank, "exitcode": 0, "alive": False} for rank in range(2)],
                  "leaked_pids": [], "rendezvous_removed": True, "port_listening": False,
                  "port_reusable": True, "clean": True}, "error": None}]
    return profile


@pytest.fixture
def combination():
    planner = make_planner(layers=2, nm=5, r_dp=(2,), r_pp=(1, 2))
    state = survivor_state(planner, 3, generation=2)
    profile = planner.estimator.profile
    required = required_metadata(profile)
    # Survivor fragments of an original symmetric DP2/PP2 layout.
    first = {"embedding", "blocks.0"}
    inventories = tuple(WorkerInventory(w, state.committed_global_step,
                        tuple(t for t in required if (t.module_id in first) == (index == 0)
                              or (index == 0 and t.module_id == "lm_head")))
                        for index, w in enumerate(state.workers))
    sources = build_state_source_map(state, required, inventories)
    dynamic = planner.best_dynamic_plan(state)
    restorer = Restorer(profile, expected_identity=profile["identity"])
    return planner, sources, dynamic, restorer


def bound_calibration(combination, sizes):
    from chameleon.estimators import Estimator
    from chameleon.planner import Planner

    original, sources, _, _ = combination
    profile = calibrated_profile(original.estimator.profile, sizes)
    estimator = Estimator(profile, expected_identity=profile["identity"], layer_modules=original.estimator.layer_modules)
    planner = Planner(estimator, config=original.config, r_dp=original.r_dp, r_pp=original.r_pp,
                      memory_capacity_bytes=original.memory_capacity_bytes)
    restorer = Restorer(profile, expected_identity=profile["identity"])
    return restorer, restorer.plan(planner.best_dynamic_plan(sources.state), sources), profile


def test_planner_estimator_restorer_manifest_against_independent_byte_oracle(combination):
    planner, sources, dynamic, restorer = combination
    manifest = restorer.plan(dynamic, sources)
    assert dynamic.pipeline_lengths == (1, 2)
    slots = tuple(slot for slot, _ in manifest.assignments)
    # Independent cost derivation uses tensor keys instead of production equality/matrix helper.
    matrix = tuple(tuple(sum(t.nbytes for t in sources.required if t.module_id in slot.modules
                             and t.key not in {local.key for local in inventory.tensors}) for slot in slots)
                   for inventory in sources.inventories)
    assert manifest.cost_matrix == matrix
    assert manifest.migration_bytes == min(sum(matrix[i][j] for i, j in enumerate(p))
                                          for p in permutations(range(len(slots))))
    expected_targets = {(worker, tensor) for slot, worker in manifest.assignments
                        for tensor in sources.required if tensor.module_id in slot.modules}
    actual_targets = {(a.destination, a.tensor) for a in manifest.actions}
    assert actual_targets == expected_targets
    assert len(manifest.actions) == len(expected_targets)
    for action in manifest.actions:
        assert action.tensor in next(i.tensors for i in sources.inventories if i.worker == action.source)
        if not action.retained:
            assert action.source in sources.sources_for(action.tensor)
    assert manifest.migration_bytes == sum(a.tensor.nbytes for a in manifest.actions if not a.retained)
    assert {a for row in manifest.migration_rounds for a in row} == {a for a in manifest.actions if not a.retained}
    for row in manifest.migration_rounds:
        devices = [w for a in row for w in (a.source, a.destination)]
        assert len(devices) == len(set(devices))
    for row in manifest.synchronization_rounds:
        devices = [w for slot, w in manifest.assignments for unit in row if unit in slot.modules]
        assert len(devices) == len(set(devices))
    assert {unit for row in manifest.synchronization_rounds for unit in row} == set(planner.estimator.profile["identity"]["module_order"])


def test_bytes_and_optimizer_inventory_change_matching_cost(combination):
    _, sources, _, _ = combination
    state = sources.state
    # Parameter-only cost favors a retaining x; optimizer bytes reverse that preference.
    required = tuple(StateTensor(f"{module}.w", module, kind, () if kind == "step" else (1,), "controlled",
                                 4 if kind == "step" else (parameter if kind == "parameter" else moment))
                     for module, parameter, moment in (("x", 80, 8), ("y", 8, 800)) for kind in ADAMW_FIELDS)
    inventories = tuple(WorkerInventory(w, state.committed_global_step,
                        tuple(t for t in required if t.module_id == ("x" if i == 0 else "y")))
                        for i, w in enumerate(state.workers))
    sources = build_state_source_map(state, required, inventories)
    slots = (TargetSlot(0, 0, ("x", "y")), TargetSlot(1, 0, ("x",)), TargetSlot(1, 1, ("y",)))
    matrix = migration_cost_matrix(sources, slots)
    assert matrix == ((1612, 0, 1612), (100, 100, 0), (100, 100, 0))
    assert 8 < 80 and matrix[0][0] > matrix[1][0]


def test_source_state_remains_held_until_all_acks(combination):
    _, sources, dynamic, restorer = combination
    manifest = restorer.plan(dynamic, sources)
    before = sources.inventories
    acknowledgements = MigrationAcknowledgements(manifest)
    assert set(manifest.held_sources) == {(i.worker, t) for i in before for t in i.tensors}
    assert manifest.release_after_ack
    with pytest.raises(RuntimeError, match="all target ACKs"):
        acknowledgements.releasable_sources()
    for action in manifest.actions[:-1]:
        acknowledgements.acknowledge(action, manifest_id=manifest.manifest_id)
    assert not acknowledgements.complete
    with pytest.raises(RuntimeError):
        acknowledgements.releasable_sources()
    acknowledgements.acknowledge(manifest.actions[-1], manifest_id=manifest.manifest_id)
    assert acknowledgements.complete
    assert acknowledgements.releasable_sources() == manifest.release_after_ack
    targets = {(a.destination, a.tensor) for a in manifest.actions}
    assert set(manifest.release_after_ack) == set(manifest.held_sources) - targets
    assert sources.inventories == before
    with pytest.raises(ValueError, match="duplicate"):
        acknowledgements.acknowledge(manifest.actions[-1], manifest_id=manifest.manifest_id)


def test_manifest_is_stable_for_reordered_survivors(combination):
    _, sources, dynamic, restorer = combination
    shuffled = build_state_source_map(replace(sources.state, workers=sources.state.workers[::-1]),
                                      sources.required[::-1], sources.inventories[::-1])
    assert restorer.plan(dynamic, sources) == restorer.plan(dynamic, shuffled)


@pytest.mark.parametrize("change", ("generation", "batch", "survivor", "profile", "layout", "length", "inventory"))
def test_reject_plan_or_inventory_mismatch(combination, change):
    planner, sources, dynamic, restorer = combination
    if change == "generation":
        dynamic = replace(dynamic, generation=1, survivors=tuple(replace(w, generation=1) for w in dynamic.survivors))
    elif change == "batch":
        dynamic = replace(dynamic, global_batch_size=100)
    elif change == "survivor":
        dynamic = replace(dynamic, survivors=(replace(dynamic.survivors[0], worker_id="dead"), *dynamic.survivors[1:]))
    elif change == "profile":
        profile = deepcopy(planner.estimator.profile)
        profile["identity"]["model_hash"] = "b" * 64
        restorer = Restorer(profile, expected_identity=profile["identity"])
    elif change == "layout":
        with pytest.raises(ValueError, match="layout"):
            replace(dynamic, layouts=tuple(tuple(stage[::-1] for stage in layout) for layout in dynamic.layouts))
        return
    elif change == "length":
        with pytest.raises(ValueError):
            replace(dynamic, pipeline_lengths=(2, 1))
        return
    else:
        required = tuple(replace(t, nbytes=t.nbytes * 2) for t in sources.required)
        inventories = tuple(replace(i, tensors=tuple(replace(t, nbytes=t.nbytes * 2) for t in i.tensors))
                            for i in sources.inventories)
        sources = build_state_source_map(sources.state, required, inventories)
    with pytest.raises(ValueError):
        restorer.plan(dynamic, sources)


@pytest.mark.parametrize("module", ("embedding", "final_norm", "lm_head"))
def test_loss_of_endpoint_source_fails_before_manifest(combination, module):
    _, sources, _, _ = combination
    inventories = tuple(replace(i, tensors=tuple(t for t in i.tensors if t.module_id != module))
                        for i in sources.inventories)
    with pytest.raises(UnrecoverableStateError, match=module):
        build_state_source_map(sources.state, sources.required, inventories)


def test_transition_refuses_missing_calibration(combination):
    _, sources, dynamic, restorer = combination
    manifest = restorer.plan(dynamic, sources)
    with pytest.raises(ValueError, match="missing transfer/bootstrap"):
        restorer.estimate_transition(manifest, unoverlapped_search_time_s=0.)


def test_calibrated_transition_matches_independent_round_oracle(combination):
    restorer, manifest, profile = bound_calibration(combination, sorted({t.nbytes for t in combination[1].required}))
    estimate = restorer.estimate_transition(manifest, unoverlapped_search_time_s=1.25)
    expected_rounds = tuple(max(a.tensor.nbytes / 100 + 2.5 for a in row) for row in manifest.migration_rounds)
    assert estimate.migration_time_s == pytest.approx(sum(expected_rounds))
    assert estimate.common_control_time_s == 3.
    assert estimate.estimated_transition_time_s == pytest.approx(1.25 + sum(expected_rounds))
    assert estimate.estimated_total_time_s == pytest.approx(1.25 + sum(expected_rounds) + 3.)
    assert "proxy" in estimate.derivation["boundary"]
    profile["calibrations"] = []
    assert restorer.estimate_transition(manifest, unoverlapped_search_time_s=1.25) == estimate
    for invalid in (-1., float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            restorer.estimate_transition(manifest, unoverlapped_search_time_s=invalid)


def test_transition_refuses_uncalibrated_tensor_size(combination):
    _, sources, dynamic, restorer = combination
    manifest = restorer.plan(dynamic, sources)
    sizes = {a.tensor.nbytes for a in manifest.actions if not a.retained}
    assert len(sizes) > 1
    restorer, manifest, _ = bound_calibration(combination, sorted(sizes)[1:])
    with pytest.raises(ValueError, match="missing P2P calibration"):
        restorer.estimate_transition(manifest, unoverlapped_search_time_s=0.)


def test_zero_migration_still_requires_bootstrap_measurement(combination):
    _, sources, dynamic, restorer = combination
    inventories = tuple(replace(i, tensors=sources.required) for i in sources.inventories)
    sources = build_state_source_map(sources.state, sources.required, inventories)
    manifest = restorer.plan(dynamic, sources)
    assert manifest.migration_bytes == 0 and manifest.migration_rounds == ()
    with pytest.raises(ValueError, match="bootstrap"):
        restorer.estimate_transition(manifest, unoverlapped_search_time_s=0.)
    restorer, manifest, _ = bound_calibration((combination[0], sources, dynamic, restorer), [4])
    estimate = restorer.estimate_transition(manifest, unoverlapped_search_time_s=0.)
    assert estimate.migration_time_s == 0. and estimate.estimated_transition_time_s == 0.
    assert estimate.common_control_time_s == estimate.estimated_total_time_s == 3.


def test_unknown_ack_and_different_profile_transition_are_rejected(combination):
    planner, sources, dynamic, restorer = combination
    manifest = restorer.plan(dynamic, sources)
    unknown = replace(manifest.actions[0], destination=WorkerIdentity("unknown", 100, 2))
    with pytest.raises(ValueError, match="unknown"):
        MigrationAcknowledgements(manifest).acknowledge(unknown, manifest_id=manifest.manifest_id)
    profile = deepcopy(planner.estimator.profile)
    profile["identity"]["model_hash"] = "b" * 64
    restorer = Restorer(profile, expected_identity=profile["identity"])
    with pytest.raises(ValueError, match="profile identity"):
        restorer.estimate_transition(manifest, unoverlapped_search_time_s=0.)


@pytest.mark.parametrize("slots", ((TargetSlot(0, 0, ("unknown",)),),
                                   tuple(TargetSlot(0, 0, ("embedding",)) for _ in range(3))))
def test_invalid_target_slots(combination, slots):
    with pytest.raises(ValueError):
        migration_cost_matrix(combination[1], slots)


@pytest.mark.parametrize("overrides", ({"pipeline": -1}, {"stage": True}, {"modules": ()},
                                       {"modules": ["embedding"]}, {"modules": ("x", "x")}))
def test_slot_metadata_validation(overrides):
    with pytest.raises(ValueError):
        replace(TargetSlot(0, 0, ("embedding",)), **overrides)


@pytest.mark.parametrize("counts", ((0, 5), (-1, 6), (1, 1), (5, 5)))
def test_changed_batch_distribution_is_rejected(combination, counts):
    _, sources, dynamic, restorer = combination
    with pytest.raises(ValueError, match="micro-batch"):
        restorer.plan(replace(dynamic, pipeline_micro_batches=counts), sources)


def test_same_total_changed_batch_distribution_is_rejected(combination):
    _, sources, dynamic, restorer = combination
    counts = dynamic.pipeline_micro_batches[::-1]
    assert counts != dynamic.pipeline_micro_batches
    with pytest.raises(ValueError, match="micro-batch"):
        restorer.plan(replace(dynamic, pipeline_micro_batches=counts), sources)


def test_plan_binds_full_worker_identity(combination):
    _, sources, dynamic, restorer = combination
    workers = tuple(replace(w, rank=w.rank + 10) for w in sources.state.workers)
    state = replace(sources.state, workers=workers)
    inventories = tuple(replace(i, worker=workers[index]) for index, i in enumerate(sources.inventories))
    sources = build_state_source_map(state, sources.required, inventories)
    with pytest.raises(ValueError, match="survivor"):
        restorer.plan(dynamic, sources)


def test_plan_binds_entire_profile_content(combination):
    planner, sources, dynamic, _ = combination
    profile = deepcopy(planner.estimator.profile)
    profile["ema_alpha"] = .25
    restorer = Restorer(profile, expected_identity=profile["identity"])
    with pytest.raises(ValueError, match="profile"):
        restorer.plan(dynamic, sources)


def test_ack_is_bound_to_recovery_step(combination):
    _, sources, dynamic, restorer = combination
    old = restorer.plan(dynamic, sources)
    state = replace(sources.state, committed_global_step=sources.state.committed_global_step + 1)
    inventories = tuple(replace(i, committed_global_step=state.committed_global_step) for i in sources.inventories)
    current = restorer.plan(dynamic, build_state_source_map(state, sources.required, inventories))
    assert old.manifest_id != current.manifest_id
    acknowledgements = MigrationAcknowledgements(current)
    with pytest.raises(ValueError, match="manifest"):
        acknowledgements.acknowledge(old.actions[0], manifest_id=old.manifest_id)
    assert not acknowledgements.complete


def test_shared_bootstrap_is_separate_from_policy_transition(combination):
    sizes = sorted({t.nbytes for t in combination[1].required})
    restorer, manifest, _ = bound_calibration(combination, sizes)
    result = restorer.estimate_transition(manifest, unoverlapped_search_time_s=1.25)
    assert result.estimated_transition_time_s == result.unoverlapped_search_time_s + result.migration_time_s
    assert result.common_control_time_s == 3.
    assert result.estimated_total_time_s == result.estimated_transition_time_s + result.common_control_time_s


def test_repeated_transition_does_not_reparse_calibration(combination, monkeypatch):
    import chameleon.transfer_calibration as calibration

    sizes = sorted({t.nbytes for t in combination[1].required})
    restorer, manifest, _ = bound_calibration(combination, sizes)
    original = calibration.validate_calibration
    calls = []

    def count(report):
        calls.append(report)
        return original(report)

    monkeypatch.setattr(calibration, "validate_calibration", count)
    result = restorer.estimate_transition(manifest, unoverlapped_search_time_s=1.25)
    assert restorer.estimate_transition(manifest, unoverlapped_search_time_s=1.25) == result
    assert not calls


def test_time_estimate_must_match_target_layout(combination):
    _, sources, dynamic, restorer = combination
    first, second = dynamic.layouts[1]
    layouts = (dynamic.layouts[0], ((*first, second[0]), second[1:]))
    assert all(layouts[1])
    with pytest.raises(ValueError, match="layout"):
        restorer.plan(replace(dynamic, layouts=layouts), sources)


def test_time_estimate_must_match_profile_content(combination):
    from chameleon.estimators import Estimator

    planner, sources, dynamic, restorer = combination
    profile = deepcopy(planner.estimator.profile)
    profile["ema_alpha"] = .25
    estimator = Estimator(profile, expected_identity=profile["identity"], layer_modules=planner.estimator.layer_modules)
    time = estimator.dynamic_time(dynamic.layouts, dynamic.pipeline_micro_batches,
                                  global_micro_batches=planner.global_micro_batches)
    with pytest.raises(ValueError, match="profile"):
        restorer.plan(replace(dynamic, time=time), sources)


@pytest.mark.parametrize("container", ("layouts", "layout", "stage", "batches"))
def test_plan_identity_inputs_are_immutable(combination, container):
    dynamic = combination[2]
    if container == "layouts":
        overrides = {"layouts": list(dynamic.layouts)}
    elif container == "layout":
        overrides = {"layouts": (list(dynamic.layouts[0]), *dynamic.layouts[1:])}
    elif container == "stage":
        overrides = {"layouts": ((list(dynamic.layouts[0][0]),), *dynamic.layouts[1:])}
    else:
        overrides = {"pipeline_micro_batches": list(dynamic.pipeline_micro_batches), "time": None}
    with pytest.raises(ValueError, match="tuple"):
        replace(dynamic, **overrides)


def test_time_estimate_copies_mutable_input_distributions(combination):
    planner, _, dynamic, _ = combination
    layouts = [[list(stage) for stage in layout] for layout in dynamic.layouts]
    counts = list(dynamic.pipeline_micro_batches)
    result = planner.estimator.dynamic_time(layouts, counts, global_micro_batches=planner.global_micro_batches)
    layouts[0][0][0] = "changed"
    counts[0] += 1
    assert result.derivation["layouts"] == dynamic.layouts
    assert tuple(result.derivation["pipeline_micro_batches"]) == dynamic.pipeline_micro_batches
