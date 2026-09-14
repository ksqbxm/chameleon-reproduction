from dataclasses import asdict, replace

import pytest

from chameleon.contracts import UnrecoverableStateError
from chameleon.recovery import validate_manifest, validate_transfer_reports
from test_plan_restorer import combination


def test_live_manifest_validates_without_tensor_values(combination):
    _, sources, dynamic, restorer = combination
    validate_manifest(restorer.plan(dynamic, sources), sources)


@pytest.mark.parametrize("fault", ["stale_manifest", "stale_step", "round_conflict"])
def test_manifest_binding_and_communication_order_are_checked(combination, fault):
    _, sources, dynamic, restorer = combination
    manifest = restorer.plan(dynamic, sources)
    if fault == "stale_manifest":
        manifest = replace(manifest, manifest_id="stale")
    elif fault == "stale_step":
        sources = replace(sources, state=replace(sources.state, committed_global_step=8))
    else:
        transfers = tuple(a for row in manifest.migration_rounds for a in row)
        manifest = replace(manifest, migration_rounds=(transfers,))
    with pytest.raises(ValueError):
        validate_manifest(manifest, sources)


def test_missing_unique_endpoint_inventory_is_unrecoverable(combination):
    from chameleon.state_sources import build_state_source_map
    _, sources, _, _ = combination
    inventories = tuple(replace(i, tensors=tuple(t for t in i.tensors
                        if not (t.module_id == "lm_head" and t.kind == "exp_avg"))) for i in sources.inventories)
    with pytest.raises(UnrecoverableStateError, match="lm_head"):
        build_state_source_map(sources.state, sources.required, inventories)


@pytest.mark.parametrize("field", ["actions", "migration_rounds", "held_sources", "release_after_ack"])
def test_incomplete_manifest_fails_before_transfer(combination, field):
    _, sources, dynamic, restorer = combination
    manifest = restorer.plan(dynamic, sources)
    with pytest.raises((ValueError, UnrecoverableStateError)):
        validate_manifest(replace(manifest, **{field: getattr(manifest, field)[1:]}), sources)


@pytest.mark.parametrize("field", ["migration_bytes", "cost_matrix", "assignments", "source"])
def test_manifest_rejects_wrong_matching_and_source(combination, field):
    _, sources, dynamic, restorer = combination
    manifest = restorer.plan(dynamic, sources)
    if field == "source":
        action = next(a for a in manifest.actions if not a.retained)
        wrong = replace(action, source=action.destination)
        value = replace(manifest, actions=tuple(wrong if a == action else a for a in manifest.actions))
    else:
        replacement = {"migration_bytes": manifest.migration_bytes + 1, "cost_matrix": (),
                       "assignments": manifest.assignments[:-1]}[field]
        value = replace(manifest, **{field: replacement})
    with pytest.raises((ValueError, UnrecoverableStateError)):
        validate_manifest(value, sources)


def transfer_reports(manifest):
    def digest(tensor):
        return tensor.parameter_name + ":" + tensor.kind
    result = []
    for _, worker in manifest.assignments:
        target = [a for a in manifest.actions if a.destination == worker]
        def row(a):
            return dict(key=list(a.tensor.key), source=a.source.worker_id, destination=a.destination.worker_id,
                        digest=digest(a.tensor), tensor_bytes=a.tensor.nbytes)
        result.append(dict(worker=worker, manifest_id=manifest.manifest_id,
            tensors=[asdict(a.tensor) for a in target],
            hashes=[dict(key=list(a.tensor.key), digest=digest(a.tensor)) for a in target],
            sends=[row(a) for a in manifest.actions if not a.retained and a.source == worker],
            receives=[row(a) for a in target if not a.retained],
            retained=[dict(key=list(a.tensor.key), digest=digest(a.tensor), same_object=True) for a in target if a.retained],
            held_before_ack=[dict(key=list(t.key), digest=digest(t), unchanged=True)
                             for w, t in manifest.held_sources if w == worker]))
    return result


def test_complete_source_and_target_audit_accepts_metadata_oracle(combination):
    _, sources, dynamic, restorer = combination
    manifest = restorer.plan(dynamic, sources)
    validate_transfer_reports(manifest, transfer_reports(manifest))


@pytest.mark.parametrize("fault", ["missing_target", "digest", "released_source", "copied_local",
                                   "missing_send", "duplicate_receive", "manifest", "missing_worker"])
def test_partial_or_invalid_target_ack_cannot_release_sources(combination, fault):
    _, sources, dynamic, restorer = combination
    manifest = restorer.plan(dynamic, sources)
    reports = transfer_reports(manifest)
    if fault == "missing_worker":
        reports.pop()
    elif fault == "missing_target":
        reports[0]["hashes"].pop()
    elif fault == "digest":
        reports[0]["hashes"][0]["digest"] = "corrupt"
    elif fault == "released_source":
        reports[0]["held_before_ack"].pop()
    elif fault == "copied_local":
        next(r for r in reports if r["retained"])["retained"][0]["same_object"] = False
    elif fault == "missing_send":
        next(r for r in reports if r["sends"])["sends"].pop()
    elif fault == "duplicate_receive":
        row = next(r for r in reports if r["receives"])
        row["receives"].append(row["receives"][0])
    else:
        reports[0]["manifest_id"] = "stale"
    with pytest.raises((ValueError, UnrecoverableStateError)):
        validate_transfer_reports(manifest, reports)


def _metadata_recovery_worker(topology, rank, device, backend, dtype, store, timeout, connection,
                              origin, lr, weight_decay, behavior, directory, capture, *, required, fault):
    """Control protocol only: no model, training tensors, Gloo/NCCL, or mocked groups."""
    import os
    from chameleon.runtime import _write_reply
    worker, state = topology.ranks[rank], topology.state
    def inspect():
        p, s = topology.location(rank)
        rows = [t for t in required if t.module_id in topology.layouts[p][s]]
        return dict(tensors=[asdict(t) for t in rows],
                    hashes=[dict(key=list(t.key), digest=t.parameter_name + ":" + t.kind) for t in rows])
    def reply(kind, **values):
        _write_reply(connection, directory, rank, kind, worker,
                     **({} if kind == "ready" else dict(step_id=state.committed_global_step + (kind == "ack"))),
                     **values)
    try:
        reply("ready", pid=os.getpid(), parameter_names=[t["key"][0] for t in inspect()["hashes"] if t["key"][1] == "parameter"])
        while True:
            token = connection.recv_bytes(maxlength=7)
            if token == b"stop":
                return
            if token == b"inspect":
                reply("inspect", **inspect())
            elif token == b"step":
                p, s = topology.location(rank)
                batches = topology.micro_batches(state, p)
                reply("ack", pipeline=p, stage=s, profile_step=None,
                    loss_batches=[dict(sample_ids=list(b), loss_sum=float(len(b))) for b in batches]
                                 if s == topology.pipeline_lengths[p] - 1 else [],
                    global_sample_count=state.global_batch_size, loss_global_sum=float(state.global_batch_size))
                assert connection.recv_bytes(maxlength=7) == b"commit"
                state = replace(state, committed_global_step=state.committed_global_step + 1)
                reply("safe")
            else:
                assert token == b"recover"
                target, manifest, _ = connection.recv()
                new_worker = next(w for w in target.ranks if w.worker_id == worker.worker_id)
                reply("joined", pid=os.getpid(), new_worker=asdict(new_worker))
                assert connection.recv_bytes(maxlength=7) == b"move"
                report = (next(r for r in transfer_reports(manifest) if r["worker"] == worker)
                          if manifest is not None else inspect())
                if fault == "digest" and rank == 0:
                    report["hashes"][0]["digest"] = "corrupt"
                reply("moved", **{k: v for k, v in report.items() if k != "worker"})
                assert connection.recv_bytes(maxlength=7) == b"groups"
                reply("staged", pid=os.getpid(), backend="metadata", new_worker=asdict(new_worker),
                      parameter_names=sorted({t["parameter_name"] for t in report["tensors"]}))
                assert connection.recv_bytes(maxlength=7) == b"install"
                reply("set", new_worker=asdict(new_worker), released_after_ack=True)
                topology, rank, worker, state = target, new_worker.rank, new_worker, target.state
    finally:
        connection.close()


def _metadata_decision(recovery, config):
    """Independent controlled manifest oracle for the controller protocol."""
    from chameleon.coloring import conflict_graph, dsatur
    from chameleon.decision_center import PolicyCandidate, select_policy
    from chameleon.estimators import TimeEstimate
    from chameleon.hungarian import hungarian
    from chameleon.planner import DynamicPlan
    from chameleon.profiler import _hash
    from chameleon.restorer import MigrationManifest, TargetSlot, TensorAction, TransitionEstimate, synchronization_rounds
    from chameleon.state_sources import build_state_source_map

    sources = build_state_source_map(recovery.survivor_state, recovery.required, recovery.inventories)
    modules = ("embedding", "blocks.0", "blocks.1", "final_norm", "lm_head")
    layouts = ((modules,), (("embedding", "blocks.0"), modules[2:]))
    time = TimeEstimate(1., dict(layouts=layouts, profile_hash="controlled", pipeline_micro_batches=(2, 4),
        global_micro_batches=6, profile_identity={"config_hash": _hash(asdict(config))}))
    plan = DynamicPlan(11, sources.state.generation, sources.state.workers, "controlled", (1, 2), (2, 4), layouts, time, ())
    slots = tuple(TargetSlot(p, s, row) for p, layout in enumerate(layouts) for s, row in enumerate(layout))
    costs = tuple(tuple(sum(t.nbytes for t in sources.required if t.module_id in slot.modules and t not in i.tensors)
                        for slot in slots) for i in sources.inventories)
    matching = hungarian(costs)
    assignments = tuple(sorted([(slots[col], i.worker) for col, i in zip(matching.columns, sources.inventories)],
                               key=lambda pair: (pair[0].pipeline, pair[0].stage)))
    local = {i.worker: set(i.tensors) for i in sources.inventories}
    actions = tuple(TensorAction(t, w if t in local[w] else sources.sources_for(t)[0], w, slot)
                    for slot, w in assignments for t in sources.required if t.module_id in slot.modules)
    transfers = {str(i): a for i, a in enumerate(actions) if not a.retained}
    rounds = tuple(tuple(transfers[key] for key in row) for row in dsatur(conflict_graph(
        {key: (a.source.worker_id, a.destination.worker_id) for key, a in transfers.items()})))
    held = tuple((i.worker, t) for i in sources.inventories for t in i.tensors)
    targets = {(a.destination, a.tensor) for a in actions}
    identity = "migration-" + _hash(dict(plan_id=plan.plan_id, committed_global_step=sources.state.committed_global_step,
                                       actions=tuple(asdict(a) for a in actions)))
    sync = synchronization_rounds({m: tuple(w.worker_id for slot, w in assignments if m in slot.modules) for m in modules})
    manifest = MigrationManifest(identity, plan, assignments, costs, matching.total_cost, actions, rounds, sync,
                                 held, tuple(pair for pair in held if pair not in targets))
    transition = TransitionEstimate(0., .01, None, {"source": "controlled metadata protocol test"}, identity)
    candidate = PolicyCandidate(plan.plan_id, "dynamic", 11, sources.state.generation, 1., transition, None, (), manifest, {})
    return select_policy((candidate,), 100.)


@pytest.fixture
def metadata_recovery_runtime(combination, monkeypatch):
    from functools import partial
    from chameleon import ClusterState, ModelConfig, WorkerIdentity
    from chameleon import runtime as module
    from chameleon.runtime import SymmetricRuntime, SymmetricTopology

    _, sources, _, _ = combination
    required = tuple(t for t in sources.required if t.module_id != "extra")
    worker = partial(_metadata_recovery_worker, required=required, fault="none")
    monkeypatch.setattr(module, "validate_device", lambda *args: "metadata")
    monkeypatch.setattr(module, "_runtime_worker", worker)
    config = ModelConfig(num_layers=2, global_batch_size=11, micro_batch_size=2)
    topology = SymmetricTopology(ClusterState(tuple(WorkerIdentity(f"stable-{i}", i, 0) for i in range(4)), 11),
                                config, (("embedding",), ("blocks.0", "blocks.1", "final_norm", "lm_head")))
    return SymmetricRuntime(topology, timeout_s=15)


def _prepare_failure(runtime):
    from conftest import _kill_at_safe_point
    for _ in range(3):
        runtime.train_step()
    runtime.inspect_state()
    failure = _kill_at_safe_point(runtime, (1,))
    return failure, runtime.recovery_state(failure)


@pytest.mark.parametrize("fault", ["none", "digest"])
def test_survivor_control_protocol_commits_only_after_all_target_acks(metadata_recovery_runtime, monkeypatch, fault):
    from functools import partial
    from chameleon import runtime as module
    runtime = metadata_recovery_runtime
    monkeypatch.setattr(module, "_runtime_worker", partial(module._runtime_worker, fault=fault))
    topology = runtime.topology
    with runtime:
        failure, recovery = _prepare_failure(runtime)
        decision = _metadata_decision(recovery, topology.config)
        if fault == "none":
            runtime.recover(failure, decision)
            assert runtime.state.generation == 1 and runtime.state.committed_global_step == 3
            assert len(runtime.processes) == 3
            assert runtime.train_step()["step_id"] == 4
        else:
            with pytest.raises(UnrecoverableStateError, match="target state differs"):
                runtime.recover(failure, decision)
            assert runtime.state.generation == 0 and runtime.state.committed_global_step == 3
            assert runtime.topology is topology and not runtime.recoveries
    assert runtime.audit["clean"] and not runtime.audit["leaked_pids"]
    assert len(runtime.audit["workers"]) == 4


def _rerouting_decision(recovery):
    from chameleon.decision_center import PolicyCandidate, ReroutingPlan, select_policy
    from chameleon.estimators import TimeEstimate
    from chameleon.restorer import TransitionEstimate
    plan = ReroutingPlan(recovery, "controlled", TimeEstimate(1., {}), (), ())
    candidate = PolicyCandidate(plan.plan_id, "rerouting", recovery.cluster.global_batch_size,
        recovery.cluster.generation, 1., TransitionEstimate(0., 0., None, {}, None), None, (), plan, {})
    return select_policy((candidate,), 100.)


def test_rerouting_rejects_changed_retained_tensor_values(metadata_recovery_runtime, monkeypatch):
    from functools import partial
    from chameleon import runtime as module
    runtime = metadata_recovery_runtime
    monkeypatch.setattr(module, "_runtime_worker", partial(module._runtime_worker, fault="digest"))
    with runtime:
        failure, recovery = _prepare_failure(runtime)
        with pytest.raises(UnrecoverableStateError, match="survivor state changed"):
            runtime.recover(failure, _rerouting_decision(recovery))
        assert runtime.state.generation == 0 and not runtime.recoveries
    assert runtime.audit["clean"]


def test_dynamic_recovery_binds_sources_to_pre_rebuild_hashes(metadata_recovery_runtime, monkeypatch):
    from chameleon import recovery as module
    runtime = metadata_recovery_runtime
    with runtime:
        failure, recovery = _prepare_failure(runtime)
        decision = _metadata_decision(recovery, runtime.topology.config)
        validate = module.validate_transfer_reports
        def change_source_and_target(manifest, reports):
            # A changed source and consistently changed targets used to pass all post-rebuild ACK checks.
            for report in reports:
                for field in ("hashes", "held_before_ack", "retained", "sends", "receives"):
                    for row in report[field]:
                        row["digest"] = "changed:" + row["digest"]
            validate(manifest, reports)
        monkeypatch.setattr(module, "validate_transfer_reports", change_source_and_target)
        with pytest.raises(UnrecoverableStateError, match="survivor state changed"):
            runtime.recover(failure, decision)
        assert runtime.state.generation == 0 and not runtime.recoveries
    assert runtime.audit["clean"]


def test_dynamic_recovery_rejects_other_model_config_before_group_rebuild(metadata_recovery_runtime):
    from chameleon.runtime import RuntimeErrorWithAudit
    runtime = metadata_recovery_runtime
    with runtime:
        failure, recovery = _prepare_failure(runtime)
        decision = _metadata_decision(recovery, replace(runtime.topology.config, seed=999))
        with pytest.raises(RuntimeErrorWithAudit, match="model config"):
            runtime.recover(failure, decision)
        assert len(runtime._stores) == 1 and runtime.state.generation == 0
    assert runtime.audit["clean"]


def test_recovery_preflight_consumes_the_same_hard_timeout(metadata_recovery_runtime, monkeypatch):
    import time
    from chameleon import recovery as module
    from chameleon.runtime import RuntimeErrorWithAudit
    runtime = metadata_recovery_runtime
    with runtime:
        failure, recovery = _prepare_failure(runtime)
        decision = _metadata_decision(recovery, runtime.topology.config)
        validate, monotonic = module.validate_manifest, time.monotonic
        def expire_after_preflight(*args):
            validate(*args)
            monkeypatch.setattr(time, "monotonic", lambda: monotonic() + runtime.timeout_s + 1)
        monkeypatch.setattr(module, "validate_manifest", expire_after_preflight)
        with pytest.raises(RuntimeErrorWithAudit, match="hard timeout"):
            runtime.recover(failure, decision)
        assert len(runtime._stores) == 1 and runtime.state.generation == 0
    assert runtime.audit["clean"]


def test_recovery_state_preserves_existing_missing_logical_slots(metadata_recovery_runtime):
    from conftest import _kill_at_safe_point
    runtime = metadata_recovery_runtime
    with runtime:
        failure, recovery = _prepare_failure(runtime)
        runtime.recover(failure, _rerouting_decision(recovery))
        # One stage-zero replica survives a further loss; the already-missing tail slot stays absent.
        failure = _kill_at_safe_point(runtime, (0,))
        recovery = runtime.recovery_state(failure)
        assert recovery.pipeline_workers[0][1] is None
        assert len(recovery.survivor_state.workers) == 2
        runtime.recover(failure, _rerouting_decision(recovery))
        assert runtime.state.generation == 2 and runtime.state.committed_global_step == 3
        assert runtime.topology.pipeline_ranks == ((None, None), (0, 1))
        assert len(runtime.processes) == 2
    assert runtime.audit["clean"]
    assert len(runtime.audit["workers"]) == 4
    assert len(runtime.audit["rendezvous_files"]) == 3


def test_recovery_refreshes_live_state_after_cached_schema_was_inspected(metadata_recovery_runtime):
    from conftest import _kill_at_safe_point
    runtime = metadata_recovery_runtime
    with runtime:
        for _ in range(3):
            runtime.train_step()
        runtime.inspect_state()
        runtime.train_step()
        failure = _kill_at_safe_point(runtime, (1,))
        recovery = runtime.recovery_state(failure)
        assert all(i.committed_global_step == 4 for i in recovery.inventories)
        runtime.recover(failure, _metadata_decision(recovery, runtime.topology.config))
        assert runtime.state.committed_global_step == 4
        assert runtime.train_step()["step_id"] == 5
    assert runtime.audit["clean"]
