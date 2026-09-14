from collections import Counter
from copy import deepcopy
from dataclasses import asdict, replace
from itertools import cycle, permutations
import json

import pytest

from chameleon.contracts import ClusterState, DecisionResult, FailureEvent, UnrecoverableStateError, WorkerIdentity
from chameleon.decision_center import DecisionCenter, NoUsablePolicyError, RecoveryState
from chameleon.estimators import Estimator
from chameleon.planner import Planner
from chameleon.restorer import MigrationManifest, Restorer
from chameleon.state_sources import WorkerInventory
from test_dynamic_planner_oracle import controlled_profile, exhaustive_oracle
from test_plan_restorer import calibrated_profile, required_metadata


def scenario(*, nm=4, dp=2, failures=(0,), capacity=10000, r_dp=(1, 2, 3),
             r_pp=(1, 2, 3, 4), calibration=True):
    config, profile = controlled_profile(2, nm)
    required = required_metadata(profile)
    if calibration:
        profile = calibrated_profile(profile, sorted({t.nbytes for t in required}))
    workers = tuple(WorkerIdentity(f"worker-{i}", i, 2) for i in range(dp * 2))
    cluster = ClusterState(workers, config.global_batch_size, 2, 7)
    layout = (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head", "extra"))
    pipelines = tuple(tuple(workers[2 * i:2 * i + 2]) for i in range(dp))
    inventories = tuple(WorkerInventory(w, 7, tuple(t for t in required if t.module_id in layout[i % 2]))
                        for i, w in enumerate(workers) if i not in failures)
    event = FailureEvent(tuple(workers[i].worker_id for i in failures), 2, 7)
    partitions = tuple(nm // dp + (i < nm % dp) for i in range(dp))
    state = RecoveryState(cluster, event, (layout,) * dp, pipelines, partitions, required, inventories)
    center = DecisionCenter(config, expected_identity=profile["identity"], r_dp=r_dp, r_pp=r_pp,
                            memory_capacity_bytes=capacity)
    return center, state, profile


@pytest.fixture(autouse=True)
def controlled_search_clock(monkeypatch):
    # Controlled mathematical timing only; the real search itself still executes.
    times = cycle((100., 100.25))
    monkeypatch.setattr("chameleon.decision_center.perf_counter", lambda: next(times))


def score_oracle(candidate, duration):
    # Independent algebra: useful samples per window / duration; no production scorer.
    if not candidate.feasible or duration <= candidate.estimated_transition_time_s:
        return None
    samples = (duration - candidate.estimated_transition_time_s) / candidate.estimated_step_time_s * candidate.global_batch_size
    return samples / duration


def test_evaluation_combines_real_planner_estimator_and_restorer_without_selecting(monkeypatch):
    center, state, profile = scenario()
    calls = Counter()
    for cls, name in ((Planner, "best_dynamic_plan"), (Estimator, "dynamic_time"),
                      (Estimator, "stage_durations"), (Restorer, "plan"), (Restorer, "estimate_transition")):
        original = getattr(cls, name)

        def counted(*args, _original=original, _name=name, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(cls, name, counted)
    before = deepcopy((state, profile))
    rerouting, dynamic = center.evaluate_candidates(state, profile)
    assert rerouting.feasible and dynamic.feasible
    assert not isinstance(rerouting, DecisionResult) and not isinstance(dynamic, DecisionResult)
    assert "score" not in asdict(dynamic) and "D" not in dynamic.derivation
    assert all(calls[name] > 0 for name in ("best_dynamic_plan", "dynamic_time", "stage_durations", "plan", "estimate_transition"))
    assert calls["best_dynamic_plan"] == calls["plan"] == calls["estimate_transition"] == 1
    assert (state, profile) == before
    # Eq.12 hand calculation: DP2/PP2, global Nm4 -> Nm2, one lost stage.
    forward = max(.5 + 1., 1.25 + .25 + .75 + .125)
    backward = max(.25 + 2., 2.5 + .5 + 1. + .25)
    assert rerouting.estimated_step_time_s == (2 + 2 - 1 + 2) * (forward + backward)
    assert rerouting.derivation["time"]["equation"] == 12
    assert rerouting.estimated_transition_time_s == 0.
    assert rerouting.common_control_time_s == dynamic.common_control_time_s == 3.
    estimator = Estimator(profile, expected_identity=profile["identity"], layer_modules=("blocks.0", "blocks.1"))
    planner = Planner(estimator, config=center.config, r_dp=center.r_dp, r_pp=center.r_pp,
                      memory_capacity_bytes=center.memory_capacity_bytes)
    oracle = exhaustive_oracle(planner, len(state.survivor_state.workers))
    assert dynamic.estimated_step_time_s == min(time for time, _ in oracle.values() if time is not None)
    manifest = dynamic.execution
    assert isinstance(manifest, MigrationManifest)
    costs = manifest.cost_matrix
    assert manifest.migration_bytes == min(sum(costs[i][j] for i, j in enumerate(p))
                                          for p in permutations(range(len(costs))))
    # Slower direction src=1, endpoint rank=1, mean(iteration 0,1)=.5.
    migration_time = sum(max(action.tensor.nbytes / 100 + 2.5 for action in row)
                         for row in manifest.migration_rounds)
    assert dynamic.estimated_transition_time_s == pytest.approx(.25 + migration_time)
    assert dynamic.transition.migration_time_s == pytest.approx(migration_time)


def test_best_dynamic_is_independent_of_duration_and_final_policy_switches():
    center, state, profile = scenario()
    rerouting, dynamic = center.evaluate_candidates(state, profile)
    assert dynamic.estimated_step_time_s < rerouting.estimated_step_time_s
    # Solve equality independently: D = transition / (1 - dynamic_step/rerouting_step).
    crossover = dynamic.estimated_transition_time_s / (1 - dynamic.estimated_step_time_s / rerouting.estimated_step_time_s)
    short, long = crossover * .75, crossover * 2.
    first, second = center.select(state, profile, short), center.select(state, profile, long)
    assert first.plan.policy == "rerouting" and second.plan.policy == "dynamic"
    first_dynamic = next(row for row in first.derivation["candidates"] if row["policy"] == "dynamic")
    second_dynamic = next(row for row in second.derivation["candidates"] if row["policy"] == "dynamic")
    for field in ("plan_id", "estimated_step_time_s", "estimated_transition_time_s", "memory", "execution"):
        assert first_dynamic[field] == second_dynamic[field]
    assert first_dynamic["plan_id"] == second_dynamic["plan_id"] == dynamic.plan_id
    for result, duration in ((first, short), (second, long)):
        expected = {c.policy: score_oracle(c, duration) for c in (rerouting, dynamic)}
        assert result.plan.policy == max(expected, key=expected.get)
        assert result.score == pytest.approx(expected[result.plan.policy])
        assert result.candidate.execution is not None
        json.dumps(result.derivation, allow_nan=False)
    assert state.cluster.committed_global_step == state.survivor_state.committed_global_step == 7
    assert state.cluster.generation == 2


@pytest.mark.parametrize("dp,nm,failures", [(2, 4, (0,)), (3, 6, (0, 1)), (4, 20, (0, 2))])
def test_rerouting_preserves_layout_and_routes_all_logical_tasks_evenly(dp, nm, failures):
    center, state, profile = scenario(dp=dp, nm=nm, failures=failures, r_dp=(1,), r_pp=(dp * 2 - len(failures),))
    rerouting, _ = center.evaluate_candidates(state, profile)
    assert rerouting.feasible
    execution = rerouting.execution
    assert execution.state.layouts == state.layouts
    assert execution.state.pipeline_workers == state.pipeline_workers
    assert execution.state.pipeline_micro_batches == (nm // dp,) * dp
    keys = [(r.pipeline, r.stage, r.micro_batch) for r in execution.routes]
    assert set(keys) == {(p, s, m) for p in range(dp) for s in range(2) for m in range(nm // dp)}
    assert len(keys) == len(set(keys)) == 2 * nm
    alive = set(state.survivor_state.workers)
    for route in execution.routes:
        original = state.pipeline_workers[route.pipeline][route.stage]
        assert route.worker in alive
        assert route.worker in {pipeline[route.stage] for pipeline in state.pipeline_workers}
        if original in alive:
            assert route.worker == original
    for stage in range(2):
        healthy = [pipeline[stage] for pipeline in state.pipeline_workers if pipeline[stage] in alive]
        loads = Counter(r.worker for r in execution.routes if r.stage == stage)
        assert max(loads[w] for w in healthy) - min(loads[w] for w in healthy) <= 1
    counts = tuple(sum(pipeline[s] not in alive for pipeline in state.pipeline_workers) for s in range(2))
    slots = 2 + nm / dp - 1 + sum((nm / dp) * fi / (dp - fi) for fi in counts)
    derivation = rerouting.derivation
    assert rerouting.estimated_step_time_s == pytest.approx(slots * (max(derivation["stage_forward_s"]) + max(derivation["stage_backward_s"])))
    assert derivation["time"]["equation"] == (12 if len(failures) == 1 else 13)
    # Reordered inventory/cluster listing cannot change route or candidate identities.
    shuffled = replace(state, cluster=replace(state.cluster, workers=state.cluster.workers[::-1]),
                       failure=replace(state.failure, failed_worker_ids=state.failure.failed_worker_ids[::-1]),
                       required=state.required[::-1], inventories=state.inventories[::-1])
    other = center.evaluate_candidates(shuffled, profile)[0]
    assert other.execution_plan == rerouting.execution_plan
    assert other.execution.routes == rerouting.execution.routes
    assert other.memory == rerouting.memory


def test_rerouting_activation_memory_counts_each_logical_stream_but_static_state_once():
    center, state, profile = scenario()
    rerouting, _ = center.evaluate_candidates(state, profile)
    estimator = Estimator(profile, expected_identity=profile["identity"], layer_modules=("blocks.0", "blocks.1"))
    baseline = estimator.memory(state.layouts[0], (10000, 10000))
    for row in rerouting.memory[0].stages:
        stage = row["stage"]
        base = baseline.stages[stage]
        static = base["static_layer_bytes"] + baseline.derivation["extra_static_bytes"][stage]
        # Remove static bytes from Eq.14 to obtain one stream's live activations.
        activation = base["peak_bytes"] - static
        streams = 2 if row["worker_id"] == "worker-2" else 1
        assert row["activation_streams"] == streams
        assert row["peak_bytes"] == static + streams * activation


@pytest.mark.parametrize("module,kind", [("embedding", "parameter"), ("blocks.0", "exp_avg"),
                                        ("final_norm", "step"), ("lm_head", "exp_avg_sq"), ("extra", "parameter")])
def test_last_complete_state_source_lost_excludes_both_policies(module, kind):
    center, state, profile = scenario()
    inventories = tuple(replace(i, tensors=tuple(t for t in i.tensors if (t.module_id, t.kind) != (module, kind)))
                        for i in state.inventories)
    broken = replace(state, inventories=inventories)
    candidates = center.evaluate_candidates(broken, profile)
    assert all(not c.feasible for c in candidates)
    assert all(any("no complete survivor source" in reason for reason in c.reasons) for c in candidates)
    with pytest.raises(NoUsablePolicyError, match="no complete survivor source"):
        center.select(broken, profile, 1000.)
    assert broken.cluster.committed_global_step == 7


def test_missing_native_stage_state_can_only_be_recovered_dynamically():
    center, state, profile = scenario(failures=(1,))
    # worker-0 loses one field, but worker-2 retains a complete same-stage source.
    inventories = tuple(replace(i, tensors=tuple(t for t in i.tensors if not (
                        i.worker.worker_id == "worker-0" and t.module_id == "embedding" and t.kind == "step")))
                        for i in state.inventories)
    state = replace(state, inventories=inventories)
    rerouting, dynamic = center.evaluate_candidates(state, profile)
    assert not rerouting.feasible and dynamic.feasible
    assert "lacks complete stage" in rerouting.reasons[0]
    assert center.select(state, profile, 1000.).plan.policy == "dynamic"


def test_oom_produces_numeric_memory_diagnostics_and_no_selected_policy():
    center, state, profile = scenario(capacity=0)
    candidates = center.evaluate_candidates(state, profile)
    assert all(not c.feasible for c in candidates)
    assert all(c.memory and any(row["peak_bytes"] > row["capacity_bytes"] for m in c.memory for row in m.stages)
               for c in candidates)
    with pytest.raises(NoUsablePolicyError, match="capacity"):
        center.select(state, profile, 1000.)


@pytest.mark.parametrize("capacity,r_dp,r_pp,winner", [(400, (3,), (1,), "rerouting"),
                                                      (380, (1,), (3,), "dynamic")])
def test_real_memory_estimates_can_exclude_either_policy_independently(capacity, r_dp, r_pp, winner):
    center, state, profile = scenario(capacity=capacity, r_dp=r_dp, r_pp=r_pp, failures=(1,))
    candidates = center.evaluate_candidates(state, profile)
    assert [c.policy for c in candidates if c.feasible] == [winner]
    rejected = next(c for c in candidates if not c.feasible)
    assert any(row["oom_reason"] for memory in rejected.memory for row in memory.stages)
    assert center.select(state, profile, 1000.).plan.policy == winner


@pytest.mark.parametrize("calibration", [False, True])
def test_missing_bootstrap_or_exact_transfer_calibration_is_reported(calibration):
    center, state, profile = scenario(calibration=calibration)
    if calibration:
        profile = calibrated_profile(profile, [4])
    rerouting, dynamic = center.evaluate_candidates(state, profile)
    assert rerouting.feasible and not dynamic.feasible
    assert dynamic.estimated_step_time_s is not None and dynamic.estimated_transition_time_s is None
    assert "missing" in dynamic.reasons[0] and "calibration" in dynamic.reasons[0]
    assert center.select(state, profile, 1000.).plan.policy == "rerouting"
    if not calibration:
        assert rerouting.common_control_time_s is None


def test_single_rerouting_candidate_when_no_legal_dynamic_topology():
    center, state, profile = scenario(r_dp=(2,), r_pp=(2,))
    rerouting, dynamic = center.evaluate_candidates(state, profile)
    assert rerouting.feasible and not dynamic.feasible
    assert "no legal dynamic plan" in dynamic.reasons[0]
    assert center.select(state, profile, 1000.).plan.policy == "rerouting"


def test_unequal_original_microbatch_counts_are_explicitly_outside_rerouting_equations():
    center, state, profile = scenario(nm=5)
    rerouting, dynamic = center.evaluate_candidates(state, profile)
    assert not rerouting.feasible and dynamic.feasible
    assert "equal integer micro-batches" in rerouting.reasons[0]
    assert center.select(state, profile, 1000.).plan.policy == "dynamic"


def test_fi_equal_dp_has_no_peer_or_zero_denominator_and_lost_source_is_unrecoverable():
    center, state, profile = scenario(failures=(0, 2))
    rerouting, dynamic = center.evaluate_candidates(state, profile)
    assert rerouting.estimated_step_time_s is None
    assert any("Fi=2 >= Ndp=2" in reason for reason in rerouting.reasons)
    assert not dynamic.feasible
    with pytest.raises(NoUsablePolicyError):
        center.select(state, profile, 1000.)


def test_no_healthy_stage_peer_does_not_exclude_dynamic_if_survivor_still_holds_state():
    center, state, profile = scenario(failures=(0, 2), r_dp=(1, 2), r_pp=(1, 2))
    # Extra old units retained in survivor memory, as permitted before migration ACK.
    state = replace(state, inventories=(replace(state.inventories[0], tensors=state.required), state.inventories[1]))
    rerouting, dynamic = center.evaluate_candidates(state, profile)
    assert not rerouting.feasible and dynamic.feasible
    assert any("no healthy stage peer" in reason for reason in rerouting.reasons)
    assert center.select(state, profile, 1000.).plan.policy == "dynamic"


def test_no_survivor_is_an_explicit_unrecoverable_error():
    center, state, profile = scenario(failures=(0, 1, 2, 3))
    with pytest.raises(UnrecoverableStateError, match="no surviving workers"):
        center.evaluate_candidates(state, profile)


def test_d_at_transition_excludes_dynamic_after_real_candidate_evaluation():
    center, state, profile = scenario()
    _, dynamic = center.evaluate_candidates(state, profile)
    decision = center.select(state, profile, dynamic.estimated_transition_time_s)
    assert decision.plan.policy == "rerouting"
    assert decision.derivation["candidates"][1]["score"] is None


@pytest.mark.parametrize("duration", [None, 0, -1, True, float("inf"), float("nan")])
def test_invalid_or_missing_d_prevents_search_and_cannot_produce_a_recovery_decision(monkeypatch, duration):
    center, state, profile = scenario()

    def forbidden(*args):
        pytest.fail("invalid D must be rejected before candidate search")

    monkeypatch.setattr(center, "evaluate_candidates", forbidden)
    with pytest.raises(ValueError, match="inter_fault_duration_s"):
        center.select(state, profile, duration)


@pytest.mark.parametrize("change", ["step", "generation", "unknown_worker", "duplicate_topology", "rank"])
def test_recovery_input_binds_original_topology_and_failure_to_full_safe_point_identity(change):
    _, state, _ = scenario()
    with pytest.raises(ValueError):
        if change == "step":
            replace(state, failure=replace(state.failure, committed_global_step=6))
        elif change == "generation":
            replace(state, failure=replace(state.failure, generation=1))
        elif change == "unknown_worker":
            replace(state, failure=replace(state.failure, failed_worker_ids=("unknown",)))
        elif change == "rank":
            replace(state, pipeline_workers=((replace(state.pipeline_workers[0][0], rank=99),
                                              state.pipeline_workers[0][1]), state.pipeline_workers[1]))
        else:
            replace(state, pipeline_workers=(state.pipeline_workers[0], state.pipeline_workers[0]))


@pytest.mark.parametrize("change", ["batch", "inventory_model", "inventory_step", "inventory_worker", "layout", "profile"])
def test_evaluation_rejects_inconsistent_inputs(change):
    center, state, profile = scenario()
    if change == "batch":
        state = replace(state, cluster=replace(state.cluster, global_batch_size=11))
    elif change == "inventory_model":
        state = replace(state, required=tuple(t for t in state.required if t.module_id != "extra"))
    elif change == "inventory_step":
        state = replace(state, inventories=(replace(state.inventories[0], committed_global_step=6), *state.inventories[1:]))
    elif change == "inventory_worker":
        state = replace(state, inventories=state.inventories[1:])
    elif change == "layout":
        layout = state.layouts[0]
        state = replace(state, layouts=((layout[0], layout[1][::-1]), state.layouts[1]))
    else:
        profile = deepcopy(profile)
        profile["identity"]["model_hash"] = "b" * 64
    with pytest.raises(ValueError):
        center.evaluate_candidates(state, profile)


@pytest.mark.parametrize("policy", ["rerouting", "dynamic"])
@pytest.mark.parametrize("field,value", [("plan_id", "forged"), ("global_batch_size", 11),
                                        ("generation", 3), ("estimated_step_time_s", 100.)])
def test_candidate_metrics_and_identity_are_bound_to_the_actual_execution(policy, field, value):
    center, state, profile = scenario()
    candidate = next(c for c in center.evaluate_candidates(state, profile) if c.policy == policy)
    with pytest.raises(ValueError, match="execution"):
        replace(candidate, **{field: value})


def test_oom_diagnostics_do_not_restart_algorithm1_search(monkeypatch):
    center, state, profile = scenario(capacity=0)
    original = Planner.candidates
    calls = []

    def counted(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(Planner, "candidates", counted)
    center.evaluate_candidates(state, profile)
    assert len(calls) == 1


def test_asymmetric_original_topology_still_allows_independent_dynamic_selection():
    center, original, profile = scenario()
    workers = original.cluster.workers[:3]
    layout = original.layouts[0]
    state = RecoveryState(cluster=replace(original.cluster, workers=workers), failure=original.failure,
                          layouts=((tuple(profile["identity"]["module_order"]),), layout),
                          pipeline_workers=((workers[0],), (workers[1], workers[2])),
                          pipeline_micro_batches=(2, 2), required=original.required,
                          inventories=tuple(WorkerInventory(w, 7, tuple(t for t in original.required
                                            if t.module_id in layout[i])) for i, w in enumerate(workers[1:])))
    rerouting, dynamic = center.evaluate_candidates(state, profile)
    assert not rerouting.feasible and dynamic.feasible
    assert "Eq.12/13" in rerouting.reasons[0]
    assert center.select(state, profile, 1000.).plan.policy == "dynamic"


def test_original_microbatch_partitions_are_explicit_and_cannot_be_reinterpreted():
    center, state, profile = scenario()
    state = replace(state, pipeline_micro_batches=(1, 3))
    rerouting, dynamic = center.evaluate_candidates(state, profile)
    assert not rerouting.feasible and dynamic.feasible
    assert rerouting.derivation["original_micro_batches"] == (1, 3)
    assert "Eq.12/13" in rerouting.reasons[0]
    assert center.select(state, profile, 1000.).plan.policy == "dynamic"


@pytest.mark.parametrize("partitions", [(1, 2), (3, 3)])
def test_original_microbatch_partitions_must_conserve_global_count(partitions):
    center, state, profile = scenario()
    with pytest.raises(ValueError, match="conserve global_micro_batches"):
        center.evaluate_candidates(replace(state, pipeline_micro_batches=partitions), profile)


def test_transition_internal_errors_are_not_hidden_as_a_policy_rejection(monkeypatch):
    center, state, profile = scenario()

    def invalid_transition(*args, **kwargs):
        raise ValueError("internal transition invariant failed")

    monkeypatch.setattr(Restorer, "estimate_transition", invalid_transition)
    with pytest.raises(ValueError, match="internal transition invariant"):
        center.select(state, profile, 1000.)


def test_common_control_reads_cached_calibration_without_reaggregation(monkeypatch):
    _, _, profile = scenario()
    restorer = Restorer(profile, expected_identity=profile["identity"])

    def forbidden(*args):
        pytest.fail("cached common control must not aggregate calibration again")

    monkeypatch.setattr("chameleon.restorer.calibration_times_s", forbidden)
    assert restorer.common_control_time_s == restorer.common_control_time_s == 3.


@pytest.mark.parametrize("policy", ["rerouting", "dynamic"])
def test_candidate_memory_must_match_execution_and_common_control_must_match_transition(policy):
    center, state, profile = scenario()
    candidate = next(c for c in center.evaluate_candidates(state, profile) if c.policy == policy)
    with pytest.raises(ValueError, match="execution"):
        replace(candidate, memory=())
    with pytest.raises(ValueError, match="common control"):
        replace(candidate, common_control_time_s=4.)


def test_transition_is_bound_to_the_exact_migration_manifest():
    center, state, profile = scenario(r_dp=(3,), r_pp=(1,))
    first = center.evaluate_candidates(state, profile)[1]
    cached = replace(state, inventories=(replace(state.inventories[0], tensors=state.required), *state.inventories[1:]))
    second = center.evaluate_candidates(cached, profile)[1]
    assert first.plan_id == second.plan_id
    assert first.execution.manifest_id != second.execution.manifest_id
    assert first.estimated_transition_time_s != second.estimated_transition_time_s
    with pytest.raises(ValueError, match="transition.*manifest"):
        replace(first, transition=second.transition)


def test_dynamic_transition_cannot_be_used_as_rerouting_paper_transition():
    center, state, profile = scenario()
    rerouting, dynamic = center.evaluate_candidates(state, profile)
    with pytest.raises(ValueError, match="rerouting.*transition"):
        replace(rerouting, transition=dynamic.transition)
