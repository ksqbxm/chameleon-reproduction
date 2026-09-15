"""Single safe-point kill selects and executes both adaptive recovery policies."""

from collections import Counter
from dataclasses import replace
import json

import pytest

from chameleon import ClusterState, ModelConfig, WorkerIdentity
from chameleon.decision_center import DecisionCenter
from chameleon.restorer import MigrationManifest
from chameleon.runtime import DistributedRuntime, DynamicTopology, ReroutingTopology, SymmetricTopology
from conftest import _kill_workers, _logged_hashes, _reply_hashes, assert_numerical_step


def _score(candidate, duration):
    return (candidate.global_batch_size / candidate.estimated_step_time_s) * (
        (duration - candidate.estimated_transition_time_s) / duration)


def _run_case(case, topology, profile, device, request):
    from conftest import _controlled_adaptive_profile, _select_candidates

    runtime = DistributedRuntime(topology, device=device, capture_state=True,
                               lr=.007, weight_decay=.125)
    with runtime:
        initial_ready = list(runtime.ready)
        steps = [runtime.train_step() for _ in range(3)]
        before = runtime.inspect_state()
        failure, kill_audit = _kill_workers(runtime, (1,))
        killed_pid = kill_audit[0]["pid"]
        recovery_state = runtime.recovery_state(failure)
        selection_profile = _controlled_adaptive_profile(
            profile, recovery_state.required, revision=0)
        center = DecisionCenter(topology.config, expected_identity=selection_profile["identity"],
                                r_dp=(topology.dp_size,), r_pp=(1, 2),
                                memory_capacity_bytes=10**15)
        candidates = center.evaluate_candidates(recovery_state, selection_profile)
        candidates = tuple(replace(candidate, derivation=dict(candidate.derivation,
            timing_input="controlled Task13 functional profile; not a measured performance claim"))
                           for candidate in candidates)
        rerouting, dynamic = candidates
        assert rerouting.feasible and dynamic.feasible
        assert isinstance(dynamic.execution, MigrationManifest) and dynamic.execution.migration_bytes > 0
        assert dynamic.estimated_step_time_s < rerouting.estimated_step_time_s
        transition = dynamic.estimated_transition_time_s
        crossover = transition / (1 - dynamic.estimated_step_time_s / rerouting.estimated_step_time_s)
        duration = (transition + crossover) / 2 if case == "short" else crossover * 2
        decision = _select_candidates(candidates, duration)
        recovery = runtime.recover(failure, decision)
        steps.extend(runtime.train_step() for _ in range(2))
    request.config._chameleon_reports.add(runtime.report_path)
    return dict(case=case, runtime=runtime, initial_ready=initial_ready, before=before, steps=steps,
                recovery_state=recovery_state, candidates=candidates, crossover=crossover,
                duration=duration, decision=decision, recovery=recovery,
                killed_pid=killed_pid)


@pytest.fixture(scope="module")
def adaptive_policy_runs(request):
    import torch
    from conftest import _adaptive_base_profile
    from chameleon.environment import environment_report, validate_container, validate_device
    from chameleon.model import build_initial_model
    from chameleon.reference import ReferenceTrainer

    device = request.config.getoption("--device")
    expected_size = 6 if device == "cpu" else 8
    if request.config.getoption("--world-size") != expected_size:
        pytest.fail(f"Task13 acceptance requires --world-size {expected_size} on {device}")
    validate_device(device, expected_size)
    if device == "cuda":
        validate_container(environment_report())
    torch.set_num_threads(1)
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=23, micro_batch_size=2)
    stages = (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head"))
    workers = tuple(WorkerIdentity(f"adaptive-{rank:02d}", rank, 13)
                    for rank in range(expected_size))
    base_profile = _adaptive_base_profile(config, device)

    runs = {}
    for case in ("short", "long"):
        topology = SymmetricTopology(ClusterState(workers, config.global_batch_size, generation=13),
                                     config, stages)
        runs[case] = _run_case(case, topology, base_profile, device, request)

    reference = ReferenceTrainer(build_initial_model(config, device=device),
        ClusterState((WorkerIdentity("reference", 0, 0),), config.global_batch_size),
        lr=.007, weight_decay=.125)
    return dict(runs=runs, reference=[reference.train_step() for _ in range(5)],
                device=device, dp=expected_size // 2)


def test_break_even_selects_rerouting_short_and_dynamic_long(adaptive_policy_runs):
    runs = adaptive_policy_runs["runs"]
    assert runs["short"]["recovery_state"].recovery_id == runs["long"]["recovery_state"].recovery_id
    assert _reply_hashes(runs["short"]["before"]) == _reply_hashes(runs["long"]["before"])
    assert [candidate.plan_id for candidate in runs["short"]["candidates"]] == [
        candidate.plan_id for candidate in runs["long"]["candidates"]]
    assert runs["short"]["decision"].plan.policy == "rerouting"
    assert runs["long"]["decision"].plan.policy == "dynamic"
    for case, result in runs.items():
        rerouting, dynamic = result["candidates"]
        assert dynamic.estimated_step_time_s < rerouting.estimated_step_time_s
        assert dynamic.estimated_transition_time_s < result["crossover"]
        assert ((dynamic.estimated_transition_time_s < result["duration"] < result["crossover"])
                if case == "short" else result["duration"] > result["crossover"])
        scores = {candidate.policy: _score(candidate, result["duration"])
                  for candidate in result["candidates"]}
        assert result["decision"].score == pytest.approx(scores[result["decision"].plan.policy])
        assert result["decision"].plan.policy == max(scores, key=scores.get)
        rows = result["decision"].derivation["candidates"]
        assert result["decision"].derivation["B"] == 23
        assert result["decision"].derivation["D"] == result["duration"]
        assert {row["policy"] for row in rows} == {"rerouting", "dynamic"}
        assert sum(row["selected"] for row in rows) == 1
        assert all(row["derivation"]["timing_input"].startswith("controlled Task13") for row in rows)


def test_selected_recovery_path_really_executes(adaptive_policy_runs):
    short, long = (adaptive_policy_runs["runs"][case] for case in ("short", "long"))
    rerouted = short["runtime"].topology
    assert isinstance(rerouted, ReroutingTopology)
    assert rerouted.layouts == short["recovery_state"].layouts
    assert rerouted.pipeline_ranks[0][1] is None
    for step in short["steps"][3:]:
        delegated = [(report, row) for report in step["reports"] for row in report["trace"]
                     if row["kind"] == "forward" and row["pipeline"] == 0 and row["stage"] == 1]
        assert delegated
        assert all(report["pipeline"] != 0 and report["stage"] == 1 for report, _ in delegated)
    assert _logged_hashes(short["recovery"]["targets"]) == _logged_hashes(short["recovery"]["source_hashes"])

    dynamic = long["runtime"].topology
    manifest = long["decision"].candidate.execution
    assert isinstance(dynamic, DynamicTopology)
    assert dynamic.layouts != long["recovery_state"].layouts
    assert sorted(dynamic.pipeline_lengths) == [1] + [2] * (adaptive_policy_runs["dp"] - 1)
    sent = Counter((row["source"], row["destination"], tuple(row["key"]))
                   for target in long["recovery"]["targets"] for row in target["sends"])
    received = Counter((row["source"], row["destination"], tuple(row["key"]))
                       for target in long["recovery"]["targets"] for row in target["receives"])
    assert sent == received
    assert sum(row["tensor_bytes"] for target in long["recovery"]["targets"]
               for row in target["sends"]) == manifest.migration_bytes > 0
    sources = _logged_hashes(long["recovery"]["source_hashes"])
    targets = {row["worker"]["worker_id"]: {tuple(item["key"]): item["digest"]
                                            for item in row["hashes"]}
               for row in long["recovery"]["targets"]}
    for action in manifest.actions:
        expected = sources[action.source.worker_id][action.tensor.key]
        assert targets[action.destination.worker_id][action.tensor.key] == expected
    for target in long["recovery"]["targets"]:
        for transfer in target["sends"] + target["receives"]:
            assert transfer["digest"] == sources[transfer["source"]][tuple(transfer["key"])]


def test_both_recovered_runs_match_uninterrupted_reference(adaptive_policy_runs):
    reference = adaptive_policy_runs["reference"]
    for result in adaptive_policy_runs["runs"].values():
        topology = result["runtime"].topology
        for index, (actual, expected) in enumerate(zip(result["steps"], reference)):
            owner_counts = ({name: adaptive_policy_runs["dp"] for name in expected.parameters}
                            if index < 3 else
                            {name: len(next(owners for module, owners in topology.module_owners.items()
                                            if name == module or name.startswith(module + ".")))
                             for name in expected.parameters})
            assert_numerical_step(actual, expected, adaptive_policy_runs["device"], owner_counts=owner_counts)
        assert result["recovery"]["committed_global_step"] == 3
        assert result["runtime"].state.committed_global_step == 5
        assert [step["step_id"] for step in result["steps"]] == [1, 2, 3, 4, 5]


def test_adaptive_recovery_log_and_process_cleanup_are_complete(adaptive_policy_runs):
    for result in adaptive_policy_runs["runs"].values():
        runtime, recovery = result["runtime"], result["recovery"]
        killed = recovery["killed"]
        assert len(killed) == 1 and killed[0]["pid"] == result["killed_pid"]
        assert not killed[0]["alive"] and killed[0]["exitcode"] not in (None, 0)
        original = {row["worker"].worker_id: row for row in result["initial_ready"]
                    if row["pid"] != result["killed_pid"]}
        assert {row["worker"].worker_id for row in runtime.ready} == set(original)
        assert all(row["pid"] == original[row["worker"].worker_id]["pid"] for row in runtime.ready)
        assert {row["module_id"] for row in recovery["state_sources"]} == {
            "embedding", "blocks.0", "blocks.1", "final_norm", "lm_head"}
        assert all(row["worker_ids"] for row in recovery["state_sources"])
        assert result["decision"].plan.policy == recovery["actual_topology"]["policy"]
        assert recovery["actual_topology"]["layouts"] == runtime.topology.layouts
        assert recovery["actual_topology"]["pipeline_ranks"] == runtime.topology.pipeline_ranks
        assert recovery["actual_topology"]["pipeline_micro_batches"] == runtime.topology.pipeline_micro_batches
        before = _reply_hashes(result["before"])
        sources = _logged_hashes(recovery["source_hashes"])
        assert sources == {worker_id: before[worker_id] for worker_id in sources}
        assert len(sources) == len(runtime.topology.ranks)
        assert all(len(digest) == 64 for hashes in sources.values() for digest in hashes.values())
        required = {}
        for tensor in result["recovery_state"].required:
            required.setdefault(tensor.module_id, set()).add(tensor.key)
        for row in recovery["state_sources"]:
            assert all(required[row["module_id"]] <= sources[worker_id].keys()
                       for worker_id in row["worker_ids"])
        for field in ("actual_group_rebuild_s", "actual_transfer_validation_s",
                      "actual_training_group_install_s"):
            assert recovery[field] > 0
        assert runtime.audit["clean"] and not runtime.audit["leaked_pids"]
        assert all(not worker["alive"] for worker in runtime.audit["workers"])
        assert all(row["removed"] for row in runtime.audit["rendezvous_files"])
        payload = json.loads(runtime.report_path.read_text(encoding="utf-8"))
        assert payload["audit"] == runtime.audit
        persisted = payload["recoveries"]
        assert len(persisted) == 1
        record = persisted[0]
        assert record["decision"] == json.loads(json.dumps(recovery["decision"]))
        assert record["state_sources"] == json.loads(json.dumps(recovery["state_sources"]))
        assert record["source_hashes"] == json.loads(json.dumps(recovery["source_hashes"]))
        assert record["actual_topology"] == json.loads(json.dumps(recovery["actual_topology"]))
        assert record["decision"]["B"] == 23 and record["decision"]["D"] == result["duration"]
        assert all(row["estimated_step_time_s"] > 0 and row["estimated_transition_time_s"] >= 0
                   and row["score"] > 0 for row in record["decision"]["candidates"])
        for field in ("actual_group_rebuild_s", "actual_transfer_validation_s",
                      "actual_training_group_install_s"):
            assert record[field] == recovery[field]
        assert record["policy"] == {"short": "rerouting", "long": "dynamic"}[result["case"]]
