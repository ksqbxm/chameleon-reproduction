"""Two safe-point kills must replan from the current generation and live state."""

from collections import Counter
from dataclasses import replace
import json

import pytest

from chameleon import ClusterState, ModelConfig, WorkerIdentity
from chameleon.contracts import FailureEvent
from chameleon.decision_center import DecisionCenter, select_policy
from chameleon.profiler import _hash
from chameleon.restorer import MigrationManifest
from chameleon.runtime import DynamicTopology, ReroutingTopology, SymmetricRuntime, SymmetricTopology


def _kill_slot(runtime, pipeline, stage):
    rank = runtime.topology.pipeline_ranks[pipeline][stage]
    assert rank is not None
    worker, process = runtime.topology.ranks[rank], runtime.processes[rank]
    process.kill()
    process.join(10)
    assert not process.is_alive() and process.exitcode not in (None, 0)
    return (FailureEvent((worker.worker_id,), runtime.state.generation,
                         runtime.state.committed_global_step),
            dict(worker_id=worker.worker_id, pid=process.pid, exitcode=process.exitcode))


def _choose(candidates, policy):
    rerouting, dynamic = candidates
    assert rerouting.feasible and dynamic.feasible
    assert dynamic.estimated_step_time_s < rerouting.estimated_step_time_s
    transition = dynamic.estimated_transition_time_s
    crossover = transition / (1 - dynamic.estimated_step_time_s / rerouting.estimated_step_time_s)
    duration = ((transition + crossover) / 2 if policy == "rerouting" else crossover * 2)
    decision = select_policy(candidates, duration)
    assert decision.plan.policy == policy
    return decision, duration, crossover


def _module_id(parameter_name):
    parts = parameter_name.split(".")
    return ".".join(parts[:2]) if parts[0] == "blocks" else parts[0]


def _owner_counts(topology):
    return {module: len(owners) for module, owners in topology.module_owners.items()}


def _reply_hashes(replies):
    return {row["worker"].worker_id: {tuple(item["key"]): item["digest"]
                                      for item in row["hashes"]}
            for row in replies}


def _logged_hashes(rows):
    return {row["worker"]["worker_id"]: {tuple(item["key"]): item["digest"]
                                          for item in row["hashes"]}
            for row in rows}


def _expected_sources(recovery_state, replies):
    survivor_ids = {worker.worker_id for worker in recovery_state.survivor_state.workers}
    hashes = {worker_id: rows for worker_id, rows in _reply_hashes(replies).items()
              if worker_id in survivor_ids}
    required = {}
    for tensor in recovery_state.required:
        required.setdefault(tensor.module_id, set()).add(tensor.key)
    return {module: tuple(sorted(worker_id for worker_id, rows in hashes.items()
                                 if keys <= rows.keys()))
            for module, keys in required.items()}


def _assert_numerical_step(actual, expected, device, module_owner_counts):
    import torch

    tolerance = dict(rtol=1e-7, atol=1e-9) if device == "cuda" else dict(rtol=1e-8, atol=1e-10)
    assert actual["sample_ids"] == list(expected.sample_ids)
    assert actual["global_sample_count"] == expected.global_sample_count
    assert actual["step_id"] == expected.committed_global_step
    assert actual["loss_global_sum"] == pytest.approx(expected.loss_global_sum,
                                                     rel=tolerance["rtol"], abs=tolerance["atol"])
    owners = Counter()
    for report, snapshot in zip(actual["reports"], actual["snapshots"]):
        assert set(report["synchronized_parameters"]) == snapshot["parameters"].keys()
        for name, parameter in snapshot["parameters"].items():
            owners[name] += 1
            torch.testing.assert_close(parameter, expected.parameters[name], **tolerance)
            torch.testing.assert_close(snapshot["gradients"][name], expected.gradients[name], **tolerance)
            state = snapshot["optimizer_state"][name]
            assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
            for field, value in state.items():
                torch.testing.assert_close(value, expected.optimizer_state[name][field], **tolerance)
    expected_owners = Counter({name: module_owner_counts[_module_id(name)]
                               for name in expected.parameters})
    assert owners == expected_owners


@pytest.fixture(scope="module")
def consecutive_kill_run(request):
    import torch
    from unittest.mock import patch
    from conftest import (_adaptive_base_profile, _controlled_adaptive_profile,
                          _guarded_recovery_worker)
    from chameleon.environment import environment_report, validate_container, validate_device
    from chameleon.model import build_initial_model
    from chameleon.reference import ReferenceTrainer

    device = request.config.getoption("--device")
    expected_size = 6 if device == "cpu" else 8
    if request.config.getoption("--world-size") != expected_size:
        pytest.fail(f"Task14 acceptance requires --world-size {expected_size} on {device}")
    validate_device(device, expected_size)
    if device == "cuda":
        validate_container(environment_report())
    torch.set_num_threads(1)
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=23, micro_batch_size=2)
    stages = (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head"))
    workers = tuple(WorkerIdentity(f"consecutive-{rank:02d}", rank, 14)
                    for rank in range(expected_size))
    topology = SymmetricTopology(ClusterState(workers, 23, generation=14), config, stages)
    base_profile = _adaptive_base_profile(config, device)
    runtime = SymmetricRuntime(topology, device=device, capture_state=True,
                               lr=.007, weight_decay=.125)
    with patch("chameleon.runtime._runtime_worker", _guarded_recovery_worker), runtime:
        initial_ready = list(runtime.ready)
        steps = [runtime.train_step() for _ in range(3)]
        owner_counts = [_owner_counts(runtime.topology)] * 3
        first_before = runtime.inspect_state()

        first_failure, first_kill = _kill_slot(runtime, 0, 1)
        first_state = runtime.recovery_state(first_failure)
        first_profile = _controlled_adaptive_profile(
            base_profile, first_state.required, revision=first_state.cluster.generation)
        first_center = DecisionCenter(config, expected_identity=first_profile["identity"],
                                      r_dp=(topology.dp_size,), r_pp=(1, 2),
                                      memory_capacity_bytes=10**15)
        first_candidates = tuple(replace(candidate, derivation=dict(candidate.derivation,
            timing_input="controlled Task14 generation 14 functional profile; not a measured performance claim"))
                                 for candidate in first_center.evaluate_candidates(first_state, first_profile))
        first_decision, first_duration, first_crossover = _choose(first_candidates, "rerouting")
        first_recovery = runtime.recover(first_failure, first_decision)
        first_topology = runtime.topology
        assert isinstance(first_topology, ReroutingTopology)
        for _ in range(2):
            steps.append(runtime.train_step())
            owner_counts.append(_owner_counts(runtime.topology))
        second_before = runtime.inspect_state()

        second_failure, second_kill = _kill_slot(runtime, 1, 0)
        generation_before_stale_rejection = runtime.state.generation
        with pytest.raises(ValueError, match="decision differs from the failed topology"):
            runtime.recover(second_failure, first_decision)
        assert not runtime._closed and runtime.state.generation == generation_before_stale_rejection
        assert all(process.is_alive() for rank, process in enumerate(runtime.processes)
                   if rank != runtime.topology.ranks.index(
                       next(worker for worker in runtime.topology.ranks
                            if worker.worker_id == second_kill["worker_id"])))

        second_state = runtime.recovery_state(second_failure)
        second_profile = _controlled_adaptive_profile(
            base_profile, second_state.required, revision=second_state.cluster.generation)
        second_center = DecisionCenter(config, expected_identity=second_profile["identity"],
                                       r_dp=(topology.dp_size,), r_pp=(1, 2),
                                       memory_capacity_bytes=10**15)
        second_candidates = tuple(replace(candidate, derivation=dict(candidate.derivation,
            timing_input="controlled Task14 generation 15 functional profile; not a measured performance claim"))
                                  for candidate in second_center.evaluate_candidates(second_state, second_profile))
        second_decision, second_duration, second_crossover = _choose(second_candidates, "dynamic")
        second_recovery = runtime.recover(second_failure, second_decision)
        second_topology = runtime.topology
        assert isinstance(second_topology, DynamicTopology)
        for _ in range(2):
            steps.append(runtime.train_step())
            owner_counts.append(_owner_counts(runtime.topology))
    request.config._chameleon_reports.add(runtime.report_path)

    reference = ReferenceTrainer(build_initial_model(config, device=device),
        ClusterState((WorkerIdentity("reference", 0, 0),), 23), lr=.007, weight_decay=.125)
    return dict(runtime=runtime, device=device, config=config, initial_ready=initial_ready,
                steps=steps, owner_counts=owner_counts,
                first=dict(before=first_before, failure=first_failure, kill=first_kill, state=first_state,
                           profile=first_profile, candidates=first_candidates, decision=first_decision,
                           duration=first_duration, crossover=first_crossover,
                           recovery=first_recovery, topology=first_topology),
                second=dict(before=second_before, failure=second_failure, kill=second_kill,
                            state=second_state, profile=second_profile, candidates=second_candidates,
                            decision=second_decision, duration=second_duration,
                            crossover=second_crossover, recovery=second_recovery,
                            topology=second_topology),
                reference=[reference.train_step() for _ in range(7)])


def test_each_failure_replans_from_fresh_generation_profile_and_sources(consecutive_kill_run):
    first, second = consecutive_kill_run["first"], consecutive_kill_run["second"]
    assert first["failure"].generation == 14 and second["failure"].generation == 15
    initial_size = len(consecutive_kill_run["initial_ready"])
    assert len(first["state"].survivor_state.workers) == initial_size - 1
    assert len(second["state"].survivor_state.workers) == initial_size - 2
    assert first["state"].recovery_id != second["state"].recovery_id
    assert _hash(first["profile"]) != _hash(second["profile"])
    assert {candidate.generation for candidate in first["candidates"]} == {14}
    assert {candidate.generation for candidate in second["candidates"]} == {15}
    assert {candidate.plan_id for candidate in first["candidates"]}.isdisjoint(
        {candidate.plan_id for candidate in second["candidates"]})
    assert all(inventory.worker.generation == 14 for inventory in first["state"].inventories)
    assert all(inventory.worker.generation == 15 for inventory in second["state"].inventories)
    killed_ids = {first["kill"]["worker_id"], second["kill"]["worker_id"]}
    assert killed_ids.isdisjoint({worker.worker_id for worker in second["state"].survivor_state.workers})
    dp = initial_size // 2
    expected_counts = (
        {"embedding": dp, "blocks.0": dp, "blocks.1": dp - 1,
         "final_norm": dp - 1, "lm_head": dp - 1},
        {module: dp - 1 for module in ("embedding", "blocks.0", "blocks.1",
                                       "final_norm", "lm_head")},
    )
    for result, counts in zip((first, second), expected_counts):
        survivor_ids = {worker.worker_id for worker in result["state"].survivor_state.workers}
        expected_hashes = {worker_id: hashes for worker_id, hashes
                           in _reply_hashes(result["before"]).items()
                           if worker_id in survivor_ids}
        assert _logged_hashes(result["recovery"]["source_hashes"]) == expected_hashes
        assert {WorkerIdentity(**row["worker"]) for row in result["recovery"]["source_hashes"]} == set(
            result["state"].survivor_state.workers)
        logged_sources = {row["module_id"]: tuple(row["worker_ids"])
                          for row in result["recovery"]["state_sources"]}
        assert logged_sources == _expected_sources(result["state"], result["before"])
        assert {module: len(worker_ids) for module, worker_ids in logged_sources.items()} == counts


def test_external_d_selects_rerouting_then_dynamic_and_executes_both(consecutive_kill_run):
    first, second = consecutive_kill_run["first"], consecutive_kill_run["second"]
    assert first["decision"].plan.policy == "rerouting"
    assert second["decision"].plan.policy == "dynamic"
    for expected_policy, result in (("rerouting", first), ("dynamic", second)):
        rerouting, dynamic = result["candidates"]
        assert dynamic.estimated_transition_time_s < result["crossover"]
        assert ((dynamic.estimated_transition_time_s < result["duration"] < result["crossover"])
                if expected_policy == "rerouting" else result["duration"] > result["crossover"])
        scores = {candidate.policy: (candidate.global_batch_size / candidate.estimated_step_time_s)
                  * ((result["duration"] - candidate.estimated_transition_time_s) / result["duration"])
                  for candidate in result["candidates"]}
        assert result["decision"].plan.policy == max(scores, key=scores.get)
        assert result["decision"].score == pytest.approx(scores[expected_policy])
        assert result["recovery"]["actual_topology"]["policy"] == expected_policy

    assert isinstance(first["topology"], ReroutingTopology)
    assert first["topology"].layouts == first["state"].layouts
    delegated = [(report, row) for step in consecutive_kill_run["steps"][3:5]
                 for report in step["reports"] for row in report["trace"]
                 if row["kind"] == "forward" and row["pipeline"] == 0 and row["stage"] == 1]
    assert delegated and all(report["pipeline"] != 0 for report, _ in delegated)

    manifest = second["decision"].candidate.execution
    assert isinstance(second["topology"], DynamicTopology)
    assert isinstance(manifest, MigrationManifest) and manifest.migration_bytes > 0
    assert second["topology"].layouts != second["state"].layouts
    sent = Counter((row["source"], row["destination"], tuple(row["key"]))
                   for target in second["recovery"]["targets"] for row in target["sends"])
    received = Counter((row["source"], row["destination"], tuple(row["key"]))
                       for target in second["recovery"]["targets"] for row in target["receives"])
    assert sent == received
    assert sum(row["tensor_bytes"] for target in second["recovery"]["targets"]
               for row in target["sends"]) == manifest.migration_bytes
    source_hashes = _logged_hashes(second["recovery"]["source_hashes"])
    target_hashes = {row["worker"]["worker_id"]: {
        tuple(item["key"]): item["digest"] for item in row["hashes"]}
        for row in second["recovery"]["targets"]}
    for action in manifest.actions:
        assert (target_hashes[action.destination.worker_id][action.tensor.key]
                == source_hashes[action.source.worker_id][action.tensor.key])
    for target in second["recovery"]["targets"]:
        for field in ("sends", "receives"):
            for row in target[field]:
                assert row["digest"] == source_hashes[row["source"]][tuple(row["key"])]


def test_consecutive_recovery_matches_uninterrupted_reference(consecutive_kill_run):
    for actual, expected, owners in zip(consecutive_kill_run["steps"],
                                        consecutive_kill_run["reference"],
                                        consecutive_kill_run["owner_counts"]):
        _assert_numerical_step(actual, expected, consecutive_kill_run["device"], owners)
    runtime = consecutive_kill_run["runtime"]
    assert runtime.state.committed_global_step == 7
    assert [step["step_id"] for step in consecutive_kill_run["steps"]] == list(range(1, 8))
    assert [step["sample_ids"] for step in consecutive_kill_run["steps"]] == [
        list(range(step * 23, (step + 1) * 23)) for step in range(7)]
    assert all(row["step"] == 5 for reply in consecutive_kill_run["second"]["before"]
               for row in reply["parameter_steps"])


def test_consecutive_kill_report_has_all_pids_generations_and_cleanup(consecutive_kill_run):
    runtime = consecutive_kill_run["runtime"]
    first, second = consecutive_kill_run["first"], consecutive_kill_run["second"]
    assert [recovery["generation"] for recovery in runtime.recoveries] == [15, 16]
    assert [recovery["committed_global_step"] for recovery in runtime.recoveries] == [3, 5]
    assert {row["pid"] for row in runtime.ready}.isdisjoint(
        {first["kill"]["pid"], second["kill"]["pid"]})
    original = {row["worker"].worker_id: row for row in consecutive_kill_run["initial_ready"]}
    assert all(row["pid"] == original[row["worker"].worker_id]["pid"] for row in runtime.ready)
    assert all(row["worker"].generation == 16 for row in runtime.ready)
    for recovery in runtime.recoveries:
        assert len(recovery["killed"]) == 1
        assert recovery["killed"][0]["exitcode"] not in (None, 0)
        assert not recovery["killed"][0]["alive"]
        for row in recovery["targets"]:
            assert row["initialization_calls"] == {"model": 1, "constructor": 1}
            assert row["checkpoint_reads"] == 0
    assert runtime.audit["clean"] and not runtime.audit["leaked_pids"]
    assert len(runtime.audit["workers"]) == len(consecutive_kill_run["initial_ready"])
    assert len(runtime.audit["rendezvous_files"]) == 3
    assert runtime.audit["rendezvous_port"] is None and runtime.audit["rendezvous_removed"]
    assert all(not row["alive"] for row in runtime.audit["workers"])
    assert all(row["removed"] for row in runtime.audit["rendezvous_files"])
    payload = json.loads(runtime.report_path.read_text(encoding="utf-8"))
    assert payload["audit"] == runtime.audit
    assert len(payload["recoveries"]) == 2
    assert [row["decision"]["selected_policy"] for row in payload["recoveries"]] == [
        "rerouting", "dynamic"]
