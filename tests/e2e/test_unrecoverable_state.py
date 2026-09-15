"""Losing the last live parameter/AdamW replica is unrecoverable and atomic."""

from dataclasses import asdict

import pytest

from chameleon import ClusterState, ModelConfig, UnrecoverableStateError, WorkerIdentity
from chameleon.contracts import FailureEvent
from chameleon.decision_center import DecisionCenter, select_policy
from chameleon.runtime import ReroutingTopology, SymmetricRuntime, SymmetricTopology
from chameleon.state_sources import ADAMW_FIELDS, build_state_source_map


def _hashes(replies):
    return {row["worker"].worker_id: {tuple(item["key"]): item["digest"]
                                      for item in row["hashes"]}
            for row in replies}


def _kill_next_owner(runtime, stage):
    rank = next(row[stage] for row in runtime.topology.pipeline_ranks if row[stage] is not None)
    worker, process = runtime.topology.ranks[rank], runtime.processes[rank]
    process.kill()
    process.join(10)
    assert not process.is_alive() and process.exitcode not in (None, 0)
    return FailureEvent((worker.worker_id,), runtime.state.generation,
                        runtime.state.committed_global_step), worker.worker_id, process.pid


@pytest.fixture(scope="module")
def unrecoverable_environment(request):
    import torch
    from conftest import _adaptive_base_profile
    from chameleon.environment import environment_report, validate_container, validate_device

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
    return device, expected_size, config, _adaptive_base_profile(config, device)


@pytest.mark.parametrize(("target_parameter", "stage", "lost_modules"), (
    ("blocks.0.linear1.weight", 0, {"embedding", "blocks.0"}),
    ("embedding.weight", 0, {"embedding", "blocks.0"}),
    ("final_norm.weight", 1, {"blocks.1", "final_norm", "lm_head"}),
    ("lm_head.weight", 1, {"blocks.1", "final_norm", "lm_head"}),
))
def test_last_parameter_and_adamw_replica_loss_is_unrecoverable(
        target_parameter, stage, lost_modules, unrecoverable_environment, request):
    from unittest.mock import patch
    from conftest import _guarded_recovery_worker

    device, size, config, profile = unrecoverable_environment
    dp = size // 2
    generation = 140 + (0 if target_parameter.startswith("blocks") else
                        10 if target_parameter.startswith("embedding") else
                        20 if target_parameter.startswith("final_norm") else 30)
    stages = (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head"))
    workers = tuple(WorkerIdentity(f"last-{generation}-{rank:02d}", rank, generation)
                    for rank in range(size))
    initial = SymmetricTopology(ClusterState(workers, 23, generation=generation), config, stages)
    runtime = SymmetricRuntime(initial, device=device, lr=.007, weight_decay=.125)
    killed_ids, killed_pids, decisions, recoveries = [], [], [], []
    error = None
    try:
        with patch("chameleon.runtime._runtime_worker", _guarded_recovery_worker), runtime:
            initial_ready = list(runtime.ready)
            for _ in range(3):
                runtime.train_step()
            baseline_replies = runtime.inspect_state()
            baseline_hashes = _hashes(baseline_replies)
            required = runtime._required
            expected_names = {name for row in runtime.ready for name in row["parameter_names"]}
            assert {tensor.parameter_name for tensor in required} == expected_names
            assert {tensor.kind for tensor in required} == set(ADAMW_FIELDS)
            assert {(tensor.parameter_name, tensor.kind) for tensor in required
                    if tensor.parameter_name == target_parameter} == {
                        (target_parameter, kind) for kind in ADAMW_FIELDS}

            for loss_index in range(dp):
                topology_before = runtime.topology
                state_before = runtime.state
                recovery_count = len(runtime.recoveries)
                failure, worker_id, pid = _kill_next_owner(runtime, stage)
                killed_ids.append(worker_id)
                killed_pids.append(pid)
                if loss_index == dp - 1:
                    with pytest.raises(UnrecoverableStateError) as captured:
                        runtime.recovery_state(failure)
                    error = str(captured.value)
                    assert runtime.topology is topology_before
                    assert runtime.state == state_before
                    assert len(runtime.recoveries) == recovery_count
                    break

                recovery_state = runtime.recovery_state(failure)
                sources = build_state_source_map(recovery_state.survivor_state,
                                                 recovery_state.required,
                                                 recovery_state.inventories)
                target_module = next(tensor.module_id for tensor in required
                                     if tensor.parameter_name == target_parameter)
                assert len(dict(sources.module_sources)[target_module]) == dp - loss_index - 1
                center = DecisionCenter(config, expected_identity=profile["identity"],
                                        r_dp=(dp,), r_pp=(1, 2), memory_capacity_bytes=10**15)
                candidates = center.evaluate_candidates(recovery_state, profile)
                decision = select_policy(candidates, inter_fault_duration_s=1.)
                assert decision.plan.policy == "rerouting"
                decisions.append(decision)
                recoveries.append(runtime.recover(failure, decision))
                assert isinstance(runtime.topology, ReroutingTopology)
                assert runtime.state.committed_global_step == 3
                live = runtime.inspect_state()
                assert all(row["step"] == 3 for reply in live for row in reply["parameter_steps"])
                live_hashes = _hashes(live)
                for reply in live:
                    worker = reply["worker"].worker_id
                    if (target_parameter, "parameter") in live_hashes[worker]:
                        assert all(live_hashes[worker][target_parameter, kind]
                                   == baseline_hashes[worker][target_parameter, kind]
                                   for kind in ADAMW_FIELDS)
    finally:
        runtime.close(error or "unrecoverable-state test cleanup")
        if hasattr(runtime, "report_path"):
            request.config._chameleon_reports.add(runtime.report_path)

    assert error is not None
    assert all(module in error for module in lost_modules)
    assert target_parameter.split(".")[0] in error
    assert runtime.topology is topology_before
    assert runtime.state == state_before
    assert runtime.state.committed_global_step == 3
    assert len(runtime.recoveries) == dp - 1 == len(recoveries) == len(decisions)
    assert [decision.plan.generation for decision in decisions] == list(range(generation, generation + dp - 1))
    assert [row["generation"] for row in recoveries] == list(range(generation + 1, generation + dp))
    assert len(set(killed_ids)) == dp and len(set(killed_pids)) == dp
    assert all(recovery["policy"] == "rerouting" for recovery in recoveries)
    assert all(row["initialization_calls"] == {"model": 1, "constructor": 1}
               and row["checkpoint_reads"] == 0
               for recovery in recoveries for row in recovery["targets"])
    assert runtime.audit["clean"] and not runtime.audit["leaked_pids"]
    assert len(runtime.audit["workers"]) == size
    assert all(not row["alive"] for row in runtime.audit["workers"])
    assert {row["pid"] for row in runtime.audit["workers"]} == {
        row["pid"] for row in initial_ready}
    assert len(runtime.audit["rendezvous_files"]) == dp
    assert all(row["removed"] for row in runtime.audit["rendezvous_files"])
    assert runtime.audit["rendezvous_port"] is None and runtime.audit["rendezvous_removed"]
    assert error in runtime.audit["error"]
    assert not any(key in runtime.__dict__ for key in ("model", "optimizer", "reference", "checkpoint"))
    assert all("snapshots" not in step for step in runtime.steps)
    assert asdict(runtime.state)["generation"] == generation + dp - 1
