"""Repeated short controller timeouts must not leak workers or rendezvous resources."""

import pytest

from chameleon import ClusterState, ModelConfig, WorkerIdentity
from chameleon.runtime import RuntimeErrorWithAudit, SymmetricRuntime, SymmetricTopology


@pytest.mark.parametrize("round_id", range(3))
def test_short_timeout_cleanup_stress_has_no_deadlock_pid_or_port_leak(round_id, request):
    from chameleon.environment import environment_report, validate_container, validate_device

    device = request.config.getoption("--device")
    size = 6 if device == "cpu" else 8
    if request.config.getoption("--world-size") != size:
        pytest.fail(f"Task14 acceptance requires --world-size {size} on {device}")
    validate_device(device, size)
    if device == "cuda":
        validate_container(environment_report())
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=24, micro_batch_size=2)
    stages = (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head"))
    workers = tuple(WorkerIdentity(f"timeout-{round_id}-{rank:02d}", rank, 200 + round_id)
                    for rank in range(size))
    topology = SymmetricTopology(ClusterState(workers, 24, generation=200 + round_id),
                                 config, stages)
    runtime = SymmetricRuntime(topology, device=device, behavior="hang")
    with runtime:
        pids = [row["pid"] for row in runtime.ready]
        assert len(set(pids)) == size
        runtime.timeout_s = 1
        with pytest.raises(RuntimeErrorWithAudit, match="hard timeout") as captured:
            runtime.train_step()
        assert captured.value.audit is runtime.audit
    request.config._chameleon_reports.add(runtime.report_path)

    assert runtime.state.committed_global_step == 0 and not runtime.steps
    assert runtime.audit["clean"] and not runtime.audit["leaked_pids"]
    assert runtime.audit["error"] and "hard timeout" in runtime.audit["error"]
    assert {row["pid"] for row in runtime.audit["workers"]} == set(pids)
    assert all(not row["alive"] for row in runtime.audit["workers"])
    assert runtime.audit["rendezvous_backend"] == "FileStore"
    assert runtime.audit["rendezvous_port"] is None
    assert runtime.audit["rendezvous_file_removed"] and runtime.audit["rendezvous_removed"]
    assert len(runtime.audit["rendezvous_files"]) == 1
    assert all(row["removed"] for row in runtime.audit["rendezvous_files"])
