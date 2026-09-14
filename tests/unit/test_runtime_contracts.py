from dataclasses import replace
from functools import partial
import json
import multiprocessing as mp
from multiprocessing.connection import Connection
from pathlib import Path
import socket
import struct
import tempfile
import time

import pytest

from chameleon import ClusterState, ModelConfig, WorkerIdentity
from chameleon.runtime import RuntimeErrorWithAudit, SymmetricRuntime, SymmetricTopology, compare_runtime_profile
from chameleon.step import StepCommit


def topology(*, batch=11, micro=2, stages=None, workers=4):
    config = ModelConfig(num_layers=2, global_batch_size=batch, micro_batch_size=micro)
    identities = tuple(WorkerIdentity(f"persistent-{20 - rank}", rank, 3) for rank in reversed(range(workers)))
    state = ClusterState(identities, batch, generation=3)
    stages = stages or (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head"))
    return SymmetricTopology(state, config, stages)


def test_stable_ids_are_independent_of_dense_rank_and_input_order():
    value = topology()
    assert value.dp_size == value.pp_size == 2
    assert [worker.rank for worker in value.ranks] == [0, 1, 2, 3]
    assert [worker.worker_id for worker in value.ranks] == ["persistent-20", "persistent-19", "persistent-18", "persistent-17"]


def test_equal_micro_batch_counts_preserve_partial_samples_and_committed_ids():
    value = topology()
    for step in range(3):
        state = replace(value.state, committed_global_step=step)
        batches = [value.micro_batches(state, pipeline) for pipeline in range(2)]
        assert tuple(map(len, batches)) == (3, 3)
        assert [[len(batch) for batch in pipeline] for pipeline in batches] == [[2, 2, 2], [2, 2, 1]]
        assert tuple(sample_id for pipeline in batches for batch in pipeline for sample_id in batch) == tuple(range(11 * step, 11 * (step + 1)))


def test_single_micro_batch_and_less_than_pipeline_depth_are_valid():
    value = topology(batch=2, micro=1)
    assert value.micro_batches(value.state, 0) == ((0,),)
    deeper = topology(batch=2, micro=1, stages=(("embedding",), ("blocks.0",), ("blocks.1",), ("final_norm", "lm_head")))
    assert deeper.dp_size == 1 and deeper.pp_size == 4
    assert deeper.micro_batches(deeper.state, 0) == ((0,), (1,))


@pytest.mark.parametrize("stages", [
    (), (("embedding",), ()), (("blocks.0", "embedding"), ("blocks.1", "final_norm", "lm_head")),
    (("embedding", "blocks.0"), ("blocks.1", "lm_head")),
    (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head", "lm_head")),
])
def test_invalid_layouts(stages):
    value = topology()
    with pytest.raises(ValueError, match="partition"):
        replace(value, stage_modules=stages)


@pytest.mark.parametrize("change,match", [
    ({"global_batch_size": 12}, "batch sizes"),
    ({"workers": tuple(WorkerIdentity(f"w{i}", i + 1, 3) for i in range(4))}, "dense ranks"),
    ({"workers": tuple(WorkerIdentity(f"w{i}", i, 3) for i in range(3))}, "fill"),
])
def test_invalid_cluster(change, match):
    value = topology()
    with pytest.raises(ValueError, match=match):
        replace(value, state=replace(value.state, **change))


def test_initial_runtime_rejects_reinitializing_committed_topology():
    current = topology()
    recovered = replace(current, state=replace(current.state, committed_global_step=3))
    with pytest.raises(ValueError, match="already committed"):
        SymmetricRuntime(recovered)


@pytest.mark.parametrize("batch,micro", [(1, 1), (5, 2)])
def test_unequal_or_zero_micro_batch_partitions_are_outside_symmetric_task(batch, micro):
    with pytest.raises(ValueError, match="equal positive"):
        topology(batch=batch, micro=micro)


@pytest.mark.parametrize("pipeline", [-1, True, 2])
def test_invalid_pipeline(pipeline):
    value = topology()
    with pytest.raises(ValueError, match="pipeline"):
        value.micro_batches(value.state, pipeline)


def test_step_state_cannot_change_generation_or_batch():
    value = topology()
    other = replace(value.state, workers=tuple(replace(w, generation=4) for w in value.state.workers), generation=4)
    with pytest.raises(ValueError, match="topology"):
        value.micro_batches(other, 0)


@pytest.mark.parametrize("options", [
    {"timeout_s": 0}, {"timeout_s": float("inf")}, {"dtype": "float16"},
    {"lr": -1}, {"weight_decay": -1}, {"behavior": "unknown"},
])
def test_invalid_runtime_options_fail_before_backend_import(options):
    with pytest.raises(ValueError):
        SymmetricRuntime(topology(), **options)


@pytest.fixture
def metadata_runtime(monkeypatch):
    from chameleon import runtime as module
    # Lifecycle/transport contracts use only stdlib metadata, never simulated training/NCCL.
    monkeypatch.setattr(module, "validate_device", lambda *_: "metadata")
    return SymmetricRuntime(topology())


@pytest.mark.parametrize("status", ["closed", "open"])
def test_runtime_cannot_reenter_or_reopen_before_resource_allocation(metadata_runtime, status):
    runtime = metadata_runtime
    if status == "closed":
        runtime.close()
    else:
        runtime._directory = object()
    runtime.root = None  # Any attempted allocation is a bug; the lifecycle guard runs first.
    with pytest.raises(RuntimeError, match="only be opened once"):
        runtime.__enter__()


def test_runtime_does_not_reserve_a_tcp_port(metadata_runtime, monkeypatch):
    from chameleon import runtime as module

    def fail_bind(*_):
        raise OSError("TCP ports are unavailable")

    monkeypatch.setattr(socket.socket, "bind", fail_bind)
    monkeypatch.setattr(module, "_runtime_worker", partial(_metadata_worker, fault="none"))
    with metadata_runtime:
        assert metadata_runtime.snapshot_profiles() == [None] * 4
    assert metadata_runtime.audit["clean"]
    assert metadata_runtime.audit["rendezvous_removed"]


def test_directory_creation_error_keeps_original_error_without_starting_workers(metadata_runtime, monkeypatch):
    def fail_mkdir(*_, **__):
        raise OSError("injected directory creation failure")

    monkeypatch.setattr(Path, "mkdir", fail_mkdir)
    with pytest.raises(RuntimeErrorWithAudit, match="injected directory creation failure") as caught:
        metadata_runtime.__enter__()
    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.audit["clean"]
    assert caught.value.audit["workers"] == []


def test_spawn_failure_cleans_workers_already_started(metadata_runtime, monkeypatch):
    from chameleon import runtime as module
    process_type = type(mp.get_context("spawn").Process())
    original_start = process_type.start
    started = []

    def fail_second_start(process):
        if started:
            raise OSError("injected process start failure")
        original_start(process)
        started.append(process.pid)

    monkeypatch.setattr(process_type, "start", fail_second_start)
    monkeypatch.setattr(module, "_runtime_worker", partial(_metadata_worker, fault="none"))
    with pytest.raises(RuntimeErrorWithAudit, match="injected process start failure") as caught:
        metadata_runtime.__enter__()
    assert isinstance(caught.value.__cause__, OSError)
    assert caught.value.audit["clean"]
    assert [worker["pid"] for worker in caught.value.audit["workers"]] == started
    assert metadata_runtime.audit["rendezvous_file_removed"]


def test_concurrent_runtimes_have_independent_rendezvous_files(metadata_runtime, monkeypatch):
    from chameleon import runtime as module
    monkeypatch.setattr(module, "_runtime_worker", partial(_metadata_worker, fault="none"))
    other = SymmetricRuntime(metadata_runtime.topology)
    with metadata_runtime:
        with other:
            assert metadata_runtime.rendezvous_file != other.rendezvous_file
            for runtime in (metadata_runtime, other):
                assert {row["rendezvous_file"] for row in runtime.ready} == {str(runtime.rendezvous_file)}
                # Control-only evidence also verifies stale store files are removed on exit.
                runtime.rendezvous_file.write_bytes(b"metadata-only")
                assert runtime.snapshot_profiles() == [None] * 4
    assert all(runtime.audit["clean"] and runtime.audit["rendezvous_file_removed"]
               for runtime in (metadata_runtime, other))


def test_closing_first_runtime_does_not_claim_second_workers_as_leaks(metadata_runtime, monkeypatch):
    from chameleon import runtime as module
    monkeypatch.setattr(module, "_runtime_worker", partial(_metadata_worker, fault="none"))
    other = SymmetricRuntime(metadata_runtime.topology)
    try:
        metadata_runtime.__enter__()
        other.__enter__()
        metadata_runtime.close()
        assert other.snapshot_profiles() == [None] * 4
        assert all(process.is_alive() for process in other.processes)
    finally:
        other.close()
        metadata_runtime.close()
    assert all(runtime.audit["clean"] and not runtime.audit["leaked_pids"]
               for runtime in (metadata_runtime, other))


def _collect_incomplete_frame(incoming, result, value):
    connection = Connection(incoming.detach())
    runtime = object.__new__(SymmetricRuntime)
    runtime.topology, runtime.commit = value, StepCommit(value.state)
    runtime.connections, runtime.processes = [connection], []
    runtime.timeout_s, runtime.origin = .5, time.monotonic()
    try:
        runtime._collect("ready", runtime.origin + runtime.timeout_s)
        result.send("unexpected success")
    except BaseException as exc:
        result.send(str(exc))
    finally:
        connection.close()
        result.close()


def test_pending_worker_error_is_read_before_exit_status(metadata_runtime, monkeypatch):
    from types import SimpleNamespace
    from chameleon import runtime as module
    from chameleon.contracts import UnrecoverableStateError
    runtime = metadata_runtime
    runtime.root.mkdir(parents=True, exist_ok=True)
    pipes = [mp.Pipe() for _ in range(2)]
    try:
        with tempfile.TemporaryDirectory(prefix="runtime-error-", dir=runtime.root) as directory:
            runtime.directory, runtime.origin = Path(directory), time.monotonic()
            runtime.connections = [parent for parent, _ in pipes]
            runtime.processes = [SimpleNamespace(exitcode=None), SimpleNamespace(exitcode=1)]
            module._write_reply(pipes[0][1], directory, 0, "ready", runtime.topology.ranks[0])
            module._write_reply(pipes[1][1], directory, 1, "error", runtime.topology.ranks[1],
                                error="missing lm_head exp_avg", unrecoverable=True)
            pipes[1][1].close()
            # The healthy reply can become readable first while another worker has already exited.
            monkeypatch.setattr(module, "wait", lambda pending, timeout: pending[:1])
            with pytest.raises(UnrecoverableStateError, match="missing lm_head exp_avg"):
                runtime._collect("ready", runtime.origin + runtime.timeout_s)
    finally:
        for parent, child in pipes:
            parent.close()
            child.close()


def test_worker_exit_without_reply_is_reported(metadata_runtime):
    runtime = metadata_runtime
    parent, child = mp.Pipe()
    child.close()
    try:
        runtime.connections, runtime.processes = [parent], []
        runtime.origin = time.monotonic()
        with pytest.raises(RuntimeError, match="worker exited before"):
            runtime._collect("ready", runtime.origin + runtime.timeout_s)
    finally:
        parent.close()


def test_oversized_incomplete_control_frame_cannot_bypass_deadline():
    root = Path("artifacts/test-results")
    root.mkdir(parents=True, exist_ok=True)
    process = None
    baseline = {child.pid for child in mp.active_children()}
    error = None
    audit = {}
    with tempfile.TemporaryDirectory(prefix="runtime-frame-", dir=root) as directory:
        audit["rendezvous_dir"] = directory
        sender, receiver = socket.socketpair()
        parent, child = mp.get_context("spawn").Pipe()
        try:
            process = mp.get_context("spawn").Process(target=_collect_incomplete_frame,
                                                      args=(receiver, child, topology()))
            process.start()
            receiver.close()
            child.close()
            # Reproduce a large framed report stalled immediately after its length prefix.
            sender.sendall(struct.pack("!i", 1024 * 1024))
            if parent.poll(5):
                error = parent.recv()
            else:
                error = "collector blocked beyond hard timeout"
        finally:
            sender.close()
            receiver.close()
            if process is not None and process.pid is not None:
                process.join(2)
                if process.is_alive():
                    process.terminate()
                process.join(2)
                if process.is_alive():
                    process.kill()
                process.join(2)
                audit["pid"], audit["exitcode"], audit["alive"] = process.pid, process.exitcode, process.is_alive()
                if not process.is_alive():
                    process.close()
            parent.close()
            child.close()
    audit.update(rendezvous_removed=not Path(directory).exists(), port=None,
                 leaked_pids=sorted({child.pid for child in mp.active_children()} - baseline), error=error)
    Path(root, "task09-review-frame-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    assert not audit["alive"] and not audit["leaked_pids"] and audit["rendezvous_removed"]
    assert error == "bad message length"
    assert audit["exitcode"] == 0


def _metadata_worker(value, rank, device, backend, dtype, rendezvous_file, timeout, connection,
                     origin, lr, weight_decay, behavior, directory, capture_state, *, fault):
    from chameleon.runtime import _write_reply

    worker = value.ranks[rank]
    try:
        _write_reply(connection, directory, rank, "ready", worker, pid=mp.current_process().pid,
                     parameter_names=["protocol-only"], groups=[], device=device, backend=backend,
                     rendezvous_file=rendezvous_file)
        while True:
            token = connection.recv_bytes(maxlength=7)
            if token == b"stop":
                break
            assert token == b"profile"
            if fault == "partial" and rank == 0:
                # Simulate interruption during report publication, before the control token.
                Path(directory, "rank-0-profile.tmp").write_text('{"padding":"' + 'x' * 65536, encoding="utf-8")
                time.sleep(3600)
            identity = replace(worker, generation=worker.generation + 1) if fault == "identity" and rank == 0 else worker
            step_id = False if fault == "step" and rank == 0 else 0
            _write_reply(connection, directory, rank, "profile", identity, step_id=step_id,
                         profile=None, padding="x" * 65536)
    finally:
        connection.close()


def _step_metadata_worker(value, rank, device, backend, dtype, rendezvous_file, timeout, connection,
                          origin, lr, weight_decay, behavior, directory, capture_state, *, fault="none"):
    """Exercise control/commit ordering with metadata; no training tensors or backend."""
    from chameleon.runtime import _write_reply

    worker, state = value.ranks[rank], value.state
    pipeline, stage = value.location(rank)
    try:
        _write_reply(connection, directory, rank, "ready", worker, pid=mp.current_process().pid,
                     parameter_names=["protocol-only"], groups=[], device=device, backend=backend)
        while True:
            token = connection.recv_bytes(maxlength=7)
            if token == b"stop":
                break
            assert token == b"step"
            batches = value.micro_batches(state, pipeline)
            last = stage == value.pipeline_lengths[pipeline] - 1
            step_id = state.committed_global_step + 1
            report = dict(pipeline=pipeline, stage=stage,
                          micro_batches=[list(batch) for batch in batches], profile_step=None,
                          loss_batches=[dict(sample_ids=list(batch), loss_sum=float(len(batch)))
                                        for batch in batches] if last else [],
                          loss_global_sum=float(state.global_batch_size), global_sample_count=state.global_batch_size)
            identity = replace(worker, generation=worker.generation + 1) if fault == "identity" and rank == 0 else worker
            acknowledged = True if fault == "step" and rank == 0 else step_id
            _write_reply(connection, directory, rank, "ack", identity, step_id=acknowledged, **report)
            assert connection.recv_bytes(maxlength=7) == b"commit"
            state = replace(state, committed_global_step=step_id)
            _write_reply(connection, directory, rank, "safe", worker, step_id=step_id)
    finally:
        connection.close()


@pytest.mark.parametrize("failure", ["profile", "snapshot", "deadline"])
def test_controller_report_failure_is_checked_before_commit(metadata_runtime, monkeypatch, failure):
    from chameleon import runtime as module
    monkeypatch.setattr(module, "_runtime_worker", _step_metadata_worker)

    def comparison(*_):
        if failure == "profile":
            raise ValueError("injected profile comparison failure")
        return None

    monkeypatch.setattr(module, "compare_runtime_profile", comparison)
    runtime = metadata_runtime
    match = {"profile": "injected profile", "snapshot": "torch|step-1-rank-0", "deadline": "hard timeout"}[failure]
    with pytest.raises(RuntimeErrorWithAudit, match=match):
        with runtime:
            if failure == "snapshot":
                runtime.capture_state = True
            elif failure == "deadline":
                collect = runtime._collect
                monotonic = time.monotonic

                def collect_then_expire(kind, deadline):
                    result = collect(kind, deadline)
                    if kind == "ack":
                        monkeypatch.setattr(time, "monotonic", lambda: monotonic() + runtime.timeout_s + 1)
                    return result

                # Expire the real step deadline only after all ACKs have arrived.
                monkeypatch.setattr(runtime, "_collect", collect_then_expire)
            runtime.train_step()
    assert runtime.state.committed_global_step == 0
    assert runtime.steps == []
    assert runtime.audit["clean"]


def test_single_ack_envelope_preserves_identity_steps_and_sample_accounting(metadata_runtime, monkeypatch):
    from chameleon import runtime as module
    monkeypatch.setattr(module, "_runtime_worker", _step_metadata_worker)
    runtime = metadata_runtime
    with runtime:
        for step_id in range(1, 4):
            step = runtime.train_step()
            assert runtime.state.committed_global_step == step_id
            assert step["sample_ids"] == list(range((step_id - 1) * 11, step_id * 11))
            assert step["global_sample_count"] == step["loss_global_sum"] == 11
            assert step["safe_worker_ids"] == [worker.worker_id for worker in runtime.topology.ranks]
            for rank, report in enumerate(step["reports"]):
                worker = runtime.topology.ranks[rank]
                assert report["worker"] == dict(worker_id=worker.worker_id, rank=rank, generation=3)
                assert report["step_id"] == step_id
                wire = json.loads(Path(runtime.directory, f"rank-{rank}-ack.json").read_text(encoding="utf-8"))
                assert "report" not in wire
                assert wire["worker"] == report["worker"] and wire["step_id"] == step_id
    assert runtime.audit["clean"] and len(runtime.steps) == 3
    assert all(worker["exitcode"] == 0 for worker in runtime.audit["workers"])


@pytest.mark.parametrize("fault,match", [("identity", "identity"), ("step", "step_id must be an integer")])
def test_ack_identity_and_integer_step_are_checked_before_commit(metadata_runtime, monkeypatch, fault, match):
    from chameleon import runtime as module
    monkeypatch.setattr(module, "_runtime_worker", partial(_step_metadata_worker, fault=fault))
    with pytest.raises(RuntimeErrorWithAudit, match=match):
        with metadata_runtime:
            metadata_runtime.train_step()
    assert metadata_runtime.state.committed_global_step == 0
    assert metadata_runtime.steps == []
    assert metadata_runtime.audit["clean"]


@pytest.mark.parametrize("fault", ["none", "partial", "identity", "step"])
def test_atomic_metadata_reply_protocol_and_cleanup(metadata_runtime, monkeypatch, fault):
    from chameleon import runtime as module

    monkeypatch.setattr(module, "_runtime_worker", partial(_metadata_worker, fault=fault))
    runtime = metadata_runtime
    if fault == "none":
        with runtime:
            assert runtime.snapshot_profiles() == [None] * 4
            assert all((runtime.directory / f"rank-{rank}-profile.json").stat().st_size > 65536 for rank in range(4))
            assert not list(runtime.directory.glob("*.tmp"))
            assert len({row["pid"] for row in runtime.ready}) == 4
            assert {row["rendezvous_file"] for row in runtime.ready} == {str(runtime.rendezvous_file)}
            assert runtime.rendezvous_file.parent == runtime.directory
        assert all(row["exitcode"] == 0 for row in runtime.audit["workers"])
    else:
        match = {"partial": "hard timeout", "identity": "identity", "step": "step_id must be an integer"}[fault]
        with pytest.raises(RuntimeErrorWithAudit, match=match):
            with runtime:
                runtime.timeout_s = 1
                runtime.snapshot_profiles()
    assert runtime.state.committed_global_step == 0
    assert runtime.audit["clean"]
    assert runtime.audit["backend"] == "metadata"
    assert runtime.audit["rendezvous_file_removed"]
    assert not any(key.startswith("port") for key in runtime.audit)
    assert len(runtime.audit["workers"]) == 4
    assert json.loads(runtime.report_path.read_text(encoding="utf-8"))["audit"] == runtime.audit


def test_estimate_uses_only_current_step_profile_with_correct_global_count():
    value = topology(batch=2, micro=1)
    reports = []
    for pipeline in range(2):
        for stage in range(2):
            trace = [{"pipeline": pipeline, "stage": stage, "micro_batch": 0,
                      "kind": kind, "phase": "steady" if stage == 1 else phase,
                      "start_s": 10 + start, "end_s": 10 + end}
                     for kind, phase, start, end in (("forward", "warmup", 0, 1), ("backward", "cooldown", 1, 3))]
            reports.append({"pipeline": pipeline, "stage": stage, "micro_batches": [[pipeline]],
                            "profile_step": {"step_id": 2, "trace": trace},
                            "pipeline_wall_time_s": 9, "training_wall_time_s": 10})
    result = compare_runtime_profile(reports, value)
    assert result["estimated_compute_time_s"] == 6  # F0,F1,B1,B0 = 1+1+2+2.
    assert result["equation9"]["step_time_s"] == 6
    assert result["equation9"]["derivation"]["global_micro_batches"] == 2
    assert result["equation9"]["derivation"]["pipeline_micro_batches"] == (1, 1)
    assert result["measured_training_time_s"] == 10
    reports[0]["profile_step"] = None
    assert compare_runtime_profile(reports, value) is None


def test_snapshot_profiles_rejects_closed_runtime(metadata_runtime):
    metadata_runtime.close()
    with pytest.raises(RuntimeError, match="open at a safe point"):
        metadata_runtime.snapshot_profiles()
