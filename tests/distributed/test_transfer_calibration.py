from copy import deepcopy
import math
from pathlib import Path

import pytest

from chameleon.transfer_calibration import (
    CalibrationError, group_bootstrap_time_s, run_transfer_calibration,
    transfer_time_s, validate_calibration,
)


def _failed_worker(rank, device, backend, port, directory, *args):
    from chameleon.environment import _write_record
    _write_record(directory, rank, {"rank": rank, "error": "injected calibration failure"})
    raise RuntimeError("injected calibration failure")


@pytest.fixture
def schema_calibration():
    # Handwritten oracle for schema/aggregation tests; no fake transport or backend.
    device = {"type": "cpu", "index": None, "torch": "schema-fixture",
              "name": "schema-fixture", "system": "schema-fixture"}
    records = []
    for rank in range(2):
        transfers = [{"source": src, "destination": dst, "tensor_bytes": 64, "iteration": iteration,
                      "execution_time_s": (1 + iteration * 2 + rank if src == 0 else 4 + iteration * 2 + rank * 2),
                      "wall_time_s": 10.}
                     for src, dst in ((0, 1), (1, 0)) for iteration in range(2)]
        records.append({"rank": rank, "worker_id": f"calibration-{rank}", "generation": 0,
                        "pid": 101 + rank, "device_identity": deepcopy(device),
                        "group_bootstrap_s": [4 + rank * 2, 6 + rank * 2],
                        "transfers": transfers, "verified": True})
    return {"schema_version": 1, "device": "cpu", "backend": "gloo", "world_size": 2,
            "tensor_bytes": [64], "iterations": 2, "bootstrap_rounds": 2, "records": records,
            "audit": {"port": 12345, "rendezvous_dir": "artifacts/schema-fixture",
                      "workers": [{"pid": 101 + rank, "exitcode": 0, "alive": False} for rank in range(2)],
                      "leaked_pids": [], "rendezvous_removed": True, "port_listening": False,
                      "port_reusable": True, "clean": True}, "error": None}


def test_calibration_estimates_use_measured_endpoint_means(schema_calibration):
    assert transfer_time_s(schema_calibration, 0, 1, 64) == 3.
    assert transfer_time_s(schema_calibration, 1, 0, 64) == 7.
    assert group_bootstrap_time_s(schema_calibration) == 7.
    with pytest.raises(ValueError, match="missing P2P calibration"):
        transfer_time_s(schema_calibration, 0, 1, 128)
    with pytest.raises(ValueError):
        group_bootstrap_time_s({})


@pytest.mark.parametrize("arguments", [(False, 1, 64), (0, True, 64), (0, 1, 64.)])
def test_transfer_lookup_requires_integer_identity(schema_calibration, arguments):
    with pytest.raises(ValueError):
        transfer_time_s(schema_calibration, *arguments)


@pytest.mark.parametrize("path,value", [
    (("schema_version",), 2), (("backend",), "nccl"),
    (("audit", "clean"), False), (("audit", "leaked_pids"), [999]),
    (("audit", "port_listening"), True), (("audit", "workers", 0, "alive"), True),
    (("records", 1, "pid"), 101), (("records", 1, "rank"), 0),
    (("records", 0, "verified"), False),
    (("records", 0, "group_bootstrap_s"), []),
    (("records", 0, "transfers"), []),
    (("records", 0, "transfers", 0, "execution_time_s"), float("nan")),
    (("records", 0, "transfers", 0, "wall_time_s"), -1.),
    (("records", 0, "transfers", 1, "iteration"), 0),
    (("records", 0, "group_bootstrap_s"), None),
    (("audit", "leaked_pids"), {}),
    (("audit", "workers", 0, "pid"), 101.),
    (("audit", "workers", 0, "alive"), 0),
    (("records", 1, "device_identity", "torch"), "different-build"),
    (("records", 0, "rank"), []), (("records", 0, "pid"), {}),
])
def test_reject_incomplete_or_invalid_calibrations(schema_calibration, path, value):
    payload = deepcopy(schema_calibration)
    parent = payload
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError):
        validate_calibration(payload)


@pytest.mark.parametrize("overrides", [
    {"world_size": 1}, {"world_size": 3}, {"world_size": True},
    {"tensor_bytes": ()}, {"tensor_bytes": (64, 64)}, {"tensor_bytes": (7,)},
    {"tensor_bytes": (True,)}, {"warmup": -1}, {"iterations": 0},
    {"tensor_bytes": ([64],)},
    {"bootstrap_rounds": 0}, {"timeout_s": 0}, {"timeout_s": float("inf")},
])
def test_invalid_calibration_inputs_fail_before_spawn(overrides):
    with pytest.raises(ValueError):
        run_transfer_calibration("cpu", **overrides)


def test_two_real_ranks_transfer_tensors_and_roundtrip_profile(distributed_environment, device,
                                                            world_size, request):
    import torch
    from chameleon import ClusterState, ModelConfig, WorkerIdentity
    from chameleon.model import build_initial_model
    from chameleon.profiler import Profiler, export_profile, load_profile, train_profile_step

    assert world_size == 2, "this contract requires two real workers"
    report = run_transfer_calibration(device, world_size)
    request.config._chameleon_reports.add(Path(f"artifacts/test-results/transfer-calibration-{device}.json"))
    assert report["audit"]["clean"]
    assert len({record["pid"] for record in report["records"]}) == 2
    for record in report["records"]:
        assert record["verified"]
        assert len(record["group_bootstrap_s"]) == 2
        assert len(record["transfers"]) == 2 * 3 * 3
        assert all(math.isfinite(row["execution_time_s"]) and row["execution_time_s"] >= 0
                   for row in record["transfers"])
    for size in report["tensor_bytes"]:
        assert transfer_time_s(report, 0, 1, size) >= 0
        assert transfer_time_s(report, 1, 0, size) >= 0
    assert group_bootstrap_time_s(report) > 0
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=1, num_heads=1,
                         sequence_length=3, global_batch_size=3, micro_batch_size=2)
    model = build_initial_model(config, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), amsgrad=False)
    state = ClusterState((WorkerIdentity("local-profiler", 0, 0),), 3)
    state, _ = train_profile_step(model, optimizer, state)
    profiler = Profiler(model, optimizer)
    profiler.add_calibration(report)
    for record in report["records"]:
        metric = f"calibration.rank{record['rank']}.group_bootstrap_s"
        assert profiler.measurements.metrics[metric]["samples"] == record["group_bootstrap_s"]
    train_profile_step(model, optimizer, state, profiler=profiler)
    path = f"artifacts/test-results/calibrated-profile-{device}.json"
    export_profile(profiler.snapshot(), path, expected_identity=profiler.identity)
    assert load_profile(path, expected_identity=profiler.identity) == profiler.snapshot()


def test_calibration_hard_timeout_audits_cleanup(distributed_environment, device, world_size, request):
    assert world_size == 2
    directory = "artifacts/test-results/transfer-timeout"
    with pytest.raises(CalibrationError, match="hard timeout") as caught:
        run_transfer_calibration(device, world_size, timeout_s=0.01, artifact_dir=directory)
    request.config._chameleon_reports.add(Path(directory, f"transfer-calibration-{device}.json"))
    audit = caught.value.audit
    assert len(audit["workers"]) == 2 and audit["clean"]
    assert not audit["leaked_pids"] and audit["rendezvous_removed"]
    assert not audit["port_listening"] and audit["port_reusable"]
    assert all(not worker["alive"] and worker["exitcode"] is not None for worker in audit["workers"])


def test_worker_exception_audits_cleanup(distributed_environment, device, world_size, request, monkeypatch):
    import chameleon.transfer_calibration as calibration
    assert world_size == 2
    directory = "artifacts/test-results/transfer-exception"
    # Actual spawned workers fail before communication; no collective or P2P is replaced.
    monkeypatch.setattr(calibration, "_calibration_worker", _failed_worker)
    with pytest.raises(CalibrationError, match="exited abnormally") as caught:
        run_transfer_calibration(device, world_size, artifact_dir=directory)
    request.config._chameleon_reports.add(Path(directory, f"transfer-calibration-{device}.json"))
    assert caught.value.audit["clean"]
    assert len(caught.value.audit["workers"]) == 2
    assert any(worker["exitcode"] != 0 for worker in caught.value.audit["workers"])
