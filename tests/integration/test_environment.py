import json
from pathlib import Path

import pytest

from chameleon.environment import SmokeError, run_spawn_smoke


def test_environment_contract(distributed_environment, device):
    report = distributed_environment
    assert report["python"] and report["torch"] and report["executable"]
    if device == "cuda":
        assert len(report["gpus"]) == report["visible_gpu_count"] == 8
        assert [g["index"] for g in report["gpus"]] == list(range(8))


def test_real_spawn_sum(distributed_environment, device, world_size):
    report = run_spawn_smoke(device, world_size)
    assert report["audit"]["clean"]
    assert len({r["pid"] for r in report["records"]}) == world_size
    assert {r["rank"] for r in report["records"]} == set(range(world_size))
    assert all(r["sum"] == world_size * (world_size + 1) / 2 for r in report["records"])
    assert all(w["exitcode"] == 0 for w in report["audit"]["workers"])
    if device == "cuda":
        assert {r["cuda_device"] for r in report["records"]} == set(range(world_size))


def test_worker_exception_cleanup(distributed_environment, device, world_size):
    with pytest.raises(SmokeError, match="abnormally") as caught:
        run_spawn_smoke(device, world_size, behavior="error")
    audit = caught.value.audit
    assert audit["clean"]
    assert len(audit["workers"]) == world_size
    assert any(w["exitcode"] != 0 for w in audit["workers"])
    report = json.loads(Path(f"artifacts/test-results/smoke-{device}-error.json").read_text(encoding="utf-8"))
    assert any("injected worker failure" in r.get("error", "") for r in report["records"])


def test_worker_hard_timeout_cleanup(distributed_environment, device, world_size):
    with pytest.raises(SmokeError, match="hard timeout") as caught:
        run_spawn_smoke(device, world_size, timeout_s=15, behavior="hang")
    assert caught.value.audit["clean"]
    assert len(caught.value.audit["workers"]) == world_size
    report = json.loads(Path(f"artifacts/test-results/smoke-{device}-hang.json").read_text(encoding="utf-8"))
    assert any(r.get("status") == "injected hang" for r in report["records"])
