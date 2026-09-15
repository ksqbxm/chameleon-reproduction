import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest

from chameleon.cli import load_config


ROOT = Path(__file__).parents[2]


def _run_cli(workdir, *arguments):
    environment = os.environ.copy()
    source = str(ROOT / "src")
    environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "chameleon", *arguments],
        cwd=workdir,
        env=environment,
        text=True,
        capture_output=True,
        timeout=240,
        check=False,
    )


def _copy_config(workdir, device):
    source = ROOT / "configs" / ("tiny_8gpu.json" if device == "cuda" else "tiny_cpu.json")
    destination = workdir / "config.json"
    shutil.copyfile(source, destination)
    return destination.name


@pytest.fixture
def cli_workdir():
    with tempfile.TemporaryDirectory(prefix="chameleon-cli-") as directory:
        yield Path(directory)


def test_tiny_configs_are_strict_and_describe_real_cpu_or_eight_gpu_runs():
    cpu = load_config("configs/tiny_cpu.json")
    gpu = load_config("configs/tiny_8gpu.json")
    assert (cpu["device"], cpu["backend"], cpu["world_size"]) == ("cpu", "gloo", 4)
    assert (gpu["device"], gpu["backend"], gpu["world_size"]) == ("cuda", "nccl", 8)
    for config in (cpu, gpu):
        assert config["parallel"]["dp_size"] * config["parallel"]["pp_size"] == config["world_size"]
        assert config["model"].dropout == 0
        assert config["model"].amsgrad is False
        assert not Path(config["artifact_dir"]).is_absolute()


def test_recover_demo_requires_caller_supplied_d_before_execution(cli_workdir):
    config = _copy_config(cli_workdir, "cpu")
    result = _run_cli(cli_workdir, "recover-demo", "--config", config, "--profile", "profile.json")
    assert result.returncode == 2
    assert "--inter-fault-duration-s" in result.stderr
    assert not (cli_workdir / "artifacts").exists()


def test_execution_failure_writes_failed_json_report(cli_workdir):
    config = _copy_config(cli_workdir, "cpu")
    result = _run_cli(cli_workdir, "train", "--config", config, "--steps", "0",
                      "--output", "failed.json")
    assert result.returncode == 1
    report = json.loads((cli_workdir / "failed.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["command"] == "train"
    assert report["error"]["type"] == "ValueError"
    assert "steps" in report["error"]["message"]


def test_profile_train_and_recover_demo_clis_execute_and_write_complete_records(cli_workdir, device):
    config = _copy_config(cli_workdir, device)
    profile = _run_cli(cli_workdir, "profile", "--config", config, "--output", "profile.json",
                       "--report", "profile-report.json")
    assert profile.returncode == 0, profile.stderr
    profile_payload = json.loads((cli_workdir / "profile.json").read_text(encoding="utf-8"))
    profile_report = json.loads((cli_workdir / "profile-report.json").read_text(encoding="utf-8"))
    assert profile_payload["schema_version"] == 3
    assert profile_payload["steps"] and profile_payload["calibrations"]
    assert profile_report["status"] == "passed"
    assert profile_report["profile_hash"]
    assert profile_report["sample_ids"] == list(range(
        profile_report["committed_global_step"] * profile_report["B"]))
    assert profile_report["global_sample_count"] == len(profile_report["sample_ids"])

    train = _run_cli(cli_workdir, "train", "--config", config, "--steps", "1",
                     "--output", "train.json")
    assert train.returncode == 0, train.stderr
    train_report = json.loads((cli_workdir / "train.json").read_text(encoding="utf-8"))
    assert train_report["status"] == "passed"
    assert train_report["B"] == len(train_report["sample_ids"])
    assert train_report["committed_global_step"] == 1
    assert train_report["state_hashes"]
    assert train_report["resource_audit"]["clean"] is True
    assert all(worker["exitcode"] == 0 for worker in train_report["resource_audit"]["workers"])

    recovered = _run_cli(
        cli_workdir,
        "recover-demo",
        "--config", config,
        "--profile", "profile.json",
        "--inter-fault-duration-s", "1000",
        "--output", "recover.json",
    )
    assert recovered.returncode == 0, recovered.stderr
    report = json.loads((cli_workdir / "recover.json").read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["D"] == 1000
    assert report["duration_source"] == "caller supplied inter-fault duration; not predicted"
    assert report["selected_policy"] in {"rerouting", "dynamic"}
    assert {row["policy"] for row in report["candidates"]} == {"rerouting", "dynamic"}
    assert all("estimated_step_time_s" in row and "estimated_transition_time_s" in row
               for row in report["candidates"])
    assert report["precompute_completed_before_kill"] is True
    assert report["cache"] == {"entries": 1, "hits": 1, "misses": 1}
    assert report["kill"]["exitcode"] not in (None, 0)
    assert report["source_map"] and report["state_hashes"]
    assert report["committed_global_step"] == 4
    assert report["actual_times_s"]
    assert all(key.endswith("_s") and isinstance(value, (int, float))
               for key, value in report["actual_times_s"].items())
    assert report["resource_audit"]["clean"] is True
    assert not report["resource_audit"]["leaked_pids"]
