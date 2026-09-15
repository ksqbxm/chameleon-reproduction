"""Shared CPU/GPU pytest matrix with fail-fast JSON, JUnit and log reporting."""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

SINGLE_PROCESS = (
    "tests/integration/test_profile_roundtrip.py",
    "tests/integration/test_profile_estimator.py",
    "tests/integration/test_partitioned_gradient_oracle.py",
    "tests/integration/test_decision_center_oracle.py",
    "tests/integration/test_plan_restorer.py",
    "tests/integration/test_dynamic_planner_oracle.py",
    "tests/integration/test_routing_accounting.py",
    "tests/integration/test_recovery_contracts.py",
    "tests/integration/test_restorer_inventory.py",
    "tests/integration/test_plan_cache.py",
)


def _stage(name, world_size, *paths):
    return {"name": name, "world_size": world_size, "paths": paths}


def matrix(device):
    large = 8 if device == "cuda" else 7
    adaptive = 8 if device == "cuda" else 6
    return (
        _stage("unit", 1 if device == "cuda" else None, "tests/unit"),
        _stage("single-process-integration", 1, *SINGLE_PROCESS),
        _stage("environment", 2, "tests/integration/test_environment.py"),
        _stage("transfer-calibration", 2, "tests/distributed/test_transfer_calibration.py"),
        _stage("symmetric-runtime", 4, "tests/integration/test_runtime_profile.py",
               "tests/distributed/test_symmetric_training.py"),
        _stage("complete-recovery", 4, "tests/distributed/test_full_state_transfer.py",
               "tests/e2e/test_kill_group_rebuild.py"),
        _stage("rerouting", 5, "tests/distributed/test_rerouted_training.py"),
        _stage("scaled-rerouting", 7, "tests/distributed/test_rerouting_scale.py"),
        _stage("asymmetric-runtime", large, "tests/integration/test_planner_runtime.py",
               "tests/distributed/test_asymmetric_training.py",
               "tests/distributed/test_colored_allreduce.py"),
        _stage("adaptive-kill", adaptive, "tests/e2e/test_kill_adaptive_policy.py"),
        _stage("consecutive-and-unrecoverable", adaptive,
               "tests/e2e/test_consecutive_kills.py",
               "tests/e2e/test_unrecoverable_state.py",
               "tests/e2e/test_resource_cleanup.py"),
        _stage("cli", 8 if device == "cuda" else 4, "tests/integration/test_cli.py"),
    )


def _validate_coverage(stages):
    expected = {path.relative_to(ROOT).as_posix()
                for directory in ("tests/integration", "tests/distributed", "tests/e2e")
                for path in (ROOT / directory).glob("test_*.py")}
    listed = [path for stage in stages for path in stage["paths"] if path != "tests/unit"]
    if len(listed) != len(set(listed)):
        raise RuntimeError("regression matrix lists a test file more than once")
    missing, extra = expected - set(listed), set(listed) - expected
    if missing or extra:
        raise RuntimeError(f"regression matrix coverage mismatch; missing={sorted(missing)}, extra={sorted(extra)}")


def _junit_counts(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    return {
        "tests": len(cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
    }


@contextmanager
def _execution_root():
    """Keep Windows FileStore paths ASCII while running the same repository files."""
    if os.name != "nt" or str(ROOT).isascii():
        yield ROOT
        return
    scratch = Path(tempfile.mkdtemp(prefix="chameleon-regression-"))
    alias = scratch / "workspace"
    created = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(alias), str(ROOT)],
        text=True,
        capture_output=True,
        check=False,
    )
    if created.returncode != 0:
        scratch.rmdir()
        raise RuntimeError(f"could not create ASCII workspace junction: {created.stderr.strip()}")
    try:
        yield alias
    finally:
        alias.rmdir()
        scratch.rmdir()


def _run_stage(stage, device, report_dir, execution_root):
    junit = report_dir / f"final-{device}-{stage['name']}.xml"
    log = report_dir / f"final-{device}-{stage['name']}.log"
    command = [sys.executable, "-m", "pytest", *stage["paths"], "-q", "--device", device]
    if stage["world_size"] is not None:
        command.extend(("--world-size", str(stage["world_size"])))
    if device == "cuda":
        command.append("--require-gpu")
    command.append(f"--junitxml={junit.relative_to(ROOT).as_posix()}")
    print(f"[{stage['name']}] {' '.join(command)}", flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=execution_root, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, errors="replace")
    try:
        with log.open("w", encoding="utf-8") as stream:
            for line in process.stdout:
                print(line, end="", flush=True)
                stream.write(line)
        exit_code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    counts = _junit_counts(junit) if junit.exists() else None
    passed = (exit_code == 0 and counts is not None and counts["tests"] > 0
              and counts["failures"] == counts["errors"] == counts["skipped"] == 0)
    return {
        "name": stage["name"],
        "world_size": stage["world_size"],
        "command": command,
        "exit_code": exit_code,
        "duration_s": time.monotonic() - started,
        "junit": junit.relative_to(ROOT).as_posix(),
        "log": log.relative_to(ROOT).as_posix(),
        "counts": counts,
        "passed": passed,
    }


def run(device, *, world_size=None, backend=None):
    from chameleon.environment import environment_report, validate_container, validate_device

    report_dir = ROOT / "artifacts" / "test-results"
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / f"final-{device}-summary.json"
    report = {
        "schema_version": 1,
        "device": device,
        "backend": backend or "gloo",
        "requested_world_size": world_size,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "failed_stage": "preflight",
        "environment": None,
        "stages": [],
    }
    try:
        if device == "cuda" and (world_size != 8 or backend != "nccl"):
            raise ValueError("GPU final regression requires --world-size 8 --backend nccl")
        if device == "cpu" and (world_size is not None or backend is not None):
            raise ValueError("CPU final regression takes no world-size/backend override")
        report["environment"] = environment_report()
        if device == "cuda":
            validate_device("cuda", 8)
            validate_container(report["environment"])
        else:
            validate_device("cpu", 2)
        stages = matrix(device)
        _validate_coverage(stages)
        with _execution_root() as execution_root:
            report["execution_root"] = str(execution_root)
            for stage in stages:
                report["failed_stage"] = stage["name"]
                result = _run_stage(stage, device, report_dir, execution_root)
                report["stages"].append(result)
                if not result["passed"]:
                    break
            else:
                report["status"] = "passed"
                report["failed_stage"] = None
    except BaseException as error:
        field = "preflight_error" if report["failed_stage"] == "preflight" else "stage_error"
        report[field] = f"{type(error).__name__}: {error}"
        print(report[field], file=sys.stderr, flush=True)
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        summary_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Final report: {summary_path.relative_to(ROOT).as_posix()}", flush=True)
    return 0 if report["status"] == "passed" else 1


def gpu_main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--backend", required=True)
    arguments = parser.parse_args(argv)
    return run("cuda", world_size=arguments.world_size, backend=arguments.backend)
