"""Read-only environment checks and a bounded, real spawn communication smoke."""

import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import re
import socket
import sys
import tempfile
import time
import traceback
from datetime import timedelta


EXPECTED_PACKAGES = {
    "numpy": "1.26.4", "scipy": "1.15.3", "pandas": "2.2.3",
    "networkx": "3.5", "PuLP": "3.2.1", "matplotlib": "3.10.3",
    "PyYAML": "6.0.2", "pytest": "8.1.1",
}


def validate_device(device: str, world_size: int) -> str:
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be an integer >= 1")
    import torch
    import torch.distributed as dist

    if not dist.is_available():
        raise RuntimeError("PyTorch distributed is unavailable")
    if device == "cpu":
        if not dist.is_gloo_available():
            raise RuntimeError("CPU mode requires real Gloo")
        return "gloo"
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError(f"CUDA requires {world_size} real visible GPUs; "
                           f"found {torch.cuda.device_count()}")
    if not dist.is_nccl_available():
        raise RuntimeError("CUDA mode requires real NCCL")
    return "nccl"


def environment_report() -> dict:
    import torch
    import torch.distributed as dist

    packages = {}
    for name in EXPECTED_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    nccl = torch.cuda.nccl.version() if dist.is_available() and dist.is_nccl_available() else None
    if isinstance(nccl, tuple):
        nccl = ".".join(map(str, nccl))
    return {
        "system": platform.system(),
        "os_release": platform.freedesktop_os_release() if sys.platform == "linux" else {},
        "python": platform.python_version(), "executable": sys.executable,
        "cwd": str(Path.cwd()), "torch": torch.__version__,
        "cuda": torch.version.cuda, "nccl": nccl, "packages": packages,
        "container_cuda_version": os.environ.get("CUDA_VERSION"),
        "visible_gpu_count": torch.cuda.device_count(),
        "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
                  "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                 for i in range(torch.cuda.device_count())],
    }


def validate_container(report: dict) -> None:
    """Check the fixed server contract without modifying its environment."""
    expected = {
        "system": "Linux", "python": "3.12.3", "executable": "/usr/bin/python",
        "cwd": "/workspace",
        "cuda": "12.9", "nccl": "2.27.3", "visible_gpu_count": 8,
    }
    errors = [f"{key}: expected {value!r}, got {report.get(key)!r}"
              for key, value in expected.items() if report.get(key) != value]
    # NGC expands the Git revision and appends its container release to torch's version.
    torch_version = report.get("torch")
    if (not isinstance(torch_version, str)
            or not re.fullmatch(r"2\.8\.0a0\+5228986[0-9a-f]*(?:\.nv25\.06)?", torch_version)):
        errors.append(f"torch: expected 2.8.0a0+5228986 with optional NGC 25.06 metadata, "
                      f"got {torch_version!r}")
    release = report.get("os_release", {})
    if release.get("ID") != "ubuntu" or release.get("VERSION_ID") != "24.04":
        errors.append("OS must be Ubuntu 24.04")
    for name, version in EXPECTED_PACKAGES.items():
        if report.get("packages", {}).get(name) != version:
            errors.append(f"{name}: expected {version}, got "
                          f"{report.get('packages', {}).get(name)!r}")
    # torch.version.cuda exposes major.minor only, so verify the toolkit patch separately.
    # NGC's CUDA_VERSION may append a numeric toolkit build to major.minor.patch.
    cuda_version = report.get("container_cuda_version")
    if (not isinstance(cuda_version, str)
            or not re.fullmatch(r"12\.9\.1(?:\.[0-9]+)?", cuda_version)):
        errors.append(f"container_cuda_version: expected 12.9.1 with optional numeric build, "
                      f"got {cuda_version!r}")
    if errors:
        raise RuntimeError("Container contract mismatch:\n" + "\n".join(errors))


class SmokeError(RuntimeError):
    def __init__(self, message: str, audit: dict):
        super().__init__(message)
        self.audit = audit


def _write_record(directory, rank, record):
    path = Path(directory, f"rank-{rank}.json")
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)


def _worker(rank, world_size, device, backend, port, timeout_s, output_dir, behavior):
    import torch
    import torch.distributed as dist

    record = {"worker_id": f"worker-{rank}", "generation": 0,
              "rank": rank, "pid": os.getpid(), "device": device, "backend": backend}
    try:
        if behavior == "hang" and rank == 0:
            record["status"] = "injected hang"
            _write_record(output_dir, rank, record)
            time.sleep(3600)
        if behavior == "error" and rank == 0:
            raise RuntimeError("injected worker failure")
        if device == "cuda":
            torch.cuda.set_device(rank)
            record["cuda_device"] = torch.cuda.current_device()
        dist.init_process_group(
            backend, init_method=f"tcp://127.0.0.1:{port}", rank=rank,
            world_size=world_size, timeout=timedelta(seconds=timeout_s + 30),
        )
        value = torch.tensor([rank + 1.0], dtype=torch.float64,
                             device=f"cuda:{rank}" if device == "cuda" else "cpu")
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        record["sum"] = value.item()
    except BaseException:
        record["error"] = traceback.format_exc()
        raise
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            _write_record(output_dir, rank, record)


def run_spawn_smoke(device: str, world_size: int, *, timeout_s: float = 60,
                    artifact_dir: str = "artifacts/test-results",
                    behavior: str = "normal") -> dict:
    """Spawn actual workers, SUM one tensor, and audit cleanup even on failure."""
    if (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
            or not 0 < timeout_s < float("inf")):
        raise ValueError("timeout_s must be finite and positive")
    if behavior not in ("normal", "error", "hang"):
        raise ValueError("behavior must be normal, error or hang")
    if Path(artifact_dir).is_absolute():
        raise ValueError("artifact_dir must be relative to the project working directory")
    backend = validate_device(device, world_size)
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Hold an ephemeral port until immediately before launching the workers.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    baseline = {p.pid for p in mp.active_children()}
    processes = []
    records = []
    error = None
    audit = {"backend": backend, "port": port, "world_size": world_size, "device": device}
    context = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="smoke-", dir=root) as directory:
        audit["rendezvous_dir"] = directory
        deadline = time.monotonic() + timeout_s
        try:
            for rank in range(world_size):
                process = context.Process(
                    target=_worker,
                    args=(rank, world_size, device, backend, port, timeout_s, directory, behavior),
                )
                process.start()
                processes.append(process)
            while any(p.is_alive() for p in processes):
                if any(p.exitcode not in (None, 0) for p in processes):
                    raise RuntimeError("spawn worker exited abnormally")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"spawn smoke exceeded hard timeout {timeout_s}s")
                time.sleep(0.02)
            if any(p.exitcode != 0 for p in processes):
                raise RuntimeError("spawn worker exited abnormally")
        except BaseException as exc:
            error = exc
        finally:
            # Terminate all workers before joining; a stuck communicator cannot block cleanup.
            for p in processes:
                if p.is_alive():
                    p.terminate()
            cleanup_deadline = time.monotonic() + 5
            for p in processes:
                p.join(max(0, cleanup_deadline - time.monotonic()))
            for p in processes:
                if p.is_alive():
                    p.kill()
            kill_deadline = time.monotonic() + 5
            for p in processes:
                p.join(max(0, kill_deadline - time.monotonic()))
            audit["workers"] = [{"pid": p.pid, "exitcode": p.exitcode, "alive": p.is_alive()}
                                for p in processes]
            audit["leaked_pids"] = sorted({p.pid for p in mp.active_children()} - baseline)
            for path in sorted(Path(directory).glob("rank-*.json")):
                records.append(json.loads(path.read_text(encoding="utf-8")))
            for p in processes:
                if not p.is_alive():
                    p.close()
    audit["rendezvous_removed"] = not Path(audit["rendezvous_dir"]).exists()
    # Probe both listening state and rebinding after the rendezvous server exits.
    with socket.socket() as probe:
        probe.settimeout(0.2)
        audit["port_listening"] = probe.connect_ex(("127.0.0.1", port)) == 0
    with socket.socket() as probe:
        if os.name != "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
            audit["port_reusable"] = True
        except OSError:
            audit["port_reusable"] = False
    audit["clean"] = (not audit["leaked_pids"] and audit["rendezvous_removed"]
                      and not audit["port_listening"] and audit["port_reusable"]
                      and all(not w["alive"] for w in audit["workers"]))
    if not audit["clean"]:
        error = RuntimeError(f"spawn smoke resource cleanup failed; original error: {error}")
    expected = world_size * (world_size + 1) / 2
    if not error and (len(records) != world_size or len({r["pid"] for r in records}) != world_size
            or any(r.get("sum") != expected or "error" in r for r in records)):
        error = RuntimeError("worker records or distributed SUM are invalid")
    if not error and device == "cuda" and {r.get("cuda_device") for r in records} != set(range(world_size)):
        error = RuntimeError("workers must own distinct CUDA devices")
    report = {"audit": audit, "records": records, "error": str(error) if error else None}
    Path(root, f"smoke-{device}-{behavior}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    if error:
        raise SmokeError(str(error), audit) from error
    return report
