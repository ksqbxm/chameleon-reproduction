"""Measured two-worker P2P and group startup costs, with bounded spawn cleanup."""

from datetime import timedelta
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import tempfile
import time
import traceback

from .contracts import _finite, _integer
from .environment import _write_record, validate_device


class CalibrationError(RuntimeError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def _validate_sizes(sizes):
    if not isinstance(sizes, (tuple, list)) or not sizes:
        raise ValueError("tensor_bytes must be a nonempty sequence")
    for size in sizes:
        _integer("tensor_bytes", size)
        if size % 8:
            raise ValueError("tensor_bytes must describe whole FP64 elements")
    if len(set(sizes)) != len(sizes):
        raise ValueError("tensor_bytes must be unique")


def _calibration_worker(rank, device, backend, port, directory, sizes, warmup,
                        iterations, bootstrap_rounds, timeout_s):
    import torch
    import torch.distributed as dist
    from .profiler import device_identity

    target = torch.device(f"cuda:{rank}" if device == "cuda" else "cpu")
    record = {"rank": rank, "worker_id": f"calibration-{rank}", "generation": 0,
              "pid": os.getpid(), "device_identity": None,
              "group_bootstrap_s": [], "transfers": []}
    try:
        if device == "cuda":
            torch.cuda.set_device(target)
        record["device_identity"] = device_identity(target)
        for _ in range(bootstrap_rounds):
            if dist.is_initialized():
                dist.destroy_process_group()
            start = time.monotonic()
            dist.init_process_group(
                backend, init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2,
                timeout=timedelta(seconds=timeout_s),
            )
            # Force lazy communicator initialization; init_process_group alone is insufficient for NCCL.
            dist.barrier()
            if device == "cuda":
                torch.cuda.synchronize(target)
            record["group_bootstrap_s"].append(time.monotonic() - start)
        for size in sizes:
            value = torch.empty(size // 8, dtype=torch.float64, device=target)
            for source, destination in ((0, 1), (1, 0)):
                for iteration in range(warmup + iterations):
                    value.fill_(source + 1 if rank == source else 0)
                    dist.barrier()
                    if device == "cuda":
                        torch.cuda.synchronize(target)
                        start_event, end_event = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                        start_event.record()
                    start = time.monotonic()
                    if rank == source:
                        dist.send(value, dst=destination)
                    else:
                        dist.recv(value, src=source)
                    if device == "cuda":
                        end_event.record()
                        end_event.synchronize()
                    wall_s = time.monotonic() - start
                    execution_s = start_event.elapsed_time(end_event) / 1000 if device == "cuda" else wall_s
                    if not torch.equal(value, torch.full_like(value, source + 1)):
                        raise RuntimeError("P2P tensor contents differ from source")
                    if iteration >= warmup:
                        record["transfers"].append({"source": source, "destination": destination,
                                                    "tensor_bytes": size, "iteration": iteration - warmup,
                                                    "execution_time_s": execution_s, "wall_time_s": wall_s})
        record["verified"] = True
    except BaseException:
        record["error"] = traceback.format_exc()
        raise
    finally:
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            _write_record(directory, rank, record)


def run_transfer_calibration(device: str, world_size: int = 2, *,
                             tensor_bytes: tuple[int, ...] = (64, 4096, 65536),
                             warmup: int = 2, iterations: int = 3, bootstrap_rounds: int = 2,
                             timeout_s: float = 60,
                             artifact_dir: str = "artifacts/test-results") -> dict:
    _integer("world_size", world_size)
    if world_size != 2:
        raise ValueError("transfer calibration requires exactly two real ranks")
    for name, value in (("warmup", warmup), ("iterations", iterations), ("bootstrap_rounds", bootstrap_rounds)):
        _integer(name, value, 0 if name == "warmup" else 1)
    _finite("timeout_s", timeout_s, positive=True)
    _validate_sizes(tensor_bytes)
    if Path(artifact_dir).is_absolute():
        raise ValueError("artifact_dir must be relative to the project working directory")
    backend = validate_device(device, world_size)
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    baseline = {p.pid for p in mp.active_children()}
    context = mp.get_context("spawn")
    processes, records, error = [], [], None
    audit = {"port": port}
    with tempfile.TemporaryDirectory(prefix="calibration-", dir=root) as directory:
        audit["rendezvous_dir"] = directory
        deadline = time.monotonic() + timeout_s
        try:
            for rank in range(2):
                process = context.Process(target=_calibration_worker,
                    args=(rank, device, backend, port, directory, tensor_bytes, warmup,
                          iterations, bootstrap_rounds, timeout_s))
                process.start()
                processes.append(process)
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"calibration exceeded hard timeout {timeout_s}s")
                alive = any(p.is_alive() for p in processes)
                if any(p.exitcode not in (None, 0) for p in processes):
                    raise RuntimeError("calibration worker exited abnormally")
                if not alive:
                    break
                time.sleep(0.02)
        except BaseException as exc:
            error = exc
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            cleanup_deadline = time.monotonic() + 5
            for process in processes:
                process.join(max(0, cleanup_deadline - time.monotonic()))
            for process in processes:
                if process.is_alive():
                    process.kill()
            cleanup_deadline = time.monotonic() + 5
            for process in processes:
                process.join(max(0, cleanup_deadline - time.monotonic()))
            audit["workers"] = [{"pid": p.pid, "exitcode": p.exitcode, "alive": p.is_alive()}
                                for p in processes]
            audit["leaked_pids"] = sorted({p.pid for p in mp.active_children()} - baseline)
            records = [json.loads(path.read_text(encoding="utf-8"))
                       for path in sorted(Path(directory).glob("rank-*.json"))]
            for process in processes:
                if not process.is_alive():
                    process.close()
    audit["rendezvous_removed"] = not Path(audit["rendezvous_dir"]).exists()
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
        error = RuntimeError(f"calibration resource cleanup failed; original error: {error}")
    report = {"schema_version": 1, "device": device, "backend": backend, "world_size": 2,
              "tensor_bytes": list(tensor_bytes), "iterations": iterations,
              "bootstrap_rounds": bootstrap_rounds, "records": records, "audit": audit,
              "error": str(error) if error else None}
    if not error:
        try:
            validate_calibration(report)
        except ValueError as exc:
            error = exc
            report["error"] = str(exc)
    Path(root, f"transfer-calibration-{device}.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if error:
        raise CalibrationError(str(error), audit) from error
    return report


def validate_calibration(report: dict) -> None:
    from .profiler import _keys, _validate_device_identity
    _keys(report, ("schema_version", "device", "backend", "world_size", "tensor_bytes", "iterations",
                   "bootstrap_rounds", "records", "audit", "error"), "calibration")
    if (type(report["schema_version"]) is not int or report["schema_version"] != 1
            or type(report["world_size"]) is not int or report["world_size"] != 2
            or report["device"] not in ("cpu", "cuda")
            or report["backend"] != ("nccl" if report["device"] == "cuda" else "gloo")):
        raise ValueError("invalid calibration schema/device/backend")
    for name in ("iterations", "bootstrap_rounds"):
        _integer(name, report[name])
    sizes = report["tensor_bytes"]
    if not isinstance(sizes, list):
        raise ValueError("invalid calibrated tensor sizes")
    _validate_sizes(sizes)
    records = report["records"]
    if (not isinstance(records, list) or len(records) != 2
            or any(not isinstance(r, dict) for r in records)):
        raise ValueError("calibration requires records from two distinct real ranks")
    for record in records:
        _keys(record, ("rank", "worker_id", "generation", "pid", "device_identity",
                       "group_bootstrap_s", "transfers", "verified"), "rank calibration")
        _integer("rank", record["rank"], 0)
        _integer("pid", record["pid"])
        _integer("generation", record["generation"], 0)
    if {r["rank"] for r in records} != {0, 1} or len({r["pid"] for r in records}) != 2:
        raise ValueError("calibration requires records from two distinct real ranks")
    audit = report["audit"]
    _keys(audit, ("port", "rendezvous_dir", "workers", "leaked_pids", "rendezvous_removed",
                  "port_listening", "port_reusable", "clean"), "calibration audit")
    if not isinstance(audit["workers"], list) or len(audit["workers"]) != 2:
        raise ValueError("invalid calibration worker audit")
    for worker in audit["workers"]:
        _keys(worker, ("pid", "exitcode", "alive"), "worker audit")
        _integer("worker pid", worker["pid"])
        if worker["alive"] is not False or type(worker["exitcode"]) is not int or worker["exitcode"] != 0:
            raise ValueError("calibration workers must exit successfully")
    if (report["error"] is not None or audit["clean"] is not True or audit["leaked_pids"]
            or not isinstance(audit["leaked_pids"], list)
            or audit["rendezvous_removed"] is not True or audit["port_listening"] is not False
            or audit["port_reusable"] is not True
            or {w["pid"] for w in audit["workers"]} != {r["pid"] for r in records}):
        raise ValueError("calibration failed or resources were not cleaned up")
    expected = {(src, dst, size, iteration) for src, dst in ((0, 1), (1, 0))
                for size in sizes for iteration in range(report["iterations"])}
    for record in records:
        _validate_device_identity(record["device_identity"])
        if not isinstance(record["group_bootstrap_s"], list):
            raise ValueError("invalid group bootstrap samples")
        if (record["verified"] is not True or record["worker_id"] != f"calibration-{record['rank']}"
                or record["generation"] != 0 or record["device_identity"]["type"] != report["device"]
                or len(record["group_bootstrap_s"]) != report["bootstrap_rounds"]):
            raise ValueError("invalid rank calibration identity/bootstrap")
        if report["device"] == "cuda" and record["device_identity"]["index"] != record["rank"]:
            raise ValueError("CUDA ranks must use distinct real devices")
        for duration in record["group_bootstrap_s"]:
            _finite("group bootstrap", duration, positive=True)
        transfers = record["transfers"]
        if not isinstance(transfers, list):
            raise ValueError("invalid transfer sample list")
        observed = set()
        for row in transfers:
            _keys(row, ("source", "destination", "tensor_bytes", "iteration",
                        "execution_time_s", "wall_time_s"), "transfer sample")
            for name in ("source", "destination", "iteration"):
                _integer(name, row[name], 0)
            _integer("tensor_bytes", row["tensor_bytes"])
            key = row["source"], row["destination"], row["tensor_bytes"], row["iteration"]
            if key in observed:
                raise ValueError("duplicate transfer sample")
            observed.add(key)
            _finite("P2P execution time", row["execution_time_s"])
            _finite("P2P wall time", row["wall_time_s"], positive=True)
        if observed != expected:
            raise ValueError("missing or unexpected transfer samples")
    devices = [r["device_identity"] for r in records]
    if (devices[0]["torch"] != devices[1]["torch"]
            or (report["device"] == "cpu" and devices[0] != devices[1])
            or (report["device"] == "cuda" and devices[0]["cuda"] != devices[1]["cuda"])):
        raise ValueError("calibration ranks must share the execution environment")


def transfer_time_s(report: dict, source: int, destination: int, tensor_bytes: int) -> float:
    """Use the slower measured endpoint mean; uncalibrated sizes require new measurements."""
    validate_calibration(report)
    for name, value in (("source", source), ("destination", destination), ("tensor_bytes", tensor_bytes)):
        _integer(name, value, 1 if name == "tensor_bytes" else 0)
    if (source, destination) not in ((0, 1), (1, 0)) or tensor_bytes not in report["tensor_bytes"]:
        raise ValueError("missing P2P calibration for this edge/tensor size")
    return max(sum(row["execution_time_s"] for row in record["transfers"]
                   if (row["source"], row["destination"], row["tensor_bytes"]) == (source, destination, tensor_bytes))
               / report["iterations"] for record in report["records"])


def group_bootstrap_time_s(report: dict) -> float:
    validate_calibration(report)
    return max(sum(r["group_bootstrap_s"]) / report["bootstrap_rounds"] for r in report["records"])
