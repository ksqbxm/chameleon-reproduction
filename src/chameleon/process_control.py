"""Shared bounded process supervision and rendezvous resource auditing."""

import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import time


class ProcessAuditError(RuntimeError):
    def __init__(self, message: str, audit: dict):
        super().__init__(message)
        self.audit = audit


def write_json_atomic(path, payload) -> None:
    destination = Path(path)
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    temporary.replace(destination)


def reserve_tcp_ports(count: int) -> tuple[int, ...]:
    reservations = []
    try:
        for _ in range(count):
            reservation = socket.socket()
            try:
                reservation.bind(("127.0.0.1", 0))
            except BaseException:
                reservation.close()
                raise
            reservations.append(reservation)
        return tuple(reservation.getsockname()[1] for reservation in reservations)
    finally:
        for reservation in reservations:
            reservation.close()


def wait_processes(processes, deadline: float, label: str) -> None:
    while any(process.is_alive() for process in processes):
        if any(process.exitcode not in (None, 0) for process in processes):
            raise RuntimeError(f"{label} worker exited abnormally")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"{label} exceeded its hard timeout")
        time.sleep(0.02)
    if any(process.exitcode != 0 for process in processes):
        raise RuntimeError(f"{label} worker exited abnormally")


def terminate_processes(processes, *, timeout_s: float = 5) -> list[dict]:
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + timeout_s
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.kill()
    deadline = time.monotonic() + timeout_s
    for process in processes:
        process.join(max(0, deadline - time.monotonic()))
    return [{"pid": process.pid, "exitcode": process.exitcode, "alive": process.is_alive()}
            for process in processes]


def close_processes(processes) -> None:
    for process in processes:
        if not process.is_alive():
            process.close()


def child_process_leaks(baseline: set[int], *, owned_pids=None) -> list[int]:
    leaked = {process.pid for process in mp.active_children()} - baseline
    if owned_pids is not None:
        leaked &= set(owned_pids)
    return sorted(leaked)


def audit_tcp_ports(ports) -> tuple[bool, bool]:
    listening, reusable = [], []
    for port in ports:
        with socket.socket() as probe:
            probe.settimeout(0.2)
            listening.append(probe.connect_ex(("127.0.0.1", port)) == 0)
        with socket.socket() as probe:
            if os.name != "nt":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
                reusable.append(True)
            except OSError:
                reusable.append(False)
    return any(listening), all(reusable)
