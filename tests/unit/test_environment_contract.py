from copy import deepcopy
from types import SimpleNamespace

import pytest

from chameleon.environment import EXPECTED_PACKAGES, run_spawn_smoke, validate_container, validate_device


@pytest.mark.parametrize("device,size", [("gpu", 2), ("cpu", 0), ("cpu", True)])
def test_invalid_device_contract(device, size):
    with pytest.raises(ValueError):
        validate_device(device, size)


@pytest.mark.parametrize("overrides", [
    {"timeout_s": 0}, {"timeout_s": float("nan")}, {"timeout_s": float("inf")},
    {"timeout_s": True}, {"behavior": "skip"},
])
def test_invalid_smoke_controls(overrides):
    with pytest.raises(ValueError):
        run_spawn_smoke("cpu", 2, **overrides)


def container_metadata():
    # Independent fixture of the documented environment; this is not a GPU mock.
    return {
        "system": "Linux", "os_release": {"ID": "ubuntu", "VERSION_ID": "24.04"},
        "python": "3.12.3", "executable": "/usr/bin/python", "cwd": "/workspace",
        "torch": "2.8.0a0+5228986c39.nv25.06", "cuda": "12.9",
        "container_cuda_version": "12.9.1.010",
        "nccl": "2.27.3", "visible_gpu_count": 8,
        "packages": {"numpy": "1.26.4", "scipy": "1.15.3", "pandas": "2.2.3",
                     "networkx": "3.5", "PuLP": "3.2.1", "matplotlib": "3.10.3",
                     "PyYAML": "6.0.2", "pytest": "8.1.1"},
    }


@pytest.mark.parametrize("key,value", [
    ("system", "Windows"), ("python", "3.13.0"), ("executable", "python"),
    ("cwd", "."), ("torch", "2.8.0"), ("cuda", "12.8"),
    ("container_cuda_version", None), ("nccl", "2.26.0"), ("visible_gpu_count", 2),
    ("os_release", {"ID": "debian", "VERSION_ID": "24.04"}),
])
def test_reject_container_mismatch(key, value):
    report = container_metadata()
    report[key] = value
    with pytest.raises(RuntimeError, match="contract mismatch"):
        validate_container(report)


@pytest.mark.parametrize("name", list(EXPECTED_PACKAGES))
def test_reject_missing_or_changed_package(name):
    report = container_metadata()
    report["packages"][name] = None
    with pytest.raises(RuntimeError, match=name):
        validate_container(report)


@pytest.mark.parametrize("torch_version", [
    "2.8.0a0+5228986", "2.8.0a0+5228986c39",
    "2.8.0a0+5228986.nv25.06", "2.8.0a0+5228986c39.nv25.06",
])
@pytest.mark.parametrize("cuda_version", ["12.9.1", "12.9.1.010"])
def test_valid_container_check_does_not_mutate_metadata(torch_version, cuda_version):
    report = container_metadata()
    report["torch"] = torch_version
    report["container_cuda_version"] = cuda_version
    before = deepcopy(report)
    validate_container(report)
    assert report == before


@pytest.mark.parametrize("key,value", [
    ("torch", None), ("torch", 2.8),
    ("torch", "2.9.0a0+5228986c39.nv25.06"),
    ("torch", "2.8.0a1+5228986c39.nv25.06"),
    ("torch", "2.8.0+5228986c39.nv25.06"),
    ("torch", "2.8.0a0+5228987c39.nv25.06"),
    ("torch", "2.8.0a0+5228986g39.nv25.06"),
    ("torch", "2.8.0a0+5228986c39.nv25.05"),
    ("torch", "2.8.0a0+5228986c39.nv25.060"),
    ("torch", "2.8.0a0+5228986c39.nv25.06.extra"),
    ("torch", "2.8.0a0+5228986c39.nv25.06\n"),
    ("container_cuda_version", 12.9),
    ("container_cuda_version", "12.9"),
    ("container_cuda_version", "12.9.10"),
    ("container_cuda_version", "12.9.2.010"),
    ("container_cuda_version", "12.8.1.010"),
    ("container_cuda_version", "12.9.1."),
    ("container_cuda_version", "12.9.1.010.1"),
    ("container_cuda_version", "12.9.1.abc"),
    ("container_cuda_version", "12.9.1.010\n"),
])
def test_reject_ngc_base_version_or_build_mismatch(key, value):
    report = container_metadata()
    report[key] = value
    with pytest.raises(RuntimeError, match=key):
        validate_container(report)


@pytest.fixture
def torch_module():
    import torch
    return torch


@pytest.mark.parametrize("distributed_available", [False, True])
@pytest.mark.parametrize("nccl_available", [False, True])
@pytest.mark.parametrize("gpu_count", [0, 8])
def test_nccl_report_uses_build_support_not_gpu_visibility(torch_module, monkeypatch,
                                                         distributed_available, nccl_available, gpu_count):
    from chameleon.environment import environment_report
    torch = torch_module
    dist = torch.distributed
    calls = []
    # Feature flags and metadata only; no backend or collective is replaced.
    monkeypatch.setattr(dist, "is_available", lambda: distributed_available)

    def nccl_support():
        assert distributed_available
        return nccl_available

    def nccl_version():
        assert distributed_available and nccl_available
        calls.append("version")
        return (2, 27, 3)

    monkeypatch.setattr(dist, "is_nccl_available", nccl_support)
    monkeypatch.setattr(torch.cuda.nccl, "version", nccl_version)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: gpu_count > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: gpu_count)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"metadata-device-{index}")
    monkeypatch.setattr(torch.cuda, "get_device_properties",
                        lambda index: SimpleNamespace(total_memory=1024 + index))
    report = environment_report()
    expected = "2.27.3" if distributed_available and nccl_available else None
    assert report["nccl"] == expected
    assert calls == (["version"] if expected is not None else [])
    assert report["visible_gpu_count"] == len(report["gpus"]) == gpu_count
    assert [gpu["index"] for gpu in report["gpus"]] == list(range(gpu_count))


def test_absent_nccl_allows_cpu_and_rejects_cuda(torch_module, monkeypatch):
    torch = torch_module
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_gloo_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_nccl_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 8)
    assert validate_device("cpu", 2) == "gloo"
    with pytest.raises(RuntimeError, match="CUDA mode requires real NCCL"):
        validate_device("cuda", 2)
