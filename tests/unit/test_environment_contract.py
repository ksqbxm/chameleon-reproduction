from copy import deepcopy

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
        "torch": "2.8.0a0+5228986", "cuda": "12.9", "container_cuda_version": "12.9.1",
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


def test_valid_container_check_does_not_mutate_metadata():
    report = container_metadata()
    before = deepcopy(report)
    validate_container(report)
    assert report == before
