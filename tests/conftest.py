import pytest


def pytest_addoption(parser):
    parser.addoption("--device", choices=("cpu", "cuda"), default="cpu")
    parser.addoption("--world-size", type=int, default=2)
    parser.addoption("--require-gpu", action="store_true", default=False)


def pytest_configure(config):
    config._chameleon_reports = set()
    if config.getoption("--world-size") < 1:
        raise pytest.UsageError("--world-size must be >= 1")
    if config.getoption("--require-gpu") and config.getoption("--device") != "cuda":
        raise pytest.UsageError("--require-gpu requires --device cuda")
    # Fail even a unit-only CUDA invocation when the requested GPUs are absent.
    if config.getoption("--device") == "cuda":
        from chameleon.environment import validate_device
        try:
            validate_device("cuda", config.getoption("--world-size"))
        except (ImportError, RuntimeError, ValueError) as exc:
            raise pytest.UsageError(str(exc)) from exc


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def world_size(request):
    return request.config.getoption("--world-size")


@pytest.fixture
def distributed_environment(request, device, world_size):
    import json
    from pathlib import Path
    from chameleon.environment import environment_report, validate_container, validate_device
    validate_device(device, world_size)
    report = environment_report()
    root = Path("artifacts/test-results")
    root.mkdir(parents=True, exist_ok=True)
    previous = {p: p.stat().st_mtime_ns for p in root.glob(f"smoke-{device}-*.json")}
    path = root / f"{device}-environment.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    request.config._chameleon_reports.add(path)
    if device == "cuda":
        validate_container(report)
    yield report
    request.config._chameleon_reports.update(
        p for p in root.glob(f"smoke-{device}-*.json")
        if previous.get(p) != p.stat().st_mtime_ns
    )


@pytest.fixture
def cpu_gloo():
    from chameleon.environment import validate_device
    return validate_device("cpu", 2)


@pytest.fixture
def cuda_nccl(world_size):
    from chameleon.environment import validate_device
    return validate_device("cuda", world_size)


def pytest_terminal_summary(terminalreporter):
    for path in sorted(terminalreporter.config._chameleon_reports):
        terminalreporter.write_line(f"{path}:\n{path.read_text(encoding='utf-8')}")
