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


@pytest.fixture(scope="module")
def symmetric_training(request):
    """Three real DP2/PP2 updates; reference tensors never enter the runtime."""
    import torch
    import time
    from chameleon import ClusterState, ModelConfig, WorkerIdentity
    from chameleon.environment import environment_report, validate_container
    from chameleon.model import build_initial_model
    from chameleon.reference import ReferenceTrainer
    from chameleon.runtime import SymmetricRuntime, SymmetricTopology

    device = request.config.getoption("--device")
    if request.config.getoption("--world-size") != 4:
        pytest.fail("Task09 DP2/PP2 acceptance requires --world-size 4")
    if device == "cuda":
        validate_container(environment_report())
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=11, micro_batch_size=2)
    workers = tuple(WorkerIdentity(f"stable-{20 - rank}", rank, 2) for rank in reversed(range(4)))
    topology = SymmetricTopology(ClusterState(workers, 11, generation=2), config,
                                (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head")))
    runtime = SymmetricRuntime(topology, device=device, capture_state=True, lr=.007, weight_decay=.125)
    with runtime:
        steps = []
        for step in range(3):
            assert runtime.state.committed_global_step == step
            steps.append(runtime.train_step())
            assert runtime.state.committed_global_step == step + 1
            assert all(process.is_alive() for process in runtime.processes)
            assert [process.pid for process in runtime.processes] == [row["pid"] for row in runtime.ready]
            time.sleep(.1)
            assert not any(connection.poll() for connection in runtime.connections)
            assert runtime.state.committed_global_step == step + 1
        profiles = runtime.snapshot_profiles()
    assert runtime.audit["clean"]
    assert all(worker["exitcode"] == 0 for worker in runtime.audit["workers"])
    torch.set_num_threads(1)
    reference = ReferenceTrainer(build_initial_model(config, device=device),
                                 ClusterState((WorkerIdentity("reference", 0, 0),), 11),
                                 lr=.007, weight_decay=.125)
    return {"runtime": runtime, "steps": steps, "reference": [reference.train_step() for _ in range(3)],
            "profiles": profiles, "config": config, "device": device}


def pytest_terminal_summary(terminalreporter):
    for path in sorted(terminalreporter.config._chameleon_reports):
        terminalreporter.write_line(f"{path}:\n{path.read_text(encoding='utf-8')}")


@pytest.fixture(scope="module")
def asymmetric_training(request):
    """Task10 initial topology only; three real SUM/AdamW steps and live profiles."""
    import torch
    from chameleon import ClusterState, ModelConfig, WorkerIdentity
    from chameleon.environment import environment_report, validate_container
    from chameleon.model import build_initial_model
    from chameleon.reference import ReferenceTrainer
    from chameleon.runtime import DynamicTopology, SymmetricRuntime

    device = request.config.getoption("--device")
    size = 8 if device == "cuda" else 7
    if request.config.getoption("--world-size") != size:
        pytest.fail(f"Task10 acceptance requires --world-size {size} on {device}")
    if device == "cuda":
        validate_container(environment_report())
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=19, micro_batch_size=2)
    short = (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head"))
    long = (("embedding",), ("blocks.0",), ("blocks.1", "final_norm", "lm_head"))
    other_long = (("embedding", "blocks.0"), ("blocks.1",), ("final_norm", "lm_head"))
    layouts = (short, long, other_long) if device == "cuda" else (short, short, long)
    workers = tuple(WorkerIdentity(f"asymmetric-{20 - r}", r, 2) for r in reversed(range(size)))
    topology = DynamicTopology(ClusterState(workers, 19, generation=2), config, layouts, (5, 3, 2))
    runtime = SymmetricRuntime(topology, device=device, capture_state=True, lr=.007, weight_decay=.125)
    with runtime:
        steps = [runtime.train_step() for _ in range(3)]
        profiles = runtime.snapshot_profiles()
        assert runtime.state.committed_global_step == 3
    assert runtime.audit["clean"]
    assert all(w["exitcode"] == 0 for w in runtime.audit["workers"])
    request.config._chameleon_reports.add(runtime.report_path)
    torch.set_num_threads(1)
    reference = ReferenceTrainer(build_initial_model(config, device=device),
                                 ClusterState((WorkerIdentity("reference", 0, 0),), 19), lr=.007, weight_decay=.125)
    return dict(runtime=runtime, topology=topology, config=config, steps=steps,
                profiles=profiles, reference=[reference.train_step() for _ in range(3)], device=device)
