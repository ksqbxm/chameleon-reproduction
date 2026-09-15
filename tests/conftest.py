import pytest


def _select_candidates(candidates, inter_fault_duration_s):
    """Exercise the public Equation 8 entry with a controlled evaluator stub."""
    from chameleon.decision_center import DecisionCenter

    center = object.__new__(DecisionCenter)
    center.evaluate_candidates = lambda state, profile: tuple(candidates)
    return center.select(None, None, inter_fault_duration_s)


def assert_numerical_step(actual, expected, device, *, fp32=False, owner_counts=None):
    from collections import Counter
    import torch

    tolerance = (dict(rtol=1e-4, atol=1e-6) if fp32 else
                 dict(rtol=1e-7, atol=1e-9) if device == "cuda" else dict(rtol=1e-8, atol=1e-10))
    assert actual["sample_ids"] == list(expected.sample_ids)
    assert actual["global_sample_count"] == expected.global_sample_count
    assert actual["step_id"] == expected.committed_global_step
    assert actual["loss_global_sum"] == pytest.approx(expected.loss_global_sum,
                                                     rel=tolerance["rtol"], abs=tolerance["atol"])
    owners = Counter()
    for report, snapshot in zip(actual["reports"], actual["snapshots"]):
        assert set(snapshot) == {"parameters", "gradients", "optimizer_state"}
        assert snapshot["parameters"].keys() == snapshot["gradients"].keys() == snapshot["optimizer_state"].keys()
        assert set(report["synchronized_parameters"]) == snapshot["parameters"].keys()
        for name, parameter in snapshot["parameters"].items():
            owners[name] += 1
            torch.testing.assert_close(parameter, expected.parameters[name], **tolerance,
                                       msg=lambda message: f"step {actual['step_id']}, {name}, parameter: {message}")
            torch.testing.assert_close(snapshot["gradients"][name], expected.gradients[name], **tolerance,
                                       msg=lambda message: f"step {actual['step_id']}, {name}, gradient: {message}")
            state = snapshot["optimizer_state"][name]
            assert set(state) == {"step", "exp_avg", "exp_avg_sq"}
            for field, value in state.items():
                torch.testing.assert_close(value, expected.optimizer_state[name][field], **tolerance,
                                           msg=lambda message: f"step {actual['step_id']}, {name}, AdamW.{field}: {message}")
    dp_size = len({report["pipeline"] for report in actual["reports"]})
    assert owners == Counter(owner_counts if owner_counts is not None else
                             {name: dp_size for name in expected.parameters})
    assert {name.split(".")[0] for name in owners} == {"embedding", "blocks", "final_norm", "lm_head"}


def _kill_workers(runtime, ranks, *, timeout_s=10):
    from chameleon.contracts import FailureEvent

    ranks = tuple(ranks)
    workers = tuple(runtime.topology.ranks[rank] for rank in ranks)
    audit = []
    for rank in ranks:
        process = runtime.processes[rank]
        process.kill()
        process.join(timeout_s)
        assert not process.is_alive() and process.exitcode not in (None, 0)
        audit.append(dict(worker_id=runtime.topology.ranks[rank].worker_id, pid=process.pid,
                          exitcode=process.exitcode))
    return (FailureEvent(tuple(worker.worker_id for worker in workers), runtime.state.generation,
                         runtime.state.committed_global_step), tuple(audit))


def _reply_hashes(replies):
    return {row["worker"].worker_id: {tuple(item["key"]): item["digest"] for item in row["hashes"]}
            for row in replies}


def _logged_hashes(rows):
    return {row["worker"]["worker_id"]: {tuple(item["key"]): item["digest"] for item in row["hashes"]}
            for row in rows}


def pytest_addoption(parser):
    parser.addoption("--device", choices=("cpu", "cuda"), default="cpu")
    parser.addoption("--world-size", type=int, default=None,
                     help="Worker count (default: 2; Task12 recovery: 4)")
    parser.addoption("--require-gpu", action="store_true", default=False)


def _world_size(config, default=2):
    requested = config.getoption("--world-size")
    return default if requested is None else requested


def pytest_configure(config):
    config._chameleon_reports = set()
    if _world_size(config) < 1:
        raise pytest.UsageError("--world-size must be >= 1")
    if config.getoption("--require-gpu") and config.getoption("--device") != "cuda":
        raise pytest.UsageError("--require-gpu requires --device cuda")
    # Fail even a unit-only CUDA invocation when the requested GPUs are absent.
    if config.getoption("--device") == "cuda":
        from chameleon.environment import validate_device
        try:
            validate_device("cuda", _world_size(config))
        except (ImportError, RuntimeError, ValueError) as exc:
            raise pytest.UsageError(str(exc)) from exc


@pytest.fixture
def device(request):
    return request.config.getoption("--device")


@pytest.fixture
def world_size(request):
    return _world_size(request.config)


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
    from chameleon.runtime import DistributedRuntime, SymmetricTopology

    device = request.config.getoption("--device")
    if _world_size(request.config) != 4:
        pytest.fail("Task09 DP2/PP2 acceptance requires --world-size 4")
    if device == "cuda":
        validate_container(environment_report())
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=11, micro_batch_size=2)
    workers = tuple(WorkerIdentity(f"stable-{20 - rank}", rank, 2) for rank in reversed(range(4)))
    topology = SymmetricTopology(ClusterState(workers, 11, generation=2), config,
                                (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head")))
    runtime = DistributedRuntime(topology, device=device, capture_state=True, lr=.007, weight_decay=.125)
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
    from chameleon.runtime import DistributedRuntime, DynamicTopology

    device = request.config.getoption("--device")
    size = 8 if device == "cuda" else 7
    if _world_size(request.config) != size:
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
    runtime = DistributedRuntime(topology, device=device, capture_state=True, lr=.007, weight_decay=.125)
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


@pytest.fixture(scope="module")
def rerouted_run(request):
    """Initial missing slots only; real workers and independent reference after shutdown."""
    import torch
    from chameleon import ClusterState, ModelConfig, WorkerIdentity
    from chameleon.environment import environment_report, validate_container, validate_device
    from chameleon.model import build_initial_model
    from chameleon.reference import ReferenceTrainer
    from chameleon.runtime import DistributedRuntime, ReroutingTopology

    device = request.config.getoption("--device")
    size = _world_size(request.config)
    validate_device(device, size)
    if device == "cuda":
        validate_container(environment_report())

    def run(slots, counts, *, stages=None, batch=19, dtype="float64"):
        config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                             sequence_length=3, global_batch_size=batch, micro_batch_size=2)
        stages = stages or (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head"))
        workers = tuple(WorkerIdentity(f"peer-{20 - r}", r, 2) for r in reversed(range(size)))
        topology = ReroutingTopology(ClusterState(workers, batch, generation=2), config, stages, counts, slots)
        runtime = DistributedRuntime(topology, device=device, dtype=dtype, capture_state=True, lr=.007, weight_decay=.125)
        with runtime:
            steps = [runtime.train_step() for _ in range(3)]
            assert all(process.is_alive() for process in runtime.processes)
        assert runtime.audit["clean"] and all(worker["exitcode"] == 0 for worker in runtime.audit["workers"])
        torch.set_num_threads(1)
        reference = ReferenceTrainer(build_initial_model(config, device=device, dtype=getattr(torch, dtype)),
                                     ClusterState((WorkerIdentity("reference", 0, 0),), batch), lr=.007, weight_decay=.125)
        return dict(runtime=runtime, topology=topology, config=config, steps=steps,
                    reference=[reference.train_step() for _ in range(3)], device=device, dtype=dtype)

    return run


@pytest.fixture(scope="module")
def rerouted_training(rerouted_run, request):
    if _world_size(request.config) != 5:
        pytest.fail("Task11 DP3/PP2 acceptance requires --world-size 5")
    return rerouted_run(((0, None), (1, 2), (3, 4)), (5, 3, 2))


def _adaptive_base_profile(config, device):
    import torch
    from chameleon import ClusterState, WorkerIdentity
    from chameleon.model import build_initial_model
    from chameleon.profiler import Profiler, train_profile_step

    if device == "cuda":
        torch.cuda.set_device(0)
    model = build_initial_model(config, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.007, weight_decay=.125, amsgrad=False)
    state = ClusterState((WorkerIdentity("profile", 0, 0),), config.global_batch_size)
    state, _ = train_profile_step(model, optimizer, state)
    profiler = Profiler(model, optimizer)
    for _ in range(2):
        state, _ = train_profile_step(model, optimizer, state, profiler=profiler)
    profile = profiler.snapshot()
    del profiler, optimizer, model
    if device == "cuda":
        torch.cuda.empty_cache()
    return profile


def _controlled_adaptive_profile(base, required, *, revision):
    """Build one schema-valid deterministic profile for adaptive-policy functional tests."""
    from copy import deepcopy

    profile = deepcopy(base)
    durations = {"embedding": .01, "blocks.0": 1., "blocks.1": .05,
                 "final_norm": .01, "lm_head": .01}
    for module, duration in durations.items():
        for kind in ("forward_s", "backward_s"):
            series = profile["metrics"][f"modules.{module}.{kind}"]
            series["samples"] = [duration] * len(series["samples"])
            series["ema"] = duration
    operation_s = sum(durations.values())
    step_times = []
    for step in profile["steps"]:
        elapsed = 0.
        for row in step["trace"]:
            row["start_s"] = elapsed
            elapsed += operation_s
            row["end_s"] = elapsed
        step_times.append(elapsed)
    for name in ("step_time_s", "step_wall_time_s"):
        series = profile["metrics"][name]
        series["samples"] = step_times
        ema = step_times[0]
        for sample in step_times[1:]:
            ema = profile["ema_alpha"] * sample + (1 - profile["ema_alpha"]) * ema
        series["ema"] = ema

    device = profile["identity"]["device"]["type"]
    identities = [deepcopy(profile["identity"]["device"]) for _ in range(2)]
    if device == "cuda":
        identities[1]["index"] = 1
    sizes = sorted({tensor.nbytes for tensor in required})
    transfer_s = .25 + revision * 1e-6
    records = []
    for rank in range(2):
        transfers = [dict(source=source, destination=destination, tensor_bytes=size,
                          iteration=iteration, execution_time_s=transfer_s, wall_time_s=transfer_s)
                     for source, destination in ((0, 1), (1, 0))
                     for size in sizes for iteration in range(2)]
        records.append(dict(rank=rank, worker_id=f"calibration-{rank}", generation=0,
                            pid=13130 + revision * 2 + rank,
                            device_identity=identities[rank], group_bootstrap_s=[.05, .05],
                            transfers=transfers, verified=True))
    profile["calibrations"] = [dict(schema_version=1, device=device,
        backend="nccl" if device == "cuda" else "gloo", world_size=2,
        tensor_bytes=sizes, iterations=2, bootstrap_rounds=2, records=records,
        audit=dict(port=13130 + revision,
                   rendezvous_dir="artifacts/test-results/controlled-adaptive-profile",
                   workers=[dict(pid=13130 + revision * 2 + rank, exitcode=0, alive=False)
                            for rank in range(2)],
                   leaked_pids=[], rendezvous_removed=True, port_listening=False,
                   port_reusable=True, clean=True), error=None)]
    return profile


def _guarded_recovery_worker(*args, fault="normal"):
    """Spies delegate real training/group operations; recovery cannot initialize or load."""
    import time
    import torch
    from chameleon import model, recovery, runtime

    calls = dict(model=0, constructor=0)
    initial, constructor = model.build_initial_model, model.TinyTransformer.__init__
    def initial_spy(*values, **options):
        calls["model"] += 1
        if calls["model"] != 1:
            raise AssertionError("recovery called model initialization")
        return initial(*values, **options)
    def constructor_spy(*values, **options):
        calls["constructor"] += 1
        if calls["constructor"] != 1:
            raise AssertionError("recovery constructed initialized parameters")
        return constructor(*values, **options)
    def forbidden_load(*values, **options):
        raise AssertionError("worker recovery read a checkpoint/test state")
    model.build_initial_model, model.TinyTransformer.__init__ = initial_spy, constructor_spy
    torch.load = forbidden_load
    write = runtime._write_reply
    def audited_reply(*values, **options):
        options.update(initialization_calls=dict(calls), checkpoint_reads=0)
        return write(*values, **options)
    runtime._write_reply = audited_reply
    if fault == "missing_endpoint_source":
        inspect = recovery.inspect_live_state
        inspections = 0
        def missing_endpoint(model_value, optimizer_value, step):
            nonlocal inspections
            inspections += 1
            if args[1] == 3 and inspections == 2:
                parameter = next(p for name, p in model_value.named_parameters() if name.startswith("lm_head."))
                del optimizer_value.state[parameter]["exp_avg"]
            return inspect(model_value, optimizer_value, step)
        recovery.inspect_live_state = missing_endpoint
    if fault in ("transfer_error", "source_mutation"):
        transfer = recovery.transfer_state
        def guarded_transfer(manifest, identity, new_ranks, structure, model_value, optimizer_value,
                             device, committed_step):
            if fault == "source_mutation":
                parameter = next(model_value.parameters())
                optimizer_value.state[parameter]["exp_avg"].add_(1)
            result = transfer(manifest, identity, new_ranks, structure, model_value, optimizer_value,
                              device, committed_step)
            if fault == "transfer_error":
                raise RuntimeError("injected state transfer failure")
            return result
        recovery.transfer_state = guarded_transfer
    if fault == "group_timeout":
        create = runtime._create_training_groups
        count = 0
        def delayed_groups(*values, **options):
            nonlocal count
            count += 1
            if count == 2:
                time.sleep(3600)
            return create(*values, **options)
        runtime._create_training_groups = delayed_groups
    runtime._runtime_worker(*args)


@pytest.fixture(scope="module")
def recovery_setup(request):
    if _world_size(request.config, default=4) != 4:
        pytest.fail("Task12 DP2/PP2 acceptance requires --world-size 4")
    import torch
    from chameleon import ClusterState, ModelConfig, WorkerIdentity
    from chameleon.environment import environment_report, validate_container, validate_device
    from chameleon.model import build_initial_model
    from chameleon.profiler import Profiler, train_profile_step
    from chameleon.runtime import SymmetricTopology

    device = request.config.getoption("--device")
    validate_device(device, 4)
    if device == "cuda":
        validate_container(environment_report())
    torch.set_num_threads(1)
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=11, micro_batch_size=2)
    # Independent profiling supplies metadata/timings only; destroy its tensors before runtime startup.
    model = build_initial_model(config, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.007, weight_decay=.125, amsgrad=False)
    state = ClusterState((WorkerIdentity("profile", 0, 0),), 11)
    state, _ = train_profile_step(model, optimizer, state)
    profiler = Profiler(model, optimizer)
    for _ in range(2):
        state, _ = train_profile_step(model, optimizer, state, profiler=profiler)
    profile = profiler.snapshot()
    del profiler, optimizer, model
    workers = tuple(WorkerIdentity(f"survivor-{20 - r}", r, 4) for r in reversed(range(4)))
    topology = SymmetricTopology(ClusterState(workers, 11, generation=4), config,
        (("embedding",), ("blocks.0", "blocks.1", "final_norm", "lm_head")))
    return topology, profile, device


def _recovery_decision(recovery_state, profile):
    """Both real candidates, measured compute, and explicitly controlled dynamic transition."""
    from dataclasses import replace
    from chameleon.decision_center import DecisionCenter
    from chameleon.restorer import MigrationManifest, TransitionEstimate
    from chameleon.contracts import ModelConfig
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=11, micro_batch_size=2)
    center = DecisionCenter(config, expected_identity=profile["identity"], r_dp=(2,), r_pp=(1, 2),
                            memory_capacity_bytes=10**12)
    rerouting, dynamic = center.evaluate_candidates(recovery_state, profile)
    assert isinstance(dynamic.execution, MigrationManifest)
    assert dynamic.reasons == ("missing transfer/bootstrap calibration for dynamic transition",)
    manifest = dynamic.execution
    transition = TransitionEstimate(0., .01, None, {"source": "controlled selection-test transition, not measured"},
                                    manifest.manifest_id)
    dynamic = replace(dynamic, transition=transition, reasons=(),
                      derivation=dict(dynamic.derivation, transition_source="controlled Task12 selection fixture"))
    decision = _select_candidates((rerouting, dynamic), 100.)
    assert decision.plan.policy == "dynamic"
    return decision


@pytest.fixture(scope="module")
def recovered_training(request, recovery_setup):
    from unittest.mock import patch
    from chameleon import ClusterState, WorkerIdentity
    from chameleon.model import build_initial_model
    from chameleon.reference import ReferenceTrainer
    from chameleon.runtime import DistributedRuntime

    topology, profile, device = recovery_setup
    runtime = DistributedRuntime(topology, device=device, capture_state=True, lr=.007, weight_decay=.125)
    with patch("chameleon.runtime._runtime_worker", _guarded_recovery_worker), runtime:
        original = list(runtime.ready)
        steps = [runtime.train_step() for _ in range(3)]
        before = runtime.inspect_state()
        failure, _ = _kill_workers(runtime, (1,), timeout_s=5)
        decision = _recovery_decision(runtime.recovery_state(failure), profile)
        recovered = runtime.recover(failure, decision)
        assert runtime.state.committed_global_step == 3
        steps += [runtime.train_step() for _ in range(2)]
    request.config._chameleon_reports.add(runtime.report_path)
    assert runtime.audit["clean"]
    reference = ReferenceTrainer(build_initial_model(topology.config, device=device),
        ClusterState((WorkerIdentity("reference", 0, 0),), 11), lr=.007, weight_decay=.125)
    return dict(runtime=runtime, original=original, before=before, recovery=recovered, decision=decision,
                steps=steps, reference=[reference.train_step() for _ in range(5)], device=device)


@pytest.fixture
def recovery_failure(request, recovery_setup):
    from dataclasses import replace
    from functools import partial
    from unittest.mock import patch
    from chameleon.contracts import UnrecoverableStateError
    from chameleon.runtime import DistributedRuntime, RuntimeErrorWithAudit

    topology, profile, device = recovery_setup
    def run(fault):
        worker = partial(_guarded_recovery_worker, fault=fault)
        runtime = DistributedRuntime(topology, device=device, lr=.007, weight_decay=.125)
        expectation_failure = None
        try:
            with patch("chameleon.runtime._runtime_worker", worker), runtime:
                for _ in range(3):
                    runtime.train_step()
                runtime.inspect_state()
                failure, _ = _kill_workers(runtime, (1,), timeout_s=5)
                try:
                    if fault == "missing_endpoint_source":
                        with pytest.raises(UnrecoverableStateError, match="lm_head"):
                            runtime.recovery_state(failure)
                    else:
                        decision = _recovery_decision(runtime.recovery_state(failure), profile)
                        if fault == "missing_manifest_tensor":
                            manifest = decision.candidate.execution
                            broken = replace(manifest, actions=manifest.actions[:-1])
                            decision = replace(decision, candidate=replace(decision.candidate, execution=broken))
                        if fault == "group_timeout":
                            runtime.timeout_s = 5
                        error = UnrecoverableStateError if fault in ("missing_manifest_tensor", "source_mutation") else RuntimeErrorWithAudit
                        message = {"missing_manifest_tensor": "manifest lacks complete",
                                   "source_mutation": "survivor state changed",
                                   "transfer_error": "injected state transfer failure",
                                   "group_timeout": "runtime staged exceeded hard timeout"}[fault]
                        with pytest.raises(error, match=message):
                            runtime.recover(failure, decision)
                except BaseException as exc:
                    expectation_failure = exc
                atomic = dict(generation=runtime.state.generation == topology.state.generation,
                              committed_step=runtime.state.committed_global_step == 3,
                              topology=runtime.topology is topology,
                              no_recovery=not runtime.recoveries)
        finally:
            runtime.close("failure test cleanup")
            if hasattr(runtime, "report_path"):
                request.config._chameleon_reports.add(runtime.report_path)
        audit = runtime.audit
        checks = dict(atomic, clean=audit["clean"], no_leaked_pids=not audit["leaked_pids"],
                      original_workers=len(audit["workers"]) == len(topology.ranks),
                      workers_stopped=all(not row["alive"] for row in audit["workers"]),
                      stores_removed=all(row["removed"] for row in audit["rendezvous_files"]),
                      rendezvous_removed=audit["rendezvous_removed"])
        assert all(checks.values()), checks
        if expectation_failure is not None:
            raise expectation_failure
    return run
