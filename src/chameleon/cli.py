"""Command-line entry points for profiling, training and safe-point recovery demos."""

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

from .contracts import ClusterState, FailureEvent, ModelConfig, WorkerIdentity, _finite, _integer
from .environment import environment_report
from .model import build_initial_model
from .plan_cache import PlanCache
from .profiler import Profiler, _hash, export_profile, load_profile, profile_identity, train_profile_step
from .runtime import SymmetricRuntime, SymmetricTopology
from .transfer_calibration import run_transfer_calibration


CONFIG_FIELDS = {
    "schema_version", "device", "backend", "world_size", "dtype", "artifact_dir",
    "model", "parallel", "training", "planner", "profile",
}


def _relative_path(value, name):
    path = Path(value) if isinstance(value, str) else None
    if (path is None or not value.strip() or path.is_absolute() or path.drive
            or ".." in path.parts):
        raise ValueError(f"{name} must be a nonempty path relative to the working directory")
    return path


def _fields(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"invalid {name} fields")


def _number(name, value, *, positive=False):
    _finite(name, value, positive=positive)
    return value


def load_config(path: str) -> dict:
    """Load the strict versioned CLI configuration without changing the environment."""
    source = _relative_path(path, "config")
    payload = json.loads(source.read_text(encoding="utf-8"))
    _fields(payload, CONFIG_FIELDS, "config")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported CLI config schema version")
    if payload["device"] not in ("cpu", "cuda"):
        raise ValueError("config device must be cpu or cuda")
    expected_backend = "nccl" if payload["device"] == "cuda" else "gloo"
    if payload["backend"] != expected_backend:
        raise ValueError("config backend must match its device")
    _integer("world_size", payload["world_size"])
    if payload["dtype"] not in ("float64", "float32"):
        raise ValueError("config dtype must be float64 or float32")
    payload["artifact_dir"] = str(_relative_path(payload["artifact_dir"], "artifact_dir"))

    _fields(payload["model"], ModelConfig.__dataclass_fields__, "model")
    model = ModelConfig(**payload["model"])
    if model.paper_path is not None:
        _relative_path(model.paper_path, "paper_path")

    parallel = payload["parallel"]
    _fields(parallel, ("dp_size", "pp_size", "stage_modules", "failure_rank"), "parallel")
    _integer("dp_size", parallel["dp_size"])
    _integer("pp_size", parallel["pp_size"])
    _integer("failure_rank", parallel["failure_rank"], 0)
    if parallel["dp_size"] * parallel["pp_size"] != payload["world_size"]:
        raise ValueError("dp_size * pp_size must equal world_size")
    if parallel["failure_rank"] >= payload["world_size"]:
        raise ValueError("failure_rank must belong to the initial topology")
    stages = tuple(tuple(stage) for stage in parallel["stage_modules"])
    if len(stages) != parallel["pp_size"]:
        raise ValueError("stage_modules must match pp_size")
    expected_modules = ("embedding", *(f"blocks.{i}" for i in range(model.num_layers)),
                        "final_norm", "lm_head")
    if (any(not stage for stage in stages)
            or tuple(module for stage in stages for module in stage) != expected_modules):
        raise ValueError("stage_modules must partition the complete model in order")
    parallel["stage_modules"] = stages

    training = payload["training"]
    _fields(training, ("steps", "resume_steps", "lr", "weight_decay", "timeout_s"), "training")
    _integer("training steps", training["steps"])
    _integer("resume_steps", training["resume_steps"])
    _number("lr", training["lr"], positive=True)
    _number("weight_decay", training["weight_decay"])
    _number("timeout_s", training["timeout_s"], positive=True)

    planner = payload["planner"]
    _fields(planner, ("r_dp", "r_pp", "memory_capacity_bytes", "max_precompute_failures"), "planner")
    for name in ("r_dp", "r_pp"):
        if not isinstance(planner[name], list) or not planner[name]:
            raise ValueError(f"{name} must be a nonempty list")
        for value in planner[name]:
            _integer(name, value)
    _integer("memory_capacity_bytes", planner["memory_capacity_bytes"], 0)
    _integer("max_precompute_failures", planner["max_precompute_failures"])

    profiling = payload["profile"]
    _fields(profiling, ("steps", "ema_alpha", "calibration_warmup", "calibration_iterations",
                        "calibration_bootstrap_rounds"), "profile")
    _integer("profile steps", profiling["steps"])
    _number("ema_alpha", profiling["ema_alpha"], positive=True)
    if profiling["ema_alpha"] > 1:
        raise ValueError("ema_alpha must not exceed 1")
    _integer("calibration_warmup", profiling["calibration_warmup"], 0)
    _integer("calibration_iterations", profiling["calibration_iterations"])
    _integer("calibration_bootstrap_rounds", profiling["calibration_bootstrap_rounds"])
    return dict(payload, model=model)


def _dtype(name):
    import torch

    return getattr(torch, name)


def _workers(config):
    return tuple(WorkerIdentity(f"cli-worker-{rank}", rank, 0)
                 for rank in range(config["world_size"]))


def _topology(config):
    model = config["model"]
    state = ClusterState(_workers(config), model.global_batch_size)
    return SymmetricTopology(state, model, config["parallel"]["stage_modules"])


def _runtime(config):
    training = config["training"]
    return SymmetricRuntime(
        _topology(config),
        device=config["device"],
        dtype=config["dtype"],
        timeout_s=training["timeout_s"],
        artifact_dir=config["artifact_dir"],
        lr=training["lr"],
        weight_decay=training["weight_decay"],
    )


def _jsonable(value):
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path, payload):
    destination = _relative_path(path, "output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False), encoding="utf-8")


def _tolerances(config):
    if config["dtype"] == "float32":
        return {"rtol": 1e-4, "atol": 1e-6}
    if config["device"] == "cuda":
        return {"rtol": 1e-7, "atol": 1e-9}
    return {"rtol": 1e-8, "atol": 1e-10}


def _base_report(command, config):
    return {
        "schema_version": 1,
        "command": command,
        "status": "passed",
        "environment": environment_report(),
        "device": config["device"],
        "backend": config["backend"],
        "world_size": config["world_size"],
        "dtype": config["dtype"],
        "seed": config["model"].seed,
        "B": config["model"].global_batch_size,
        "tolerances": _tolerances(config),
    }


def _failure_report(command, config, error):
    report = (_base_report(command, config) if config is not None else {
        "schema_version": 1,
        "command": command,
    })
    report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
    audit = getattr(error, "audit", None)
    if audit is not None:
        report["resource_audit"] = audit
    return report


def profile_command(config, output):
    import torch

    torch.set_num_threads(1)
    model_config = config["model"]
    model = build_initial_model(model_config, device=config["device"], dtype=_dtype(config["dtype"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"],
                                  weight_decay=config["training"]["weight_decay"], amsgrad=False)
    state = ClusterState((WorkerIdentity("cli-profiler", 0, 0),), model_config.global_batch_size)
    sample_ids = list(range(model_config.global_batch_size))
    state, _ = train_profile_step(model, optimizer, state)
    profiler = Profiler(model, optimizer, ema_alpha=config["profile"]["ema_alpha"])
    for _ in range(config["profile"]["steps"]):
        start = state.committed_global_step * model_config.global_batch_size
        sample_ids.extend(range(start, start + model_config.global_batch_size))
        state, _ = train_profile_step(model, optimizer, state, profiler=profiler)
    sizes = {parameter.numel() * parameter.element_size() for parameter in model.parameters()
             if parameter.requires_grad}
    for parameter_state in optimizer.state.values():
        sizes.update(tensor.numel() * tensor.element_size() for tensor in parameter_state.values())
    calibration = run_transfer_calibration(
        config["device"],
        2,
        tensor_bytes=tuple(sorted(sizes)),
        warmup=config["profile"]["calibration_warmup"],
        iterations=config["profile"]["calibration_iterations"],
        bootstrap_rounds=config["profile"]["calibration_bootstrap_rounds"],
        timeout_s=config["training"]["timeout_s"],
        artifact_dir=config["artifact_dir"],
    )
    profiler.add_calibration(calibration)
    snapshot = profiler.snapshot()
    output_path = _relative_path(output, "profile output")
    export_profile(snapshot, str(output_path), expected_identity=profiler.identity)
    report = _base_report("profile", config)
    report.update(profile_path=output, profile_hash=_hash(snapshot), profile_schema_version=snapshot["schema_version"],
                  committed_global_step=state.committed_global_step,
                  sample_ids=sample_ids, global_sample_count=len(sample_ids),
                  profile_steps=[row["step_id"] for row in snapshot["steps"]],
                  calibration_audit=calibration["audit"])
    return report


def train_command(config, steps):
    _integer("steps", steps)
    runtime = _runtime(config)
    inspected = []
    with runtime:
        results = [runtime.train_step() for _ in range(steps)]
        inspected = runtime.inspect_state()
    report = _base_report("train", config)
    sample_ids = [sample for result in results for sample in result["sample_ids"]]
    report.update(committed_global_step=runtime.state.committed_global_step,
                  sample_ids=sample_ids, global_sample_count=len(sample_ids),
                  steps=[{key: value for key, value in result.items() if key != "snapshots"}
                         for result in results],
                  state_hashes=[{"worker": row["worker"], "hashes": row["hashes"]} for row in inspected],
                  runtime_report=str(runtime.report_path), resource_audit=runtime.audit)
    return report


def _load_matching_profile(config, path):
    import torch

    model = build_initial_model(config["model"], device=config["device"], dtype=_dtype(config["dtype"]))
    expected_identity = profile_identity(model)
    profile = load_profile(str(_relative_path(path, "profile")), expected_identity=expected_identity)
    del model
    if config["device"] == "cuda":
        torch.cuda.empty_cache()
    return profile, expected_identity


def recover_demo_command(config, profile_path, duration):
    from .decision_center import DecisionCenter, select_policy

    _finite("inter_fault_duration_s", duration, positive=True)
    profile, expected_identity = _load_matching_profile(config, profile_path)
    cache = PlanCache()
    planner = config["planner"]
    center = DecisionCenter(
        config["model"],
        expected_identity=expected_identity,
        r_dp=planner["r_dp"],
        r_pp=planner["r_pp"],
        memory_capacity_bytes=planner["memory_capacity_bytes"],
        plan_cache=cache,
    )
    runtime = _runtime(config)
    failure = decision = recovery = None
    before, results = [], []
    killed = {}
    with runtime:
        results.extend(runtime.train_step() for _ in range(config["training"]["steps"]))
        rank = config["parallel"]["failure_rank"]
        worker, process = runtime.topology.ranks[rank], runtime.processes[rank]
        failure = FailureEvent((worker.worker_id,), runtime.state.generation,
                               runtime.state.committed_global_step)
        preview, before = runtime.preview_recovery_state(failure)
        precomputed = center.precompute_dynamic(
            (preview,), profile,
            max_failures=planner["max_precompute_failures"],
        )
        process.kill()
        process.join(10)
        if process.is_alive() or process.exitcode in (None, 0):
            raise RuntimeError("recover-demo failed to confirm the requested real worker kill")
        killed = {"worker_id": worker.worker_id, "pid": process.pid, "exitcode": process.exitcode}
        recovery_state = runtime.recovery_state(failure)
        if recovery_state.recovery_id != preview.recovery_id:
            raise RuntimeError("live recovery state differs from the precomputed safe-point scenario")
        candidates = center.evaluate_candidates(recovery_state, profile)
        decision = select_policy(candidates, duration)
        recovery = runtime.recover(failure, decision)
        results.extend(runtime.train_step() for _ in range(config["training"]["resume_steps"]))
    report = _base_report("recover-demo", config)
    rows = decision.derivation["candidates"]
    sample_ids = [sample for result in results for sample in result["sample_ids"]]
    report.update(
        D=duration,
        duration_source="caller supplied inter-fault duration; not predicted",
        committed_global_step=runtime.state.committed_global_step,
        sample_ids=sample_ids,
        global_sample_count=len(sample_ids),
        candidates=[{
            "plan_id": row["plan_id"],
            "policy": row["policy"],
            "estimated_step_time_s": row["estimated_step_time_s"],
            "estimated_transition_time_s": row["estimated_transition_time_s"],
            "score": row["score"],
            "selected": row["selected"],
            "rejection_reasons": row["rejection_reasons"],
        } for row in rows],
        selected_policy=decision.plan.policy,
        selected_plan_id=decision.plan.plan_id,
        selected_score=decision.score,
        profile_hash=_hash(profile),
        precomputed=precomputed,
        precompute_completed_before_kill=True,
        cache=cache.stats,
        failure=asdict(failure),
        kill=killed,
        source_map=recovery["state_sources"],
        state_hashes=recovery["source_hashes"],
        actual_topology=recovery["actual_topology"],
        actual_times_s={key: value for key, value in recovery.items()
                        if key.startswith("actual_") and key.endswith("_s")},
        safe_point_before_kill=[{"worker": row["worker"], "hashes": row["hashes"]} for row in before],
        runtime_report=str(runtime.report_path),
        resource_audit=runtime.audit,
    )
    return report


def build_parser():
    parser = argparse.ArgumentParser(prog="chameleon")
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser("profile", help="measure a local model profile and P2P calibration")
    profile.add_argument("--config", required=True)
    profile.add_argument("--output", required=True)
    profile.add_argument("--report", default="artifacts/test-results/cli-profile.json")
    train = commands.add_parser("train", help="run the configured real DP/PP workers")
    train.add_argument("--config", required=True)
    train.add_argument("--steps", type=int)
    train.add_argument("--output", default="artifacts/test-results/cli-train.json")
    recover = commands.add_parser("recover-demo", help="kill one worker and select/recover at a safe point")
    recover.add_argument("--config", required=True)
    recover.add_argument("--profile", required=True)
    recover.add_argument("--inter-fault-duration-s", type=float, required=True,
                         help="caller-supplied D; Chameleon does not predict MTBF")
    recover.add_argument("--output", default="artifacts/test-results/cli-recover-demo.json")
    return parser


def main(argv=None):
    arguments = build_parser().parse_args(argv)
    output = arguments.report if arguments.command == "profile" else arguments.output
    config = None
    try:
        config = load_config(arguments.config)
        if arguments.command == "profile":
            report = profile_command(config, arguments.output)
        elif arguments.command == "train":
            steps = config["training"]["steps"] if arguments.steps is None else arguments.steps
            report = train_command(config, steps)
        else:
            report = recover_demo_command(config, arguments.profile, arguments.inter_fault_duration_s)
    except Exception as error:
        report = _failure_report(arguments.command, config, error)
        _write_json(output, report)
        print(json.dumps({"status": "failed", "command": arguments.command, "report": output,
                          "error": report["error"]}, allow_nan=False))
        return 1
    _write_json(output, report)
    print(json.dumps({"status": "passed", "command": arguments.command, "report": output}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
