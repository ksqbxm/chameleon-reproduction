from copy import deepcopy
import json
from pathlib import Path
import tempfile

import pytest

from chameleon.profiler import export_profile, load_profile, validate_snapshot


@pytest.fixture
def profile_directory():
    root = Path("artifacts/test-results")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="profile-roundtrip-", dir=root) as directory:
        yield Path(directory)


@pytest.fixture
def schema_profile():
    # Handwritten schema fixture, never substituted for live CPU/CUDA measurements.
    return {
        "schema_version": 3,
        "identity": {"model_hash": "a" * 64, "config_hash": "b" * 64,
                     "module_parameter_bytes": {"embedding": 80},
                     "module_order": ["embedding"],
                     "device": {"type": "cpu", "index": None, "torch": "schema-fixture",
                                "name": "schema-fixture", "system": "schema-fixture"},
                     "parallel": {"dp_size": 1, "pp_size": 1, "rank": 0}},
        "ema_alpha": 0.25,
        "metrics": {"step_time_s": {"samples": [2.], "ema": 2.},
                    "step_wall_time_s": {"samples": [3.], "ema": 3.},
                    **{f"modules.embedding.{field}": {"samples": [value], "ema": value}
                       for field, value in (("forward_s", 0.8), ("backward_s", 1.),
                                            ("parameter_bytes", 80), ("gradient_bytes", 80),
                                            ("adamw_bytes", 164), ("output_activation_bytes", 32),
                                            ("saved_activation_bytes", 32))},
                    "modules.loss_and_runtime.saved_activation_bytes": {"samples": [16], "ema": 16}},
        "steps": [{"step_id": 2, "trace": [
            {"kind": "forward", "micro_batch": 0, "pipeline": 0, "stage": 0,
             "phase": "steady", "start_s": 0., "end_s": 0.8},
            {"kind": "backward", "micro_batch": 0, "pipeline": 0, "stage": 0,
             "phase": "steady", "start_s": 0.8, "end_s": 1.8},
        ], "memory": {"kind": "logical_tensor_bytes",
                      "modules": {"embedding": {"parameter_bytes": 80, "gradient_bytes": 80,
                                                 "adamw_bytes": 164}},
                      "output_activation_bytes": {"embedding": 32},
                      "saved_activation_bytes": {"embedding": 32, "loss_and_runtime": 16},
                      "peak_allocated_bytes": None, "peak_reserved_bytes": None}}],
        "calibrations": [],
    }


def test_schema_json_roundtrip_is_explicit_and_lossless(schema_profile, profile_directory):
    path = profile_directory / "profile.json"
    assert not path.exists()
    export_profile(schema_profile, str(path), expected_identity=schema_profile["identity"])
    assert load_profile(str(path), expected_identity=schema_profile["identity"]) == schema_profile
    assert json.loads(path.read_text(encoding="utf-8")) == schema_profile


@pytest.mark.parametrize("ids", [(1,), (7,), (1, 0), (0, 2)])
def test_trace_rejects_gapped_or_reordered_micro_batch_ids(schema_profile, ids):
    count = len(ids)
    trace = schema_profile["steps"][0]["trace"]
    schema_profile["steps"][0]["trace"] = [dict(row, micro_batch=mb, start_s=row["start_s"] + 2 * i,
                                               end_s=row["end_s"] + 2 * i)
                                           for i, mb in enumerate(ids) for row in trace]
    for metric in ("step_time_s", "step_wall_time_s"):
        schema_profile["metrics"][metric] = dict(samples=[3. * count], ema=3. * count)
    for metric, series in schema_profile["metrics"].items():
        if metric.endswith(("forward_s", "backward_s", "output_activation_bytes", "saved_activation_bytes")):
            series["samples"] *= count
    with pytest.raises(ValueError, match="1F1B"):
        validate_snapshot(schema_profile, schema_profile["identity"])


@pytest.mark.parametrize("lengths", [[2, 2, 3], [2, 3, 3]])
def test_asymmetric_parallel_schema_roundtrip_and_stale_depth_rejection(schema_profile, profile_directory, lengths):
    # Schema-only fixture for Task10; real profiles run in test_planner_runtime.
    schema_profile["identity"]["parallel"] = dict(dp_size=3, pp_size=3, rank=1, pipeline_lengths=lengths)
    for row in schema_profile["steps"][0]["trace"]:
        row["stage"] = 1  # Last stage of pipeline0 (PP2), F0/B0 are both steady.
    path = profile_directory / "asymmetric.json"
    export_profile(schema_profile, str(path), expected_identity=schema_profile["identity"])
    assert load_profile(str(path), expected_identity=schema_profile["identity"]) == schema_profile
    expected = deepcopy(schema_profile["identity"])
    expected["parallel"]["pipeline_lengths"][0] = 3
    with pytest.raises(ValueError, match="identity mismatch"):
        load_profile(str(path), expected_identity=expected)


@pytest.mark.parametrize("field", ["model_hash", "config_hash", "device", "parallel"])
def test_reject_stale_identity(schema_profile, profile_directory, field):
    path = profile_directory / "profile.json"
    export_profile(schema_profile, str(path), expected_identity=schema_profile["identity"])
    expected = deepcopy(schema_profile["identity"])
    if field.endswith("hash"):
        expected[field] = "c" * 64
    elif field == "device":
        expected[field]["name"] = "different-device"
    else:
        expected[field]["dp_size"] = 2
    with pytest.raises(ValueError, match="identity mismatch"):
        load_profile(str(path), expected_identity=expected)


@pytest.mark.parametrize("field", ["schema_version", "identity", "ema_alpha", "metrics", "steps", "calibrations"])
def test_reject_missing_top_level_fields(schema_profile, field):
    payload = deepcopy(schema_profile)
    del payload[field]
    with pytest.raises(ValueError, match="fields"):
        validate_snapshot(payload, schema_profile["identity"])


@pytest.mark.parametrize("path,value", [
    (("schema_version",), 1), (("schema_version",), 2), (("schema_version",), 4), (("schema_version",), True),
    (("metrics", "step_time_s", "samples"), []),
    (("metrics", "step_time_s", "samples"), [float("nan")]),
    (("metrics", "step_time_s", "ema"), 9.),
    (("steps", 0, "memory", "modules", "embedding", "parameter_bytes"), -1),
    (("steps", 0, "memory", "modules", "embedding", "adamw_bytes"), 1.5),
    (("steps", 0, "memory", "peak_allocated_bytes"), 1024),
    (("steps", 0, "memory", "kind"), "cuda_hbm"),
    (("steps", 0, "memory", "saved_activation_bytes"), {}),
    (("steps", 0, "trace", 0, "kind"), "backward"),
    (("steps", 0, "trace", 1, "kind"), "forward"),
    (("steps", 0, "trace", 0, "phase"), "prediction"),
    (("steps", 0, "trace", 1, "start_s"), 0.2),
    (("steps", 0, "trace", 1, "end_s"), float("inf")),
    (("steps", 0, "trace", 0, "stage"), 1),
    (("steps", 0, "trace"), []),
    (("metrics",), {}), (("steps",), {}), (("calibrations",), {}),
])
def test_reject_invalid_measurement_fields(schema_profile, path, value):
    payload = deepcopy(schema_profile)
    parent = payload
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    with pytest.raises(ValueError):
        validate_snapshot(payload, schema_profile["identity"])


def test_reject_unexpected_fields_and_malformed_identity(schema_profile):
    payload = deepcopy(schema_profile)
    payload["checkpoint"] = {}
    with pytest.raises(ValueError, match="fields"):
        validate_snapshot(payload, schema_profile["identity"])
    payload = deepcopy(schema_profile)
    payload["identity"]["model_hash"] = "invalid"
    with pytest.raises(ValueError, match="hash"):
        validate_snapshot(payload, payload["identity"])


def test_reject_missing_backward_and_duplicate_step(schema_profile):
    payload = deepcopy(schema_profile)
    payload["steps"][0]["trace"].pop()
    with pytest.raises(ValueError, match="missing backward"):
        validate_snapshot(payload, schema_profile["identity"])
    payload = deepcopy(schema_profile)
    payload["steps"].append(deepcopy(payload["steps"][0]))
    for series in payload["metrics"].values():
        series["samples"] *= 2
    with pytest.raises(ValueError, match="IDs must increase"):
        validate_snapshot(payload, schema_profile["identity"])


def test_reject_all_forward_all_backward_trace(schema_profile):
    payload = deepcopy(schema_profile)
    original = payload["steps"][0]["trace"]
    payload["steps"][0]["trace"] = [
        dict(original[0 if kind == "forward" else 1], kind=kind, micro_batch=mb,
             start_s=index * 0.25, end_s=(index + 1) * 0.25)
        for index, (kind, mb) in enumerate((
            ("forward", 0), ("forward", 1), ("backward", 0), ("backward", 1)))
    ]
    with pytest.raises(ValueError, match="1F1B"):
        validate_snapshot(payload, schema_profile["identity"])


def test_reject_trace_outside_step_time(schema_profile):
    payload = deepcopy(schema_profile)
    payload["steps"][0]["trace"][-1]["end_s"] = 3.
    with pytest.raises(ValueError, match="step time"):
        validate_snapshot(payload, schema_profile["identity"])


def test_parameter_bytes_must_match_model_identity(schema_profile):
    payload = deepcopy(schema_profile)
    payload["steps"][0]["memory"]["modules"]["embedding"]["parameter_bytes"] += 8
    with pytest.raises(ValueError, match="parameter bytes"):
        validate_snapshot(payload, schema_profile["identity"])


@pytest.mark.parametrize("field", ["forward_s", "backward_s", "parameter_bytes",
                                   "output_activation_bytes", "saved_activation_bytes"])
def test_reject_missing_module_samples(schema_profile, field):
    payload = deepcopy(schema_profile)
    del payload["metrics"][f"modules.embedding.{field}"]
    with pytest.raises(ValueError, match="module measurement"):
        validate_snapshot(payload, schema_profile["identity"])


def test_reject_incomplete_module_inventory_from_model_manifest(schema_profile):
    payload = deepcopy(schema_profile)
    payload["identity"]["module_parameter_bytes"]["lm_head"] = 20
    payload["identity"]["module_order"].append("lm_head")
    with pytest.raises(ValueError, match="module inventory"):
        validate_snapshot(payload, payload["identity"])


def test_reject_raw_memory_samples_that_disagree_with_inventory(schema_profile):
    payload = deepcopy(schema_profile)
    payload["metrics"]["modules.embedding.gradient_bytes"] = {"samples": [88], "ema": 88}
    with pytest.raises(ValueError, match="memory samples"):
        validate_snapshot(payload, schema_profile["identity"])


def test_reject_module_timing_that_exceeds_actual_operation(schema_profile):
    payload = deepcopy(schema_profile)
    payload["metrics"]["modules.embedding.backward_s"] = {"samples": [2.], "ema": 2.}
    with pytest.raises(ValueError, match="module timing"):
        validate_snapshot(payload, schema_profile["identity"])


def test_cuda_peak_samples_are_required_and_match_memory(schema_profile):
    payload = deepcopy(schema_profile)
    payload["identity"]["device"] = {"type": "cuda", "index": 0, "torch": "schema-fixture",
                                     "name": "schema-fixture", "total_memory_bytes": 4096,
                                     "capability": [8, 0], "cuda": "12.9"}
    payload["steps"][0]["memory"].update(kind="cuda_hbm", peak_allocated_bytes=512,
                                       peak_reserved_bytes=1024)
    with pytest.raises(ValueError, match="peak measurement"):
        validate_snapshot(payload, payload["identity"])
    for field, value in (("peak_allocated_bytes", 512), ("peak_reserved_bytes", 1024)):
        payload["metrics"][field] = {"samples": [value], "ema": value}
    validate_snapshot(payload, payload["identity"])
    payload["metrics"]["peak_allocated_bytes"] = {"samples": [768], "ema": 768}
    with pytest.raises(ValueError, match="peak samples"):
        validate_snapshot(payload, payload["identity"])


@pytest.mark.parametrize("field", ["peak_allocated_bytes", "peak_reserved_bytes"])
def test_cpu_profile_cannot_contain_measured_hbm_metrics(schema_profile, field):
    payload = deepcopy(schema_profile)
    payload["metrics"][field] = {"samples": [512], "ema": 512}
    with pytest.raises(ValueError, match="CPU profiles"):
        validate_snapshot(payload, schema_profile["identity"])


@pytest.mark.parametrize("pp_size,operations", [
    (2, (("forward", 0, "warmup"), ("forward", 1, "steady"),
         ("backward", 0, "steady"), ("backward", 1, "cooldown"))),
    (4, (("forward", 0, "warmup"), ("backward", 0, "cooldown"))),
])
def test_1f1b_phase_order_matches_handwritten_stage_zero_oracle(schema_profile, pp_size, operations):
    payload = deepcopy(schema_profile)
    payload["identity"]["parallel"]["pp_size"] = pp_size
    time_s, trace = 0., []
    for kind, mb, phase in operations:
        duration = 0.8 if kind == "forward" else 1.
        trace.append({"kind": kind, "micro_batch": mb, "pipeline": 0, "stage": 0,
                      "phase": phase, "start_s": time_s, "end_s": time_s + duration})
        time_s += duration
    payload["steps"][0]["trace"] = trace
    payload["metrics"]["step_time_s"] = {"samples": [time_s + 0.2], "ema": time_s + 0.2}
    payload["metrics"]["step_wall_time_s"] = {"samples": [time_s + 0.3], "ema": time_s + 0.3}
    for field in ("forward_s", "backward_s", "output_activation_bytes", "saved_activation_bytes"):
        payload["metrics"][f"modules.embedding.{field}"]["samples"] *= len(operations) // 2
    payload["metrics"]["modules.loss_and_runtime.saved_activation_bytes"]["samples"] *= len(operations) // 2
    validate_snapshot(payload, payload["identity"])
    payload["steps"][0]["trace"][-1]["phase"] = "steady"
    with pytest.raises(ValueError, match="1F1B"):
        validate_snapshot(payload, payload["identity"])


@pytest.fixture
def torch_module():
    import torch
    return torch


def test_live_profile_roundtrip_contains_measurements_only(torch_module, device, profile_directory):
    from chameleon import ClusterState, ModelConfig, WorkerIdentity
    from chameleon.model import build_initial_model
    from chameleon.profiler import Profiler, profile_identity, train_profile_step
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=1, num_heads=1,
                         sequence_length=3, global_batch_size=3, micro_batch_size=2)
    model = build_initial_model(config, device=device)
    optimizer = torch_module.optim.AdamW(model.parameters(), amsgrad=False)
    state = ClusterState((WorkerIdentity("profile-worker", 0, 0),), 3)
    state, _ = train_profile_step(model, optimizer, state)
    profiler = Profiler(model, optimizer)
    original_identity = deepcopy(profiler.identity)
    for _ in range(2):
        state, _ = train_profile_step(model, optimizer, state, profiler=profiler)
    assert list(profile_directory.iterdir()) == []  # Normal profiling performs no disk I/O.
    assert profiler.identity == profile_identity(model) == original_identity  # Weight updates aren't staleness.
    payload = profiler.snapshot()
    path = profile_directory / "live-profile.json"
    export_profile(payload, str(path), expected_identity=original_identity)
    assert load_profile(str(path), expected_identity=original_identity) == payload
    serialized = path.read_text(encoding="utf-8")
    assert all(key not in serialized for key in ("exp_avg", "exp_avg_sq", "MTBF", "inter_fault_duration"))
    payload["metrics"]["step_time_s"]["samples"].append(999.)
    assert len(profiler.snapshot()["metrics"]["step_time_s"]["samples"]) == 2
    model.extra = torch_module.nn.Linear(4, 4, device=device, dtype=torch_module.float64)
    with pytest.raises(ValueError, match="identity mismatch"):
        load_profile(str(path), expected_identity=profile_identity(model))
