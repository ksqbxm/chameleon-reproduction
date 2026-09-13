from dataclasses import asdict
import json
import math
from pathlib import Path
import tempfile

import pytest

from chameleon import ClusterState, ModelConfig, WorkerIdentity
from chameleon.estimators import Estimator, estimate_operation_time, estimate_symmetric_time
from chameleon.profiler import Profiler, export_profile, load_profile, train_profile_step


@pytest.fixture
def live_profile(device):
    import torch
    from chameleon.model import build_initial_model

    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=5, micro_batch_size=2)
    model = build_initial_model(config, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.007, amsgrad=False)
    state = ClusterState((WorkerIdentity("estimator-profile", 0, 0),), config.global_batch_size)
    state, _ = train_profile_step(model, optimizer, state)
    profiler = Profiler(model, optimizer)
    for _ in range(2):
        state, _ = train_profile_step(model, optimizer, state, profiler=profiler)
    payload = profiler.snapshot()
    root = Path("artifacts/test-results")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="profile-estimator-", dir=root) as directory:
        path = Path(directory) / "profile.json"
        export_profile(payload, str(path), expected_identity=profiler.identity)
        loaded = load_profile(str(path), expected_identity=profiler.identity)
    assert loaded == payload
    return loaded, profiler.identity


def test_actual_profile_feeds_asymmetric_time_with_all_endpoints(live_profile, device):
    profile, identity = live_profile
    estimator = Estimator(profile, expected_identity=identity, layer_modules=("blocks.0", "blocks.1"))
    layouts = ((("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head")),
               (("embedding", "blocks.0", "blocks.1", "final_norm", "lm_head"),))
    result = estimator.dynamic_time(layouts, (2, 1), global_micro_batches=3)
    metrics = profile["metrics"]
    for pipeline, layout in enumerate(layouts):
        expected_forward = tuple(sum(metrics[f"modules.{name}.forward_s"]["ema"] for name in stage)
                                 for stage in layout)
        expected_backward = tuple(sum(metrics[f"modules.{name}.backward_s"]["ema"] for name in stage)
                                  for stage in layout)
        derivation = result.derivation["pipelines"][pipeline]
        assert derivation["stage_forward_s"] == expected_forward
        assert derivation["stage_backward_s"] == expected_backward
    # One-stage pipeline has no bubbles: independently sum both passes of its single micro-batch.
    one_stage = result.derivation["pipelines"][1]
    assert result.derivation["pipeline_times_s"][1] == pytest.approx(
        one_stage["stage_forward_s"][0] + one_stage["stage_backward_s"][0])
    assert result.step_time_s == max(result.derivation["pipeline_times_s"])
    assert result.feasible and len(result.trace) == 10
    Path(f"artifacts/test-results/task05-{device}-profile-estimator.json").write_text(
        json.dumps({"profile": profile, "estimate": asdict(result)}, indent=2, allow_nan=False), encoding="utf-8")


def test_actual_trace_durations_feed_equation11_without_performance_claim(live_profile):
    profile, identity = live_profile
    trace = profile["steps"][-1]["trace"]
    nm = 3  # Real batches contain [2, 2, 1] samples, not three equally sized mock batches.
    durations = {(row["pipeline"], row["stage"], row["micro_batch"], row["kind"]): row["end_s"] - row["start_s"]
                 for row in trace}
    result = estimate_operation_time(durations, num_stages=1, pipeline_micro_batches=nm)
    assert result.step_time_s == pytest.approx(sum(row["end_s"] - row["start_s"] for row in trace))
    assert result.step_time_s <= profile["metrics"]["step_time_s"]["samples"][-1] + 1e-12
    estimator = Estimator(profile, expected_identity=identity, layer_modules=("blocks.0", "blocks.1"))
    forward, backward = estimator.stage_durations((tuple(identity["module_order"]),))
    symmetric = estimate_symmetric_time(num_stages=1, global_micro_batches=nm, dp_size=1,
                                        forward_s=forward[0], backward_s=backward[0])
    dynamic = estimator.dynamic_time(((tuple(identity["module_order"]),),),
                                     (nm,), global_micro_batches=nm)
    assert symmetric.step_time_s == pytest.approx(dynamic.step_time_s)


def test_actual_tensor_inventory_memory_boundary_and_endpoint_oom(live_profile):
    profile, identity = live_profile
    estimator = Estimator(profile, expected_identity=identity, layer_modules=("blocks.0", "blocks.1"))
    layout = (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head"))
    result = estimator.memory(layout, (10**12, 10**12))
    memory = profile["steps"][-1]["memory"]
    layers = ("blocks.0", "blocks.1")
    static_layer = sum(2 * memory["modules"][name]["parameter_bytes"] + memory["modules"][name]["adamw_bytes"]
                       for name in layers) / 2
    activation_layer = sum(memory["saved_activation_bytes"][name] for name in layers) / 2
    for stage, endpoints in enumerate((("embedding",), ("final_norm", "lm_head"))):
        extra_static = sum(2 * memory["modules"][name]["parameter_bytes"] + memory["modules"][name]["adamw_bytes"]
                           for name in endpoints)
        extra_activation = sum(memory["saved_activation_bytes"][name] for name in endpoints)
        if stage == 1:
            extra_activation += memory["saved_activation_bytes"]["loss_and_runtime"]
        expected = static_layer + extra_static + (2 - stage) * (activation_layer + extra_activation)
        assert result.stages[stage]["peak_bytes"] == pytest.approx(expected)
        assert extra_static > 0
    assert result.derivation["source_kind"] == ("logical_tensor_bytes" if identity["device"]["type"] == "cpu" else "cuda_hbm")
    capacities = tuple(math.ceil(row["peak_bytes"]) for row in result.stages)
    assert estimator.memory(layout, capacities).feasible
    oom = estimator.memory(layout, (capacities[0] - 1, capacities[1]))
    assert not oom.feasible and "stage 0" in oom.reasons[0]
