from dataclasses import asdict
import json
from pathlib import Path

import pytest

from chameleon.estimators import estimate_pipeline_time
from chameleon.profiler import export_profile, load_profile


def test_live_local_stage_profiles_roundtrip_with_all_endpoints(symmetric_training):
    steps = symmetric_training["steps"]
    assert steps[0]["profile_comparison"] is None
    for report, payload in zip(steps[2]["reports"], symmetric_training["profiles"]):
        identity = payload["identity"]
        assert identity["parallel"] == {"dp_size": 2, "pp_size": 2, "rank": report["worker"]["rank"]}
        modules = (("embedding", "blocks.0") if report["stage"] == 0
                   else ("blocks.1", "final_norm", "lm_head"))
        assert identity["module_order"] == list(modules)
        assert [step["step_id"] for step in payload["steps"]] == [2, 3]
        path = Path("artifacts/test-results") / f"task09-{symmetric_training['device']}-rank{report['worker']['rank']}-profile.json"
        export_profile(payload, str(path), expected_identity=identity)
        assert load_profile(str(path), expected_identity=identity) == payload
        for name in modules:
            for kind in ("forward_s", "backward_s"):
                assert len(payload["metrics"][f"modules.{name}.{kind}"]["samples"]) == 6
        for step in payload["steps"]:
            memory = step["memory"]
            assert set(memory["modules"]) == set(modules)
            assert all(row["parameter_bytes"] > 0 and row["adamw_bytes"] > 0 and row["gradient_bytes"] > 0
                       for row in memory["modules"].values())
            assert memory["kind"] == ("cuda_hbm" if symmetric_training["device"] == "cuda" else "logical_tensor_bytes")
            if symmetric_training["device"] == "cuda":
                assert memory["peak_reserved_bytes"] >= memory["peak_allocated_bytes"] > 0
            else:
                assert memory["peak_allocated_bytes"] is memory["peak_reserved_bytes"] is None


def test_profiler_trace_matches_actual_runtime_operations(symmetric_training):
    for step in symmetric_training["steps"][1:]:
        for report in step["reports"]:
            measured = report["profile_step"]["trace"]
            full_profile = symmetric_training["profiles"][report["worker"]["rank"]]
            assert report["profile_step"] == full_profile["steps"][step["step_id"] - 2]
            fields = ("kind", "micro_batch", "pipeline", "stage", "phase")
            assert [tuple(row[field] for field in fields) for row in measured] == [tuple(row[field] for field in fields) for row in report["trace"]]
            assert len(measured) == 6
            assert all(row["end_s"] > row["start_s"] for row in measured)


def test_estimator_reports_compute_and_measured_step_from_live_trace(symmetric_training):
    for step in symmetric_training["steps"][1:]:
        comparison = step["profile_comparison"]
        for pipeline, estimate in enumerate(comparison["equation11_pipelines"]):
            rows = {(row["stage"], row["micro_batch"], row["kind"]): row
                    for report in step["reports"] if report["pipeline"] == pipeline
                    for row in report["profile_step"]["trace"]}
            # Independent hand recurrence for PP2/Nm3 (including warmup and cooldown).
            f0, f1, b0, b1 = ([rows[stage, mb, kind]["end_s"] - rows[stage, mb, kind]["start_s"] for mb in range(3)]
                              for stage, kind in ((0, "forward"), (1, "forward"), (0, "backward"), (1, "backward")))
            first_f0 = f0[0]
            first_f1 = first_f0 + f1[0]
            first_b1 = first_f1 + b1[0]
            second_f0 = first_f0 + f0[1]
            first_b0 = max(second_f0, first_b1) + b0[0]
            second_f1 = max(second_f0, first_b1) + f1[1]
            second_b1 = second_f1 + b1[1]
            third_f0 = first_b0 + f0[2]
            second_b0 = max(third_f0, second_b1) + b0[1]
            third_f1 = max(third_f0, second_b1) + f1[2]
            third_b1 = third_f1 + b1[2]
            third_b0 = max(second_b0, third_b1) + b0[2]
            assert estimate["step_time_s"] == pytest.approx(max(third_b0, third_b1))
            assert estimate["derivation"]["pipeline_micro_batches"] == 3
        assert comparison["estimated_compute_time_s"] == max(item["step_time_s"] for item in comparison["equation11_pipelines"])
        assert comparison["equation9"]["derivation"]["global_micro_batches"] == 6
        assert comparison["equation9"]["derivation"]["pipeline_micro_batches"] == (3, 3)
        assert comparison["measured_training_time_s"] >= comparison["measured_pipeline_time_s"] > 0
        assert comparison["estimated_compute_time_s"] > 0
        assert "P2P" in comparison["boundary"]
    path = Path("artifacts/test-results") / f"task09-{symmetric_training['device']}-runtime-estimator.json"
    path.write_text(json.dumps([step["profile_comparison"] for step in symmetric_training["steps"][1:]], indent=2), encoding="utf-8")


def test_local_module_measurements_feed_estimator_including_endpoints(symmetric_training):
    step = symmetric_training["steps"][-1]
    for pipeline in range(2):
        reports = [report for report in step["reports"] if report["pipeline"] == pipeline]
        profiles = [symmetric_training["profiles"][report["worker"]["rank"]] for report in reports]
        forward, backward = (tuple(sum(profile["metrics"][f"modules.{name}.{kind}_s"]["ema"]
                                        for name in profile["identity"]["module_order"])
                                  for profile in profiles) for kind in ("forward", "backward"))
        estimate = estimate_pipeline_time(forward, backward, 3, pipeline=pipeline)
        assert estimate.step_time_s > 0 and len(estimate.trace) == 12
        assert estimate.derivation["stage_forward_s"] == forward
        assert estimate.derivation["stage_backward_s"] == backward
        path = Path("artifacts/test-results") / f"task09-{symmetric_training['device']}-pipeline{pipeline}-module-estimator.json"
        path.write_text(json.dumps(asdict(estimate), indent=2), encoding="utf-8")


def test_default_runtime_retains_metadata_without_training_tensor_backup(distributed_environment, device, world_size):
    import torch
    from chameleon import ClusterState, ModelConfig, WorkerIdentity
    from chameleon.runtime import DistributedRuntime, SymmetricTopology

    assert world_size == 4, "Task09 acceptance requires --world-size 4"
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=4, micro_batch_size=1)
    topology = SymmetricTopology(ClusterState(tuple(WorkerIdentity(f"metadata-{rank}", rank, 0) for rank in range(4)), 4),
                                 config, (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head")))
    runtime = DistributedRuntime(topology, device=device)
    with runtime:
        first, second = runtime.train_step(), runtime.train_step()
        assert "snapshots" not in first and "snapshots" not in second
        assert not list(runtime.directory.glob("*.pt"))
        assert not hasattr(runtime, "model") and not hasattr(runtime, "optimizer")
        assert second["profile_comparison"] is not None
        assert all("profile" not in report for step in runtime.steps for report in step["reports"])
        assert all(report["profile_step"]["step_id"] == 2 for report in second["reports"])
        profiles = runtime.snapshot_profiles()
        assert all([step["step_id"] for step in profile["steps"]] == [2] for profile in profiles)

    def check_metadata(value):
        assert not isinstance(value, torch.Tensor)
        if isinstance(value, dict):
            for child in value.values():
                check_metadata(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check_metadata(child)

    check_metadata(runtime.steps)
    assert runtime.audit["clean"]
