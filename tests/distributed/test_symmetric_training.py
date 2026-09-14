from collections import Counter
import json

import pytest

from chameleon import ClusterState, ModelConfig, WorkerIdentity
from chameleon.runtime import RuntimeErrorWithAudit, SymmetricRuntime, SymmetricTopology


def assert_p2p_warmup(runtime):
    config = runtime.topology.config
    shape = [min(config.micro_batch_size, config.global_batch_size), config.sequence_length, config.hidden_size]
    element_size = 4 if runtime.dtype == "float32" else 8
    for ranks in runtime.topology.pipeline_ranks:
        for stage, rank in enumerate(ranks):
            peers = set(ranks[max(0, stage - 1):stage] + ranks[stage + 1:stage + 2])
            rows = runtime.ready[rank]["p2p_warmup"]
            assert Counter((row["action"], row["peer_rank"]) for row in rows) == Counter(
                (action, peer) for peer in peers for action in ("send", "recv"))
            for row in rows:
                assert row["shape"] == shape
                assert row["tensor_bytes"] == shape[0] * shape[1] * shape[2] * element_size
                assert row["device"] == (f"cuda:{rank}" if runtime.device == "cuda" else "cpu")
                assert row["value"] == (rank if row["action"] == "send" else row["peer_rank"]) + 1


def assert_numerical_step(actual, expected, device, *, fp32=False, owner_counts=None):
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


def assert_trace_dependencies(reports, stages, count):
    operations = {}
    for report in reports:
        trace = report["trace"]
        warmup = min(stages - report["stage"] - 1, count)
        # Independently construct the hand-specified FIFO 1F1B order; no production scheduler.
        expected = [("forward", mb, "warmup") for mb in range(warmup)]
        for mb in range(count - warmup):
            expected += [("forward", mb + warmup, "steady"), ("backward", mb, "steady")]
        expected += [("backward", mb, "cooldown") for mb in range(count - warmup, count)]
        assert [(row["kind"], row["micro_batch"], row["phase"]) for row in trace] == expected
        previous = None
        for row in trace:
            key = row["pipeline"], row["stage"], row["micro_batch"], row["kind"]
            assert key not in operations
            assert row["end_s"] >= row["start_s"]
            if previous:
                assert row["start_s"] >= previous["end_s"]
            operations[key] = row
            previous = row
        assert report["optimizer_completed_s"] >= trace[-1]["end_s"]
    for (pipeline, stage, mb, kind), row in operations.items():
        if kind == "forward" and stage:
            assert row["start_s"] >= operations[pipeline, stage - 1, mb, "forward"]["end_s"]
        if kind == "backward":
            assert row["start_s"] >= operations[pipeline, stage, mb, "forward"]["end_s"]
            if stage + 1 < stages:
                assert row["start_s"] >= operations[pipeline, stage + 1, mb, "backward"]["end_s"]


def test_all_parameters_gradients_and_adamw_match_three_fp64_steps(symmetric_training):
    result = symmetric_training
    for actual, expected in zip(result["steps"], result["reference"]):
        assert_numerical_step(actual, expected, result["device"])


def test_real_pids_stable_identity_and_owner_groups(symmetric_training):
    runtime = symmetric_training["runtime"]
    assert_p2p_warmup(runtime)
    assert len({row["pid"] for row in runtime.ready}) == 4
    assert [row["worker"].rank for row in runtime.ready] == [0, 1, 2, 3]
    assert [row["worker"].worker_id for row in runtime.ready] == ["stable-20", "stable-19", "stable-18", "stable-17"]
    assert all(row["worker"].generation == 2 for row in runtime.ready)
    assert all(row["groups"] == [[0, 1], [2, 3], [0, 2], [1, 3]] for row in runtime.ready)
    for step in symmetric_training["steps"]:
        assert [row["pid"] for row in step["reports"]] == [row["pid"] for row in runtime.ready]
    assert runtime.audit["clean"] and runtime.audit["committed_global_step"] == 3


def test_partial_samples_sum_count_and_all_worker_commit(symmetric_training):
    for index, step in enumerate(symmetric_training["steps"]):
        assert step["sample_ids"] == list(range(index * 11, (index + 1) * 11))
        logical_ids = [sample_id for report in step["reports"] if report["stage"] == 1 for sample_id in report["sample_ids"]]
        assert Counter(logical_ids) == Counter(step["sample_ids"])
        assert [len(report["sample_ids"]) for report in step["reports"] if report["stage"] == 1] == [6, 5]
        for report in step["reports"]:
            assert report["sample_ids"] == step["reports"][report["pipeline"] * 2]["sample_ids"]
            assert report["global_sample_count"] == 11
        assert all(row["before"] == row["after"] == index and not row["advanced"] for row in step["commits"][:-1])
        last = step["commits"][-1]
        assert last["worker_id"] == "stable-17" and last["before"] == index and last["after"] == index + 1 and last["advanced"]
        assert min(row["commit_s"] for row in step["commits"]) >= max(row["ack_received_s"] for row in step["commits"])
        assert all(row["ack_received_s"] >= report["optimizer_completed_s"]
                   for row, report in zip(step["commits"], step["reports"]))
        assert set(step["safe_worker_ids"]) == {"stable-20", "stable-19", "stable-18", "stable-17"}


def test_real_trace_has_fifo_order_and_cross_worker_dependencies(symmetric_training):
    for step in symmetric_training["steps"]:
        assert_trace_dependencies(step["reports"], stages=2, count=3)


def test_actual_activation_and_gradient_transfers_are_matched(symmetric_training):
    for step in symmetric_training["steps"]:
        messages = Counter()
        for report in step["reports"]:
            for row in report["communication"]:
                source, target = ((row["rank"], row["peer_rank"]) if row["action"] == "send"
                                  else (row["peer_rank"], row["rank"]))
                messages[row["kind"], row["micro_batch"], source, target, row["action"]] += 1
                assert row["tensor_bytes"] == row["shape"][0] * 3 * 4 * 8
                assert row["shape"] == [len(report["micro_batches"][row["micro_batch"]]), 3, 4]
                assert row["start_s"] <= row["end_s"] <= report["optimizer_completed_s"]
                assert row["worker_id"] == report["worker"]["worker_id"]
        expected = Counter({(kind, mb, source, target, action): 1
                            for first in (0, 2) for mb in range(3) for action in ("send", "recv")
                            for kind, source, target in (("activation", first, first + 1), ("gradient", first + 1, first))})
        assert messages == expected


def test_training_and_collectives_use_requested_device(symmetric_training):
    device = symmetric_training["device"]
    for step in symmetric_training["steps"]:
        for rank, report in enumerate(step["reports"]):
            assert report["device"] == (f"cuda:{rank}" if device == "cuda" else "cpu")
            assert report["backend"] == ("nccl" if device == "cuda" else "gloo")
            assert all(row["device"] == report["device"] for row in report["communication"])


@pytest.mark.parametrize("case", ["single_partial", "less_than_depth", "single_stage", "fp32"])
def test_small_schedules_and_fp32_smoke(distributed_environment, device, world_size, case):
    import torch
    from chameleon.model import build_initial_model
    from chameleon.reference import ReferenceTrainer

    assert world_size == 4, "Task09 acceptance requires --world-size 4"
    stages = (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head"))
    batch, micro = (3, 2) if case == "single_partial" else (2, 1) if case == "less_than_depth" else (8, 2)
    if case == "less_than_depth":
        stages = (("embedding",), ("blocks.0",), ("blocks.1",), ("final_norm", "lm_head"))
    if case == "single_stage":
        stages = (("embedding", "blocks.0", "blocks.1", "final_norm", "lm_head"),)
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=batch, micro_batch_size=micro)
    topology = SymmetricTopology(ClusterState(tuple(WorkerIdentity(f"small-{rank}", rank, 0) for rank in range(4)), batch), config, stages)
    dtype = "float32" if case == "fp32" else "float64"
    runtime = SymmetricRuntime(topology, device=device, dtype=dtype, capture_state=True)
    with runtime:
        steps = [runtime.train_step() for _ in range(3 if case == "fp32" else 1)]
    assert_p2p_warmup(runtime)
    if case == "fp32" and device == "cuda":
        assert all(row["float32_matmul_precision"] == "highest" for row in runtime.ready)
    reference = ReferenceTrainer(build_initial_model(config, device=device, dtype=getattr(torch, dtype)),
                                 ClusterState((WorkerIdentity("reference", 0, 0),), batch))
    for actual in steps:
        assert_numerical_step(actual, reference.train_step(), device, fp32=case == "fp32")
        assert_trace_dependencies(actual["reports"], stages=len(stages), count=topology.global_micro_batches // topology.dp_size)
    assert runtime.state.committed_global_step == len(steps)
    assert runtime.audit["clean"] and all(worker["exitcode"] == 0 for worker in runtime.audit["workers"])


@pytest.mark.parametrize("behavior", ["error", "hang"])
def test_runtime_failure_and_hard_timeout_never_commit_and_clean(distributed_environment, device, world_size, behavior):
    assert world_size == 4, "Task09 acceptance requires --world-size 4"
    config = ModelConfig(num_layers=2, global_batch_size=2, micro_batch_size=1)
    topology = SymmetricTopology(ClusterState(tuple(WorkerIdentity(f"failure-{rank}", rank, 0) for rank in range(4)), 2),
                                 config, (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head")))
    runtime = SymmetricRuntime(topology, device=device, behavior=behavior)
    with pytest.raises(RuntimeErrorWithAudit, match="hard timeout" if behavior == "hang" else "injected|abnormally") as caught:
        with runtime:
            runtime.timeout_s = 5  # Startup has its own longer bound; the stalled training step has a hard bound.
            runtime.train_step()
    assert caught.value.audit["clean"]
    assert runtime.state.committed_global_step == 0
    assert len(runtime.audit["workers"]) == 4
    assert any(worker["exitcode"] != 0 for worker in runtime.audit["workers"])
    assert json.loads(runtime.report_path.read_text(encoding="utf-8"))["audit"] == runtime.audit
