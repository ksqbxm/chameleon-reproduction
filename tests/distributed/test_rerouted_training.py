from collections import Counter
import json

import pytest

from chameleon import ClusterState, ModelConfig, WorkerIdentity
from chameleon.runtime import DistributedRuntime, ReroutingTopology, RuntimeErrorWithAudit
from conftest import assert_numerical_step
from test_symmetric_training import assert_trace_dependencies


def assert_rerouted_numerics(result):
    topology = result["topology"]
    counts = {}
    for name in result["reference"][0].parameters:
        module = ".".join(name.split(".")[:2]) if name.startswith("blocks.") else name.split(".")[0]
        counts[name] = len(topology.module_owners[module])
    for actual, expected in zip(result["steps"], result["reference"]):
        assert_numerical_step(actual, expected, result["device"], fp32=result["dtype"] == "float32", owner_counts=counts)


def assert_rerouted_trace(result):
    topology = result["topology"]
    for step in result["steps"]:
        for p, count in enumerate(topology.pipeline_micro_batches):
            logical = [dict(stage=s, trace=sorted([row for report in step["reports"] for row in report["trace"]
                                                   if row["pipeline"] == p and row["stage"] == s],
                                                  key=lambda row: row["start_s"]),
                            optimizer_completed_s=min(report["optimizer_completed_s"] for report in step["reports"]))
                       for s in range(topology.pp_size)]
            assert_trace_dependencies(logical, stages=topology.pp_size, count=count)
        # A peer's native and rerouted computations never overlap on the device.
        for report in step["reports"]:
            for previous, following in zip(report["trace"], report["trace"][1:]):
                assert previous["end_s"] <= following["start_s"]


def assert_rerouted_transfers(result):
    topology = result["topology"]
    for step in result["steps"]:
        actual, expected = Counter(), Counter()
        for report in step["reports"]:
            rank = report["worker"]["rank"]
            assert report["device"] == (f"cuda:{rank}" if result["device"] == "cuda" else "cpu")
            assert report["backend"] == ("nccl" if result["device"] == "cuda" else "gloo")
            for row in report["communication"]:
                p, s, mb = row["pipeline"], row["stage"], row["micro_batch"]
                actual[p, s, mb, row["kind"], row["action"], rank, row["peer_rank"]] += 1
                assert row["shape"] == [len(row["sample_ids"]), 3, 4]
                assert row["tensor_bytes"] == len(row["sample_ids"]) * 3 * 4 * (4 if result["dtype"] == "float32" else 8)
                assert row["device"] == report["device"]
                assert row["peer_worker_id"] == topology.ranks[row["peer_rank"]].worker_id
                assert row["start_s"] <= row["end_s"] <= report["optimizer_completed_s"]
        for p, count in enumerate(topology.pipeline_micro_batches):
            for s in range(topology.pp_size - 1):
                for mb in range(count):
                    source, target = topology.task_rank(p, s, mb), topology.task_rank(p, s + 1, mb)
                    expected[p, s, mb, "activation", "send", source, target] += 1
                    expected[p, s, mb, "activation", "recv", target, source] += 1
                    expected[p, s + 1, mb, "gradient", "send", target, source] += 1
                    expected[p, s + 1, mb, "gradient", "recv", source, target] += 1
        assert actual == expected


def assert_rerouted_owners_and_cleanup(result):
    topology, runtime = result["topology"], result["runtime"]
    specifications = [list(range(len(topology.ranks)))] + [list(row) for row in dict.fromkeys(topology.module_owners.values())]
    for ready in runtime.ready:
        rank = ready["worker"].rank
        assert ready["groups"] == specifications
        initial = ready["initial_topology"]
        assert initial["physical_workers"] == len(topology.ranks)
        assert initial["logical_slots"] == topology.logical_slots
        assert initial["pipeline_ranks"] == [list(row) for row in topology.pipeline_ranks]
        assert initial["layouts"] == [[list(stage) for stage in layout] for layout in topology.layouts]
        assert initial["pipeline_micro_batches"] == list(topology.pipeline_micro_batches)
        peers = {target if source == rank else source for source, target in topology.transfer_edges if rank in (source, target)}
        assert Counter((row["action"], row["peer_rank"]) for row in ready["p2p_warmup"]) == Counter(
            (action, peer) for action in ("send", "recv") for peer in peers)
        for row in ready["p2p_warmup"]:
            assert row["value"] == (rank if row["action"] == "send" else row["peer_rank"]) + 1
    for step in result["steps"]:
        for report, ready in zip(step["reports"], runtime.ready):
            assert report["pid"] == ready["pid"]
            assert set(ready["parameter_owners"]) == set(report["synchronized_parameters"])
            assert Counter(row["parameter"] for row in report["allreduces"]) == Counter(report["synchronized_parameters"])
            for row in report["allreduces"]:
                assert row["owner_ranks"] == list(topology.module_owners[row["module"]])
                assert row["owner_ranks"] == ready["parameter_owners"][row["parameter"]]
                assert row["reduction"] == "SUM" and row["divisor"] == topology.config.global_batch_size
                assert row["async_op"] and row["module"] in topology.synchronization_rounds[row["color"]]
    assert len({ready["pid"] for ready in runtime.ready}) == len(topology.ranks)
    assert runtime.audit["committed_global_step"] == 3 and runtime.audit["clean"]
    assert runtime.audit["rendezvous_removed"] and runtime.audit["rendezvous_file_removed"]
    assert runtime.audit["rendezvous_backend"] == "FileStore" and runtime.audit["rendezvous_port"] is None
    assert not runtime.audit["leaked_pids"]
    assert all(not row["alive"] and row["exitcode"] == 0 for row in runtime.audit["workers"])
    assert json.loads(runtime.report_path.read_text(encoding="utf-8"))["audit"] == runtime.audit


def test_all_parameters_gradients_loss_and_adamw_match_three_steps(rerouted_training):
    assert_rerouted_numerics(rerouted_training)


def test_real_native_and_rerouted_1f1b_dependencies(rerouted_training):
    assert_rerouted_trace(rerouted_training)


def test_real_activation_and_gradient_transfers_keep_logical_pipeline(rerouted_training):
    result = rerouted_training
    assert_rerouted_transfers(result)
    for step in result["steps"]:
        tasks = sorted((row["micro_batch"], report["worker"]["rank"]) for report in step["reports"]
                       for row in report["trace"] if row["pipeline"] == 0 and row["stage"] == 1 and row["kind"] == "forward")
        assert tasks == list(enumerate((4, 2, 4, 2, 4)))
        # The healthy predecessor of the missing stage still computes its five tasks.
        assert len([row for row in step["reports"][0]["trace"] if row["kind"] == "forward"]) == 5


def test_fixed_samples_and_optimizer_ack_before_all_worker_commit(rerouted_training):
    result = rerouted_training
    for index, step in enumerate(result["steps"]):
        assert step["sample_ids"] == list(range(index * 19, (index + 1) * 19))
        losses = [batch for report in step["reports"] for batch in report["loss_batches"]]
        assert Counter(sample for batch in losses for sample in batch["sample_ids"]) == Counter(step["sample_ids"])
        assert Counter(batch["pipeline"] for batch in losses) == Counter({0: 5, 1: 3, 2: 2})
        assert [sum(len(batch["sample_ids"]) for batch in losses if batch["pipeline"] == p) for p in range(3)] == [10, 6, 3]
        assert next(batch for batch in losses if (batch["pipeline"], batch["micro_batch"]) == (2, 1))["sample_ids"] == [index * 19 + 18]
        assert all(c["before"] == c["after"] == index and not c["advanced"] for c in step["commits"][:-1])
        assert step["commits"][-1]["after"] == index + 1 and step["commits"][-1]["advanced"]
        assert min(c["commit_s"] for c in step["commits"]) >= max(c["ack_received_s"] for c in step["commits"])
        assert all(c["ack_received_s"] >= report["optimizer_completed_s"] for c, report in zip(step["commits"], step["reports"]))
        assert len(step["safe_worker_ids"]) == 5


def test_real_owner_groups_native_processes_and_cleanup(rerouted_training):
    assert_rerouted_owners_and_cleanup(rerouted_training)


@pytest.mark.parametrize("case", ["first_stage", "multiple_stages", "fp32", "single_partial", "one_stage"])
def test_additional_missing_stages_and_partial_micro_batches(rerouted_run, world_size, case):
    assert world_size == 5
    slots, counts, options = ((0, None), (1, 2), (3, 4)), (5, 3, 2), {}
    if case == "first_stage":
        slots = ((None, 0), (1, 2), (3, 4))
    elif case == "multiple_stages":
        slots = ((None, 0, 1), (None, 2, None), (3, None, 4))
        options["stages"] = (("embedding",), ("blocks.0",), ("blocks.1", "final_norm", "lm_head"))
    elif case == "fp32":
        options["dtype"] = "float32"
    elif case == "single_partial":
        counts, options["batch"] = (1, 1, 1), 5
    else:
        slots, counts, options = ((None,), (0,), (1,), (2,), (3,), (4,)), (2, 2, 2, 2, 1, 1), dict(
            batch=19, stages=(("embedding", "blocks.0", "blocks.1", "final_norm", "lm_head"),))
    result = rerouted_run(slots, counts, **options)
    assert_rerouted_numerics(result)
    assert_rerouted_trace(result)
    assert_rerouted_transfers(result)
    assert_rerouted_owners_and_cleanup(result)


@pytest.mark.parametrize("behavior", ["error", "hang"])
def test_rerouted_worker_error_or_timeout_never_commit_and_cleanup(distributed_environment, device, world_size, behavior):
    assert world_size == 5
    config = ModelConfig(num_layers=2, global_batch_size=19, micro_batch_size=2)
    topology = ReroutingTopology(ClusterState(tuple(WorkerIdentity(f"failure-{r}", r, 0) for r in range(5)), 19),
                                config, (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head")),
                                (5, 3, 2), ((0, None), (1, 2), (3, 4)))
    runtime = DistributedRuntime(topology, device=device, behavior=behavior)
    with pytest.raises(RuntimeErrorWithAudit, match="hard timeout" if behavior == "hang" else "injected|abnormally") as caught:
        with runtime:
            runtime.timeout_s = 5
            runtime.train_step()
    assert caught.value.audit["clean"] and runtime.state.committed_global_step == 0
    assert len(runtime.audit["workers"]) == 5 and not runtime.audit["leaked_pids"]
    assert runtime.audit["rendezvous_removed"] and runtime.audit["rendezvous_file_removed"]
    assert any(worker["exitcode"] != 0 for worker in runtime.audit["workers"])
