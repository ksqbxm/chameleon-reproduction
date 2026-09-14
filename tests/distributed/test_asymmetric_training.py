from collections import Counter
import json

import pytest

from chameleon import ClusterState, ModelConfig, WorkerIdentity
from chameleon.runtime import DynamicTopology, RuntimeErrorWithAudit, SymmetricRuntime
from test_symmetric_training import assert_numerical_step, assert_p2p_warmup, assert_trace_dependencies


def test_every_gradient_parameter_and_adamw_matches_three_fp64_steps(asymmetric_training):
    result = asymmetric_training
    for actual, expected in zip(result["steps"], result["reference"]):
        assert_numerical_step(actual, expected, result["device"])


def test_unequal_sample_counts_partial_batch_and_all_worker_commits(asymmetric_training):
    result = asymmetric_training
    topology = result["topology"]
    for index, step in enumerate(result["steps"]):
        assert step["sample_ids"] == list(range(index * 19, (index + 1) * 19))
        endpoints = [r for r in step["reports"] if r["loss_batches"]]
        assert [len(r["micro_batches"]) for r in endpoints] == [5, 3, 2]
        assert [len(r["sample_ids"]) for r in endpoints] == [10, 6, 3]
        assert endpoints[-1]["micro_batches"][-1] == [index * 19 + 18]
        assert Counter(sample for r in endpoints for sample in r["sample_ids"]) == Counter(step["sample_ids"])
        for report in step["reports"]:
            p, s = topology.location(report["worker"]["rank"])
            assert (report["pipeline"], report["stage"]) == (p, s)
            assert report["sample_ids"] == endpoints[p]["sample_ids"]
        assert all(c["before"] == c["after"] == index and not c["advanced"] for c in step["commits"][:-1])
        assert step["commits"][-1]["after"] == index + 1
        assert step["commits"][-1]["advanced"]
        assert len(step["safe_worker_ids"]) == len(topology.ranks)
        assert min(c["commit_s"] for c in step["commits"]) >= max(c["ack_received_s"] for c in step["commits"])


def test_pipeline_means_then_mean_is_detectably_wrong(asymmetric_training):
    import torch
    from chameleon.data import make_batch
    from chameleon.model import build_initial_model
    from chameleon.reference import sample_losses

    result = asymmetric_training
    model = build_initial_model(result["config"], device=result["device"])
    batch = make_batch(range(19), result["config"], device=result["device"])
    losses = sample_losses(model(batch.inputs), batch.targets)
    wrong_loss = torch.stack((losses[:10].mean(), losses[10:16].mean(), losses[16:].mean())).mean()
    wrong_loss.backward()
    tolerance = dict(rtol=1e-7, atol=1e-9) if result["device"] == "cuda" else dict(rtol=1e-8, atol=1e-10)
    expected = result["reference"][0]
    assert not torch.isclose(wrong_loss.detach().cpu(), torch.tensor(expected.loss_global_mean, dtype=torch.float64), **tolerance)
    for module in ("embedding", "blocks.0", "blocks.1", "final_norm", "lm_head"):
        assert any(not torch.allclose(p.grad.cpu(), expected.gradients[name], **tolerance)
                   for name, p in model.named_parameters() if name.startswith(module + "."))


def test_actual_nonuniform_1f1b_has_fifo_and_cross_worker_dependencies(asymmetric_training):
    result = asymmetric_training
    for step in result["steps"]:
        for p, depth in enumerate(result["topology"].pipeline_lengths):
            assert_trace_dependencies([r for r in step["reports"] if r["pipeline"] == p],
                                      stages=depth, count=(5, 3, 2)[p])


def test_actual_p2p_shapes_peers_and_requested_device(asymmetric_training):
    result = asymmetric_training
    topology = result["topology"]
    for step in result["steps"]:
        messages = Counter()
        for report in step["reports"]:
            rank = report["worker"]["rank"]
            assert report["device"] == (f"cuda:{rank}" if result["device"] == "cuda" else "cpu")
            assert report["backend"] == ("nccl" if result["device"] == "cuda" else "gloo")
            for row in report["communication"]:
                src, dst = ((rank, row["peer_rank"]) if row["action"] == "send" else (row["peer_rank"], rank))
                messages[row["kind"], row["micro_batch"], src, dst, row["action"]] += 1
                assert row["shape"] == [len(report["micro_batches"][row["micro_batch"]]), 3, 4]
                assert row["tensor_bytes"] == row["shape"][0] * 3 * 4 * 8
                assert row["device"] == report["device"]
                assert row["peer_worker_id"] == topology.ranks[row["peer_rank"]].worker_id
        expected = Counter()
        for p, ranks in enumerate(topology.pipeline_ranks):
            for src, dst in zip(ranks, ranks[1:]):
                for mb in range((5, 3, 2)[p]):
                    for action in ("send", "recv"):
                        expected["activation", mb, src, dst, action] = 1
                        expected["gradient", mb, dst, src, action] = 1
        assert messages == expected


def test_persistent_processes_and_resource_audit(asymmetric_training):
    result = asymmetric_training
    runtime = result["runtime"]
    assert_p2p_warmup(runtime)
    size = 8 if result["device"] == "cuda" else 7
    assert len({r["pid"] for r in runtime.ready}) == size
    for step in result["steps"]:
        assert [r["pid"] for r in step["reports"]] == [r["pid"] for r in runtime.ready]
    assert runtime.audit["clean"] and runtime.audit["committed_global_step"] == 3
    assert runtime.audit["rendezvous_removed"] and runtime.audit["rendezvous_file_removed"]
    assert not runtime.audit["leaked_pids"]
    assert json.loads(runtime.report_path.read_text(encoding="utf-8"))["audit"] == runtime.audit


@pytest.mark.parametrize("case", ["fp32", "error", "hang"])
def test_fp32_or_failure_cleans_nonuniform_topology(distributed_environment, device, world_size, case):
    import torch
    from chameleon.model import build_initial_model
    from chameleon.reference import ReferenceTrainer

    size = 8 if device == "cuda" else 7
    assert world_size == size
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=19, micro_batch_size=2)
    short = (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head"))
    long = (("embedding",), ("blocks.0",), ("blocks.1", "final_norm", "lm_head"))
    topology = DynamicTopology(ClusterState(tuple(WorkerIdentity(f"smoke-{r}", r, 0) for r in range(size)), 19),
                               config, (short, long if device == "cuda" else short, long), (5, 3, 2))
    runtime = SymmetricRuntime(topology, device=device, capture_state=case == "fp32", dtype="float32",
                               behavior=case if case != "fp32" else "normal")
    if case == "fp32":
        with runtime:
            actual = runtime.train_step()
        assert_p2p_warmup(runtime)
        if device == "cuda":
            assert all(row["float32_matmul_precision"] == "highest" for row in runtime.ready)
        reference = ReferenceTrainer(build_initial_model(config, device=device, dtype=torch.float32),
                                     ClusterState((WorkerIdentity("reference", 0, 0),), 19))
        assert_numerical_step(actual, reference.train_step(), device, fp32=True)
    else:
        with pytest.raises(RuntimeErrorWithAudit, match="hard timeout" if case == "hang" else "injected|abnormally"):
            with runtime:
                runtime.timeout_s = 5
                runtime.train_step()
        assert runtime.state.committed_global_step == 0
    assert runtime.audit["clean"]
