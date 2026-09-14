from collections import Counter
from dataclasses import replace
from functools import partial
from itertools import product
import multiprocessing as mp

import pytest

from chameleon import ClusterState, ModelConfig, WorkerIdentity
from chameleon.runtime import ReroutingTopology, RuntimeErrorWithAudit, SymmetricRuntime
from chameleon.rerouting import execution_rounds, validate_routing_reports


def topology(*, slots=((0, None), (1, 2), (3, 4)), counts=(5, 3, 2), stages=None):
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=19, micro_batch_size=2)
    workers = tuple(WorkerIdentity(f"peer-{20 - r}", r, 2) for r in reversed(range(5)))
    stages = (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head")) if stages is None else stages
    return ReroutingTopology(ClusterState(workers, 19, generation=2), config, stages, counts, slots)


def test_missing_stage_keeps_layout_native_slots_and_all_samples():
    current = topology()
    assert current.dp_size == 3 and current.pp_size == 2
    assert current.logical_slots == 6 and len(current.ranks) == 5
    assert current.layouts == (current.stage_modules,) * 3
    assert current.module_owners == {"embedding": (0, 1, 3), "blocks.0": (0, 1, 3),
                                     "blocks.1": (2, 4), "final_norm": (2, 4), "lm_head": (2, 4)}
    assert Counter(current.task_rank(0, 1, mb) for mb in range(5)) == Counter({4: 3, 2: 2})
    for step in range(3):
        state = replace(current.state, committed_global_step=step)
        batches = [current.micro_batches(state, p) for p in range(3)]
        assert [sum(map(len, row)) for row in batches] == [10, 6, 3]
        assert batches[-1][-1] == (step * 19 + 18,)
        assert [sample for row in batches for batch in row for sample in batch] == list(range(step * 19, (step + 1) * 19))
        for p, slots in enumerate(current.pipeline_ranks):
            for s, rank in enumerate(slots):
                for mb in range(current.pipeline_micro_batches[p]):
                    owner = current.task_rank(p, s, mb)
                    assert current.location(owner)[1] == s
                    if rank is not None:
                        assert owner == rank


def test_multiple_missing_slots_balance_extra_work_across_same_stage_peers():
    # DP4/PP2 with three missing slots and exactly five native workers.
    current = topology(slots=((None, None), (None, 0), (1, 2), (3, 4)), counts=(3, 3, 2, 2))
    assert Counter(current.task_rank(p, 0, mb) for p in (0, 1) for mb in range(3)) == Counter({1: 3, 3: 3})
    assert Counter(current.task_rank(0, 1, mb) for mb in range(3)) == Counter({0: 1, 2: 1, 4: 1})
    assert current.logical_slots == 8


@pytest.mark.parametrize("slots", [((0, None), (1, None), (2, None), (3, None), (4, None)),
                                   ((None, 0), (None, 1), (None, 2), (None, 3), (None, 4))])
def test_last_stage_peer_missing_is_infeasible(slots):
    with pytest.raises(ValueError, match="Fi >= Ndp"):
        topology(slots=slots, counts=(2, 2, 2, 2, 2))


@pytest.mark.parametrize("slots", [((0, None), (1, 2), (3, 3)), ((0, None), (1, 2), (3, 5)),
                                   ((0, None), (1, 2), (3, True)), ((0,), (1, 2), (3, 4)),
                                   [(0, None), (1, 2), (3, 4)], ((None, None), (0, 1), (2, 3))])
def test_duplicate_unknown_or_idle_physical_workers_are_rejected(slots):
    with pytest.raises(ValueError, match="rank|slot"):
        topology(slots=slots)


@pytest.mark.parametrize("counts", [(0, 8, 2), (True, 7, 2), (5, 3), (5, 3, 1), [5, 3, 2]])
def test_invalid_batch_allocations(counts):
    with pytest.raises(ValueError, match="micro-batch"):
        topology(counts=counts)


@pytest.mark.parametrize("stages", [(("embedding", "blocks.0"), ("blocks.1", "lm_head")),
                                    (("blocks.0", "embedding"), ("blocks.1", "final_norm", "lm_head")),
                                    (("embedding",), ()), ()])
def test_rerouting_cannot_repartition_or_omit_model_modules(stages):
    with pytest.raises(ValueError, match="partition"):
        topology(stages=stages)


@pytest.mark.parametrize("task", [(-1, 0, 0), (3, 0, 0), (0, 2, 0), (0, 1, 5), (0, 1, True)])
def test_invalid_pipeline_stage_or_micro_batch(task):
    with pytest.raises(ValueError):
        topology().task_rank(*task)


def test_committed_state_cannot_be_initialized_and_step_identity_cannot_change():
    current = topology()
    with pytest.raises(ValueError, match="already committed"):
        SymmetricRuntime(replace(current, state=replace(current.state, committed_global_step=1)))
    with pytest.raises(ValueError, match="topology"):
        current.micro_batches(replace(current.state, generation=3,
                                      workers=tuple(replace(w, generation=3) for w in current.state.workers)), 0)


def test_execution_rounds_preserve_hand_counted_fifo_and_dependencies():
    current = topology()
    rounds = execution_rounds(current)
    completed, order = set(), {}
    for operations in rounds:
        owners = [current.task_rank(op.pipeline, op.stage, op.micro_batch) for op in operations]
        assert len(owners) == len(set(owners))
        for op in operations:
            assert op.key not in completed
            assert set(op.dependencies) <= completed
            order.setdefault((op.pipeline, op.stage), []).append((op.kind, op.micro_batch))
        completed.update(op.key for op in operations)
    assert len(completed) == 40
    # Independent PP2 oracle: stage0 F0,F1,B0,F2,B1,...,B(Nm-1); stage1 F0,B0,...
    for p, count in enumerate((5, 3, 2)):
        first = [("forward", 0)]
        for mb in range(count - 1):
            first += [("forward", mb + 1), ("backward", mb)]
        first += [("backward", count - 1)]
        assert order[p, 0] == first
        assert order[p, 1] == [(kind, mb) for mb in range(count) for kind in ("forward", "backward")]


def routing_reports(current, state=None):
    """Metadata-only reports for validation; never simulate tensors or training."""
    state = current.state if state is None else state
    reports = [dict(worker=dict(rank=r), pipeline=current.location(r)[0], stage=current.location(r)[1],
                    trace=[], communication=[], loss_batches=[]) for r in range(len(current.ranks))]
    for operations in execution_rounds(current):
        for op in operations:
            p, s, mb = op.pipeline, op.stage, op.micro_batch
            rank = current.task_rank(p, s, mb)
            fields = dict(pipeline=p, stage=s, micro_batch=mb,
                          global_micro_batch=sum(current.pipeline_micro_batches[:p]) + mb,
                          sample_ids=list(current.micro_batches(state, p)[mb]))
            reports[rank]["trace"].append(dict(fields, kind=op.kind, phase=op.phase))
            if op.kind == "forward" and s == current.pp_size - 1:
                reports[rank]["loss_batches"].append(dict(fields, loss_sum=1.))
            target_stage = s + (1 if op.kind == "forward" else -1)
            if 0 <= target_stage < current.pp_size:
                peer = current.task_rank(p, target_stage, mb)
                kind = "activation" if op.kind == "forward" else "gradient"
                reports[rank]["communication"].append(dict(fields, kind=kind, action="send", rank=rank, peer_rank=peer))
                reports[peer]["communication"].append(dict(fields, kind=kind, action="recv", rank=peer, peer_rank=rank))
    return reports


def test_valid_native_and_rerouted_task_accounting():
    current = topology()
    validate_routing_reports(current, current.state, routing_reports(current))


@pytest.mark.parametrize("fault", ["duplicate", "missing", "wrong_owner", "wrong_mb", "wrong_sample", "bool_sample",
                                    "wrong_global_mb", "wrong_phase", "duplicate_message", "wrong_peer",
                                    "missing_message", "duplicate_loss", "wrong_loss_owner", "wrong_loss_sample"])
def test_routing_rejects_duplicate_missing_or_misidentified_tasks_and_messages(fault):
    current = topology()
    reports = routing_reports(current)
    if fault == "duplicate":
        reports[0]["trace"].append(reports[0]["trace"][0])
    elif fault == "missing":
        reports[0]["trace"].pop()
    elif fault == "wrong_owner":
        reports[1]["trace"].append(reports[0]["trace"].pop())
    elif fault == "wrong_mb":
        reports[0]["trace"][0]["micro_batch"] = True
    elif fault == "wrong_sample":
        reports[0]["trace"][0]["sample_ids"] = [18]
    elif fault == "bool_sample":
        reports[0]["trace"][0]["sample_ids"] = [False, 1]
    elif fault == "wrong_global_mb":
        reports[0]["trace"][0]["global_micro_batch"] = 5
    elif fault == "wrong_phase":
        reports[0]["trace"][0]["phase"] = "cooldown"
    elif fault == "duplicate_message":
        reports[0]["communication"].append(reports[0]["communication"][0])
    elif fault == "wrong_peer":
        reports[0]["communication"][0]["peer_rank"] = 1
    elif fault == "missing_message":
        reports[0]["communication"].pop()
    elif fault == "duplicate_loss":
        reports[4]["loss_batches"].append(reports[4]["loss_batches"][0])
    elif fault == "wrong_loss_owner":
        reports[0]["loss_batches"].append(reports[4]["loss_batches"].pop())
    else:
        reports[4]["loss_batches"][0]["sample_ids"] = [18]
    with pytest.raises(ValueError, match="routing|micro_batch"):
        validate_routing_reports(current, current.state, reports)


@pytest.mark.parametrize("slots,counts,stages", [
    (((None, 0), (1, 2), (3, 4)), (5, 3, 2), None),
    (((None, 0, 1), (None, 2, None), (3, None, 4)), (5, 3, 2),
     (("embedding",), ("blocks.0",), ("blocks.1", "final_norm", "lm_head"))),
    (((None,), (0,), (1,), (2,), (3,), (4,)), (2, 2, 2, 2, 1, 1),
     (("embedding", "blocks.0", "blocks.1", "final_norm", "lm_head"),)),
])
def test_other_stage_depths_complete_every_task_and_remain_deterministic(slots, counts, stages):
    current = topology(slots=slots, counts=counts, stages=stages)
    first = execution_rounds(current)
    assert first == execution_rounds(current)
    assert sum(map(len, first)) == 2 * current.global_micro_batches * current.pp_size
    validate_routing_reports(current, current.state, routing_reports(current))
    for row in first:
        ranks = [current.task_rank(op.pipeline, op.stage, op.micro_batch) for op in row]
        assert len(ranks) == len(set(ranks))
    assert all(current.location(source)[1] + 1 == current.location(target)[1] for source, target in current.transfer_edges)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_all_small_missing_slot_patterns_have_exact_tasks_and_balanced_peers(depth):
    config = ModelConfig(num_layers=2, global_batch_size=19, micro_batch_size=2)
    layouts = {1: (("embedding", "blocks.0", "blocks.1", "final_norm", "lm_head"),),
               2: (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head")),
               3: (("embedding",), ("blocks.0",), ("blocks.1", "final_norm", "lm_head"))}
    for missing in product((False, True), repeat=3 * depth):
        if any(all(missing[p * depth + s] for p in range(3)) for s in range(depth)):
            continue
        rank, slots = 0, []
        for p in range(3):
            row = []
            for s in range(depth):
                row.append(None if missing[p * depth + s] else rank)
                rank += not missing[p * depth + s]
            slots.append(tuple(row))
        workers = tuple(WorkerIdentity(f"worker-{r}", r, 0) for r in range(rank))
        current = ReroutingTopology(ClusterState(workers, 19), config, layouts[depth], (5, 3, 2), tuple(slots))
        completed = set()
        for operations in execution_rounds(current):
            owners = [current.task_rank(op.pipeline, op.stage, op.micro_batch) for op in operations]
            assert len(owners) == len(set(owners))
            for op in operations:
                assert set(op.dependencies) <= completed and op.key not in completed
            completed.update(op.key for op in operations)
        assert len(completed) == 20 * depth
        for s in range(depth):
            peers = {row[s] for row in slots if row[s] is not None}
            extra = Counter(current.task_rank(p, s, mb) for p in range(3) if slots[p][s] is None
                            for mb in range((5, 3, 2)[p]))
            assert max(extra[peer] for peer in peers) - min(extra[peer] for peer in peers) <= 1


def _routing_metadata_worker(current, rank, device, backend, dtype, rendezvous_file, timeout,
                             connection, origin, lr, weight_decay, behavior, directory, capture_state, *, fault):
    """Control protocol only; no model, optimizer or simulated distributed backend."""
    from chameleon.runtime import _write_reply

    worker, state = current.ranks[rank], current.state
    try:
        _write_reply(connection, directory, rank, "ready", worker, pid=mp.current_process().pid,
                     parameter_names=["metadata-only"])
        while True:
            token = connection.recv_bytes(maxlength=7)
            if token == b"stop":
                break
            assert token == b"step"
            report = routing_reports(current, state)[rank]
            report.pop("worker")
            if fault and rank == 0:
                report["trace"].append(report["trace"][0])
            for batch in report["loss_batches"]:
                batch["loss_sum"] = float(len(batch["sample_ids"]))
            report.update(profile_step=None, loss_global_sum=float(state.global_batch_size), global_sample_count=state.global_batch_size)
            step_id = state.committed_global_step + 1
            _write_reply(connection, directory, rank, "ack", worker, step_id=step_id, **report)
            assert connection.recv_bytes(maxlength=7) == b"commit"
            state = replace(state, committed_global_step=step_id)
            _write_reply(connection, directory, rank, "safe", worker, step_id=step_id)
    finally:
        connection.close()


@pytest.mark.parametrize("fault", [False, True])
def test_routing_audit_runs_before_commit_with_real_metadata_process_cleanup(monkeypatch, fault):
    from chameleon import runtime as module
    monkeypatch.setattr(module, "validate_device", lambda *_: "metadata")
    monkeypatch.setattr(module, "_runtime_worker", partial(_routing_metadata_worker, fault=fault))
    runtime = SymmetricRuntime(topology())
    if fault:
        with pytest.raises(RuntimeErrorWithAudit, match="routing"):
            with runtime:
                runtime.train_step()
        assert runtime.state.committed_global_step == 0 and runtime.steps == []
    else:
        with runtime:
            for step in range(3):
                result = runtime.train_step()
                assert result["sample_ids"] == list(range(step * 19, (step + 1) * 19))
                assert runtime.state.committed_global_step == step + 1
        assert all(row["exitcode"] == 0 for row in runtime.audit["workers"])
    assert len(runtime.audit["workers"]) == 5 and runtime.audit["clean"]
    assert not runtime.audit["leaked_pids"] and runtime.audit["rendezvous_removed"]
    assert runtime.audit["backend"] == "metadata"
