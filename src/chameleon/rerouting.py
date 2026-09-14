"""Logical 1F1B arbitration and real same-stage peer execution."""

from collections import Counter
import os
import time

from .contracts import _integer
from .schedule import build_1f1b_schedule


def execution_rounds(topology):
    """Ready operations run together, with at most one computation per physical rank."""
    remaining = [op for p, count in enumerate(topology.pipeline_micro_batches)
                 for queue in build_1f1b_schedule(topology.pp_size, count, pipeline=p) for op in queue]
    completed, rounds = set(), []
    while remaining:
        selected, busy = [], set()
        for op in sorted(remaining, key=lambda item: item.key):
            rank = topology.task_rank(op.pipeline, op.stage, op.micro_batch)
            if rank not in busy and set(op.dependencies) <= completed:
                selected.append(op)
                busy.add(rank)
        if not selected:
            raise ValueError("routing operation dependencies cannot make progress")
        completed.update(op.key for op in selected)
        remaining = [op for op in remaining if op.key not in completed]
        rounds.append(tuple(selected))
    return tuple(rounds)


def _task_fields(topology, batches, pipeline, stage, mb):
    return dict(pipeline=pipeline, stage=stage, micro_batch=mb,
                global_micro_batch=sum(topology.pipeline_micro_batches[:pipeline]) + mb,
                sample_ids=list(batches[pipeline][mb]))


def validate_routing_reports(topology, state, reports):
    """Reject missing/duplicate tasks and misrouted IDs or peers before controller commit."""
    batches = tuple(topology.micro_batches(state, p) for p in range(topology.dp_size))
    operations = {op.key: op for row in execution_rounds(topology) for op in row}
    expected, actual = Counter(), Counter()
    messages, expected_messages = Counter(), Counter()
    losses, expected_losses = Counter(), Counter()
    for op in operations.values():
        p, s, mb, kind = op.key
        rank = topology.task_rank(p, s, mb)
        expected[op.key, rank] += 1
        if kind == "forward" and s == topology.pp_size - 1:
            expected_losses[p, mb, rank] += 1
        target_stage = s + (1 if kind == "forward" else -1)
        if 0 <= target_stage < topology.pp_size:
            peer = topology.task_rank(p, target_stage, mb)
            message_kind = "activation" if kind == "forward" else "gradient"
            expected_messages[p, s, mb, message_kind, "send", rank, peer] += 1
            expected_messages[p, s, mb, message_kind, "recv", peer, rank] += 1

    def check_fields(row):
        p, s, mb = (row[field] for field in ("pipeline", "stage", "micro_batch"))
        topology.task_rank(p, s, mb)  # Validate integers and bounds, including bool IDs.
        _integer("routing global_micro_batch", row["global_micro_batch"], 0)
        for sample in row["sample_ids"]:
            _integer("routing sample_id", sample, 0)
        if any(row[field] != value for field, value in _task_fields(topology, batches, p, s, mb).items()):
            raise ValueError("routing task sample IDs or global micro-batch ID mismatch")
        return p, s, mb

    for report in reports:
        rank = report["worker"]["rank"]
        if (report["pipeline"], report["stage"]) != topology.location(rank):
            raise ValueError("routing report must retain the native logical slot")
        for row in report["trace"]:
            key = (*check_fields(row), row["kind"])
            if key not in operations or row["phase"] != operations[key].phase:
                raise ValueError("routing trace kind or phase mismatch")
            actual[key, rank] += 1
        for row in report["communication"]:
            p, s, mb = check_fields(row)
            if row["rank"] != rank:
                raise ValueError("routing message rank mismatch")
            messages[p, s, mb, row["kind"], row["action"], rank, row["peer_rank"]] += 1
        for row in report["loss_batches"]:
            p, s, mb = check_fields(row)
            if s != topology.pp_size - 1:
                raise ValueError("routing loss must come from the logical final stage")
            losses[p, mb, rank] += 1
    if actual != expected or messages != expected_messages or losses != expected_losses:
        raise ValueError("routing must execute every logical task and transfer exactly once on the correct peer")


def train_rerouted_step(topology, rank, model, optimizer, pp_group, owner_groups,
                        state, origin, capture_path):
    import torch
    import torch.distributed as dist
    from .data import make_batch
    from .global_loss import micro_batch_loss_sum
    from .runtime import _capture_worker_state, _complete_update, _synchronize

    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    batches = tuple(topology.micro_batches(state, p) for p in range(topology.dp_size))
    optimizer.zero_grad(set_to_none=True)
    dist.barrier()
    _synchronize(device)
    start = time.monotonic()
    incoming, graphs, losses = {}, {}, {}
    trace, communication = [], []
    for round_id, operations in enumerate(execution_rounds(topology)):
        outgoing = {}
        for op in operations:
            p, s, mb, kind = op.key
            if topology.task_rank(p, s, mb) != rank:
                continue
            left = time.monotonic() - origin
            if kind == "forward":
                batch = make_batch(batches[p][mb], model.config, device=device) if s in (0, topology.pp_size - 1) else None
                inputs = batch.inputs if s == 0 else incoming.pop(op.key).requires_grad_()
                output = model(inputs)
                if s == topology.pp_size - 1:
                    output = micro_batch_loss_sum(output, batch.targets)
                    losses[p, mb] = output.detach()
                graphs[p, s, mb] = inputs, output
                outgoing[op.key] = output
            else:
                inputs, output = graphs.pop((p, s, mb))
                if s == topology.pp_size - 1:
                    output.backward()
                else:
                    output.backward(incoming.pop(op.key))
                if s:
                    outgoing[op.key] = inputs.grad
            _synchronize(device)
            trace.append(dict(_task_fields(topology, batches, p, s, mb), kind=kind,
                              phase=op.phase, round=round_id, start_s=left, end_s=time.monotonic() - origin))

        # All endpoints post transfers in the same order, including multiple messages per peer.
        pending, rows = [], []
        for op in operations:
            p, s, mb, kind = op.key
            target_stage = s + (1 if kind == "forward" else -1)
            if not 0 <= target_stage < topology.pp_size:
                continue
            source = topology.task_rank(p, s, mb)
            target = topology.task_rank(p, target_stage, mb)
            if rank not in (source, target):
                continue
            sending = rank == source
            tensor = (outgoing[op.key].detach().contiguous() if sending else
                      torch.empty((len(batches[p][mb]), topology.config.sequence_length,
                                   topology.config.hidden_size), device=device, dtype=dtype))
            if not sending:
                key = p, target_stage, mb, kind
                if key in incoming:
                    raise RuntimeError("routing received a duplicate logical task tensor")
                incoming[key] = tensor
            peer = target if sending else source
            pending.append(dist.P2POp(dist.isend if sending else dist.irecv, tensor, peer, pp_group))
            rows.append(dict(_task_fields(topology, batches, p, s, mb),
                             kind="activation" if kind == "forward" else "gradient",
                             action="send" if sending else "recv", rank=rank, peer_rank=peer,
                             worker_id=topology.ranks[rank].worker_id, peer_worker_id=topology.ranks[peer].worker_id,
                             shape=list(tensor.shape), dtype=str(dtype), device=str(device),
                             tensor_bytes=tensor.numel() * tensor.element_size(), round=round_id,
                             start_s=time.monotonic() - origin))
        if pending:
            for request in dist.batch_isend_irecv(pending):
                request.wait()
            _synchronize(device)
        communication.extend(dict(row, end_s=time.monotonic() - origin) for row in rows)
        # No rank advances into another round while an edge in this round is in flight.
        dist.barrier()
        _synchronize(device)
    if incoming or graphs:
        raise RuntimeError("routing left in-flight tensors or autograd graphs")
    pipeline_s = time.monotonic() - start
    count = sum(len(batches[p][mb]) for p, mb in losses)
    synced, allreduces, global_loss = _complete_update(topology, rank, model, optimizer, owner_groups,
                                                      tuple(losses.values()), count, origin)
    completed_s = time.monotonic() - origin
    pipeline, stage = topology.location(rank)
    forward_tasks = [row for row in trace if row["kind"] == "forward"]
    report = dict(pid=os.getpid(), pipeline=pipeline, stage=stage,
                  sample_ids=[sample for row in forward_tasks for sample in row["sample_ids"]],
                  micro_batches=[row["sample_ids"] for row in forward_tasks], trace=trace, communication=communication,
                  synchronized_parameters=synced, allreduces=allreduces,
                  synchronization_rounds=topology.synchronization_rounds,
                  loss_batches=[dict(_task_fields(topology, batches, p, topology.pp_size - 1, mb), loss_sum=loss.item())
                                for (p, mb), loss in losses.items()],
                  loss_global_sum=global_loss[0].item(), global_sample_count=int(global_loss[1].item()),
                  pipeline_wall_time_s=pipeline_s, training_wall_time_s=completed_s - (start - origin),
                  optimizer_completed_s=completed_s, device=str(device), backend=dist.get_backend(), profile_step=None)
    _capture_worker_state(model, optimizer, capture_path)
    return report
