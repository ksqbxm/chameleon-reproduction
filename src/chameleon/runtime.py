"""Persistent spawned workers for initial DP/PP training at safe steps."""

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import tempfile
import time
import traceback

from .contracts import ClusterState, ModelConfig, WorkerIdentity, _finite, _integer
from .data import next_sample_ids
from .environment import validate_device
from .global_loss import GlobalBatchAccounting
from .schedule import build_1f1b_schedule
from .step import StepCommit
from .restorer import synchronization_rounds
from .profiler import _hash


class _Topology:
    def _validate_initial_state(self):
        if self.state.global_batch_size != self.config.global_batch_size:
            raise ValueError("cluster and model global batch sizes must match")
        if self.state.committed_global_step != 0:
            raise ValueError("initial runtime cannot initialize already committed training state")
        if {w.rank for w in self.state.workers} != set(range(len(self.state.workers))):
            raise ValueError("runtime requires dense ranks, independent of stable worker IDs")

    @property
    def global_micro_batches(self):
        return (self.config.global_batch_size + self.config.micro_batch_size - 1) // self.config.micro_batch_size

    @property
    def ranks(self):
        return tuple(sorted(self.state.workers, key=lambda worker: worker.rank))

    def location(self, rank):
        _integer("rank", rank, 0)
        for pipeline, ranks in enumerate(self.pipeline_ranks):
            if rank in ranks:
                return pipeline, ranks.index(rank)
        raise ValueError("rank must belong to the topology")

    @property
    def module_owners(self):
        return {module: tuple(sorted(self.pipeline_ranks[p][s]
                                    for p, layout in enumerate(self.layouts)
                                    for s, stage in enumerate(layout)
                                    if module in stage and self.pipeline_ranks[p][s] is not None))
                for module in (name for stage in self.layouts[0] for name in stage)}

    @property
    def synchronization_rounds(self):
        return synchronization_rounds({module: tuple(self.ranks[rank].worker_id for rank in owners)
                                       for module, owners in self.module_owners.items()})

    def micro_batches(self, state: ClusterState, pipeline: int):
        _integer("pipeline", pipeline, 0)
        if pipeline >= self.dp_size:
            raise ValueError("pipeline must belong to the topology")
        if (state.workers != self.state.workers or state.generation != self.state.generation
                or state.global_batch_size != self.config.global_batch_size):
            raise ValueError("step state must match the runtime topology")
        ids = next_sample_ids(state)
        size = self.config.micro_batch_size
        batches = tuple(ids[start:start + size] for start in range(0, len(ids), size))
        start = sum(self.pipeline_micro_batches[:pipeline])
        return batches[start:start + self.pipeline_micro_batches[pipeline]]


@dataclass(frozen=True)
class SymmetricTopology(_Topology):
    state: ClusterState
    config: ModelConfig
    stage_modules: tuple[tuple[str, ...], ...]

    def __post_init__(self):
        expected = ("embedding", *(f"blocks.{i}" for i in range(self.config.num_layers)),
                    "final_norm", "lm_head")
        if (not isinstance(self.stage_modules, tuple) or not self.stage_modules
                or any(not isinstance(stage, tuple) or not stage for stage in self.stage_modules)
                or tuple(name for stage in self.stage_modules for name in stage) != expected):
            raise ValueError("stages must partition every model module once in model order")
        self._validate_initial_state()
        if len(self.state.workers) % self.pp_size:
            raise ValueError("symmetric workers must fill every DP/PP stage")
        if self.global_micro_batches % self.dp_size:
            raise ValueError("symmetric runtime requires equal positive micro-batch counts")

    @property
    def pp_size(self):
        return len(self.stage_modules)

    @property
    def dp_size(self):
        return len(self.state.workers) // self.pp_size

    @property
    def layouts(self):
        return (self.stage_modules,) * self.dp_size

    @property
    def pipeline_lengths(self):
        return (self.pp_size,) * self.dp_size

    @property
    def pipeline_micro_batches(self):
        return (self.global_micro_batches // self.dp_size,) * self.dp_size

    @property
    def pipeline_ranks(self):
        return tuple(tuple(range(p * self.pp_size, (p + 1) * self.pp_size)) for p in range(self.dp_size))


@dataclass(frozen=True)
class DynamicTopology(_Topology):
    """Initial training layout; plan/manifest metadata never restores old tensors."""
    state: ClusterState
    config: ModelConfig
    layouts: tuple[tuple[tuple[str, ...], ...], ...]
    pipeline_micro_batches: tuple[int, ...]
    pipeline_ranks: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        expected = ("embedding", *(f"blocks.{i}" for i in range(self.config.num_layers)),
                    "final_norm", "lm_head")
        if (not isinstance(self.layouts, tuple) or not self.layouts
                or any(not isinstance(layout, tuple) or not layout
                       or any(not isinstance(stage, tuple) or not stage for stage in layout)
                       or tuple(name for stage in layout for name in stage) != expected for layout in self.layouts)):
            raise ValueError("each pipeline must partition every model module once in model order")
        self._validate_initial_state()
        if not isinstance(self.pipeline_micro_batches, tuple):
            raise ValueError("pipeline micro-batch counts must be a tuple")
        for count in self.pipeline_micro_batches:
            _integer("pipeline micro-batch count", count)
        if (len(self.pipeline_micro_batches) != self.dp_size
                or sum(self.pipeline_micro_batches) != self.global_micro_batches):
            raise ValueError("pipeline micro-batch counts must preserve the global count")
        if self.pipeline_ranks == ():
            offset, pipelines = 0, []
            for length in self.pipeline_lengths:
                pipelines.append(tuple(range(offset, offset + length)))
                offset += length
            object.__setattr__(self, "pipeline_ranks", tuple(pipelines))
        if (not isinstance(self.pipeline_ranks, tuple)
                or any(not isinstance(ranks, tuple) for ranks in self.pipeline_ranks)
                or tuple(map(len, self.pipeline_ranks)) != self.pipeline_lengths):
            raise ValueError("pipeline ranks must match layout lengths")
        ranks = tuple(rank for pipeline in self.pipeline_ranks for rank in pipeline)
        for rank in ranks:
            _integer("pipeline rank", rank, 0)
        if len(ranks) != len(self.ranks) or set(ranks) != set(range(len(self.ranks))):
            raise ValueError("pipeline ranks must cover each dense rank exactly once")

    @property
    def dp_size(self):
        return len(self.layouts)

    @property
    def pipeline_lengths(self):
        return tuple(map(len, self.layouts))

    @classmethod
    def from_plan(cls, plan, config, *, assignments=None):
        if not plan.feasible:
            raise ValueError("runtime requires a feasible dynamic plan")
        if plan.time.derivation["profile_identity"]["config_hash"] != _hash(asdict(config)):
            raise ValueError("runtime config must match the dynamic plan profile")
        ranks = ()
        if assignments is not None:
            slots = {(slot.pipeline, slot.stage): (slot, worker) for slot, worker in assignments}
            expected = {(p, s) for p, layout in enumerate(plan.layouts) for s in range(len(layout))}
            if (len(slots) != len(assignments) or set(slots) != expected
                    or any(slot.modules != plan.layouts[p][s] or worker not in plan.survivors
                           for (p, s), (slot, worker) in slots.items())):
                raise ValueError("assignments must match dynamic plan slots and survivor identities")
            ranks = tuple(tuple(slots[p, s][1].rank for s in range(length)) for p, length in enumerate(plan.pipeline_lengths))
        return cls(ClusterState(plan.survivors, plan.global_batch_size, plan.generation), config,
                   plan.layouts, plan.pipeline_micro_batches, ranks)


@dataclass(frozen=True)
class ReroutingTopology(_Topology):
    """Initial logical slots, with missing tasks served by existing same-stage peers."""
    state: ClusterState
    config: ModelConfig
    stage_modules: tuple[tuple[str, ...], ...]
    pipeline_micro_batches: tuple[int, ...]
    pipeline_ranks: tuple[tuple[int | None, ...], ...]

    def __post_init__(self):
        expected = ("embedding", *(f"blocks.{i}" for i in range(self.config.num_layers)),
                    "final_norm", "lm_head")
        if (not isinstance(self.stage_modules, tuple) or not self.stage_modules
                or any(not isinstance(stage, tuple) or not stage for stage in self.stage_modules)
                or tuple(name for stage in self.stage_modules for name in stage) != expected):
            raise ValueError("stages must partition every model module once in model order")
        self._validate_initial_state()
        if (not isinstance(self.pipeline_ranks, tuple) or not self.pipeline_ranks
                or any(not isinstance(row, tuple) or len(row) != self.pp_size for row in self.pipeline_ranks)):
            raise ValueError("logical pipeline rank slots must match the unchanged stage layout")
        ranks = tuple(rank for row in self.pipeline_ranks for rank in row if rank is not None)
        for rank in ranks:
            _integer("native rank", rank, 0)
        if len(ranks) != len(self.ranks) or set(ranks) != set(range(len(self.ranks))):
            raise ValueError("native slots must cover each physical rank exactly once; no idle workers")
        if (not isinstance(self.pipeline_micro_batches, tuple)
                or len(self.pipeline_micro_batches) != self.dp_size):
            raise ValueError("pipeline micro-batch counts must match logical pipelines")
        for count in self.pipeline_micro_batches:
            _integer("pipeline micro-batch count", count)
        if sum(self.pipeline_micro_batches) != self.global_micro_batches:
            raise ValueError("pipeline micro-batch counts must preserve the global count")
        if any(all(row[stage] is None for row in self.pipeline_ranks) for stage in range(self.pp_size)):
            raise ValueError("rerouting infeasible: Fi >= Ndp leaves no healthy same-stage peer")

    @property
    def pp_size(self):
        return len(self.stage_modules)

    @property
    def dp_size(self):
        return len(self.pipeline_ranks)

    @property
    def layouts(self):
        return (self.stage_modules,) * self.dp_size

    @property
    def pipeline_lengths(self):
        return (self.pp_size,) * self.dp_size

    @property
    def logical_slots(self):
        return self.dp_size * self.pp_size

    def task_rank(self, pipeline, stage, micro_batch):
        for name, value in (("pipeline", pipeline), ("stage", stage), ("micro_batch", micro_batch)):
            _integer(name, value, 0)
        if pipeline >= self.dp_size or stage >= self.pp_size:
            raise ValueError("task must belong to a logical pipeline/stage")
        if micro_batch >= self.pipeline_micro_batches[pipeline]:
            raise ValueError("micro_batch must belong to its logical pipeline")
        native = self.pipeline_ranks[pipeline][stage]
        if native is not None:
            return native
        # Match DecisionCenter's stable peer ordering; balance all missing slots together.
        peers = sorted((row[stage] for row in self.pipeline_ranks if row[stage] is not None),
                       key=lambda rank: self.ranks[rank].worker_id)
        offset = sum(self.pipeline_micro_batches[p] for p in range(pipeline)
                     if self.pipeline_ranks[p][stage] is None)
        return peers[(offset + micro_batch) % len(peers)]

    @property
    def transfer_edges(self):
        return tuple(sorted({(self.task_rank(p, s, mb), self.task_rank(p, s + 1, mb))
                             for p, count in enumerate(self.pipeline_micro_batches)
                             for s in range(self.pp_size - 1) for mb in range(count)}))


class RuntimeErrorWithAudit(RuntimeError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def _write_reply(connection, directory, rank, kind, worker, **values):
    """Publish a complete metadata file before sending a fixed-size control token."""
    path = Path(directory) / f"rank-{rank}-{kind}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(dict(kind=kind, worker=asdict(worker), **values),
                                    allow_nan=False), encoding="utf-8")
    temporary.replace(path)
    connection.send_bytes(kind.encode("ascii"))


def _synchronize(device):
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


def _warm_pipeline(ranks, rank, group, device, dtype, shape):
    """Connect both directions of every pipeline edge before schedule-dependent P2P."""
    stage = ranks.index(rank)
    peers = ranks[max(0, stage - 1):stage] + ranks[stage + 1:stage + 2]
    return _warm_peers(peers, rank, group, device, dtype, shape)


def _warm_peers(peers, rank, group, device, dtype, shape):
    import torch
    import torch.distributed as dist

    messages = sorted((source, target) for peer in peers
                      for source, target in ((rank, peer), (peer, rank)))
    ops, rows = [], []
    for source, target in messages:
        sending = source == rank
        tensor = torch.full(shape, source + 1 if sending else 0, device=device, dtype=dtype)
        peer = target if sending else source
        ops.append(dist.P2POp(dist.isend if sending else dist.irecv, tensor, peer, group))
        rows.append(dict(action="send" if sending else "recv", peer_rank=peer,
                         device=str(device), shape=list(shape), tensor_bytes=tensor.numel() * tensor.element_size()))
    if ops:
        for work in dist.batch_isend_irecv(ops):
            work.wait()
        _synchronize(device)
    for (source, _), op, row in zip(messages, ops, rows):
        row["value"] = op.tensor.flatten()[0].item()
        if not torch.all(op.tensor == source + 1).item():
            raise RuntimeError("pipeline connection warmup received the wrong peer value")
    return rows


def _exchange(topology, rank, group, batches, send, receive, device, dtype, origin):
    """Pair the current send with the next operation's receive, then drain both."""
    import torch
    import torch.distributed as dist

    ops, rows = [], []
    incoming = None
    start = time.monotonic() - origin
    for action, specification in (("send", send), ("recv", receive)):
        if specification is None:
            continue
        kind, mb, peer, tensor = specification
        if action == "recv":
            tensor = torch.empty((len(batches[mb]), topology.config.sequence_length,
                                  topology.config.hidden_size), device=device, dtype=dtype)
            incoming = tensor
        else:
            tensor = tensor.detach().contiguous()
        ops.append(dist.P2POp(dist.isend if action == "send" else dist.irecv,
                             tensor, peer, group))
        rows.append({"action": action, "kind": kind, "micro_batch": mb,
                     "rank": rank, "peer_rank": peer,
                     "worker_id": topology.ranks[rank].worker_id,
                     "peer_worker_id": topology.ranks[peer].worker_id,
                     "shape": list(tensor.shape), "dtype": str(dtype),
                     "device": str(device), "tensor_bytes": tensor.numel() * tensor.element_size(),
                     "start_s": start})
    if ops:
        for request in dist.batch_isend_irecv(ops):
            request.wait()
        _synchronize(device)
    end = time.monotonic() - origin
    return incoming, [dict(row, end_s=end) for row in rows]


def _reduce_gradients(topology, rank, model, owner_groups, origin):
    """Drain each owner color, then normalize each complete gradient once by B."""
    import torch.distributed as dist

    device = next(model.parameters()).device
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    module_parameters = {module: sorted((name, p) for name, p in parameters.items()
                                        if name.startswith(module + ".")) for module in model.module_ids}
    owners = topology.module_owners
    synced, allreduces = [], []
    # All pipelines finish their P2P before any rank changes collective groups.
    dist.barrier()
    _synchronize(device)
    for color, modules in enumerate(topology.synchronization_rounds):
        pending = []
        for module in modules:
            if rank not in owners[module]:
                continue
            for name, parameter in module_parameters[module]:
                if parameter.grad is None:
                    raise RuntimeError(f"missing trainable gradient: {name}")
                launched = time.monotonic() - origin
                work = dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM,
                                       group=owner_groups[module], async_op=True)
                pending.append((work, name, parameter, module, launched))
        # Each device owns at most one module in a color and uses one communicator.
        for work, *_ in pending:
            work.wait()
        _synchronize(device)
        completed = time.monotonic() - origin
        for _, name, parameter, module, launched in pending:
            parameter.grad.div_(topology.config.global_batch_size)
            synced.append(name)
            allreduces.append(dict(parameter=name, module=module, color=color,
                                   owner_ranks=dist.get_process_group_ranks(owner_groups[module]),
                                   device=str(device), backend=dist.get_backend(),
                                   tensor_bytes=parameter.grad.numel() * parameter.grad.element_size(),
                                   async_op=True, reduction="SUM", divisor=topology.config.global_batch_size,
                                   start_s=launched, end_s=completed))
        dist.barrier()
        _synchronize(device)
    if set(synced) != set(parameters) or len(synced) != len(parameters):
        raise RuntimeError("owner rounds must synchronize every trainable parameter exactly once")
    return synced, allreduces


def _train_worker_step(topology, rank, model, optimizer, pp_group, owner_groups,
                       state, profiler, origin, capture_path):
    if isinstance(topology, ReroutingTopology):
        from .rerouting import train_rerouted_step
        return train_rerouted_step(topology, rank, model, optimizer, pp_group, owner_groups,
                                   state, origin, capture_path)
    import torch.distributed as dist
    from .data import make_batch
    from .global_loss import micro_batch_loss_sum

    pipeline, stage = topology.location(rank)
    pipeline_ranks = topology.pipeline_ranks[pipeline]
    depth = len(pipeline_ranks)
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    batches = topology.micro_batches(state, pipeline)
    queue = build_1f1b_schedule(depth, len(batches), pipeline=pipeline)[stage]
    step_id = state.committed_global_step + 1
    optimizer.zero_grad(set_to_none=True)
    dist.barrier()
    _synchronize(device)
    start = time.monotonic()
    trace, communication, graphs, losses = [], [], {}, {}
    incoming, rows = _exchange(topology, rank, pp_group, batches, None,
                              ("activation", 0, pipeline_ranks[stage - 1], None) if stage else None,
                              device, dtype, origin)
    communication.extend(rows)
    with profiler.step(step_id) if profiler else nullcontext():
        for index, op in enumerate(queue):
            left = time.monotonic() - origin
            scope = (profiler.operation(op.kind, op.micro_batch, pipeline=pipeline,
                                        stage=stage, phase=op.phase) if profiler else nullcontext())
            mb = op.micro_batch
            with scope:
                if op.kind == "forward":
                    batch = (make_batch(batches[mb], model.config, device=device)
                             if stage in (0, depth - 1) else None)
                    inputs = batch.inputs if stage == 0 else incoming.requires_grad_()
                    output = model(inputs)
                    if stage == depth - 1:
                        loss = micro_batch_loss_sum(output, batch.targets)
                        losses[mb] = loss.detach()
                        output = loss
                    graphs[mb] = inputs, output
                    send = (("activation", mb, pipeline_ranks[stage + 1], output)
                            if stage + 1 < depth else None)
                else:
                    inputs, output = graphs.pop(mb)
                    if stage == depth - 1:
                        output.backward()
                    else:
                        output.backward(incoming)
                    send = ("gradient", mb, pipeline_ranks[stage - 1], inputs.grad) if stage else None
            _synchronize(device)
            trace.append({"pipeline": pipeline, "stage": stage, "micro_batch": mb,
                          "kind": op.kind, "phase": op.phase, "start_s": left,
                          "end_s": time.monotonic() - origin})
            receive = None
            if index + 1 < len(queue):
                following = queue[index + 1]
                if following.kind == "forward" and stage:
                    receive = "activation", following.micro_batch, pipeline_ranks[stage - 1], None
                if following.kind == "backward" and stage + 1 < depth:
                    receive = "gradient", following.micro_batch, pipeline_ranks[stage + 1], None
            incoming, rows = _exchange(topology, rank, pp_group, batches, send, receive,
                                       device, dtype, origin)
            communication.extend(rows)
        if graphs:
            raise RuntimeError("pipeline left in-flight autograd graphs")
        pipeline_s = time.monotonic() - start
        count = sum(len(batches[mb]) for mb in losses)
        synced, allreduces, global_loss = _complete_update(topology, rank, model, optimizer, owner_groups,
                                                          tuple(losses.values()), count, origin)
    completed_s = time.monotonic() - origin
    report = {"pid": os.getpid(), "pipeline": pipeline, "stage": stage,
              "sample_ids": [sample_id for batch in batches for sample_id in batch],
              "micro_batches": [list(batch) for batch in batches], "trace": trace,
              "communication": communication, "synchronized_parameters": synced,
              "allreduces": allreduces, "synchronization_rounds": topology.synchronization_rounds,
              "loss_batches": [{"sample_ids": list(batches[mb]), "loss_sum": loss.item()}
                               for mb, loss in losses.items()],
              "loss_global_sum": global_loss[0].item(), "global_sample_count": int(global_loss[1].item()),
              "pipeline_wall_time_s": pipeline_s, "training_wall_time_s": completed_s - (start - origin),
              "optimizer_completed_s": completed_s, "device": str(device),
              "backend": dist.get_backend(), "profile_step": profiler.steps[-1] if profiler else None}
    _capture_worker_state(model, optimizer, capture_path)
    return report


def _complete_update(topology, rank, model, optimizer, owner_groups, losses, sample_count, origin):
    """Both execution policies share owner SUM, one normalization and AdamW."""
    import torch
    import torch.distributed as dist

    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    synced, allreduces = _reduce_gradients(topology, rank, model, owner_groups, origin)
    loss_sum = torch.stack(losses).sum() if losses else torch.zeros((), device=device, dtype=dtype)
    global_loss = torch.stack((loss_sum.to(torch.float64),
                               torch.tensor(sample_count, device=device, dtype=torch.float64)))
    dist.all_reduce(global_loss, op=dist.ReduceOp.SUM)
    if global_loss[1].item() != topology.config.global_batch_size:
        raise RuntimeError("global sample count differs from fixed B")
    optimizer.step()
    _synchronize(device)
    return synced, allreduces, global_loss


def _capture_worker_state(model, optimizer, capture_path):
    # Test evidence is written only after the update; it is never a runtime state source.
    if capture_path is not None:
        import torch
        parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
        torch.save({"parameters": {name: p.detach().cpu().clone() for name, p in parameters.items()},
                    "gradients": {name: p.grad.detach().cpu().clone() for name, p in parameters.items()},
                    "optimizer_state": {name: {key: value.detach().cpu().clone()
                                               for key, value in optimizer.state[p].items()}
                                        for name, p in parameters.items()}}, capture_path)


def _runtime_worker(topology, rank, device_kind, backend, dtype_name, rendezvous_file,
                    timeout_s, connection, origin, lr, weight_decay, behavior, directory, capture_state):
    import torch
    import torch.distributed as dist
    from .model import build_initial_stage
    from .profiler import Profiler

    groups = []
    identity = topology.ranks[rank]
    try:
        torch.set_num_threads(1)
        if device_kind == "cuda":
            torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}" if device_kind == "cuda" else "cpu")
        dtype = getattr(torch, dtype_name)
        timeout = timedelta(seconds=timeout_s)
        dist.init_process_group(backend, store=dist.FileStore(rendezvous_file, len(topology.ranks)),
                                rank=rank, world_size=len(topology.ranks), timeout=timeout,
                                device_id=device if device_kind == "cuda" else None)
        pipeline, stage = topology.location(rank)
        pp_group, owner_groups = None, {}
        # Every rank creates and warms each group in the same order, before subset P2P.
        owners = topology.module_owners
        rerouting = isinstance(topology, ReroutingTopology)
        specifications = [tuple(range(len(topology.ranks)))] if rerouting else list(topology.pipeline_ranks)
        pp_group_count = len(specifications)
        specifications += list(dict.fromkeys(owners.values()))
        p2p_warmup = []
        for index, ranks in enumerate(specifications):
            group = dist.new_group(list(ranks), timeout=timeout, backend=backend)
            if rank in ranks:
                groups.append(group)
                dist.all_reduce(torch.zeros(1, device=device, dtype=dtype), group=group)
                _synchronize(device)
                if index == (0 if rerouting else pipeline):
                    pp_group = group
                    config = topology.config
                    shape = (min(config.micro_batch_size, config.global_batch_size),
                             config.sequence_length, config.hidden_size)
                    if rerouting:
                        peers = sorted({target if source == rank else source
                                        for source, target in topology.transfer_edges if rank in (source, target)})
                        p2p_warmup = _warm_peers(peers, rank, group, device, dtype, shape)
                    else:
                        p2p_warmup = _warm_pipeline(ranks, rank, group, device, dtype, shape)
                if index >= pp_group_count:
                    owner_groups.update({module: group for module, peers in owners.items() if peers == ranks})
            dist.barrier()
            _synchronize(device)
        model = build_initial_stage(topology.config, topology.layouts[pipeline][stage], device=device, dtype=dtype)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                     lr=lr, weight_decay=weight_decay, amsgrad=False)
        _write_reply(connection, directory, rank, "ready", identity, pid=os.getpid(),
                     parameter_names=[name for name, p in model.named_parameters() if p.requires_grad],
                     groups=specifications, module_owners=owners,
                     parameter_owners={name: dist.get_process_group_ranks(owner_groups[module])
                                       for name, p in model.named_parameters() if p.requires_grad
                                       for module in topology.layouts[pipeline][stage] if name.startswith(module + ".")},
                     synchronization_rounds=topology.synchronization_rounds, p2p_warmup=p2p_warmup,
                     float32_matmul_precision=torch.get_float32_matmul_precision(),
                     device=str(device), backend=backend,
                     **(dict(initial_topology=dict(logical_slots=topology.logical_slots,
                             physical_workers=len(topology.ranks), pipeline_ranks=topology.pipeline_ranks,
                             layouts=topology.layouts, pipeline_micro_batches=topology.pipeline_micro_batches))
                        if rerouting else {}))
        committed = topology.state
        profiler = None
        while True:
            command = connection.recv_bytes(maxlength=7)
            if command == b"stop":
                break
            if command == b"profile":
                _write_reply(connection, directory, rank, "profile", identity,
                             step_id=committed.committed_global_step,
                             profile=profiler.snapshot() if profiler else None)
                continue
            if command != b"step":
                raise RuntimeError("worker requires the current committed state at a safe point")
            if rank == 0 and behavior == "error":
                raise RuntimeError("injected runtime worker failure")
            if rank == 0 and behavior == "hang":
                time.sleep(3600)
            step_id = committed.committed_global_step + 1
            capture_path = (str(Path(directory) / f"step-{step_id}-rank-{rank}.pt")
                            if capture_state else None)
            report = _train_worker_step(topology, rank, model, optimizer, pp_group, owner_groups,
                                        committed, profiler, origin, capture_path)
            _write_reply(connection, directory, rank, "ack", identity, step_id=step_id, **report)
            if connection.recv_bytes(maxlength=7) != b"commit":
                raise RuntimeError("worker requires controller commit after optimizer acknowledgement")
            committed = replace(committed, committed_global_step=step_id)
            if profiler is None and not rerouting:
                options = (dict(pp_size=topology.pp_size) if isinstance(topology, SymmetricTopology)
                           else dict(pp_size=max(topology.pipeline_lengths), pipeline_lengths=topology.pipeline_lengths))
                profiler = Profiler(model, optimizer, dp_size=topology.dp_size, rank=rank, **options)
            _write_reply(connection, directory, rank, "safe", identity, step_id=committed.committed_global_step)
    except BaseException:
        try:
            _write_reply(connection, directory, rank, "error", identity, error=traceback.format_exc())
        except (OSError, EOFError):
            pass
        raise
    finally:
        try:
            for group in reversed(groups):
                dist.destroy_process_group(group)
            if dist.is_initialized():
                dist.destroy_process_group()
        finally:
            connection.close()


def compare_runtime_profile(reports, topology):
    """Estimate compute from measured scopes; report the full measured step separately."""
    from .estimators import estimate_operation_time, estimate_symmetric_time

    if any(report["profile_step"] is None for report in reports):
        return None  # The first real update warms AdamW; no synthetic state or timings.
    symmetric = isinstance(topology, SymmetricTopology)
    estimates, forwards, backwards = [], [], []
    for pipeline in range(topology.dp_size):
        durations = {}
        for report in reports:
            if report["pipeline"] != pipeline:
                continue
            trace = report["profile_step"]["trace"]
            if symmetric:
                for kind, samples in (("forward", forwards), ("backward", backwards)):
                    samples.append(sum(row["end_s"] - row["start_s"] for row in trace if row["kind"] == kind)
                                   / len(report["micro_batches"]))
            durations.update({(pipeline, row["stage"], row["micro_batch"], row["kind"]):
                              row["end_s"] - row["start_s"] for row in trace})
        estimates.append(estimate_operation_time(durations, num_stages=topology.pipeline_lengths[pipeline],
                         pipeline_micro_batches=topology.pipeline_micro_batches[pipeline],
                         pipeline=pipeline))
    uniform = (estimate_symmetric_time(num_stages=topology.pp_size,
                                     global_micro_batches=topology.global_micro_batches, dp_size=topology.dp_size,
                                     forward_s=max(forwards), backward_s=max(backwards))
               if symmetric else None)
    return {"equation9": asdict(uniform) if uniform else None,
            "equation10": {"pipeline_lengths": topology.pipeline_lengths,
                           "pipeline_micro_batches": topology.pipeline_micro_batches,
                           "global_micro_batches": topology.global_micro_batches},
            "equation11_pipelines": [asdict(item) for item in estimates],
            "estimated_compute_time_s": max(item.step_time_s for item in estimates),
            "measured_pipeline_time_s": max(report["pipeline_wall_time_s"] for report in reports),
            "measured_training_time_s": max(report["training_wall_time_s"] for report in reports),
            "boundary": "measured operation scopes estimate computation; full step includes P2P, SUM, AdamW and profiling"}


class SymmetricRuntime:
    """Controller holds identities and metadata, with no initial model backup."""

    def __init__(self, topology: SymmetricTopology | DynamicTopology | ReroutingTopology, *, device="cpu", dtype="float64",
                 timeout_s=120, artifact_dir="artifacts/test-results", capture_state=False,
                 lr=1e-3, weight_decay=.01, behavior="normal"):
        _finite("timeout_s", timeout_s, positive=True)
        _finite("lr", lr, positive=True)
        _finite("weight_decay", weight_decay)
        if dtype not in ("float64", "float32"):
            raise ValueError("runtime dtype must be float64 or float32")
        if behavior not in ("normal", "error", "hang"):
            raise ValueError("invalid runtime test behavior")
        if Path(artifact_dir).is_absolute():
            raise ValueError("artifact_dir must be relative to the project working directory")
        self.topology, self.device, self.dtype = topology, device, dtype
        self.backend = validate_device(device, len(topology.ranks))
        self.timeout_s, self.capture_state = timeout_s, capture_state
        self.lr, self.weight_decay, self.behavior = lr, weight_decay, behavior
        self.root = Path(artifact_dir)
        self.commit = StepCommit(topology.state)
        self.processes, self.connections, self.ready = [], [], []
        self.steps, self.audit = [], None
        self._directory = None
        self._closed = False

    @property
    def state(self):
        return self.commit.state

    @contextmanager
    def _abort_on_error(self):
        try:
            yield
        except BaseException as exc:
            self.close(str(exc))
            raise RuntimeErrorWithAudit(str(exc), self.audit) from exc

    def _check_deadline(self, kind, deadline):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"runtime {kind} exceeded hard timeout {self.timeout_s}s")

    def _collect(self, kind, deadline):
        pending = set(range(len(self.connections)))
        result = {}
        while pending:
            self._check_deadline(kind, deadline)
            if any(p.exitcode not in (None, 0) for p in self.processes):
                raise RuntimeError("runtime worker exited abnormally")
            pipes = [self.connections[rank] for rank in sorted(pending)]
            for connection in wait(pipes, timeout=min(.1, max(0, deadline - time.monotonic()))):
                rank = self.connections.index(connection)
                token = connection.recv_bytes(maxlength=7)
                received_s = time.monotonic() - self.origin
                if token not in (kind.encode("ascii"), b"error"):
                    raise RuntimeError("runtime reply protocol mismatch")
                message = json.loads((self.directory / f"rank-{rank}-{token.decode('ascii')}.json").read_text(encoding="utf-8"))
                worker = WorkerIdentity(**message["worker"])
                if message.get("kind") != token.decode("ascii") or worker != self.topology.ranks[rank]:
                    raise RuntimeError("runtime reply identity or protocol mismatch")
                if token == b"error":
                    raise RuntimeError(message["error"])
                if kind != "ready":
                    _integer("reply step_id", message.get("step_id"), 0)
                    if message["step_id"] != self.state.committed_global_step + (kind == "ack"):
                        raise RuntimeError("runtime reply is for the wrong global step")
                message.update(worker=worker, received_s=received_s)
                result[rank] = message
                pending.remove(rank)
        self._check_deadline(kind, deadline)
        return [result[rank] for rank in sorted(result)]

    def __enter__(self):
        if self._closed or self._directory is not None:
            raise RuntimeError("runtime can only be opened once")
        self.origin = time.monotonic()
        with self._abort_on_error():
            self.root.mkdir(parents=True, exist_ok=True)
            self._directory = tempfile.TemporaryDirectory(prefix="runtime-", dir=self.root)
            self.directory = Path(self._directory.name)
            self.rendezvous_file = self.directory / "store"
            context = mp.get_context("spawn")
            for rank in range(len(self.topology.ranks)):
                parent, child = context.Pipe()
                self.connections.append(parent)
                process = context.Process(target=_runtime_worker, args=(self.topology, rank, self.device,
                    self.backend, self.dtype, str(self.rendezvous_file), self.timeout_s, child, self.origin,
                    self.lr, self.weight_decay, self.behavior, str(self.directory), self.capture_state))
                try:
                    process.start()
                    self.processes.append(process)
                finally:
                    child.close()
            self.ready = self._collect("ready", self.origin + self.timeout_s)
            if (len({row["pid"] for row in self.ready}) != len(self.topology.ranks)
                    or any(row["pid"] != process.pid for row, process in zip(self.ready, self.processes))):
                raise RuntimeError("runtime workers must have distinct real PIDs")
            return self

    def train_step(self):
        if not self.ready or self._closed:
            raise RuntimeError("runtime must be open at a safe point")
        with self._abort_on_error():
            deadline = time.monotonic() + self.timeout_s
            step_id = self.state.committed_global_step + 1
            sample_ids = next_sample_ids(self.state)
            for connection in self.connections:
                connection.send_bytes(b"step")
            replies = self._collect("ack", deadline)
            reports = [dict({key: value for key, value in reply.items() if key not in ("kind", "received_s")},
                            worker=asdict(reply["worker"])) for reply in replies]
            if isinstance(self.topology, ReroutingTopology):
                from .rerouting import validate_routing_reports
                validate_routing_reports(self.topology, self.state, reports)
            accounting = GlobalBatchAccounting(self.state, expected_owners={
                row["worker"].worker_id: set(row["parameter_names"]) for row in self.ready})
            for report in reports:
                for batch in report["loss_batches"]:
                    accounting.add_micro_batch(batch["sample_ids"], batch["loss_sum"], sample_count=len(batch["sample_ids"]))
            loss = accounting.report()
            rtol, atol = ((1e-4, 1e-6) if self.dtype == "float32" else
                          (1e-7, 1e-9) if self.device == "cuda" else (1e-8, 1e-10))
            if any(report["global_sample_count"] != loss.global_sample_count
                   or not math.isclose(report["loss_global_sum"], loss.loss_global_sum, rel_tol=rtol, abs_tol=atol)
                   for report in reports):
                raise RuntimeError("worker global SUM does not match sample accounting")
            comparison = compare_runtime_profile(reports, self.topology)
            if self.capture_state:
                import torch
                snapshots = [torch.load(self.directory / f"step-{step_id}-rank-{rank}.pt", weights_only=True)
                             for rank in range(len(self.topology.ranks))]
            self._check_deadline("step", deadline)
            commits = []
            for reply in replies:
                before = self.state.committed_global_step
                advanced = self.commit.acknowledge(reply["worker"], reply["step_id"])
                commits.append({"worker_id": reply["worker"].worker_id, "before": before,
                                "after": self.state.committed_global_step, "advanced": advanced,
                                "ack_received_s": reply["received_s"], "commit_s": time.monotonic() - self.origin})
            for connection in self.connections:
                connection.send_bytes(b"commit")
            safe = self._collect("safe", deadline)
            result = {"step_id": step_id, "sample_ids": list(sample_ids),
                      "loss_global_sum": loss.loss_global_sum, "global_sample_count": loss.global_sample_count,
                      "loss_global_mean": loss.loss_global_mean, "commits": commits, "reports": reports,
                      "profile_comparison": comparison,
                      "safe_worker_ids": [reply["worker"].worker_id for reply in safe]}
            if self.capture_state:
                result["snapshots"] = snapshots
            self.steps.append(result)
            return result

    def snapshot_profiles(self):
        """Export cumulative versioned profiles only on explicit safe-point requests."""
        if not self.ready or self._closed:
            raise RuntimeError("runtime must be open at a safe point")
        with self._abort_on_error():
            deadline = time.monotonic() + self.timeout_s
            for connection in self.connections:
                connection.send_bytes(b"profile")
            return [reply["profile"] for reply in self._collect("profile", deadline)]

    def close(self, error=None):
        if self._closed:
            return
        self._closed = True
        if self._directory is None:
            self.audit = {"clean": True, "workers": [], "committed_global_step": self.state.committed_global_step}
            return
        close_failed = False
        try:
            if error is None:
                for connection in self.connections:
                    try:
                        connection.send_bytes(b"stop")
                    except (OSError, EOFError):
                        pass
                deadline = time.monotonic() + min(self.timeout_s, 10)
                for process in self.processes:
                    process.join(max(0, deadline - time.monotonic()))
                if any(process.is_alive() or process.exitcode != 0 for process in self.processes):
                    error = "runtime shutdown failed or exceeded hard timeout"
                    close_failed = True
        finally:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
            deadline = time.monotonic() + 5
            for process in self.processes:
                process.join(max(0, deadline - time.monotonic()))
            for process in self.processes:
                if process.is_alive():
                    process.kill()
            deadline = time.monotonic() + 5
            for process in self.processes:
                process.join(max(0, deadline - time.monotonic()))
            workers = [{"pid": p.pid, "exitcode": p.exitcode, "alive": p.is_alive()} for p in self.processes]
            for connection in self.connections:
                connection.close()
            for process in self.processes:
                if not process.is_alive():
                    process.close()
            self._directory.cleanup()
        leaked = sorted(worker["pid"] for worker in workers if worker["alive"])
        self.audit = {"workers": workers, "leaked_pids": leaked,
                      "rendezvous_backend": "FileStore", "rendezvous_port": None,
                      "rendezvous_file": str(self.rendezvous_file),
                      "rendezvous_file_removed": not self.rendezvous_file.exists(),
                      "rendezvous_dir": str(self.directory), "rendezvous_removed": not self.directory.exists(),
                      "committed_global_step": self.state.committed_global_step, "error": error,
                      "device": self.device, "backend": self.backend}
        self.audit["clean"] = not leaked and not self.directory.exists()
        if not self.audit["clean"]:
            error = self.audit["error"] = f"runtime resource cleanup failed; original error: {error}"
            close_failed = True
        serializable = [{key: value for key, value in step.items() if key != "snapshots"} for step in self.steps]
        self.report_path = self.root / f"{self.directory.name}-{self.device}-{self.dtype}-{self.behavior}.json"
        ready = [dict(row, worker=asdict(row["worker"])) for row in self.ready]
        self.report_path.write_text(json.dumps({"audit": self.audit, "ready": ready, "steps": serializable}, indent=2,
                                              allow_nan=False), encoding="utf-8")
        if close_failed:
            raise RuntimeErrorWithAudit(error, self.audit)

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close(str(exc_value) if exc_value else None)
