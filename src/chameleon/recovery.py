"""Safe-point recovery using live survivor tensors and tensor-free module structure."""

from copy import deepcopy
from dataclasses import asdict
import hashlib

from .contracts import UnrecoverableStateError
from .restorer import MigrationManifest, migration_cost_matrix, synchronization_rounds
from .hungarian import hungarian
from .state_sources import ADAMW_FIELDS, adamw_inventory
from .profiler import _hash


def validate_manifest(manifest, sources):
    """Validate the entire transfer before any process group or tensor is changed."""
    if not isinstance(manifest, MigrationManifest):
        raise ValueError("dynamic recovery requires a migration manifest")
    plan, state = manifest.plan, sources.state
    if (not plan.feasible or set(plan.survivors) != set(state.workers)
            or plan.generation != state.generation or plan.global_batch_size != state.global_batch_size):
        raise ValueError("manifest plan differs from the live survivors")
    assignments = manifest.assignments
    slots = tuple(slot for slot, _ in assignments)
    expected_slots = {(p, s) for p, layout in enumerate(plan.layouts) for s in range(len(layout))}
    if (len(assignments) != len(state.workers)
            or {(slot.pipeline, slot.stage) for slot in slots} != expected_slots
            or {worker for _, worker in assignments} != set(state.workers)
            or any(slot.modules != plan.layouts[slot.pipeline][slot.stage] for slot in slots)):
        raise ValueError("manifest assignments must cover every target and survivor once")
    model_order = tuple(m for stage in plan.layouts[0] for m in stage)
    if (set(model_order) != {t.module_id for t in sources.required}
            or any(tuple(m for stage in layout for m in stage) != model_order for layout in plan.layouts)):
        raise ValueError("manifest must preserve the complete ordered model")
    expected = {(worker, tensor, slot) for slot, worker in assignments
                for tensor in sources.required if tensor.module_id in slot.modules}
    if (len(manifest.actions) != len(expected)
            or {(a.destination, a.tensor, a.slot) for a in manifest.actions} != expected):
        raise UnrecoverableStateError("manifest lacks complete target parameter/AdamW tensors")
    local = {i.worker: set(i.tensors) for i in sources.inventories}
    for action in manifest.actions:
        if action.source not in local or action.tensor not in local[action.source]:
            raise UnrecoverableStateError("transfer source is not a live survivor tensor")
        if action.tensor in local[action.destination]:
            if not action.retained:
                raise ValueError("local intersection must be retained without transfer")
        elif action.source not in sources.sources_for(action.tensor):
            raise UnrecoverableStateError("transfer requires an intact survivor module source")
    transfers = tuple(a for row in manifest.migration_rounds for a in row)
    expected_transfers = tuple(a for a in manifest.actions if not a.retained)
    if len(transfers) != len(expected_transfers) or set(transfers) != set(expected_transfers):
        raise ValueError("migration rounds must cover each missing tensor exactly once")
    for row in manifest.migration_rounds:
        endpoints = [w for a in row for w in (a.source, a.destination)]
        if len(endpoints) != len(set(endpoints)):
            raise ValueError("migration round has conflicting workers")
    costs = migration_cost_matrix(sources, slots)
    selected_cost = sum(costs[i][next(j for j, (_, w) in enumerate(assignments) if w == inventory.worker)]
                        for i, inventory in enumerate(sources.inventories))
    if (costs != manifest.cost_matrix or selected_cost != hungarian(costs).total_cost
            or manifest.migration_bytes != selected_cost
            or selected_cost != sum(a.tensor.nbytes for a in expected_transfers)):
        raise ValueError("manifest must use the Hungarian minimum missing-state byte assignment")
    held = {(i.worker, t) for i in sources.inventories for t in i.tensors}
    targets = {(a.destination, a.tensor) for a in manifest.actions}
    if (set(manifest.held_sources) != held or len(manifest.held_sources) != len(held)
            or set(manifest.release_after_ack) != held - targets
            or len(manifest.release_after_ack) != len(held - targets)):
        raise ValueError("manifest source lifetime differs from live state")
    identity = "migration-" + _hash({"plan_id": plan.plan_id, "committed_global_step": state.committed_global_step,
                                     "actions": tuple(asdict(a) for a in manifest.actions)})
    owners = {module: tuple(w.worker_id for slot, w in assignments if module in slot.modules)
              for module in model_order}
    if manifest.manifest_id != identity or manifest.synchronization_rounds != synchronization_rounds(owners):
        raise ValueError("manifest identity/communication rounds differ from the committed state")


def live_tensors(model, optimizer):
    return {(name, kind): parameter if kind == "parameter" else optimizer.state[parameter][kind]
            for name, parameter in model.named_parameters() if parameter.requires_grad
            for kind in ADAMW_FIELDS}


def tensor_digest(tensor):
    """Hash exact tensor bytes, independently of torch.save container metadata."""
    import torch
    raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def inspect_live_state(model, optimizer, committed_step):
    inventory = adamw_inventory(model, optimizer, committed_global_step=committed_step)
    tensors = live_tensors(model, optimizer)
    return dict(tensors=[asdict(t) for t in inventory],
                hashes=[dict(key=list(t.key), digest=tensor_digest(tensors[t.key])) for t in inventory],
                parameter_steps=[dict(parameter_name=name, step=state["step"].item(),
                                      exp_avg_nonzero=state["exp_avg"].count_nonzero().item(),
                                      exp_avg_sq_nonzero=state["exp_avg_sq"].count_nonzero().item())
                                 for name, p in model.named_parameters() if p.requires_grad
                                 for state in (optimizer.state[p],)])


def rebuild_stage(structure, modules, old_model, old_optimizer, target_tensors):
    """Reuse local Parameter objects; allocate missing ones from received tensors only."""
    import torch
    from .model import PipelineStage

    model = PipelineStage(deepcopy(structure), modules)
    old_parameters = dict(old_model.named_parameters())
    for name, template in tuple(model.named_parameters()):
        value = target_tensors[name, "parameter"]
        parameter = (old_parameters[name] if name in old_parameters and value is old_parameters[name]
                     else torch.nn.Parameter(value, requires_grad=template.requires_grad))
        parent_name, leaf = name.rsplit(".", 1)
        model.get_submodule(parent_name)._parameters[leaf] = parameter
    parameter_group = dict(old_optimizer.param_groups[0])
    parameter_group["params"] = list(model.parameters())
    optimizer = torch.optim.AdamW([parameter_group])
    for name, parameter in model.named_parameters():
        optimizer.state[parameter] = {kind: target_tensors[name, kind] for kind in ADAMW_FIELDS[1:]}
    return model, optimizer


def transfer_state(manifest, identity, new_ranks, structure, model, optimizer, device, committed_step):
    """DSATUR P2P rounds over the survivor group; source memory remains held by caller."""
    import torch
    import torch.distributed as dist

    adamw_inventory(model, optimizer, committed_global_step=committed_step)
    old = live_tensors(model, optimizer)
    before = {key: tensor_digest(value) for key, value in old.items()}
    targets = {a.tensor.key: old[a.tensor.key] for a in manifest.actions
               if a.destination == identity and a.retained}
    sends, receives = [], []
    for row in manifest.migration_rounds:
        pending, buffers = [], []
        for action in row:
            tensor = action.tensor
            if action.source == identity:
                value = old[tensor.key].detach().to(device=device).contiguous()
                digest = torch.tensor(list(bytes.fromhex(before[tensor.key])), device=device, dtype=torch.uint8)
                peer = new_ranks[action.destination.worker_id]
                pending += [dist.P2POp(dist.isend, value, peer), dist.P2POp(dist.isend, digest, peer)]
                buffers.append((action, value, digest, False))
            elif action.destination == identity:
                value = torch.empty(tensor.shape, dtype=getattr(torch, tensor.dtype.removeprefix("torch.")), device=device)
                digest = torch.empty(32, device=device, dtype=torch.uint8)
                peer = new_ranks[action.source.worker_id]
                pending += [dist.P2POp(dist.irecv, value, peer), dist.P2POp(dist.irecv, digest, peer)]
                buffers.append((action, value, digest, True))
        if pending:
            for request in dist.batch_isend_irecv(pending):
                request.wait()
        for action, value, digest, received in buffers:
            actual = tensor_digest(value)
            if actual != bytes(digest.cpu().tolist()).hex():
                raise UnrecoverableStateError("P2P state tensor digest mismatch")
            record = dict(key=list(action.tensor.key), source=action.source.worker_id,
                          destination=action.destination.worker_id, digest=actual,
                          tensor_bytes=action.tensor.nbytes)
            if received:
                # Standard non-capturable AdamW keeps its step scalar on CPU, including NCCL training.
                targets[action.tensor.key] = value.cpu() if action.tensor.kind == "step" else value
                receives.append(record)
            else:
                sends.append(record)
        dist.barrier()
    slot = next(slot for slot, worker in manifest.assignments if worker == identity)
    recovered, recovered_optimizer = rebuild_stage(structure, slot.modules, model, optimizer, targets)
    inventory = adamw_inventory(recovered, recovered_optimizer, committed_global_step=committed_step)
    expected = {a.tensor for a in manifest.actions if a.destination == identity}
    if set(inventory) != expected:
        raise UnrecoverableStateError("target has incomplete recovered state")
    loaded = live_tensors(recovered, recovered_optimizer)
    retained = [dict(key=list(a.tensor.key), digest=before[a.tensor.key],
                     same_object=loaded[a.tensor.key] is old[a.tensor.key])
                for a in manifest.actions if a.destination == identity and a.retained]
    held = []
    for key, value in old.items():
        digest = tensor_digest(value)
        held.append(dict(key=list(key), digest=digest, unchanged=digest == before[key]))
    if any(not r["same_object"] for r in retained) or any(not r["unchanged"] for r in held):
        raise UnrecoverableStateError("source state changed before target ACK")
    return recovered, recovered_optimizer, dict(manifest_id=manifest.manifest_id, sends=sends, receives=receives,
        retained=retained, held_before_ack=held, **inspect_live_state(recovered, recovered_optimizer, committed_step))


def validate_transfer_reports(manifest, reports):
    """Require complete target ACKs and matching live-source/target digests."""
    by_worker = {r["worker"].worker_id: r for r in reports}
    if len(by_worker) != len(reports) or set(by_worker) != {w.worker_id for _, w in manifest.assignments}:
        raise ValueError("transfer ACKs must cover every survivor exactly once")
    sends, receives = {}, {}
    from .state_sources import StateTensor
    for worker_id, report in by_worker.items():
        if report["manifest_id"] != manifest.manifest_id:
            raise ValueError("transfer ACK manifest mismatch")
        expected = {a.tensor.key for a in manifest.actions if a.destination.worker_id == worker_id}
        metadata = tuple(StateTensor(**dict(t, shape=tuple(t["shape"]))) for t in report["tensors"])
        needed = {a.tensor for a in manifest.actions if a.destination.worker_id == worker_id}
        if len(metadata) != len(needed) or set(metadata) != needed:
            raise UnrecoverableStateError("target parameter/AdamW metadata ACK mismatch")
        hashes = report["hashes"]
        if len(hashes) != len(expected) or {tuple(r["key"]) for r in hashes} != expected:
            raise UnrecoverableStateError("incomplete target hash ACK")
        retained = {a.tensor.key for a in manifest.actions if a.destination.worker_id == worker_id and a.retained}
        if (len(report["retained"]) != len(retained) or {tuple(r["key"]) for r in report["retained"]} != retained
                or any(not r["same_object"] for r in report["retained"])):
            raise ValueError("target failed to retain local tensor intersection")
        held = {t.key for w, t in manifest.held_sources if w.worker_id == worker_id}
        if (len(report["held_before_ack"]) != len(held)
                or {tuple(r["key"]) for r in report["held_before_ack"]} != held
                or any(not r["unchanged"] for r in report["held_before_ack"])):
            raise UnrecoverableStateError("old source tensors were released or changed before ACK")
        for field, result in (("sends", sends), ("receives", receives)):
            for row in report[field]:
                key = row["source"], row["destination"], tuple(row["key"])
                if key in result or row["source" if field == "sends" else "destination"] != worker_id:
                    raise ValueError("duplicate or misattributed P2P audit")
                result[key] = row
    transfers = {(a.source.worker_id, a.destination.worker_id, a.tensor.key): a
                 for a in manifest.actions if not a.retained}
    if sends.keys() != transfers.keys() or receives.keys() != transfers.keys():
        raise UnrecoverableStateError("missing or unexpected P2P state transfers")
    for key, action in transfers.items():
        if sends[key] != receives[key] or sends[key]["tensor_bytes"] != action.tensor.nbytes:
            raise UnrecoverableStateError("source and target transfer audit mismatch")
    for action in manifest.actions:
        destination = by_worker[action.destination.worker_id]
        digest = next(r["digest"] for r in destination["hashes"] if tuple(r["key"]) == action.tensor.key)
        source = by_worker[action.source.worker_id]
        original = next(r["digest"] for r in source["held_before_ack"] if tuple(r["key"]) == action.tensor.key)
        if digest != original:
            raise UnrecoverableStateError("target state differs from held survivor source")
