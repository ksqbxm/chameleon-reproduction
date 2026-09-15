"""Survivor-only state metadata; tensor values stay in worker memory."""

from dataclasses import dataclass

from .contracts import ClusterState, UnrecoverableStateError, WorkerIdentity, _integer


ADAMW_FIELDS = ("parameter", "step", "exp_avg", "exp_avg_sq")


@dataclass(frozen=True)
class StateTensor:
    parameter_name: str
    module_id: str
    kind: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int

    def __post_init__(self):
        for value in (self.parameter_name, self.module_id, self.dtype):
            if not isinstance(value, str) or not value.strip():
                raise ValueError("state tensor names/dtype must be nonempty")
        if self.kind not in ADAMW_FIELDS:
            raise ValueError("state tensor must be parameter or standard AdamW state")
        if not isinstance(self.shape, tuple):
            raise ValueError("shape must be a tuple")
        for size in self.shape:
            _integer("dimension", size)
        _integer("tensor bytes", self.nbytes)

    @property
    def key(self):
        return self.parameter_name, self.kind


def adamw_inventory(model, optimizer, *, committed_global_step: int) -> tuple[StateTensor, ...]:
    """Read complete live trainable state without copying or initializing tensors."""
    _integer("committed_global_step", committed_global_step)
    import torch
    from .model import parameter_inventory

    if (not isinstance(optimizer, torch.optim.AdamW)
            or any(group["amsgrad"] for group in optimizer.param_groups)):
        raise ValueError("recovery requires standard AdamW with amsgrad=False")
    parameters = dict(model.named_parameters())
    trainable = {p for p in parameters.values() if p.requires_grad}
    owned = [p for group in optimizer.param_groups for p in group["params"]]
    if set(owned) != trainable or len(owned) != len(trainable):
        raise ValueError("optimizer must own every trainable parameter exactly once")
    result = []
    for entry in parameter_inventory(model):
        parameter = parameters[entry.name]
        state = optimizer.state.get(parameter, {})
        if (set(state) != set(ADAMW_FIELDS[1:])
                or any(not isinstance(t, torch.Tensor) for t in state.values())
                or state["step"].shape not in ((), (1,))
                or state["step"].item() != committed_global_step
                or any(state[k].shape != parameter.shape or state[k].dtype != parameter.dtype
                       or state[k].device != parameter.device for k in ADAMW_FIELDS[2:])):
            raise UnrecoverableStateError(f"incomplete or stale AdamW state: {entry.name}")
        for kind, tensor in (("parameter", parameter), *state.items()):
            result.append(StateTensor(entry.name, entry.module_id, kind, tuple(tensor.shape),
                                      str(tensor.dtype), tensor.numel() * tensor.element_size()))
    return tuple(sorted(result, key=lambda t: t.key))


@dataclass(frozen=True)
class WorkerInventory:
    worker: WorkerIdentity
    committed_global_step: int
    tensors: tuple[StateTensor, ...]

    def __post_init__(self):
        if not isinstance(self.worker, WorkerIdentity):
            raise ValueError("inventory requires a WorkerIdentity")
        _integer("committed_global_step", self.committed_global_step, 0)
        if (not isinstance(self.tensors, tuple)
                or any(not isinstance(t, StateTensor) for t in self.tensors)
                or len({t.key for t in self.tensors}) != len(self.tensors)):
            raise ValueError("invalid or duplicate worker tensor inventory")


@dataclass(frozen=True)
class StateSourceMap:
    state: ClusterState
    required: tuple[StateTensor, ...]
    inventories: tuple[WorkerInventory, ...]
    module_sources: tuple[tuple[str, tuple[WorkerIdentity, ...]], ...]

    def sources_for(self, tensor: StateTensor) -> tuple[WorkerIdentity, ...]:
        if tensor not in self.required:
            raise ValueError("tensor is not in the required inventory")
        return dict(self.module_sources)[tensor.module_id]


def build_state_source_map(state: ClusterState, required: tuple[StateTensor, ...],
                           inventories: tuple[WorkerInventory, ...]) -> StateSourceMap:
    """Require one intact parameter/AdamW module replica at the committed safe point."""
    required = tuple(required)
    if not required or any(not isinstance(t, StateTensor) for t in required):
        raise ValueError("required inventory must contain state tensors")
    expected = {t.key: t for t in required}
    if len(expected) != len(required):
        raise ValueError("duplicate required tensor")
    for name in {t.parameter_name for t in required}:
        rows = {t.kind: t for t in required if t.parameter_name == name}
        if (set(rows) != set(ADAMW_FIELDS) or len({t.module_id for t in rows.values()}) != 1
                or rows["step"].shape not in ((), (1,))
                or any(rows[k].shape != rows["parameter"].shape for k in ADAMW_FIELDS[2:])):
            raise ValueError(f"required inventory lacks complete parameter/AdamW metadata: {name}")
    inventories = tuple(sorted(inventories, key=lambda i: i.worker.worker_id))
    if (len(inventories) != len(state.workers)
            or {i.worker for i in inventories} != set(state.workers)):
        raise ValueError("inventories must cover exactly the current survivor identities")
    for inventory in inventories:
        if inventory.committed_global_step != state.committed_global_step:
            raise ValueError("inventory must describe the committed global step")
        if any(expected.get(t.key) != t for t in inventory.tensors):
            raise ValueError("worker tensor metadata differs from required inventory")
    sources, missing = [], []
    for module in sorted({t.module_id for t in required}):
        needed = {t for t in required if t.module_id == module}
        peers = tuple(i.worker for i in inventories if needed <= set(i.tensors))
        if not peers:
            names = ", ".join(sorted(f"{t.parameter_name}:{t.kind}" for t in needed))
            missing.append(f"{module}: {names}")
        else:
            sources.append((module, peers))
    if missing:
        raise UnrecoverableStateError("no complete survivor source for " + "; ".join(missing))
    return StateSourceMap(state, tuple(sorted(required, key=lambda t: t.key)), inventories, tuple(sources))
