from dataclasses import replace

import pytest

from chameleon.contracts import ClusterState, UnrecoverableStateError, WorkerIdentity
from chameleon.state_sources import (
    ADAMW_FIELDS, StateTensor, WorkerInventory, adamw_inventory, build_state_source_map,
)


def metadata_inventory(modules=("embedding", "blocks.0", "final_norm", "lm_head", "extra")):
    """Controlled byte metadata, without training tensors or a transport simulation."""
    return tuple(StateTensor(f"{module}.weight", module, kind, () if kind == "step" else (8,),
                             "float32" if kind == "step" else "float64", 4 if kind == "step" else 64)
                 for module in modules for kind in ADAMW_FIELDS)


@pytest.fixture
def sources_input():
    state = ClusterState((WorkerIdentity("a", 3, 2), WorkerIdentity("b", 7, 2)), 10, 2, 5)
    required = metadata_inventory()
    inventories = tuple(WorkerInventory(w, 5, required) for w in state.workers)
    return state, required, inventories


def test_sources_cover_every_tensor_and_are_stable(sources_input):
    state, required, inventories = sources_input
    result = build_state_source_map(state, required, inventories)
    reordered = build_state_source_map(replace(state, workers=state.workers[::-1]), required[::-1], inventories[::-1])
    assert result.required == reordered.required
    assert result.inventories == reordered.inventories
    assert result.module_sources == reordered.module_sources
    for tensor in required:
        assert result.sources_for(tensor) == state.workers
    assert {t.module_id for t in result.required} == {"embedding", "blocks.0", "final_norm", "lm_head", "extra"}


@pytest.mark.parametrize("module", ("embedding", "blocks.0", "final_norm", "lm_head", "extra"))
@pytest.mark.parametrize("kind", ADAMW_FIELDS)
def test_missing_any_required_tensor_is_unrecoverable(sources_input, module, kind):
    state, required, inventories = sources_input
    partial = tuple(t for t in required if (t.module_id, t.kind) != (module, kind))
    inventories = tuple(replace(i, tensors=partial) for i in inventories)
    with pytest.raises(UnrecoverableStateError, match=module):
        build_state_source_map(state, required, inventories)


def test_does_not_stitch_incomplete_replicas(sources_input):
    state, required, inventories = sources_input
    inventories = (replace(inventories[0], tensors=tuple(t for t in required if t.kind != "step")),
                   replace(inventories[1], tensors=tuple(t for t in required if t.kind == "step")))
    with pytest.raises(UnrecoverableStateError, match="complete survivor"):
        build_state_source_map(state, required, inventories)


def test_unrecoverable_error_reports_every_missing_module(sources_input):
    state, required, inventories = sources_input
    partial = tuple(t for t in required if t.module_id not in {"embedding", "lm_head"})
    inventories = tuple(replace(i, tensors=partial) for i in inventories)
    with pytest.raises(UnrecoverableStateError) as captured:
        build_state_source_map(state, required, inventories)
    assert "embedding" in str(captured.value) and "lm_head" in str(captured.value)


def test_complete_parameters_on_different_peers_are_not_a_complete_module(sources_input):
    state, required, _ = sources_input
    required = tuple(replace(t, module_id="blocks.0") for t in required)
    inventories = tuple(WorkerInventory(w, state.committed_global_step,
                        tuple(t for t in required if (t.parameter_name.startswith("embedding")) == (i == 0)))
                        for i, w in enumerate(state.workers))
    with pytest.raises(UnrecoverableStateError, match="blocks.0"):
        build_state_source_map(state, required, inventories)


def test_incomplete_peer_does_not_hide_healthy_replica(sources_input):
    state, required, inventories = sources_input
    inventories = (replace(inventories[0], tensors=required[1:]), inventories[1])
    result = build_state_source_map(state, required, inventories)
    assert result.sources_for(required[0]) == (state.workers[1],)


@pytest.mark.parametrize("change", ("failed", "generation", "rank", "step", "duplicate", "metadata", "extra"))
def test_reject_wrong_survivor_inventory(sources_input, change):
    state, required, inventories = sources_input
    first = inventories[0]
    if change in ("failed", "generation", "rank"):
        worker = {"failed": WorkerIdentity("dead", 3, 2), "generation": WorkerIdentity("a", 3, 1),
                  "rank": WorkerIdentity("a", 2, 2)}[change]
        first = replace(first, worker=worker)
    elif change == "step":
        first = replace(first, committed_global_step=4)
    elif change == "duplicate":
        with pytest.raises(ValueError, match="duplicate"):
            replace(first, tensors=(*required, required[0]))
        return
    elif change == "metadata":
        first = replace(first, tensors=(replace(required[0], nbytes=128), *required[1:]))
    else:
        first = replace(first, tensors=(*required, replace(required[0], parameter_name="unknown")))
    with pytest.raises(ValueError):
        build_state_source_map(state, required, (first, inventories[1]))


@pytest.mark.parametrize("change", ("empty", "duplicate", "missing", "module", "shape"))
def test_required_metadata_must_be_complete(sources_input, change):
    state, required, inventories = sources_input
    if change == "empty":
        required = ()
    elif change == "duplicate":
        required = (*required, required[0])
    elif change == "missing":
        required = required[1:]
    else:
        required = (replace(required[0], **({"module_id": "wrong"} if change == "module" else {"shape": (4,)})),
                    *required[1:])
    with pytest.raises(ValueError):
        build_state_source_map(state, required, inventories)


@pytest.mark.parametrize("overrides", ({"nbytes": 0}, {"nbytes": True}, {"shape": [8]}, {"shape": (0,)},
                                       {"kind": "gradient"}, {"dtype": ""}, {"module_id": ""}))
def test_tensor_metadata_validation(overrides):
    with pytest.raises(ValueError):
        replace(metadata_inventory()[0], **overrides)


@pytest.mark.parametrize("overrides", ({"worker": "a"}, {"committed_global_step": True},
                                       {"committed_global_step": -1}, {"tensors": []}))
def test_worker_inventory_validation(sources_input, overrides):
    with pytest.raises(ValueError):
        replace(sources_input[2][0], **overrides)


@pytest.mark.parametrize("step", (True, 0, -1, 1.5))
def test_live_inventory_requires_a_materialized_committed_step(step):
    with pytest.raises(ValueError, match="committed_global_step"):
        adamw_inventory(None, None, committed_global_step=step)
