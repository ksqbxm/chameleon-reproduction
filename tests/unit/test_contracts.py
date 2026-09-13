from dataclasses import FrozenInstanceError, replace

import pytest

from chameleon import (
    ClusterState, DecisionResult, ExecutionPlan, FailureEvent, ModelConfig,
    UnrecoverableStateError, WorkerIdentity,
)


@pytest.mark.parametrize("field,value", [
    ("vocab_size", 0), ("hidden_size", -1), ("num_layers", 0),
    ("num_heads", 0), ("sequence_length", 0), ("global_batch_size", True),
    ("micro_batch_size", 0), ("seed", -1), ("dropout", 0.1),
    ("dropout", float("nan")), ("dropout", float("inf")), ("amsgrad", True),
    ("amsgrad", 0), ("hidden_size", 31), ("micro_batch_size", 9),
])
def test_invalid_model(field, value):
    with pytest.raises(ValueError):
        ModelConfig(**{field: value})


def plan(**overrides):
    values = dict(plan_id="dynamic-0", policy="dynamic", global_batch_size=8,
                  generation=1, estimated_step_time_s=1.0,
                  estimated_transition_time_s=2.0)
    return ExecutionPlan(**(values | overrides))


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_step_time_and_duration(value):
    with pytest.raises(ValueError, match="step_time"):
        plan(estimated_step_time_s=value)
    with pytest.raises(ValueError, match="inter_fault_duration"):
        DecisionResult(plan(), value, 1)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1"])
def test_invalid_transition(value):
    with pytest.raises(ValueError, match="transition"):
        plan(estimated_transition_time_s=value)


@pytest.mark.parametrize("duration", [1, 2])
def test_unavailable_window(duration):
    with pytest.raises(ValueError, match="unavailable"):
        DecisionResult(plan(), duration, 1)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_score(value):
    with pytest.raises(ValueError, match="score"):
        DecisionResult(plan(), 10, value)


@pytest.mark.parametrize("overrides", [
    {"plan_id": ""}, {"policy": "manual"}, {"generation": -1},
    {"global_batch_size": 0},
])
def test_invalid_plan(overrides):
    with pytest.raises(ValueError):
        plan(**overrides)


@pytest.mark.parametrize("workers", [
    (), (WorkerIdentity("w0", 0, 1),),
    (WorkerIdentity("w0", 0, 0), WorkerIdentity("w0", 1, 0)),
    (WorkerIdentity("w0", 0, 0), WorkerIdentity("w1", 0, 0)),
])
def test_invalid_cluster(workers):
    with pytest.raises(ValueError):
        ClusterState(workers, 8)


@pytest.mark.parametrize("field,value", [
    ("global_batch_size", 0), ("generation", -1), ("committed_global_step", -1),
    ("committed_global_step", True),
])
def test_invalid_cluster_counters(field, value):
    values = dict(workers=(WorkerIdentity("w0", 0, 0),), global_batch_size=8)
    with pytest.raises(ValueError, match=field):
        ClusterState(**(values | {field: value}))


@pytest.mark.parametrize("ids", [(), ("",), ("w0", "w0")])
def test_invalid_failure(ids):
    with pytest.raises(ValueError, match="failed_worker_ids"):
        FailureEvent(ids, 0, 0)


@pytest.mark.parametrize("generation,step", [(-1, 0), (0, -1), (True, 0)])
def test_invalid_failure_counters(generation, step):
    with pytest.raises(ValueError):
        FailureEvent(("w0",), generation, step)


@pytest.mark.parametrize("args", [("", 0, 0), ("w", -1, 0), ("w", 0, -1)])
def test_invalid_identity(args):
    with pytest.raises(ValueError):
        WorkerIdentity(*args)


def test_valid_contracts_are_immutable():
    config = ModelConfig(global_batch_size=10, micro_batch_size=3)
    assert config.dropout == 0 and config.amsgrad is False
    workers = tuple(WorkerIdentity(f"w{i}", i, 0) for i in range(2))
    state = ClusterState(workers, config.global_batch_size)
    assert state.committed_global_step == 0
    assert replace(state, generation=1,
                   workers=tuple(replace(w, generation=1) for w in workers)).workers[0].worker_id == "w0"
    with pytest.raises(FrozenInstanceError):
        state.global_batch_size = 20
    assert FailureEvent(("w0",), 0, 0).committed_global_step == 0
    assert DecisionResult(plan(), 10, 6.4).plan.policy == "dynamic"
    assert plan(policy="rerouting", estimated_transition_time_s=0).estimated_transition_time_s == 0
    assert issubclass(UnrecoverableStateError, RuntimeError)
