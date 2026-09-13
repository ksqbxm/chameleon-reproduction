import pytest

from chameleon.schedule import build_1f1b_schedule


def labels(queue):
    return [(op.kind, op.micro_batch, op.phase) for op in queue]


def test_single_stage_alternates_forward_backward():
    assert labels(build_1f1b_schedule(1, 2)[0]) == [
        ("forward", 0, "steady"), ("backward", 0, "steady"),
        ("forward", 1, "steady"), ("backward", 1, "steady"),
    ]


def test_two_stage_warmup_steady_cooldown_hand_fixture():
    queues = build_1f1b_schedule(2, 3, pipeline=7)
    assert labels(queues[0]) == [
        ("forward", 0, "warmup"), ("forward", 1, "steady"),
        ("backward", 0, "steady"), ("forward", 2, "steady"),
        ("backward", 1, "steady"), ("backward", 2, "cooldown"),
    ]
    assert labels(queues[1]) == [
        ("forward", 0, "steady"), ("backward", 0, "steady"),
        ("forward", 1, "steady"), ("backward", 1, "steady"),
        ("forward", 2, "steady"), ("backward", 2, "steady"),
    ]
    assert queues[0][2].dependencies == (
        (7, 0, 1, "forward"), (7, 0, 0, "forward"), (7, 1, 0, "backward"),
    )


def test_fewer_micro_batches_than_stages_hand_fixture():
    queues = build_1f1b_schedule(4, 1)
    for stage in range(3):
        assert labels(queues[stage]) == [("forward", 0, "warmup"), ("backward", 0, "cooldown")]
    assert labels(queues[3]) == [("forward", 0, "steady"), ("backward", 0, "steady")]


@pytest.mark.parametrize("stages", range(1, 7))
@pytest.mark.parametrize("micro_batches", range(1, 7))
def test_every_operation_once_dependencies_complete_and_acyclic(stages, micro_batches):
    queues = build_1f1b_schedule(stages, micro_batches, pipeline=2)
    operations = {op.key: op for queue in queues for op in queue}
    expected = {(2, stage, mb, kind) for stage in range(stages)
                for mb in range(micro_batches) for kind in ("forward", "backward")}
    assert len(operations) == sum(map(len, queues)) == stages * micro_batches * 2
    assert operations.keys() == expected
    for stage, queue in enumerate(queues):
        for index, op in enumerate(queue):
            assert op.pipeline == 2 and op.stage == stage
            required = {queue[index - 1].key} if index else set()
            if op.kind == "forward" and stage:
                required.add((2, stage - 1, op.micro_batch, "forward"))
            if op.kind == "backward":
                required.add((2, stage, op.micro_batch, "forward"))
                if stage + 1 < stages:
                    required.add((2, stage + 1, op.micro_batch, "backward"))
            assert set(op.dependencies) == required
            assert all(key in operations for key in op.dependencies)
    # Independent graph elimination verifies there is no cycle.
    remaining = {key: set(op.dependencies) for key, op in operations.items()}
    while remaining:
        ready = {key for key, parents in remaining.items() if not parents}
        assert ready
        remaining = {key: parents - ready for key, parents in remaining.items() if key not in ready}


@pytest.mark.parametrize("stages,batches,pipeline", [
    (0, 1, 0), (1, 0, 0), (True, 1, 0), (1, 1.5, 0), (1, 1, -1), (1, 1, True),
])
def test_invalid_schedule_counts(stages, batches, pipeline):
    with pytest.raises(ValueError):
        build_1f1b_schedule(stages, batches, pipeline=pipeline)
