from dataclasses import asdict, replace

import pytest

from chameleon import ClusterState, ModelConfig, WorkerIdentity
from chameleon.runtime import DynamicTopology, SymmetricRuntime, compare_runtime_profile
from chameleon.estimators import TimeEstimate
from chameleon.planner import DynamicPlan
from chameleon.profiler import _hash, _validate_parallel, _validate_trace
from chameleon.restorer import TargetSlot


def topology(*, batch=19, counts=(5, 3, 2), ranks=()):
    config = ModelConfig(num_layers=2, global_batch_size=batch, micro_batch_size=2)
    state = ClusterState(tuple(WorkerIdentity(f"dynamic-{20 - r}", r, 2) for r in reversed(range(7))),
                         batch, generation=2)
    short = (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head"))
    long = (("embedding",), ("blocks.0",), ("blocks.1", "final_norm", "lm_head"))
    return DynamicTopology(state, config, (short, short, long), counts, ranks)


def test_nonuniform_batches_preserve_samples_and_partial_batch_for_three_steps():
    value = topology()
    assert value.pipeline_lengths == (2, 2, 3)
    assert value.pipeline_ranks == ((0, 1), (2, 3), (4, 5, 6))
    for step in range(3):
        batches = [value.micro_batches(replace(value.state, committed_global_step=step), p) for p in range(3)]
        assert tuple(map(len, batches)) == (5, 3, 2)
        assert [sum(map(len, pipeline)) for pipeline in batches] == [10, 6, 3]
        assert batches[-1][-1] == (step * 19 + 18,)
        assert [sample for pipeline in batches for batch in pipeline for sample in batch] == list(range(step * 19, (step + 1) * 19))


def test_permuted_workers_map_modules_and_owners_without_rank_arithmetic():
    value = topology(ranks=((6, 0), (4, 2), (1, 5, 3)))
    assert value.location(6) == (0, 0)
    assert value.location(3) == (2, 2)
    assert value.module_owners == {"embedding": (1, 4, 6), "blocks.0": (4, 5, 6),
                                   "blocks.1": (0, 2, 3), "final_norm": (0, 2, 3), "lm_head": (0, 2, 3)}
    rounds = value.synchronization_rounds
    assert sorted(module for row in rounds for module in row) == sorted(value.module_owners)
    assert any(len(row) > 1 for row in rounds)
    for row in rounds:
        owners = [rank for module in row for rank in value.module_owners[module]]
        assert len(owners) == len(set(owners))


@pytest.mark.parametrize("counts", [(0, 8, 2), (True, 7, 2), (5, 3), (5, 3, 1), [5, 3, 2]])
def test_invalid_micro_batch_counts(counts):
    with pytest.raises(ValueError, match="micro-batch"):
        topology(counts=counts)


@pytest.mark.parametrize("ranks", [((0, 1), (2, 3), (4, 5, 5)), ((0, 1), (2, 3), (4, 5, 7)),
                                   ((0, 1), (2, 3), (4, 5, True)), ((0,), (2, 3), (4, 5, 6)),
                                   [(0, 1), (2, 3), (4, 5, 6)]])
def test_invalid_worker_slot_maps(ranks):
    with pytest.raises(ValueError, match="rank"):
        topology(ranks=ranks)


def test_topology_rejects_batch_change():
    current = topology()
    with pytest.raises(ValueError):
        replace(current, state=replace(current.state, global_batch_size=20))


def test_initial_runtime_rejects_reinitializing_committed_dynamic_state():
    current = topology()
    recovered = replace(current, state=replace(current.state, committed_global_step=3))
    with pytest.raises(ValueError, match="already committed"):
        SymmetricRuntime(recovered)


def test_each_pipeline_requires_complete_ordered_model_including_endpoints():
    current = topology()
    for layout in ((), (("embedding", "blocks.0"), ("blocks.1", "lm_head")),
                   (("blocks.0", "embedding"), ("blocks.1", "final_norm", "lm_head"))):
        with pytest.raises(ValueError, match="partition"):
            replace(current, layouts=(layout, *current.layouts[1:]))


def test_profile_estimator_uses_actual_pipeline_depth_count_and_maximum():
    current = topology()
    reports = []
    for p, length in enumerate(current.pipeline_lengths):
        count = current.pipeline_micro_batches[p]
        for s in range(length):
            warmup = min(length - s - 1, count)
            ops = [("forward", mb) for mb in range(warmup)]
            for mb in range(count - warmup):
                ops += [("forward", warmup + mb), ("backward", mb)]
            ops += [("backward", mb) for mb in range(count - warmup, count)]
            duration = p + 1
            trace = [dict(pipeline=p, stage=s, micro_batch=mb, kind=kind,
                          start_s=i * duration, end_s=(i + 1) * duration) for i, (kind, mb) in enumerate(ops)]
            reports.append(dict(pipeline=p, profile_step=dict(trace=trace), micro_batches=[[]] * count,
                                pipeline_wall_time_s=100 + p, training_wall_time_s=110 + p))
    actual = compare_runtime_profile(reports, current)
    # Uniform F/B durations: (Nm + PP - 1) * (F + B), independently hand counted.
    expected = [12, 16, 24]
    assert [estimate["step_time_s"] for estimate in actual["equation11_pipelines"]] == expected
    assert actual["estimated_compute_time_s"] == 24
    assert actual["estimated_compute_time_s"] != sum(expected) / 3
    assert actual["equation9"] is None
    assert actual["measured_pipeline_time_s"] == 102


def plan_for(current):
    time = TimeEstimate(1., dict(layouts=current.layouts, profile_hash="controlled",
                                pipeline_micro_batches=current.pipeline_micro_batches,
                                global_micro_batches=current.global_micro_batches,
                                profile_identity=dict(config_hash=_hash(asdict(current.config)))))
    return DynamicPlan(current.config.global_batch_size, current.state.generation, current.ranks,
                       "controlled", current.pipeline_lengths, current.pipeline_micro_batches,
                       current.layouts, time, ())


def test_feasible_plan_and_restorer_assignments_define_actual_worker_locations():
    current = topology()
    plan = plan_for(current)
    assignments = tuple((TargetSlot(p, s, stage), worker)
                        for (p, s, stage), worker in zip(
                            ((p, s, stage) for p, layout in enumerate(current.layouts) for s, stage in enumerate(layout)),
                            reversed(current.ranks)))
    actual = DynamicTopology.from_plan(plan, current.config, assignments=assignments)
    assert actual.pipeline_ranks == ((6, 5), (4, 3), (2, 1, 0))
    assert actual.layouts == plan.layouts
    assert actual.pipeline_micro_batches == (5, 3, 2)
    assert actual.state.committed_global_step == 0
    assert DynamicTopology.from_plan(plan, current.config).pipeline_ranks == current.pipeline_ranks
    invalid = assignments[:-1]
    for invalid in (invalid, (*assignments, assignments[0]),
                    ((replace(assignments[0][0], modules=("lm_head",)), assignments[0][1]), *assignments[1:]),
                    ((assignments[0][0], replace(assignments[0][1], generation=3)), *assignments[1:]),
                    ((assignments[0][0], assignments[1][1]), *assignments[1:])):
        with pytest.raises(ValueError):
            DynamicTopology.from_plan(plan, current.config, assignments=invalid)
    with pytest.raises(ValueError, match="feasible"):
        DynamicTopology.from_plan(replace(plan, time=None), current.config)
    with pytest.raises(ValueError, match="config"):
        DynamicTopology.from_plan(plan, replace(current.config, seed=43))


@pytest.mark.parametrize("lengths", [[], [2, 3], [2, 0, 3], [2, True, 3], [2, 2, 2], (2, 2, 3)])
def test_profiler_rejects_invalid_asymmetric_parallel_metadata(lengths):
    with pytest.raises(ValueError):
        _validate_parallel(dict(dp_size=3, pp_size=3, rank=6, pipeline_lengths=lengths))


def test_profiler_uses_each_pipeline_depth_instead_of_maximum_for_fifo():
    parallel = dict(dp_size=3, pp_size=3, rank=6, pipeline_lengths=[2, 2, 3])
    _validate_parallel(parallel)
    # Stage1 of PP2 is the final stage: F0 B0 F1 B1, all steady.
    trace = [dict(pipeline=0, stage=1, kind=kind, micro_batch=mb, phase="steady",
                  start_s=float(i), end_s=float(i + 1))
             for i, (kind, mb) in enumerate((("forward", 0), ("backward", 0), ("forward", 1), ("backward", 1)))]
    _validate_trace(trace, parallel)
    with pytest.raises(ValueError, match="1F1B"):
        _validate_trace(trace, dict(dp_size=3, pp_size=3, rank=6))
    with pytest.raises(ValueError, match="configuration"):
        _validate_trace([dict(row, stage=2) for row in trace], parallel)
    with pytest.raises(ValueError, match="rank"):
        _validate_parallel(dict(parallel, rank=7))
