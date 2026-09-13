from dataclasses import asdict, replace
import hashlib
import heapq
import inspect
from itertools import product
import json

import pytest

from chameleon.contracts import ClusterState, ModelConfig, WorkerIdentity
from chameleon.estimators import Estimator
from chameleon.planner import NoFeasibleDynamicPlanError, Planner


def controlled_profile(num_layers, global_nm):
    """Controlled measurements for algorithm oracles; no live training claim."""
    config = ModelConfig(num_layers=num_layers, global_batch_size=2 * global_nm - 1,
                         micro_batch_size=1 if global_nm == 1 else 2)
    specs = {"embedding": (40, .5, .25, 3)}
    specs.update({f"blocks.{i}": (10 + 10 * i, 1. + .25 * i, 2. + .5 * i, 5 + i)
                  for i in range(num_layers)})
    specs.update({"final_norm": (20, .25, .5, 2), "lm_head": (40, .75, 1., 7),
                  "extra": (8, .125, .25, 1)})
    metrics, memory = {}, {}
    # Two measured micro-batches; the search may extrapolate a different global Nm.
    for name, (size, forward, backward, activation) in specs.items():
        memory[name] = {"parameter_bytes": size, "gradient_bytes": size, "adamw_bytes": 2 * size + 4}
        for field, values in (("parameter_bytes", [size]), ("gradient_bytes", [size]),
                              ("adamw_bytes", [2 * size + 4]), ("saved_activation_bytes", [activation] * 2),
                              ("output_activation_bytes", [activation] * 2),
                              ("forward_s", [forward] * 2), ("backward_s", [backward] * 2)):
            metrics[f"modules.{name}.{field}"] = {"samples": values, "ema": values[0]}
    metrics["modules.loss_and_runtime.saved_activation_bytes"] = {"samples": [4, 4], "ema": 4}
    trace, now = [], 0.
    for mb in range(2):
        for kind, index in (("forward", 1), ("backward", 2)):
            end = now + sum(spec[index] for spec in specs.values()) + .5
            trace.append(dict(kind=kind, micro_batch=mb, pipeline=0, stage=0, phase="steady", start_s=now, end_s=end))
            now = end
    for field in ("step_time_s", "step_wall_time_s"):
        metrics[field] = {"samples": [now + 1.], "ema": now + 1.}
    profile = {"schema_version": 3, "ema_alpha": .5,
               "identity": {"model_hash": "a" * 64,
                            "config_hash": hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest(),
                            "module_parameter_bytes": {name: spec[0] for name, spec in specs.items()},
                            "module_order": list(specs),
                            "device": {"type": "cpu", "index": None, "torch": "fixture",
                                       "name": "fixture", "system": "fixture"},
                            "parallel": {"dp_size": 1, "pp_size": 1, "rank": 0}},
               "metrics": metrics, "calibrations": [],
               "steps": [{"step_id": 2, "trace": trace,
                          "memory": {"kind": "logical_tensor_bytes", "modules": memory,
                                     "output_activation_bytes": {name: spec[3] for name, spec in specs.items()},
                                     "saved_activation_bytes": {**{name: spec[3] for name, spec in specs.items()},
                                                                "loss_and_runtime": 4},
                                     "peak_allocated_bytes": None, "peak_reserved_bytes": None}}]}
    return config, profile


def make_planner(*, layers=3, nm=5, r_dp=(1, 2, 3), r_pp=(1, 2, 3, 4), capacity=10000):
    config, profile = controlled_profile(layers, nm)
    estimator = Estimator(profile, expected_identity=profile["identity"],
                          layer_modules=tuple(f"blocks.{i}" for i in range(layers)))
    return Planner(estimator, config=config, r_dp=r_dp, r_pp=r_pp, memory_capacity_bytes=capacity)


def survivor_state(planner, count, *, generation=0):
    return ClusterState(tuple(WorkerIdentity(f"worker-{i}", i, generation) for i in range(count)),
                        planner.config.global_batch_size, generation, committed_global_step=7)


def pipeline_event_oracle(layout, nm, metrics):
    """Handwritten phase queues and event simulation, independent of production DAG."""
    pp = len(layout)
    queues = []
    for stage in range(pp):
        warmup = min(pp - stage - 1, nm)
        queue = [("forward", mb) for mb in range(warmup)]
        for mb in range(nm - warmup):
            queue.extend((("forward", warmup + mb), ("backward", mb)))
        queue.extend(("backward", mb) for mb in range(nm - warmup, nm))
        queues.append(queue)
    events, completed, busy = [], set(), set()
    now = 0.
    while any(queues) or events:
        for stage, queue in enumerate(queues):
            if not queue or stage in busy:
                continue
            kind, mb = queue[0]
            if kind == "forward":
                ready = stage == 0 or (stage - 1, "forward", mb) in completed
            else:
                ready = ((stage, "forward", mb) in completed
                         and (stage == pp - 1 or (stage + 1, "backward", mb) in completed))
            if ready:
                duration = sum(metrics[f"modules.{name}.{kind}_s"]["ema"] for name in layout[stage])
                heapq.heappush(events, (now + duration, stage, kind, mb))
                busy.add(stage)
                queue.pop(0)
        assert events, "oracle deadlock"
        now = events[0][0]
        while events and events[0][0] == now:
            _, stage, kind, mb = heapq.heappop(events)
            completed.add((stage, kind, mb))
            busy.remove(stage)
    assert len(completed) == 2 * pp * nm
    return now


def exhaustive_oracle(planner, survivors):
    """Product enumeration of the defined heuristic space, without planner helpers."""
    result = {}
    profile = planner.estimator.profile
    order, layers = profile["identity"]["module_order"], planner.estimator.layer_modules
    metrics = profile["metrics"]
    sizes = profile["steps"][-1]["memory"]["modules"]
    layer_static = sum(2 * sizes[name]["parameter_bytes"] + sizes[name]["adamw_bytes"] for name in layers) / len(layers)
    layer_activation = sum(max(metrics[f"modules.{name}.saved_activation_bytes"]["samples"]) for name in layers) / len(layers)
    nm = (planner.config.global_batch_size - 1) // planner.config.micro_batch_size + 1
    for dp in planner.r_dp:
        for lengths in product(planner.r_pp, repeat=dp):
            if tuple(sorted(lengths)) != lengths or sum(lengths) != survivors or nm < dp:
                continue
            base = [nm * pp // survivors for pp in lengths]
            remaining = nm - sum(base)
            batches = set()
            for additions in product(range(remaining + 1), repeat=dp):
                if sum(additions) != remaining:
                    continue
                batch = [count + extra for count, extra in zip(base, additions)]
                while 0 in batch:
                    zero = batch.index(0)
                    donor = sorted(range(dp), key=lambda i: (-batch[i], i))[0]
                    assert batch[donor] > 1
                    batch[zero], batch[donor] = 1, batch[donor] - 1
                batches.add(tuple(batch))
            layout_options = []
            for pp in lengths:
                q = len(layers) // pp
                options = set()
                for counts in product((q, q + 1), repeat=pp):
                    if sum(counts) != len(layers):
                        continue
                    groups, owner, index = [[] for _ in range(pp)], {}, 0
                    for stage, count in enumerate(counts):
                        for _ in range(count):
                            owner[layers[index]] = stage
                            index += 1
                    stage, passed_layers = 0, 0
                    for name in order:
                        if name in owner:
                            stage = owner[name]
                            passed_layers += 1
                        elif passed_layers == len(layers):
                            stage = pp - 1
                        groups[stage].append(name)
                    if all(groups):
                        options.add(tuple(tuple(group) for group in groups))
                layout_options.append(options)
            for layouts in product(*layout_options):
                peaks = []
                for layout in layouts:
                    pipeline_peaks = []
                    for stage, modules in enumerate(layout):
                        count = sum(name in layers for name in modules)
                        endpoints = [name for name in modules if name not in layers]
                        static = count * layer_static + sum(2 * sizes[name]["parameter_bytes"] + sizes[name]["adamw_bytes"]
                                                           for name in endpoints)
                        activation = count * layer_activation + sum(max(metrics[f"modules.{name}.saved_activation_bytes"]["samples"])
                                                                   for name in endpoints)
                        if stage == len(layout) - 1:
                            activation += max(metrics["modules.loss_and_runtime.saved_activation_bytes"]["samples"])
                        pipeline_peaks.append(static + (len(layout) - stage) * activation)
                    peaks.append(tuple(pipeline_peaks))
                feasible = all(peak <= planner.memory_capacity_bytes for pipeline in peaks for peak in pipeline)
                for batch in batches:
                    time = max(pipeline_event_oracle(layout, count, metrics) for layout, count in zip(layouts, batch)) if feasible else None
                    result[lengths, batch, layouts] = time, tuple(peaks)
    return result


@pytest.mark.parametrize("survivors,layers,nm,capacity", [
    (1, 1, 1, 10000), (4, 3, 5, 10000), (5, 3, 5, 10000),
    (6, 4, 7, 10000), (5, 2, 2, 10000), (5, 3, 5, 400), (5, 3, 5, 330),
])
def test_complete_candidate_space_and_best_time_against_oracle(survivors, layers, nm, capacity):
    planner = make_planner(layers=layers, nm=nm, capacity=capacity)
    state = survivor_state(planner, survivors)
    expected = exhaustive_oracle(planner, survivors)
    plans = list(planner.candidates(state))
    actual = {(plan.pipeline_lengths, plan.pipeline_micro_batches, plan.layouts): plan for plan in plans}
    assert actual.keys() == expected.keys() and len(plans) == len(actual)
    assert len({plan.plan_id for plan in plans}) == len(plans)
    for key, (time, peaks) in expected.items():
        plan = actual[key]
        assert plan.policy == "dynamic" and sum(plan.pipeline_lengths) == survivors
        assert sum(plan.pipeline_micro_batches) == nm and min(plan.pipeline_micro_batches) >= 1
        assert plan.global_batch_size == state.global_batch_size and plan.generation == state.generation
        assert plan.survivor_worker_ids == tuple(sorted(worker.worker_id for worker in state.workers))
        assert tuple(tuple(row["peak_bytes"] for row in memory.stages) for memory in plan.memory) == peaks
        assert plan.feasible == (time is not None)
        if time is not None:
            assert plan.estimated_step_time_s == time
            assert plan.time.derivation["global_micro_batches"] == nm
            assert plan.time.derivation["pipeline_micro_batches"] == plan.pipeline_micro_batches
            assert plan.time.derivation["equation"] == 10
        else:
            assert plan.time is None and plan.reasons
    feasible = [plan for plan in plans if plan.feasible]
    if feasible:
        best = planner.best_dynamic_plan(state)
        assert best == min(feasible, key=lambda plan: (expected[plan.pipeline_lengths, plan.pipeline_micro_batches, plan.layouts][0], plan.plan_id))
    else:
        with pytest.raises(NoFeasibleDynamicPlanError):
            planner.best_dynamic_plan(state)


def test_batch_remainders_are_compared_by_time_instead_of_fixed_assignment():
    planner = make_planner(layers=3, nm=5, r_dp=(2,), r_pp=(2, 3))
    plans = list(planner.candidates(survivor_state(planner, 5)))
    assert {plan.pipeline_micro_batches for plan in plans} == {(2, 3)}  # Exact proportional split.
    planner = make_planner(layers=3, nm=7, r_dp=(2,), r_pp=(2, 3))
    plans = list(planner.candidates(survivor_state(planner, 5)))
    times = {batch: min(plan.estimated_step_time_s for plan in plans if plan.pipeline_micro_batches == batch)
             for batch in {(2, 5), (3, 4)}}
    assert times[(2, 5)] != times[(3, 4)]
    assert planner.best_dynamic_plan(survivor_state(planner, 5)).estimated_step_time_s == min(times.values())


def test_survivor_counts_and_actual_identities_are_searched_separately():
    planner = make_planner()
    for failures in (1, 2, 3):
        state = survivor_state(planner, 7 - failures, generation=failures)
        candidates = list(planner.candidates(state))
        assert candidates and all(sum(plan.pipeline_lengths) == 7 - failures for plan in candidates)
        best = planner.best_dynamic_plan(state)
        assert sum(best.pipeline_lengths) == len(state.workers)
        assert best.generation == failures
    state = survivor_state(planner, 5)
    best = planner.best_dynamic_plan(state)
    different = replace(state, workers=(WorkerIdentity("replacement", 4, 0), *state.workers[:-1]))
    other = planner.best_dynamic_plan(different)
    assert other.plan_id != best.plan_id
    assert "replacement" in other.survivor_worker_ids and "worker-4" not in other.survivor_worker_ids
    different_ranks = replace(state, workers=tuple(replace(worker, rank=worker.rank + 10) for worker in state.workers))
    assert planner.best_dynamic_plan(different_ranks).plan_id != best.plan_id
    assert planner.best_dynamic_plan(replace(state, workers=tuple(reversed(state.workers)))) == best
    assert planner.best_dynamic_plan(replace(state, committed_global_step=8)) == best


def test_exact_time_ties_use_stable_plan_id_and_ranges_have_stable_order():
    # Identical one-stage pipelines with batches [1,2] or [2,1] tie exactly.
    planner = make_planner(layers=3, nm=3, r_pp=(1,))
    state = survivor_state(planner, 2)
    plans = [plan for plan in planner.candidates(state) if plan.feasible]
    minimum = min(plan.estimated_step_time_s for plan in plans)
    ties = [plan for plan in plans if plan.estimated_step_time_s == minimum]
    assert len(ties) >= 2
    assert planner.best_dynamic_plan(state).plan_id == min(plan.plan_id for plan in ties)
    reordered = Planner(planner.estimator, config=planner.config, r_dp=(3, 2, 1, 2), r_pp=(1, 1),
                        memory_capacity_bytes=10000)
    assert list(reordered.candidates(state)) == list(planner.candidates(state))
    assert reordered.best_dynamic_plan(state) == planner.best_dynamic_plan(state)


def test_endpoint_memory_can_eliminate_otherwise_fitting_layer_layout():
    planner = make_planner(layers=3, nm=5, r_dp=(2,), r_pp=(2, 3), capacity=400)
    plans = list(planner.candidates(survivor_state(planner, 5)))
    assert any(plan.feasible for plan in plans) and any(not plan.feasible for plan in plans)
    eliminated = [row for plan in plans for memory in plan.memory for row in memory.stages
                  if row["oom_reason"] and row["static_layer_bytes"] + row["dynamic_layer_bytes"] <= 400]
    assert eliminated and all(row["extra_bytes"] > 0 for row in eliminated)
    assert any("extra=" in reason for plan in plans for reason in plan.reasons)
    assert planner.best_dynamic_plan(survivor_state(planner, 5)).feasible


def test_planner_memory_capacity_boundary_is_inclusive():
    planner = make_planner(layers=3, nm=5, r_dp=(2,), r_pp=(2, 3), capacity=388)
    assert planner.best_dynamic_plan(survivor_state(planner, 5)).feasible
    below = make_planner(layers=3, nm=5, r_dp=(2,), r_pp=(2, 3), capacity=387)
    with pytest.raises(NoFeasibleDynamicPlanError):
        below.best_dynamic_plan(survivor_state(below, 5))


def test_all_oom_reports_stage_reasons_without_fabricating_time():
    planner = make_planner(capacity=0)
    state = survivor_state(planner, 5)
    plans = list(planner.candidates(state))
    assert plans and all(not plan.feasible and plan.estimated_step_time_s is None for plan in plans)
    with pytest.raises(NoFeasibleDynamicPlanError) as caught:
        planner.best_dynamic_plan(state)
    assert caught.value.reasons and all("stage" in reason and "capacity 0" in reason for reason in caught.value.reasons)


@pytest.mark.parametrize("kwargs,survivors", [
    ({"r_dp": (2,), "r_pp": (3,)}, 5),
    ({"nm": 1, "r_dp": (2,), "r_pp": (1, 2)}, 3),
    ({"layers": 1, "r_dp": (1,), "r_pp": (4,)}, 4),
])
def test_no_legal_partition_insufficient_batches_or_empty_stages(kwargs, survivors):
    planner = make_planner(**kwargs)
    state = survivor_state(planner, survivors)
    assert list(planner.candidates(state)) == []
    with pytest.raises(NoFeasibleDynamicPlanError, match="no legal dynamic plan"):
        planner.best_dynamic_plan(state)


def test_dynamic_search_has_no_d_score_transition_or_policy_selection():
    planner = make_planner()
    best = planner.best_dynamic_plan(survivor_state(planner, 5))
    assert list(inspect.signature(planner.best_dynamic_plan).parameters) == ["state"]
    assert list(inspect.signature(planner.candidates).parameters) == ["state"]
    assert not hasattr(best, "score") and not hasattr(best, "estimated_transition_time_s")
    assert best.policy == "dynamic"
    assert not hasattr(planner, "select") and not hasattr(planner, "recover")


@pytest.mark.parametrize("updates", [{"global_batch_size": 11}, {"seed": 43}])
def test_planner_rejects_profile_config_mismatch(updates):
    planner = make_planner()
    with pytest.raises(ValueError, match="config identity"):
        Planner(planner.estimator, config=replace(planner.config, **updates), r_dp=(1,), r_pp=(1,), memory_capacity_bytes=10000)


@pytest.mark.parametrize("layers", [
    ("embedding", "blocks.0", "blocks.1"),
    ("blocks.0", "blocks.1", "lm_head"),
    ("blocks.1", "blocks.2", "final_norm"),
    ("blocks.2", "blocks.1", "blocks.0"),
    ("blocks.0", "blocks.1"),
])
def test_planner_rejects_endpoint_substitution_and_reordered_blocks(layers):
    config, profile = controlled_profile(3, 5)
    estimator = Estimator(profile, expected_identity=profile["identity"], layer_modules=layers)
    with pytest.raises(ValueError, match="configured blocks"):
        Planner(estimator, config=config, r_dp=(2,), r_pp=(2, 3), memory_capacity_bytes=10000)


def test_pipeline_memory_is_not_recomputed_for_other_pipeline_combinations(monkeypatch):
    planner = make_planner(layers=3, nm=5, r_dp=(3,), r_pp=(2,))
    calls = []
    memory = planner.estimator.memory

    def measured_memory(layout, capacities_bytes, *, pipeline=0):
        calls.append((pipeline, layout))
        return memory(layout, capacities_bytes, pipeline=pipeline)

    monkeypatch.setattr(planner.estimator, "memory", measured_memory)
    plans = list(planner.candidates(survivor_state(planner, 6)))
    assert len(plans) == 48  # 2^3 layer layouts times 6 distinct batch remainder distributions.
    expected = {(i, layout) for plan in plans for i, layout in enumerate(plan.layouts)}
    assert set(calls) == expected
    assert len(calls) == len(expected) == 6


def test_planner_rejects_changed_global_batch():
    planner = make_planner()
    with pytest.raises(ValueError, match="global batch size"):
        planner.best_dynamic_plan(replace(survivor_state(planner, 5), global_batch_size=10))


@pytest.mark.parametrize("kwargs", [{"r_dp": ()}, {"r_dp": (True,)}, {"r_pp": ()}, {"r_pp": (0,)},
                                    {"capacity": -1}, {"capacity": True}, {"capacity": 1.5}])
def test_invalid_search_bounds_and_capacity(kwargs):
    with pytest.raises(ValueError):
        make_planner(**kwargs)
