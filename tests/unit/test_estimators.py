from copy import deepcopy
import heapq
import json

import pytest

from chameleon.estimators import (
    Estimator, estimate_operation_time, estimate_pipeline_time, estimate_rerouting_time,
    estimate_stage_memory, estimate_symmetric_time,
)
from chameleon.schedule import build_1f1b_schedule


@pytest.mark.parametrize("pp,global_nm,dp,tf,tb,expected", [
    (1, 1, 1, 2., 3., 5.), (4, 6, 2, 2., 3., 30.),
    (2, 24, 3, 0.25, 0.75, 9.), (5, 2, 2, 1., 1., 10.),
])
def test_equation9_independent_hand_calculations(pp, global_nm, dp, tf, tb, expected):
    result = estimate_symmetric_time(num_stages=pp, global_micro_batches=global_nm,
                                     dp_size=dp, forward_s=tf, backward_s=tb)
    assert result.feasible and result.step_time_s == expected
    assert result.derivation["pipeline_micro_batches"] == (global_nm // dp,) * dp
    assert result.derivation["equation"] == 9


@pytest.mark.parametrize("counts,expected,extra,equation", [
    ((0, 0, 0), 35., (0., 0., 0.), 13),
    ((1, 0, 0), 35. + 25. / 3., (5. / 3., 0., 0.), 12),
    ((0, 1, 0), 35. + 25. / 3., (0., 5. / 3., 0.), 12),
    ((2, 0, 0), 60., (5., 0., 0.), 13),
    ((1, 1, 0), 35. + 50. / 3., (5. / 3., 5. / 3., 0.), 13),
    ((3, 1, 0), 35. + 75. + 25. / 3., (15., 5. / 3., 0.), 13),
])
def test_equations12_13_independent_hand_calculations(counts, expected, extra, equation):
    result = estimate_rerouting_time(global_micro_batches=20, dp_size=4,
                                     failures_per_stage=counts, forward_s=2., backward_s=3.)
    assert result.feasible
    assert result.step_time_s == pytest.approx(expected)
    assert result.derivation["extra_slots_per_stage"] == extra
    assert result.derivation["pipeline_micro_batches"] == (5, 5, 5, 5)
    assert result.derivation["equation"] == equation


@pytest.mark.parametrize("dp,counts", [(1, (1,)), (2, (0, 2)), (2, (3, 0)), (4, (4, 5))])
def test_rerouting_last_peer_missing_is_infeasible_without_division(dp, counts):
    result = estimate_rerouting_time(global_micro_batches=dp * 2, dp_size=dp,
                                     failures_per_stage=counts, forward_s=1., backward_s=1.)
    assert not result.feasible and result.step_time_s is None
    for stage, fi in enumerate(counts):
        if fi >= dp:
            assert any(f"stage {stage}" in reason and "no healthy stage peer" in reason
                       for reason in result.reasons)
    assert "extra_slots_per_stage" not in result.derivation


@pytest.mark.parametrize("global_nm,dp", [(3, 2), (1, 2), (0, 1), (True, 1), (2, 0), (2, True)])
def test_symmetric_and_rerouting_reject_invalid_or_unequal_batch_scope(global_nm, dp):
    with pytest.raises(ValueError):
        estimate_symmetric_time(num_stages=2, global_micro_batches=global_nm,
                                 dp_size=dp, forward_s=1., backward_s=1.)
    with pytest.raises(ValueError):
        estimate_rerouting_time(global_micro_batches=global_nm, dp_size=dp,
                                 failures_per_stage=(0, 0), forward_s=1., backward_s=1.)


@pytest.mark.parametrize("value", [0., -1., True, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["forward_s", "backward_s"])
def test_invalid_compute_times(value, field):
    values = {"forward_s": 1., "backward_s": 1., field: value}
    with pytest.raises(ValueError):
        estimate_symmetric_time(num_stages=1, global_micro_batches=1, dp_size=1, **values)
    with pytest.raises(ValueError):
        estimate_rerouting_time(global_micro_batches=1, dp_size=1, failures_per_stage=(0,), **values)
    with pytest.raises(ValueError):
        estimate_pipeline_time((values["forward_s"],), (values["backward_s"],), 1)


@pytest.mark.parametrize("counts", [(), (-1,), (True,), (1.5,)])
def test_invalid_failure_inventory(counts):
    with pytest.raises(ValueError):
        estimate_rerouting_time(global_micro_batches=4, dp_size=2, failures_per_stage=counts,
                                 forward_s=1., backward_s=1.)


def event_oracle(forward, backward, micro_batches, *, variable=False):
    """Independent state machine + completion-event heap. Never reads production queues/dependencies."""
    stages = len(forward)
    sent, received = set(), set()
    forward_ids = [0] * stages
    backward_ids = [0] * stages
    busy = set()
    completed = {}
    events = []
    now = 0.
    while len(completed) < 2 * stages * micro_batches:
        for stage in range(stages):
            if stage in busy or backward_ids[stage] == micro_batches:
                continue
            # The stage may initially issue P-stage-1 forwards, then keeps one more
            # forward than that credit before each FIFO backward (bounded activation window).
            credit = stages - stage
            outstanding = forward_ids[stage] - backward_ids[stage]
            if forward_ids[stage] < micro_batches and outstanding < credit:
                kind, mb = "forward", forward_ids[stage]
                if stage and (stage - 1, mb) not in sent:
                    continue
                duration = forward[stage] * (mb + 1 if variable else 1)
            else:
                kind, mb = "backward", backward_ids[stage]
                if (stage, mb) not in sent or (stage + 1 < stages and (stage + 1, mb) not in received):
                    continue
                duration = backward[stage] * (mb + 1 if variable else 1)
            busy.add(stage)
            heapq.heappush(events, (now + duration, stage, mb, kind, now))
        assert events, "independent oracle deadlocked"
        now = events[0][0]
        while events and events[0][0] == now:
            end, stage, mb, kind, start = heapq.heappop(events)
            busy.remove(stage)
            completed[stage, mb, kind] = (start, end)
            if kind == "forward":
                sent.add((stage, mb))
                forward_ids[stage] += 1
            else:
                received.add((stage, mb))
                backward_ids[stage] += 1
    return now, completed


@pytest.mark.parametrize("forward,backward", [
    ((1.,), (2.,)), ((1., 1.), (1., 1.)),
    ((1., 3., 2.), (4., 1., 2.)), ((2., 1., 4., 1.), (1., 5., 1., 3.)),
    ((0.25, 1.5, 0.5), (2.25, 0.5, 1.)),
])
@pytest.mark.parametrize("micro_batches", [1, 2, 3, 6])
def test_equation11_against_independent_discrete_event_oracle(forward, backward, micro_batches):
    result = estimate_pipeline_time(forward, backward, micro_batches, pipeline=3)
    expected, completed = event_oracle(forward, backward, micro_batches)
    assert result.step_time_s == expected
    assert len(result.trace) == 2 * len(forward) * micro_batches
    for row in result.trace:
        assert row["pipeline"] == 3
        assert (row["start_s"], row["end_s"]) == completed[row["stage"], row["micro_batch"], row["kind"]]


def test_per_operation_durations_support_unequal_micro_batches():
    forward, backward, nm = (1., 3., 2.), (2., 1., 4.), 4
    queues = build_1f1b_schedule(3, nm)
    durations = {op.key: (forward if op.kind == "forward" else backward)[op.stage] * (op.micro_batch + 1)
                 for queue in queues for op in queue}
    result = estimate_operation_time(durations, num_stages=3, pipeline_micro_batches=nm)
    expected, completed = event_oracle(forward, backward, nm, variable=True)
    assert result.step_time_s == expected
    for row in result.trace:
        assert (row["start_s"], row["end_s"]) == completed[row["stage"], row["micro_batch"], row["kind"]]


def test_operation_durations_must_match_canonical_schedule():
    with pytest.raises(ValueError, match="exactly"):
        estimate_operation_time({}, num_stages=1, pipeline_micro_batches=1)
    durations = {(1, 0, 0, "forward"): 1., (1, 0, 0, "backward"): 1.}
    with pytest.raises(ValueError, match="exactly"):
        estimate_operation_time(durations, num_stages=1, pipeline_micro_batches=1)


@pytest.mark.parametrize("forward,backward", [((), ()), ((1.,), (1., 2.)), ((1., 2.), ())])
def test_pipeline_duration_inventory_must_cover_all_stages(forward, backward):
    with pytest.raises(ValueError):
        estimate_pipeline_time(forward, backward, 1)


@pytest.mark.parametrize("stages", [0, -1, True, 1.5])
def test_equation9_requires_integer_positive_stage_count(stages):
    with pytest.raises(ValueError):
        estimate_symmetric_time(num_stages=stages, global_micro_batches=1, dp_size=1,
                                 forward_s=1., backward_s=1.)


def memory_fixture(**updates):
    values = dict(average_parameter_bytes=10., average_optimizer_bytes=24.,
                  average_activation_bytes=5., extra_static_bytes=(100., 20., 60.),
                  extra_activation_bytes=(7., 0., 9.), capacities_bytes=(239, 74, 167), pipeline=2)
    values.update(updates)
    return estimate_stage_memory((2, 1, 2), **values)


def test_equation14_independent_hand_calculation_at_capacity():
    result = memory_fixture()
    assert result.feasible and result.reasons == ()
    assert [r["static_layer_bytes"] for r in result.stages] == [88., 44., 88.]
    assert [r["dynamic_layer_bytes"] for r in result.stages] == [30., 10., 10.]
    assert [r["extra_bytes"] for r in result.stages] == [121., 20., 69.]
    assert [r["peak_bytes"] for r in result.stages] == [239., 74., 167.]
    assert result.derivation["equation"] == 14


def test_memory_oom_gives_each_stage_reason_and_endpoint_cost():
    result = memory_fixture(capacities_bytes=(238, 73, 166))
    assert not result.feasible and len(result.reasons) == 3
    assert all(f"pipeline 2 stage {stage}" in result.reasons[stage] for stage in range(3))
    assert "extra=121" in result.reasons[0]
    no_endpoints = memory_fixture(extra_static_bytes=(0., 0., 0.), extra_activation_bytes=(0., 0., 0.),
                                  capacities_bytes=(238, 73, 166))
    assert no_endpoints.feasible


def test_single_stage_memory_and_endpoint_only_stage():
    single = estimate_stage_memory((1,), average_parameter_bytes=10., average_optimizer_bytes=24.,
                                   average_activation_bytes=5.,
                                   extra_static_bytes=(0.,), extra_activation_bytes=(0.,), capacities_bytes=(49,))
    assert single.feasible and single.stages[0]["peak_bytes"] == 49.
    endpoint_only = estimate_stage_memory((0,), average_parameter_bytes=10., average_optimizer_bytes=24.,
                                          average_activation_bytes=5.,
                                          extra_static_bytes=(5.,), extra_activation_bytes=(3.,), capacities_bytes=(7,))
    assert not endpoint_only.feasible and endpoint_only.stages[0]["peak_bytes"] == 8.


@pytest.mark.parametrize("field", ["average_parameter_bytes", "average_optimizer_bytes",
                                   "average_activation_bytes"])
@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf")])
def test_memory_rejects_invalid_average_cost(field, value):
    with pytest.raises(ValueError):
        memory_fixture(**{field: value})


@pytest.mark.parametrize("updates", [
    {"extra_static_bytes": (0.,)}, {"extra_activation_bytes": (0.,)}, {"capacities_bytes": (1,)},
    {"extra_static_bytes": (0., -1., 0.)}, {"extra_activation_bytes": (0., 0., float("nan"))},
    {"capacities_bytes": (10., 20, 30)}, {"capacities_bytes": (True, 20, 30)}, {"pipeline": -1},
])
def test_memory_rejects_invalid_stage_inputs(updates):
    with pytest.raises(ValueError):
        memory_fixture(**updates)


@pytest.fixture
def measured_schema():
    # Independent artificial measurements for formula/schema tests only; live integration is separate.
    specs = {"embedding": (80, .5, .25, 3), "blocks.0": (10, 1., 2., 5),
             "blocks.1": (30, 2., 3., 9), "final_norm": (20, .25, .5, 2),
             "lm_head": (40, .75, 1., 7), "extra": (8, .125, .25, 1)}
    nm = 3
    metrics = {}
    for name, (size, tf, tb, activation) in specs.items():
        for field, samples in (("parameter_bytes", [size]), ("gradient_bytes", [size]),
                               ("adamw_bytes", [2 * size + 4]), ("saved_activation_bytes", [activation] * nm),
                               ("forward_s", [tf] * nm), ("backward_s", [tb] * nm),
                               ("output_activation_bytes", [activation] * nm)):
            metrics[f"modules.{name}.{field}"] = {"samples": samples, "ema": samples[0]}
    metrics["modules.loss_and_runtime.saved_activation_bytes"] = {"samples": [4] * nm, "ema": 4}
    trace, now = [], 0.
    for mb in range(nm):
        for kind, position in (("forward", 1), ("backward", 2)):
            end = now + sum(spec[position] for spec in specs.values()) + .5
            trace.append(dict(kind=kind, micro_batch=mb, pipeline=0, stage=0, phase="steady", start_s=now, end_s=end))
            now = end
    for field in ("step_time_s", "step_wall_time_s"):
        metrics[field] = {"samples": [now + 1.], "ema": now + 1.}
    return {"schema_version": 3, "ema_alpha": .5,
            "identity": {"model_hash": "a" * 64, "config_hash": "b" * 64,
                         "module_parameter_bytes": {name: spec[0] for name, spec in specs.items()},
                         "module_order": list(specs),
                         "device": {"type": "cpu", "index": None, "torch": "fixture",
                                    "name": "fixture", "system": "fixture"},
                         "parallel": {"dp_size": 1, "pp_size": 1, "rank": 0}},
            "metrics": metrics, "calibrations": [],
            "steps": [{"step_id": 2, "trace": trace,
                       "memory": {"kind": "logical_tensor_bytes",
                                  "modules": {name: {"parameter_bytes": spec[0], "gradient_bytes": spec[0],
                                                     "adamw_bytes": spec[0] * 2 + 4} for name, spec in specs.items()},
                                  "output_activation_bytes": {name: spec[3] for name, spec in specs.items()},
                                  "saved_activation_bytes": {**{name: spec[3] for name, spec in specs.items()},
                                                             "loss_and_runtime": 4},
                                  "peak_allocated_bytes": None, "peak_reserved_bytes": None}}]}


def make_estimator(profile):
    return Estimator(profile, expected_identity=profile["identity"], layer_modules=("blocks.0", "blocks.1"))


LAYOUT = (("embedding", "blocks.0"), ("blocks.1", "final_norm", "lm_head", "extra"))


def test_profile_time_includes_endpoints_and_equation10_slowest_pipeline(measured_schema):
    estimator = make_estimator(measured_schema)
    forward, backward = estimator.stage_durations(LAYOUT)
    assert forward == (1.5, 3.125) and backward == (2.25, 4.75)
    result = estimator.dynamic_time((LAYOUT, (tuple(measured_schema["identity"]["module_order"]),)),
                                     (5, 2), global_micro_batches=7)
    expected_a, _ = event_oracle(forward, backward, 5)
    expected_b, _ = event_oracle((4.625,), (7.,), 2)
    assert result.step_time_s == max(expected_a, expected_b)
    assert result.derivation["pipeline_times_s"] == (expected_a, expected_b)
    assert result.derivation["equation"] == 10 and result.derivation["pipeline_equation"] == 11
    json.dumps(result.derivation, allow_nan=False)
    json.dumps(result.trace, allow_nan=False)


def test_profile_memory_separates_average_layers_from_actual_endpoints(measured_schema):
    estimator = make_estimator(measured_schema)
    result = estimator.memory(LAYOUT, (428, 389))
    assert result.feasible
    assert result.derivation["average_parameter_bytes"] == 20.
    assert result.derivation["average_optimizer_bytes"] == 44.
    assert result.derivation["average_gradient_bytes"] == 20.
    assert result.derivation["average_activation_bytes"] == 7.
    assert result.derivation["extra_static_bytes"] == (324, 284)
    assert result.derivation["extra_activation_bytes"] == (3., 14.)
    assert [row["peak_bytes"] for row in result.stages] == [428., 389.]
    assert result.derivation["source_kind"] == "logical_tensor_bytes"
    assert not estimator.memory(LAYOUT, (427, 389)).feasible


def test_json_key_sorting_cannot_change_model_execution_order(measured_schema):
    sorted_profile = json.loads(json.dumps(measured_schema, sort_keys=True))
    assert make_estimator(sorted_profile).stage_durations(LAYOUT) == ((1.5, 3.125), (2.25, 4.75))


def test_cleared_gradient_snapshot_does_not_reduce_training_peak(measured_schema):
    expected = make_estimator(measured_schema).memory(LAYOUT, (10000, 10000))
    for name, row in measured_schema["steps"][0]["memory"]["modules"].items():
        row["gradient_bytes"] = 0
        measured_schema["metrics"][f"modules.{name}.gradient_bytes"] = {"samples": [0], "ema": 0}
    actual = make_estimator(measured_schema).memory(LAYOUT, (10000, 10000))
    assert [row["peak_bytes"] for row in actual.stages] == [row["peak_bytes"] for row in expected.stages]


@pytest.mark.parametrize("name,field,values", [
    ("embedding", "saved_activation_bytes", [1.5, 3, 3]),
    ("embedding", "output_activation_bytes", [1.5, 3, 3]),
    ("loss_and_runtime", "saved_activation_bytes", [1.5, 4, 4]),
])
def test_raw_byte_samples_must_be_integers_even_below_peak(measured_schema, name, field, values):
    ema = values[0]
    for value in values[1:]:
        ema = .5 * value + .5 * ema
    measured_schema["metrics"][f"modules.{name}.{field}"] = {"samples": values, "ema": ema}
    with pytest.raises(ValueError, match="byte sample"):
        make_estimator(measured_schema)


def test_layer_inventory_is_copied_with_profile(measured_schema):
    layers = ["blocks.0", "blocks.1"]
    estimator = Estimator(measured_schema, expected_identity=measured_schema["identity"], layer_modules=layers)
    layers.clear()
    assert estimator.memory(LAYOUT, (10000, 10000)).derivation["average_parameter_bytes"] == 20.


def test_short_last_micro_batch_cannot_dilute_peak_activation(measured_schema):
    # Real per-forward sizes [12,12,6] must retain peak 12 rather than mean 10.
    measured_schema["steps"][0]["memory"]["saved_activation_bytes"]["embedding"] = 12
    measured_schema["metrics"]["modules.embedding.saved_activation_bytes"] = {"samples": [12, 12, 6], "ema": 9}
    result = make_estimator(measured_schema).memory(LAYOUT, (10000, 10000))
    assert result.stages[0]["peak_bytes"] == 446.


def test_smaller_later_profile_step_does_not_erase_observed_activation_peak(measured_schema):
    later = deepcopy(measured_schema["steps"][0])
    later["step_id"] = 3
    later["memory"]["saved_activation_bytes"]["embedding"] = 6
    measured_schema["steps"].append(later)
    for series in measured_schema["metrics"].values():
        series["samples"] *= 2
    measured_schema["steps"][0]["memory"]["saved_activation_bytes"]["embedding"] = 12
    measured_schema["metrics"]["modules.embedding.saved_activation_bytes"] = {
        "samples": [12, 12, 6, 6, 6, 6], "ema": 6.375,
    }
    result = make_estimator(measured_schema).memory(LAYOUT, (10000, 10000))
    assert result.stages[0]["peak_bytes"] == 446.
    assert result.derivation["profile_step_id"] == 3


@pytest.mark.parametrize("order", [None, "embedding", [], ["embedding"] * 6,
                                  ["embedding", "blocks.0", "blocks.1", "final_norm", "lm_head", "absent"],
                                  ["embedding", "blocks.0", "blocks.1", "final_norm", "lm_head", []]])
def test_module_order_must_cover_exact_unique_inventory(measured_schema, order):
    measured_schema["identity"]["module_order"] = order
    with pytest.raises(ValueError, match="module order"):
        make_estimator(measured_schema)


def test_operation_input_cannot_omit_local_forward_backward_dependency():
    forward, backward = build_1f1b_schedule(1, 1)[0]
    durations = {forward.key: 1., backward.key: 1.}
    result = estimate_operation_time(durations, num_stages=1, pipeline_micro_batches=1)
    assert result.step_time_s == 2.
    assert result.trace[1]["dependencies"] == (forward.key,)


def test_memory_retains_paper_activation_factor_when_nm_is_smaller_than_pp(measured_schema):
    layout = tuple((module,) for module in measured_schema["identity"]["module_order"])
    result = make_estimator(measured_schema).memory(layout, (10000,) * 6)
    assert result.derivation["measured_micro_batches"] == 3
    assert [row["activation_slots"] for row in result.stages] == [6, 5, 4, 3, 2, 1]
    assert [row["layer_count"] for row in result.stages] == [0, 1, 1, 0, 0, 0]


def test_profile_is_copied_and_no_hidden_initial_state_is_used(measured_schema):
    estimator = make_estimator(measured_schema)
    before = estimator.memory(LAYOUT, (428, 389))
    measured_schema["steps"][0]["memory"]["modules"]["embedding"]["adamw_bytes"] = 99999
    assert estimator.memory(LAYOUT, (428, 389)) == before


@pytest.mark.parametrize("layout", [(), (("embedding",),), (("embedding", "embedding"),),
                                  (("blocks.0", "embedding", "blocks.1", "final_norm", "lm_head", "extra"),),
                                  ((), ("embedding", "blocks.0", "blocks.1", "final_norm", "lm_head", "extra"))])
def test_profile_rejects_missing_duplicated_reordered_or_empty_layout(measured_schema, layout):
    estimator = make_estimator(measured_schema)
    with pytest.raises(ValueError):
        estimator.stage_durations(layout)
    with pytest.raises(ValueError):
        estimator.memory(layout, (100000,))


@pytest.mark.parametrize("counts,global_nm", [((0, 3), 3), ((2, 2), 5), ((1,), 1), ((True, 2), 3)])
def test_dynamic_time_enforces_global_micro_batch_conservation(measured_schema, counts, global_nm):
    with pytest.raises(ValueError):
        make_estimator(measured_schema).dynamic_time((LAYOUT, LAYOUT), counts, global_micro_batches=global_nm)


def test_estimator_rejects_stale_identity_and_missing_measurements(measured_schema):
    expected = deepcopy(measured_schema["identity"])
    expected["config_hash"] = "c" * 64
    with pytest.raises(ValueError, match="identity mismatch"):
        Estimator(measured_schema, expected_identity=expected, layer_modules=("blocks.0", "blocks.1"))
    del measured_schema["metrics"]["modules.embedding.forward_s"]
    with pytest.raises(ValueError, match="missing module measurement"):
        make_estimator(measured_schema)


def test_estimator_requires_real_steps_and_explicit_valid_layer_inventory(measured_schema):
    for layers in ((), ("absent",), ("blocks.0", "blocks.0")):
        with pytest.raises(ValueError, match="layer_modules"):
            Estimator(measured_schema, expected_identity=measured_schema["identity"], layer_modules=layers)
    measured_schema["steps"], measured_schema["metrics"] = [], {}
    with pytest.raises(ValueError, match="measured profile steps"):
        make_estimator(measured_schema)
