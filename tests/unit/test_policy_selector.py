from dataclasses import replace
import json

import pytest

from chameleon.contracts import DecisionResult
from chameleon.decision_center import NoUsablePolicyError, PolicyCandidate
from chameleon.estimators import MemoryEstimate
from chameleon.restorer import TransitionEstimate
from conftest import _select_candidates


def candidate(policy, step, transition, *, plan_id=None, reasons=(), memory=()):
    estimate = None if transition is None else TransitionEstimate(0., transition, 3., {"source": "controlled"}, None)
    return PolicyCandidate(plan_id or policy, policy, 10, 0, step, estimate, 3.,
                           memory, None, {"source": "controlled mathematical fixture"}, reasons)


@pytest.mark.parametrize("duration,policy,score", [(15., "rerouting", 5.), (40., "dynamic", 7.5)])
def test_independent_equation8_oracle_switches_even_when_dynamic_step_is_smaller(duration, policy, score):
    plans = (candidate("rerouting", 2., 0.), candidate("dynamic", 1., 10.))
    result = _select_candidates(plans, duration)
    # Hand calculations: rerouting = 10/2; dynamic = 10*(D-10)/D.
    assert result.plan.policy == policy
    assert result.score == score
    assert isinstance(result, DecisionResult)
    assert result.candidate is next(c for c in plans if c.policy == policy)
    rows = result.derivation["candidates"]
    assert [row["score"] for row in rows] == pytest.approx([5., 10. * (duration - 10.) / duration])
    assert sum(row["selected"] for row in rows) == 1
    assert result.derivation["D"] == duration and result.derivation["B"] == 10
    assert all(row["common_control_time_s"] == 3. for row in rows)
    assert next(row for row in rows if not row["selected"])["rejection_reasons"] == ("lower Equation 8 score",)
    json.dumps(result.derivation, allow_nan=False)


def test_break_even_uses_smaller_transition_before_plan_id():
    plans = (candidate("rerouting", 2., 0., plan_id="z"), candidate("dynamic", 1., 10., plan_id="a"))
    assert _select_candidates(plans, 20.).plan.plan_id == "z"


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_tie_uses_stable_plan_id(reverse):
    plans = (candidate("rerouting", 2., 0., plan_id="z"), candidate("dynamic", 2., 0., plan_id="a"))
    assert _select_candidates(plans[::-1] if reverse else plans, 20.).plan.plan_id == "a"


@pytest.mark.parametrize("duration", [None, 0, -1, float("nan"), float("inf"), -float("inf"), True, "15"])
def test_missing_or_invalid_duration_is_rejected(duration):
    with pytest.raises(ValueError, match="inter_fault_duration_s"):
        _select_candidates((candidate("rerouting", 2., 0.),), duration)


def test_omitting_duration_does_not_select_a_default_policy():
    from chameleon.decision_center import DecisionCenter

    center = object.__new__(DecisionCenter)
    center.evaluate_candidates = lambda state, profile: (candidate("rerouting", 2., 0.),)
    with pytest.raises(TypeError, match="inter_fault_duration_s"):
        center.select(None, None)


@pytest.mark.parametrize("policy", ["rerouting", "dynamic"])
def test_single_candidate(policy):
    result = _select_candidates((candidate(policy, 2., 1.),), 10.)
    assert result.plan.policy == policy and result.score == 4.5


@pytest.mark.parametrize("duration", [5., 10.])
def test_unavailable_dynamic_has_no_negative_score(duration):
    result = _select_candidates((candidate("rerouting", 2., 0.), candidate("dynamic", 1., 10.)), duration)
    assert result.plan.policy == "rerouting"
    assert result.derivation["candidates"][1]["score"] is None
    assert "D <= t_transition" in result.derivation["candidates"][1]["rejection_reasons"][0]


@pytest.mark.parametrize("plans", [(), (candidate("dynamic", 1., 10.),),
                                   (candidate("rerouting", None, None, reasons=("lost replica",)),)])
def test_no_usable_candidates_is_an_explicit_error(plans):
    with pytest.raises(NoUsablePolicyError, match="no usable policy") as error:
        _select_candidates(plans, 10.)
    assert all(row["score"] is None for row in error.value.derivation["candidates"])


def test_oom_candidate_is_excluded_even_if_faster():
    memory = MemoryEstimate((), {"equation": 14}, ("OOM",))
    plans = (candidate("rerouting", 2., 0.), candidate("dynamic", 1., 0., memory=(memory,)))
    result = _select_candidates(plans, 10.)
    assert result.plan.policy == "rerouting"
    assert result.derivation["candidates"][1]["rejection_reasons"] == ("OOM",)


@pytest.mark.parametrize("field,value", [
    ("estimated_step_time_s", 0), ("estimated_step_time_s", -1),
    ("estimated_step_time_s", float("nan")), ("estimated_step_time_s", float("inf")),
    ("estimated_step_time_s", True), ("estimated_step_time_s", "1"),
    ("migration_time_s", -1), ("migration_time_s", float("nan")),
    ("migration_time_s", float("inf")), ("migration_time_s", True),
    ("migration_time_s", "0"), ("common_control_time_s", -1),
])
def test_invalid_candidate_times(field, value):
    with pytest.raises(ValueError, match=field):
        plan = candidate("dynamic", 1., 0.)
        if field == "migration_time_s":
            replace(plan.transition, migration_time_s=value)
        else:
            replace(plan, **{field: value})


@pytest.mark.parametrize("field", ["estimated_step_time_s", "transition"])
def test_missing_estimate_requires_explicit_infeasibility_reason(field):
    with pytest.raises(ValueError, match="require step and transition"):
        replace(candidate("dynamic", 1., 0.), **{field: None})


@pytest.mark.parametrize("field,value", [("global_batch_size", 11), ("generation", 1), ("policy", "rerouting")])
def test_mixed_batch_generation_or_duplicate_policy_is_rejected(field, value):
    plans = (candidate("rerouting", 2., 0.), replace(candidate("dynamic", 1., 0.), **{field: value}))
    with pytest.raises(ValueError, match="unique policies"):
        _select_candidates(plans, 10.)


def test_nonfinite_computed_score_is_rejected():
    with pytest.raises(ValueError, match="Equation 8 score"):
        _select_candidates((candidate("dynamic", 1e-320, 0.),), 10.)


def test_duplicate_plan_ids_are_rejected_to_keep_tie_break_deterministic():
    plans = (candidate("rerouting", 2., 0., plan_id="same"), candidate("dynamic", 2., 0., plan_id="same"))
    with pytest.raises(ValueError, match="plan IDs"):
        _select_candidates(plans, 10.)


@pytest.mark.parametrize("change", ["candidate", "score"])
def test_decision_is_bound_to_its_candidate_and_equation8_score(change):
    first = candidate("rerouting", 2., 0.)
    result = _select_candidates((first,), 15.)
    with pytest.raises(ValueError, match="decision"):
        if change == "candidate":
            replace(result, candidate=candidate("dynamic", 1., 10.))
        else:
            replace(result, score=result.score + 1.)


def test_memory_infeasibility_is_a_complete_reason_even_without_time_measurements():
    memory = MemoryEstimate((), {"equation": 14}, ("OOM",))
    unavailable = candidate("dynamic", None, None, memory=(memory,))
    assert not unavailable.feasible and unavailable.reasons == ("OOM",)


def test_infeasibility_reasons_are_reported_once():
    memory = MemoryEstimate((), {"equation": 14}, ("OOM",))
    unavailable = candidate("dynamic", None, None, memory=(memory,), reasons=("OOM",))
    with pytest.raises(NoUsablePolicyError) as error:
        _select_candidates((unavailable,), 15.)
    assert error.value.derivation["candidates"][0]["rejection_reasons"] == ("OOM",)


def test_unknown_common_control_is_distinct_from_zero_transition():
    estimate = TransitionEstimate(0., 0., None, {"source": "paper model"}, None)
    assert estimate.estimated_transition_time_s == 0.
    assert estimate.estimated_total_time_s is None


@pytest.mark.parametrize("field,value", [("unoverlapped_search_time_s", -1.),
                                        ("migration_time_s", float("inf")),
                                        ("common_control_time_s", float("nan"))])
def test_transition_components_are_validated_before_they_can_enter_scoring(field, value):
    with pytest.raises(ValueError, match=field):
        replace(candidate("dynamic", 1., 0.).transition, **{field: value})
