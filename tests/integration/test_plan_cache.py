from copy import deepcopy

import pytest

from chameleon.decision_center import DecisionCenter, select_policy
from chameleon.plan_cache import PlanCache
from chameleon.planner import Planner
from test_decision_center_oracle import scenario


def _cached_center(center, cache):
    return DecisionCenter(
        center.config,
        expected_identity=center.expected_identity,
        r_dp=center.r_dp,
        r_pp=center.r_pp,
        memory_capacity_bytes=center.memory_capacity_bytes,
        plan_cache=cache,
    )


def test_precompute_accepts_one_through_k_failures_and_reuses_dynamic_search(monkeypatch):
    original, one_failure, profile = scenario(failures=(0,))
    _, two_failures, _ = scenario(failures=(0, 1))
    cache = PlanCache()
    center = _cached_center(original, cache)
    calls = 0
    search = Planner.best_dynamic_plan

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return search(*args, **kwargs)

    monkeypatch.setattr(Planner, "best_dynamic_plan", counted)
    records = center.precompute_dynamic((one_failure, two_failures), profile, max_failures=2)
    assert calls == 2
    assert [row["failed_workers"] for row in records] == [1, 2]
    assert all(not row["cache_hit"] for row in records)
    assert cache.stats == {"entries": 2, "hits": 0, "misses": 2}

    rerouting, dynamic = center.evaluate_candidates(one_failure, profile)
    assert rerouting.feasible and dynamic.feasible
    assert calls == 2
    assert dynamic.derivation["dynamic_search_cache"]["hit"] is True
    assert dynamic.transition.unoverlapped_search_time_s == 0
    assert cache.stats == {"entries": 2, "hits": 1, "misses": 2}


def test_duration_is_not_cached_and_equation8_is_recomputed_for_each_d():
    original, state, profile = scenario()
    cache = PlanCache()
    center = _cached_center(original, cache)
    center.precompute_dynamic((state,), profile, max_failures=1)

    first = center.evaluate_candidates(state, profile)
    rerouting, dynamic = first
    crossover = dynamic.estimated_transition_time_s / (
        1 - dynamic.estimated_step_time_s / rerouting.estimated_step_time_s
    )
    short = select_policy(first, crossover * 0.75)
    second = center.evaluate_candidates(state, profile)
    long = select_policy(second, crossover * 2)

    assert short.plan.policy == "rerouting"
    assert long.plan.policy == "dynamic"
    assert short.inter_fault_duration_s != long.inter_fault_duration_s
    assert short.derivation["D"] == short.inter_fault_duration_s
    assert long.derivation["D"] == long.inter_fault_duration_s
    assert first[1].plan_id == second[1].plan_id
    assert cache.stats == {"entries": 1, "hits": 2, "misses": 1}


def test_post_failure_cache_miss_keeps_search_time_and_can_change_equation8(monkeypatch):
    original, state, profile = scenario()
    precomputed = _cached_center(original, PlanCache())
    precomputed.precompute_dynamic((state,), profile, max_failures=1)
    assert select_policy(precomputed.evaluate_candidates(state, profile), 100.).plan.policy == "dynamic"

    clock = iter((10., 110.))
    monkeypatch.setattr("chameleon.decision_center.perf_counter", lambda: next(clock))
    candidates = _cached_center(original, PlanCache()).evaluate_candidates(state, profile)
    dynamic = candidates[1]

    assert dynamic.derivation["dynamic_search_cache"]["hit"] is False
    assert dynamic.derivation["unoverlapped_search_time_s"] == 100.
    assert dynamic.transition.unoverlapped_search_time_s == 100.
    assert select_policy(candidates, 100.).plan.policy == "rerouting"


def test_state_model_config_and_full_profile_identity_are_cache_keys(monkeypatch):
    original, state, profile = scenario()
    cache = PlanCache()
    center = _cached_center(original, cache)
    calls = 0
    search = Planner.best_dynamic_plan

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return search(*args, **kwargs)

    monkeypatch.setattr(Planner, "best_dynamic_plan", counted)
    center.precompute_dynamic((state,), profile, max_failures=1)
    center.evaluate_candidates(state, profile)
    assert calls == 1

    changed_profile = deepcopy(profile)
    for record in changed_profile["calibrations"][0]["records"]:
        for transfer in record["transfers"]:
            transfer["execution_time_s"] += 0.125
            transfer["wall_time_s"] += 0.125
    center.evaluate_candidates(state, changed_profile)
    assert calls == 2

    changed_model = deepcopy(changed_profile)
    changed_model["identity"]["model_hash"] = "f" * 64
    changed_center = DecisionCenter(
        original.config,
        expected_identity=changed_model["identity"],
        r_dp=original.r_dp,
        r_pp=original.r_pp,
        memory_capacity_bytes=original.memory_capacity_bytes,
        plan_cache=cache,
    )
    changed_center.evaluate_candidates(state, changed_model)
    assert calls == 3

    _, other_state, other_profile = scenario(failures=(1,))
    center.evaluate_candidates(other_state, other_profile)
    assert calls == 4
    assert cache.stats["entries"] == 4


def test_precompute_rejects_out_of_range_or_missing_scenarios_without_search(monkeypatch):
    original, state, profile = scenario()
    cache = PlanCache()
    center = _cached_center(original, cache)
    monkeypatch.setattr(Planner, "best_dynamic_plan", lambda *_: pytest.fail("search must not run"))

    with pytest.raises(ValueError, match="nonempty"):
        center.precompute_dynamic((), profile, max_failures=1)
    with pytest.raises(ValueError, match="max_failures"):
        center.precompute_dynamic((state,), profile, max_failures=0)
    _, two_failures, _ = scenario(failures=(0, 1))
    with pytest.raises(ValueError, match="1..max_failures"):
        center.precompute_dynamic((two_failures,), profile, max_failures=1)
    assert cache.stats == {"entries": 0, "hits": 0, "misses": 0}
