from collections import Counter

import pytest

from test_rerouted_training import (assert_rerouted_numerics, assert_rerouted_owners_and_cleanup,
                                    assert_rerouted_trace, assert_rerouted_transfers)


@pytest.fixture(scope="module")
def scaled_training(rerouted_run, request):
    if request.config.getoption("--world-size") != 7:
        pytest.fail("Task11 DP4/PP2 scale acceptance requires --world-size 7")
    return rerouted_run(((0, None), (1, 2), (3, 4), (5, 6)), (6, 2, 2, 2), batch=23)


def test_seven_workers_three_peers_share_six_extra_tasks(scaled_training):
    result = scaled_training
    assert result["topology"].logical_slots == 8
    assert len(result["runtime"].ready) == 7
    for step in result["steps"]:
        extra = [(row["micro_batch"], report["worker"]["rank"]) for report in step["reports"]
                 for row in report["trace"] if row["pipeline"] == 0 and row["stage"] == 1 and row["kind"] == "forward"]
        assert sorted(extra) == list(enumerate((6, 4, 2, 6, 4, 2)))
        assert Counter(rank for _, rank in extra) == Counter({2: 2, 4: 2, 6: 2})
    assert_rerouted_owners_and_cleanup(result)


def test_scaled_training_matches_reference_and_preserves_real_1f1b_p2p(scaled_training):
    assert_rerouted_numerics(scaled_training)
    assert_rerouted_trace(scaled_training)
    assert_rerouted_transfers(scaled_training)
