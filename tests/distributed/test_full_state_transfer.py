"""Real survivor P2P, with pre/post hashes used only as test assertions."""

from collections import Counter

import pytest

from chameleon.state_sources import ADAMW_FIELDS
from test_symmetric_training import assert_numerical_step


def test_complete_model_and_adamw_hashes_survive_missing_only_p2p(recovered_training):
    result = recovered_training
    transfer = result["recovery"]
    before = {row["worker"].worker_id: {tuple(t["key"]): t["digest"] for t in row["hashes"]}
              for row in result["before"]}
    manifest = result["decision"].candidate.execution
    targets = {row["worker"]["worker_id"]: row for row in transfer["targets"]}
    for action in manifest.actions:
        actual = next(t["digest"] for t in targets[action.destination.worker_id]["hashes"]
                      if tuple(t["key"]) == action.tensor.key)
        assert actual == before[action.source.worker_id][action.tensor.key]
    sent = Counter((r["source"], r["destination"], tuple(r["key"]))
                   for target in transfer["targets"] for r in target["sends"])
    received = Counter((r["source"], r["destination"], tuple(r["key"]))
                       for target in transfer["targets"] for r in target["receives"])
    expected = Counter((a.source.worker_id, a.destination.worker_id, a.tensor.key)
                       for a in manifest.actions if not a.retained)
    assert sent == received == expected and expected
    assert sum(r["tensor_bytes"] for target in transfer["targets"] for r in target["sends"]) == manifest.migration_bytes
    modules = {a.tensor.module_id for a in manifest.actions}
    assert modules == {"embedding", "blocks.0", "blocks.1", "final_norm", "lm_head"}
    assert {a.tensor.kind for a in manifest.actions} == set(ADAMW_FIELDS)
    for worker in result["before"] + transfer["targets"]:
        expected_names = {t["key"][0] for t in worker["hashes"] if t["key"][1] == "parameter"}
        assert {s["parameter_name"] for s in worker["parameter_steps"]} == expected_names
        assert len(worker["parameter_steps"]) == len(expected_names)
        assert all(s["step"] == 3 and s["exp_avg_nonzero"] > 0 and s["exp_avg_sq_nonzero"] > 0
                   for s in worker["parameter_steps"])


def test_intersection_and_source_lifetime_are_audited_at_target_ack(recovered_training):
    result = recovered_training
    manifest = result["decision"].candidate.execution
    assert manifest.release_after_ack
    for target in result["recovery"]["targets"]:
        assert all(r["same_object"] for r in target["retained"])
        assert all(r["unchanged"] for r in target["held_before_ack"])
        expected = {t.key for w, t in manifest.held_sources if w.worker_id == target["worker"]["worker_id"]}
        assert {tuple(r["key"]) for r in target["held_before_ack"]} == expected
    assert result["recovery"]["source_release_after_all_target_acks"]
    assert all(r["released_after_ack"] for r in result["recovery"]["installed"])


def test_three_updates_then_two_resumed_updates_match_uninterrupted_reference(recovered_training):
    result = recovered_training
    for actual, expected in zip(result["steps"], result["reference"]):
        assert_numerical_step(actual, expected, result["device"])
    assert [r["step_id"] for r in result["steps"]] == [1, 2, 3, 4, 5]
    assert result["recovery"]["committed_global_step"] == 3
    assert result["runtime"].state.committed_global_step == 5


@pytest.mark.parametrize("fault", ["missing_manifest_tensor", "missing_endpoint_source", "transfer_error",
                                 "source_mutation", "group_timeout"])
def test_failed_recovery_never_commits_partial_topology(recovery_failure, fault):
    recovery_failure(fault)
