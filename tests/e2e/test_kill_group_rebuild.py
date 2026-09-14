"""DP2/PP2 safe-point kill: existing PIDs join generation+1 and resume training."""


def test_real_kill_preserves_survivor_pids_and_physical_devices(recovered_training):
    result = recovered_training
    recovery, original, runtime = result["recovery"], result["original"], result["runtime"]
    assert len(recovery["killed"]) == 1
    killed = recovery["killed"][0]
    assert killed["exitcode"] not in (None, 0) and not killed["alive"]
    assert len(runtime.ready) == 3
    survivors = {r["worker"].worker_id: r for r in original if r["pid"] != killed["pid"]}
    assert {r["worker"].worker_id for r in runtime.ready} == survivors.keys()
    assert killed["pid"] not in {r["pid"] for r in runtime.ready}
    assert any(r["worker"].rank != survivors[r["worker"].worker_id]["worker"].rank for r in runtime.ready)
    for row in runtime.ready:
        prior = survivors[row["worker"].worker_id]
        assert row["pid"] == prior["pid"]
        assert row["device"] == prior["device"]
        assert row["worker"].generation == prior["worker"].generation + 1
    assert recovery["actual_group_rebuild_s"] > 0
    assert recovery["actual_transfer_validation_s"] > 0
    assert all(row["backend"] == runtime.backend for row in runtime.ready)
    assert runtime.audit["clean"] and not runtime.audit["leaked_pids"]
    assert runtime.audit["rendezvous_port"] is None
    assert runtime.audit["rendezvous_removed"] and runtime.audit["rendezvous_file_removed"]
    assert len(runtime.audit["rendezvous_files"]) == 2
    assert all(row["removed"] for row in runtime.audit["rendezvous_files"])
    assert len(runtime.audit["workers"]) == 4
    assert sum(r["exitcode"] == 0 for r in runtime.audit["workers"]) == 3


def test_worker_initialization_and_checkpoint_read_guards_survive_recovery(recovered_training):
    for row in recovered_training["original"]:
        assert row["initialization_calls"] == {"model": 1, "constructor": 1}
    for row in recovered_training["recovery"]["targets"]:
        assert row["initialization_calls"] == {"model": 1, "constructor": 1}
        assert row["checkpoint_reads"] == 0
