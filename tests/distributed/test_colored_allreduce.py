from collections import Counter


def test_groups_cover_every_trainable_parameter_and_endpoint(asymmetric_training):
    result = asymmetric_training
    topology, runtime = result["topology"], result["runtime"]
    specifications = list(topology.pipeline_ranks) + list(dict.fromkeys(topology.module_owners.values()))
    actual = {}
    for row in runtime.ready:
        assert row["groups"] == [list(ranks) for ranks in specifications]
        assert row["module_owners"] == {m: list(ranks) for m, ranks in topology.module_owners.items()}
        assert row["parameter_owners"].keys() == set(row["parameter_names"])
        for name, ranks in row["parameter_owners"].items():
            module = ".".join(name.split(".")[:2]) if name.startswith("blocks.") else name.split(".")[0]
            assert ranks == list(topology.module_owners[module])
            assert row["worker"].rank in ranks
            assert name not in actual or actual[name] == ranks
            actual[name] = ranks
    assert actual.keys() == result["reference"][0].parameters.keys()
    assert all(len(ranks) == 3 for ranks in actual.values())


def test_actual_async_sum_launches_follow_conflict_free_dsatur_rounds(asymmetric_training):
    result = asymmetric_training
    topology = result["topology"]
    rounds = topology.synchronization_rounds
    assert any(len(row) > 1 for row in rounds)
    for modules in rounds:
        devices = [rank for module in modules for rank in topology.module_owners[module]]
        assert len(devices) == len(set(devices))
    for step in result["steps"]:
        counted = Counter()
        rows = []
        for report in step["reports"]:
            assert report["synchronization_rounds"] == [list(row) for row in rounds]
            assert len(report["allreduces"]) == len(report["synchronized_parameters"])
            for row in report["allreduces"]:
                counted[row["parameter"]] += 1
                assert row["module"] in rounds[row["color"]]
                assert row["owner_ranks"] == list(topology.module_owners[row["module"]])
                assert row["async_op"] is True and row["reduction"] == "SUM"
                assert row["divisor"] == 19 and row["tensor_bytes"] > 0
                assert row["device"] == report["device"] and row["backend"] == report["backend"]
                assert row["start_s"] <= row["end_s"] <= report["optimizer_completed_s"]
                rows.append(row)
        assert counted == Counter({name: 3 for name in result["reference"][0].parameters})
        for color in range(1, len(rounds)):
            previous = [r for r in rows if r["color"] == color - 1]
            current = [r for r in rows if r["color"] == color]
            assert min(r["start_s"] for r in current) >= max(r["end_s"] for r in previous)
