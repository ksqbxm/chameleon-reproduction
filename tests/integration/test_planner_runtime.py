from dataclasses import asdict
import json
from pathlib import Path

import pytest

from chameleon import ClusterState, ModelConfig, WorkerIdentity
from chameleon.estimators import Estimator
from chameleon.planner import Planner, repair_zero_partitions
from chameleon.profiler import Profiler, export_profile, load_profile, train_profile_step
from chameleon.restorer import Restorer
from chameleon.runtime import DynamicTopology, SymmetricRuntime
from chameleon.state_sources import WorkerInventory, adamw_inventory, build_state_source_map


@pytest.fixture(params=["ordinary", "repaired_zero"])
def planned_training(request, distributed_environment, device, world_size):
    import torch
    from chameleon.model import build_initial_model
    from chameleon.reference import ReferenceTrainer

    torch.set_num_threads(1)
    size = 8 if device == "cuda" else 7
    assert world_size == size
    batch = 5 if request.param == "repaired_zero" else 19
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=batch, micro_batch_size=2)
    # This independent test model supplies measured profile/inventory metadata only.
    # Its trained tensor values never enter the initial runtime or Restorer.
    model = build_initial_model(config, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.007, weight_decay=.125, amsgrad=False)
    profiling_state = ClusterState((WorkerIdentity("profile", 0, 0),), batch)
    profiling_state, _ = train_profile_step(model, optimizer, profiling_state)
    profiler = Profiler(model, optimizer)
    for _ in range(2):
        profiling_state, _ = train_profile_step(model, optimizer, profiling_state, profiler=profiler)
    profile = profiler.snapshot()
    required = adamw_inventory(model, optimizer, committed_global_step=3)
    workers = tuple(WorkerIdentity(f"planner-{r}", r, 2) for r in reversed(range(size)))
    state = ClusterState(workers, batch, generation=2, committed_global_step=3)
    estimator = Estimator(profile, expected_identity=profile["identity"], layer_modules=("blocks.0", "blocks.1"))
    planner = Planner(estimator, config=config, r_dp=(3,), r_pp=(2, 3), memory_capacity_bytes=10**12)
    plan = planner.best_dynamic_plan(state)
    # Complete fragments from prior replicas, deliberately assigned in reverse rank order.
    slots = [(p, s, modules) for p, layout in enumerate(plan.layouts) for s, modules in enumerate(layout)]
    inventories = tuple(WorkerInventory(worker, 3, tuple(t for t in required if t.module_id in slots[i][2]))
                        for i, worker in enumerate(workers))
    sources = build_state_source_map(state, required, inventories)
    manifest = Restorer(profile, expected_identity=profile["identity"]).plan(plan, sources)
    topology = DynamicTopology.from_plan(plan, config, assignments=manifest.assignments)
    assert topology.synchronization_rounds == manifest.synchronization_rounds
    assert topology.state.committed_global_step == 0  # New topology; Task12 restores old state.
    del profiler, model, optimizer
    runtime = SymmetricRuntime(topology, device=device, capture_state=True, lr=.007, weight_decay=.125)
    with runtime:
        steps = [runtime.train_step() for _ in range(3)]
        profiles = runtime.snapshot_profiles()
    assert runtime.audit["clean"] and all(w["exitcode"] == 0 for w in runtime.audit["workers"])
    request.config._chameleon_reports.add(runtime.report_path)
    reference = ReferenceTrainer(build_initial_model(config, device=device),
                                 ClusterState((WorkerIdentity("reference", 0, 0),), batch), lr=.007, weight_decay=.125)
    path = Path("artifacts/test-results") / f"task10-planner-{device}-{request.param}.json"
    path.write_text(json.dumps(dict(profile=profile, plan=asdict(plan),
                                    assignments=[(asdict(s), asdict(w)) for s, w in manifest.assignments],
                                    synchronization_rounds=manifest.synchronization_rounds,
                                    runtime_report=str(runtime.report_path), initial_topology_only=True),
                               indent=2, allow_nan=False), encoding="utf-8")
    return dict(plan=plan, topology=topology, manifest=manifest, steps=steps, profiles=profiles,
                reference=[reference.train_step() for _ in range(3)], device=device, case=request.param)


def test_live_profiler_estimator_planner_restorer_schedule_executes(planned_training):
    import torch

    result = planned_training
    plan, topology = result["plan"], result["topology"]
    assert plan.pipeline_lengths == ((2, 3, 3) if result["device"] == "cuda" else (2, 2, 3))
    assert tuple(rank for ranks in topology.pipeline_ranks for rank in ranks) != tuple(range(len(topology.ranks)))
    assert plan.estimated_step_time_s == max(plan.time.derivation["pipeline_times_s"])
    if result["case"] == "repaired_zero":
        lengths = plan.pipeline_lengths
        base = tuple(3 * length // sum(lengths) for length in lengths)
        assert 0 in base
        raw = (*base[:-1], 3 - sum(base[:-1]))
        assert repair_zero_partitions(raw) == plan.pipeline_micro_batches == (1, 1, 1)
    tolerance = dict(rtol=1e-7, atol=1e-9) if result["device"] == "cuda" else dict(rtol=1e-8, atol=1e-10)
    for actual, expected in zip(result["steps"], result["reference"]):
        assert actual["sample_ids"] == list(expected.sample_ids)
        assert actual["global_sample_count"] == expected.global_sample_count
        assert actual["loss_global_sum"] == pytest.approx(expected.loss_global_sum, rel=tolerance["rtol"], abs=tolerance["atol"])
        owners = {}
        for report, snapshot in zip(actual["reports"], actual["snapshots"]):
            p, s = topology.location(report["worker"]["rank"])
            assert (report["pipeline"], report["stage"]) == (p, s)
            assert len(report["micro_batches"]) == plan.pipeline_micro_batches[p]
            assert report["synchronization_rounds"] == [list(row) for row in topology.synchronization_rounds]
            for name, parameter in snapshot["parameters"].items():
                owners[name] = owners.get(name, 0) + 1
                torch.testing.assert_close(parameter, expected.parameters[name], **tolerance)
                torch.testing.assert_close(snapshot["gradients"][name], expected.gradients[name], **tolerance)
                for key, value in snapshot["optimizer_state"][name].items():
                    torch.testing.assert_close(value, expected.optimizer_state[name][key], **tolerance)
        assert owners == {name: 3 for name in expected.parameters}


def test_nonuniform_local_profiles_roundtrip_and_use_real_depth(planned_training):
    result = planned_training
    topology = result["topology"]
    for rank, profile in enumerate(result["profiles"]):
        p, s = topology.location(rank)
        assert profile["identity"]["module_order"] == list(topology.layouts[p][s])
        assert profile["identity"]["parallel"] == dict(dp_size=3, pp_size=3, rank=rank,
                                                      pipeline_lengths=list(topology.pipeline_lengths))
        count = topology.pipeline_micro_batches[p]
        assert [step["step_id"] for step in profile["steps"]] == [2, 3]
        assert all(len(step["trace"]) == 2 * count for step in profile["steps"])
        path = Path("artifacts/test-results") / f"task10-local-{result['device']}-{result['case']}-{rank}.json"
        export_profile(profile, str(path), expected_identity=profile["identity"])
        assert load_profile(str(path), expected_identity=profile["identity"]) == profile


def test_actual_nonuniform_profiles_feed_equation11_and_maximum(asymmetric_training):
    result = asymmetric_training
    topology = result["topology"]
    for step in result["steps"][1:]:
        comparison = step["profile_comparison"]
        expected = []
        for p, depth in enumerate(topology.pipeline_lengths):
            count = topology.pipeline_micro_batches[p]
            durations, previous = {}, {}
            for report in step["reports"]:
                if report["pipeline"] != p:
                    continue
                s = report["stage"]
                last = None
                warmup = min(depth - s - 1, count)
                order = [("forward", mb) for mb in range(warmup)]
                for mb in range(count - warmup):
                    order += [("forward", mb + warmup), ("backward", mb)]
                order += [("backward", mb) for mb in range(count - warmup, count)]
                measured = {(r["kind"], r["micro_batch"]): r for r in report["profile_step"]["trace"]}
                for kind, mb in order:
                    key = s, kind, mb
                    row = measured[kind, mb]
                    durations[key] = row["end_s"] - row["start_s"]
                    previous[key] = last
                    last = key
            # Independent dependency recurrence over the hand-specified FIFO queues.
            finished = {}

            def finish(key):
                if key not in finished:
                    s, kind, mb = key
                    dependencies = [previous[key]] if previous[key] is not None else []
                    if kind == "forward" and s:
                        dependencies.append((s - 1, "forward", mb))
                    if kind == "backward":
                        dependencies.append((s, "forward", mb))
                        if s + 1 < depth:
                            dependencies.append((s + 1, "backward", mb))
                    finished[key] = max((finish(k) for k in dependencies), default=0) + durations[key]
                return finished[key]

            expected.append(max(finish(key) for key in durations))
            assert comparison["equation11_pipelines"][p]["step_time_s"] == pytest.approx(expected[-1])
            assert comparison["equation11_pipelines"][p]["derivation"]["pipeline_micro_batches"] == count
        assert comparison["estimated_compute_time_s"] == pytest.approx(max(expected))
        assert comparison["equation9"] is None
        assert comparison["equation10"]["global_micro_batches"] == 10
        assert comparison["measured_pipeline_time_s"] == max(r["pipeline_wall_time_s"] for r in step["reports"])
