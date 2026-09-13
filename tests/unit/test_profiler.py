from copy import deepcopy
import math

import pytest

from chameleon import ClusterState, ModelConfig, WorkerIdentity
from chameleon.profiler import Measurements, Profiler, tensor_inventory, train_profile_step


def test_raw_samples_and_ema_match_hand_calculation():
    measurements = Measurements(0.25)
    for value in (4., 8., 0., 12.):
        measurements.add("forward_s", value)
    assert measurements.metrics["forward_s"] == {
        "samples": [4., 8., 0., 12.], "ema": 5.8125,
    }
    measurements.add("backward_s", 2.)
    assert measurements.metrics["backward_s"] == {"samples": [2.], "ema": 2.}


@pytest.mark.parametrize("alpha", [0, -1, 1.1, True, float("nan"), float("inf")])
def test_invalid_ema_alpha(alpha):
    with pytest.raises(ValueError):
        Measurements(alpha)


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf")])
def test_invalid_measurements(value):
    with pytest.raises(ValueError):
        Measurements().add("step_time_s", value)


@pytest.fixture
def torch_module():
    import torch
    return torch


@pytest.fixture
def training(torch_module, device):
    from chameleon.model import build_initial_model
    config = ModelConfig(vocab_size=7, hidden_size=4, num_layers=2, num_heads=1,
                         sequence_length=3, global_batch_size=5, micro_batch_size=2)
    model = build_initial_model(config, device=device)
    optimizer = torch_module.optim.AdamW(model.parameters(), lr=0.007, amsgrad=False)
    state = ClusterState((WorkerIdentity("profile-worker", 0, 0),), config.global_batch_size)
    return model, optimizer, state


def test_adamw_requires_real_warmup(training):
    model, optimizer, state = training
    with pytest.raises(ValueError, match="warmed up"):
        tensor_inventory(model, optimizer)
    profiler = Profiler(model, optimizer)
    with pytest.raises(ValueError, match="warmed up"):
        train_profile_step(model, optimizer, state, profiler=profiler)
    assert state.committed_global_step == 0
    assert not profiler.steps
    state, _ = train_profile_step(model, optimizer, state)
    assert state.committed_global_step == 1
    assert tensor_inventory(model, optimizer)


def test_inventory_matches_independent_tensors_and_includes_endpoints(training, torch_module):
    model, optimizer, state = training
    train_profile_step(model, optimizer, state)
    inventory = tensor_inventory(model, optimizer)
    assert set(inventory) == {"embedding", "blocks.0", "blocks.1", "final_norm", "lm_head"}
    for module_id, row in inventory.items():
        tensors = [p for name, p in model.named_parameters()
                   if p.requires_grad and name.startswith(module_id + ".")]
        assert row == {
            "parameter_bytes": sum(p.numel() * p.element_size() for p in tensors),
            "gradient_bytes": sum(p.grad.numel() * p.grad.element_size() for p in tensors),
            "adamw_bytes": sum(t.numel() * t.element_size()
                               for p in tensors for t in optimizer.state[p].values()),
        }
    # New trainable modules must appear without a hardcoded endpoint allowlist.
    model.extra = torch_module.nn.Linear(4, 4, device=next(model.parameters()).device,
                                        dtype=torch_module.float64)
    optimizer.add_param_group({"params": list(model.extra.parameters())})
    for p in model.extra.parameters():
        p.grad = torch_module.ones_like(p)
    optimizer.step()
    assert "extra" in tensor_inventory(model, optimizer)
    model.extra.bias.requires_grad_(False)
    optimizer.param_groups[-1]["params"] = [model.extra.weight]
    assert tensor_inventory(model, optimizer)["extra"]["parameter_bytes"] == 4 * 4 * 8


def test_live_timing_trace_and_output_activation_bytes(training, device, torch_module):
    model, optimizer, state = training
    state, _ = train_profile_step(model, optimizer, state)
    profiler = Profiler(model, optimizer, ema_alpha=0.25)
    state, loss = train_profile_step(model, optimizer, state, profiler=profiler)
    assert state.committed_global_step == 2 and math.isfinite(loss)
    snapshot = profiler.snapshot()
    trace = snapshot["steps"][0]["trace"]
    # Independent single-stage oracle: each of three micro-batches executes F then B.
    assert [(r["kind"], r["micro_batch"], r["pipeline"], r["stage"], r["phase"]) for r in trace] == [
        (kind, mb, 0, 0, "steady") for mb in range(3) for kind in ("forward", "backward")
    ]
    assert all(0 <= row["start_s"] <= row["end_s"] for row in trace)
    assert all(left["end_s"] <= right["start_s"] for left, right in zip(trace, trace[1:]))
    memory = snapshot["steps"][0]["memory"]
    for module_id in memory["modules"]:
        for direction in ("forward", "backward"):
            values = snapshot["metrics"][f"modules.{module_id}.{direction}_s"]["samples"]
            assert len(values) == 3
            assert all(math.isfinite(value) and value >= 0 for value in values)
        width = 7 if module_id == "lm_head" else 4
        assert snapshot["metrics"][f"modules.{module_id}.output_activation_bytes"]["samples"] == [
            samples * 3 * width * 8 for samples in (2, 2, 1)
        ]
    assert snapshot["metrics"]["step_time_s"]["ema"] > 0
    for mb in range(3):
        measured = sum(snapshot["metrics"][f"modules.{name}.backward_s"]["samples"][mb]
                       for name in memory["modules"])
        backward = trace[mb * 2 + 1]
        assert measured <= backward["end_s"] - backward["start_s"] + 1e-12
    if device == "cpu":
        assert memory["kind"] == "logical_tensor_bytes"
        assert memory["peak_allocated_bytes"] is memory["peak_reserved_bytes"] is None
    else:
        assert memory["kind"] == "cuda_hbm"
        tensor_bytes = sum(sum(row.values()) for row in memory["modules"].values())
        # AdamW step scalars reside on CPU by default, so exclude them from HBM.
        cpu_state_bytes = sum(t.numel() * t.element_size()
                              for state in optimizer.state.values() for t in state.values()
                              if t.device.type == "cpu")
        assert memory["peak_reserved_bytes"] >= memory["peak_allocated_bytes"] >= tensor_bytes - cpu_state_bytes


def test_profiling_preserves_training_values(training, torch_module, device):
    model, optimizer, state = training
    state, _ = train_profile_step(model, optimizer, state)
    baseline_model = deepcopy(model)
    baseline_optimizer = torch_module.optim.AdamW(baseline_model.parameters(), lr=0.007, amsgrad=False)
    baseline_optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
    profiler = Profiler(model, optimizer)
    baseline_state, baseline_loss = train_profile_step(baseline_model, baseline_optimizer, state)
    measured_state, measured_loss = train_profile_step(model, optimizer, state, profiler=profiler)
    tolerance = {"rtol": 1e-7, "atol": 1e-9} if device == "cuda" else {"rtol": 1e-8, "atol": 1e-10}
    assert measured_state == baseline_state
    assert measured_loss == pytest.approx(baseline_loss, rel=tolerance["rtol"], abs=tolerance["atol"])
    for p, expected in zip(model.parameters(), baseline_model.parameters()):
        torch_module.testing.assert_close(p, expected, **tolerance)
        torch_module.testing.assert_close(p.grad, expected.grad, **tolerance)
        for key, value in optimizer.state[p].items():
            torch_module.testing.assert_close(value, baseline_optimizer.state[expected][key], **tolerance)


def test_saved_activation_bytes_match_independent_autograd_inventory(training, torch_module):
    from chameleon.data import make_batch, next_sample_ids
    from chameleon.global_loss import micro_batch_loss_sum

    model, optimizer, state = training
    state, _ = train_profile_step(model, optimizer, state)
    persistent = {t.untyped_storage().data_ptr() for t in (*model.parameters(), *model.buffers())}
    saved_sizes = []
    def pack(tensor):
        if tensor.untyped_storage().data_ptr() not in persistent:
            saved_sizes[-1].append(tensor.numel() * tensor.element_size())
        return tensor.detach()
    optimizer.zero_grad(set_to_none=True)
    ids = next_sample_ids(state)
    with torch_module.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        for offset in range(0, len(ids), model.config.micro_batch_size):
            saved_sizes.append([])
            batch = make_batch(ids[offset:offset + model.config.micro_batch_size], model.config,
                               device=next(model.parameters()).device)
            micro_batch_loss_sum(model(batch.inputs), batch.targets).backward()
    profiler = Profiler(model, optimizer)
    train_profile_step(model, optimizer, state, profiler=profiler)
    snapshot = profiler.snapshot()
    actual = snapshot["steps"][0]["memory"]["saved_activation_bytes"]
    series = {name: snapshot["metrics"][f"modules.{name}.saved_activation_bytes"]["samples"] for name in actual}
    assert [sum(samples[mb] for samples in series.values()) for mb in range(3)] == list(map(sum, saved_sizes))
    assert actual == {name: max(samples) for name, samples in series.items()}
    assert sum(saved_sizes[-1]) < max(map(sum, saved_sizes))
    assert all(size > 0 for size in actual.values())


def test_repeated_steps_release_hooks_and_events(training, torch_module, device):
    model, optimizer, state = training
    state, _ = train_profile_step(model, optimizer, state)
    profiler = Profiler(model, optimizer)
    allocated = []
    for _ in range(6):
        state, _ = train_profile_step(model, optimizer, state, profiler=profiler)
        assert not profiler._active and not profiler._pending and not profiler._trace and not profiler._saved
        for module in model.modules():
            assert not module._forward_hooks and not module._forward_pre_hooks and not module._backward_hooks
        for parameter in model.parameters():
            assert not parameter._post_accumulate_grad_hooks
        if device == "cuda":
            torch_module.cuda.synchronize()
            allocated.append(torch_module.cuda.memory_allocated())
    assert len(profiler.steps) == 6
    if device == "cuda":
        # Compare live allocations after the first profiled step; allocator reservations may grow.
        assert max(allocated[1:]) - min(allocated[1:]) <= 4096


def test_exception_removes_profiling_hooks(training):
    model, optimizer, state = training
    train_profile_step(model, optimizer, state)
    profiler = Profiler(model, optimizer)
    with pytest.raises(RuntimeError, match="injected"):
        with profiler.step(2):
            raise RuntimeError("injected profiling failure")
    assert not profiler.steps and not profiler.measurements.metrics and not profiler._active
    assert all(not m._forward_hooks and not m._backward_hooks for m in model.modules())


@pytest.mark.parametrize("failure_site", ["hook", "stamp"])
def test_setup_failure_cleans_hooks_and_allows_next_step(training, monkeypatch, failure_site):
    model, optimizer, state = training
    state, _ = train_profile_step(model, optimizer, state)
    profiler = Profiler(model, optimizer)
    def fail(*args, **kwargs):
        raise RuntimeError("injected setup failure")
    with monkeypatch.context() as patch:
        if failure_site == "hook":
            patch.setattr(model.final_norm, "register_forward_hook", fail)
        else:
            patch.setattr(profiler, "_stamp", fail)
        with pytest.raises(RuntimeError, match="setup failure"):
            with profiler.step(2):
                pass
    assert not profiler._active and not profiler.steps and not profiler.measurements.metrics
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    state, _ = train_profile_step(model, optimizer, state, profiler=profiler)
    assert state.committed_global_step == 2


def test_nested_scope_rejection_preserves_enclosing_operation(training, monkeypatch):
    model, optimizer, state = training
    state, _ = train_profile_step(model, optimizer, state)
    profiler = Profiler(model, optimizer)
    original = model.embedding.forward
    def forward(inputs):
        with pytest.raises(ValueError, match="nested"):
            with profiler.operation("forward", 99):
                pass
        return original(inputs)
    monkeypatch.setattr(model.embedding, "forward", forward)
    state, _ = train_profile_step(model, optimizer, state, profiler=profiler)
    assert state.committed_global_step == 2
    assert len(profiler.steps[0]["trace"]) == 6


def test_empty_backward_cannot_be_reported_as_real_work(training):
    from chameleon.data import make_batch
    from chameleon.global_loss import micro_batch_loss_sum
    model, optimizer, state = training
    train_profile_step(model, optimizer, state)
    profiler = Profiler(model, optimizer)
    batch = make_batch((5, 6), model.config, device=next(model.parameters()).device)
    with pytest.raises(ValueError, match="incomplete module"):
        with profiler.step(2):
            with profiler.operation("forward", 0):
                loss = micro_batch_loss_sum(model(batch.inputs), batch.targets)
            with profiler.operation("backward", 0):
                pass
    assert loss.requires_grad and not profiler.steps and not profiler._active


def test_backward_label_must_match_actual_micro_batch_graph(training):
    from chameleon.data import make_batch
    from chameleon.global_loss import micro_batch_loss_sum
    model, optimizer, state = training
    train_profile_step(model, optimizer, state)
    profiler = Profiler(model, optimizer)
    batch = make_batch((5, 6), model.config, device=next(model.parameters()).device)
    with pytest.raises(ValueError, match="actual forward graph"):
        with profiler.step(2):
            with profiler.operation("forward", 0):
                loss0 = micro_batch_loss_sum(model(batch.inputs), batch.targets)
            with profiler.operation("forward", 1):
                loss1 = micro_batch_loss_sum(model(batch.inputs), batch.targets)
            with profiler.operation("backward", 0):
                loss1.backward()
    assert loss0.requires_grad and not profiler.steps and not profiler._active


def test_actual_module_options_invalidate_profile_identity(training):
    from chameleon.profiler import profile_identity
    model, optimizer, state = training
    state, _ = train_profile_step(model, optimizer, state)
    profiler = Profiler(model, optimizer)
    model.final_norm.eps *= 2
    changed = profile_identity(model)
    assert changed["config_hash"] == profiler.identity["config_hash"]
    assert changed["model_hash"] != profiler.identity["model_hash"]
    with pytest.raises(ValueError, match="identity changed"):
        train_profile_step(model, optimizer, state, profiler=profiler)


def test_parameter_accumulation_cannot_validate_an_unrecorded_graph(training):
    from chameleon.data import make_batch
    from chameleon.global_loss import micro_batch_loss_sum
    model, optimizer, state = training
    train_profile_step(model, optimizer, state)
    batch = make_batch((5, 6), model.config, device=next(model.parameters()).device)
    old_loss = micro_batch_loss_sum(model(batch.inputs), batch.targets)
    profiler = Profiler(model, optimizer)
    with pytest.raises(ValueError, match="incomplete module"):
        with profiler.step(2):
            with profiler.operation("forward", 0):
                recorded_loss = micro_batch_loss_sum(model(batch.inputs), batch.targets)
            with profiler.operation("backward", 0):
                old_loss.backward()
    assert recorded_loss.requires_grad and not profiler.steps and not profiler._active


def test_cuda_events_are_recorded_and_read_after_sync(training, device, torch_module, monkeypatch):
    model, optimizer, state = training
    state, _ = train_profile_step(model, optimizer, state)
    profiler = Profiler(model, optimizer)
    stamps = []
    original_stamp, original_elapsed = profiler._stamp, profiler._elapsed
    def stamp():
        result = original_stamp()
        assert isinstance(result, torch_module.cuda.Event if device == "cuda" else float)
        stamps.append(result)
        return result
    def elapsed(left, right):
        if device == "cuda":
            assert left.query() and right.query()
        return original_elapsed(left, right)
    monkeypatch.setattr(profiler, "_stamp", stamp)
    monkeypatch.setattr(profiler, "_elapsed", elapsed)
    train_profile_step(model, optimizer, state, profiler=profiler)
    assert len(stamps) > 10
    if device == "cuda":
        assert all(event.query() for event in stamps)
