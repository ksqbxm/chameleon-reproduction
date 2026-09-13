"""Live measurements only; profiles contain no recoverable training tensors."""

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time

from .contracts import _finite, _integer
from .schedule import build_1f1b_schedule


SCHEMA_VERSION = 3


class Measurements:
    def __init__(self, alpha: float = 0.5):
        _finite("alpha", alpha, positive=True)
        if alpha > 1:
            raise ValueError("alpha must not exceed 1")
        self.alpha = alpha
        self.metrics: dict[str, dict] = {}

    def add(self, name: str, value: float) -> None:
        _finite(name, value)
        series = self.metrics.setdefault(name, {"samples": [], "ema": value})
        if series["samples"]:
            series["ema"] = self.alpha * value + (1 - self.alpha) * series["ema"]
        series["samples"].append(value)


def _hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def profile_identity(model, *, dp_size: int = 1, pp_size: int = 1, rank: int = 0) -> dict:
    from .model import parameter_inventory

    _integer("dp_size", dp_size)
    _integer("pp_size", pp_size)
    _integer("rank", rank, 0)
    if rank >= dp_size * pp_size:
        raise ValueError("rank must belong to the parallel configuration")
    device = next(model.parameters()).device
    if any(t.device != device for t in (*model.parameters(), *model.buffers())):
        raise ValueError("a profiler measures one worker device")
    architecture = {
        "modules": [(name, type(module).__module__ + "." + type(module).__qualname__, module.extra_repr())
                    for name, module in model.named_modules()],
        "parameters": [(name, list(p.shape), str(p.dtype), p.requires_grad)
                       for name, p in model.named_parameters()],
        "buffers": [(name, list(t.shape), str(t.dtype)) for name, t in model.named_buffers()],
    }
    module_bytes = {}
    for entry in parameter_inventory(model):
        module_bytes[entry.module_id] = module_bytes.get(entry.module_id, 0) + entry.nbytes
    return {"model_hash": _hash(architecture), "config_hash": _hash(asdict(model.config)),
            "module_parameter_bytes": module_bytes, "module_order": list(module_bytes),
            "device": device_identity(device),
            "parallel": {"dp_size": dp_size, "pp_size": pp_size, "rank": rank}}


def device_identity(device) -> dict:
    import torch
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    device_info = {"type": device.type, "index": device.index, "torch": str(torch.__version__)}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_info.update(name=properties.name, total_memory_bytes=properties.total_memory,
                           capability=list(torch.cuda.get_device_capability(device)),
                           cuda=torch.version.cuda)
    elif device.type == "cpu":
        device_info.update(name=platform.processor() or platform.machine(), system=platform.system())
    else:
        raise ValueError("profiling requires cpu or cuda")
    return device_info


def tensor_inventory(model, optimizer) -> dict[str, dict[str, int]]:
    """Logical tensor bytes, grouped by the model's complete trainable inventory."""
    import torch
    from .model import parameter_inventory

    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("profiling requires AdamW")
    if any(group["amsgrad"] for group in optimizer.param_groups):
        raise ValueError("AdamW amsgrad must be False")
    parameters = dict(model.named_parameters())
    trainable = {p for p in parameters.values() if p.requires_grad}
    if {p for group in optimizer.param_groups for p in group["params"]} != trainable:
        raise ValueError("optimizer must own every trainable parameter")
    result = {}
    for entry in parameter_inventory(model):
        parameter = parameters[entry.name]
        state = optimizer.state.get(parameter, {})
        if (set(state) != {"step", "exp_avg", "exp_avg_sq"}
                or any(not isinstance(t, torch.Tensor) for t in state.values())
                or state["step"].item() <= 0):
            raise ValueError("AdamW must be warmed up to materialize its state before profiling")
        row = result.setdefault(entry.module_id, {"parameter_bytes": 0, "gradient_bytes": 0,
                                                  "adamw_bytes": 0})
        row["parameter_bytes"] += entry.nbytes
        if parameter.grad is not None:
            row["gradient_bytes"] += parameter.grad.numel() * parameter.grad.element_size()
        row["adamw_bytes"] += sum(t.numel() * t.element_size() for t in state.values())
    return result


class Profiler:
    def __init__(self, model, optimizer, *, dp_size: int = 1, pp_size: int = 1,
                 rank: int = 0, ema_alpha: float = 0.5):
        self.model = model
        self.optimizer = optimizer
        self.identity = profile_identity(model, dp_size=dp_size, pp_size=pp_size, rank=rank)
        self.device = next(model.parameters()).device
        self.measurements = Measurements(ema_alpha)
        self.steps: list[dict] = []
        self.calibrations: list[dict] = []
        self._active = False
        self._operation = None

    def _operation_key(self, kind):
        if self._operation is None or self._operation[0] != kind:
            raise ValueError(f"{kind} must run inside an actual operation scope")
        return self._operation[1]

    def _stamp(self):
        if self.device.type == "cpu":
            return time.monotonic()
        import torch
        event = torch.cuda.Event(enable_timing=True)
        event.record(torch.cuda.current_stream(self.device))
        return event

    def _elapsed(self, start, end) -> float:
        return end - start if self.device.type == "cpu" else start.elapsed_time(end) / 1000

    def _finish(self, metric, start, group=None):
        self._pending.append((metric, start, self._stamp(), group))

    @contextmanager
    def operation(self, kind: str, micro_batch: int, *, pipeline: int = 0,
                  stage: int = 0, phase: str = "steady"):
        """Wrap an actual runtime operation; never synthesize a predicted trace."""
        if not self._active:
            raise ValueError("operation requires an active profiling step")
        if self._operation is not None:
            raise ValueError("profiling operations cannot be nested")
        if kind not in ("forward", "backward") or phase not in ("warmup", "steady", "cooldown"):
            raise ValueError("invalid 1F1B operation or phase")
        for name, value in (("micro_batch", micro_batch), ("pipeline", pipeline), ("stage", stage)):
            _integer(name, value, 0)
        parallel = self.identity["parallel"]
        if pipeline >= parallel["dp_size"] or stage >= parallel["pp_size"]:
            raise ValueError("operation must belong to the parallel configuration")
        key = pipeline, stage, micro_batch
        if (kind, key) in self._executed:
            raise ValueError("duplicate profiling operation")
        self._operation = kind, key
        self._executed[kind, key] = set()
        if kind == "forward":
            self._saved[key] = dict.fromkeys((*self.identity["module_order"], "loss_and_runtime"), 0)
        try:
            start = self._stamp()
            yield
            if self._executed[kind, key] != self.identity["module_parameter_bytes"].keys():
                raise ValueError("incomplete module measurements for actual operation")
            self._trace.append(({"kind": kind, "micro_batch": micro_batch, "pipeline": pipeline,
                                 "stage": stage, "phase": phase}, start, self._stamp()))
        finally:
            self._operation = None

    def _hooks(self, inventory, handles):
        forward_starts = {}
        backwards = {}
        seen = set()

        def forward_pre(name, module, inputs):
            key = self._operation_key("forward")
            if name in forward_starts or name in self._executed["forward", key]:
                raise ValueError("a module must execute once per forward operation")
            forward_starts[name] = self._stamp(), key

        def forward_post(name, module, inputs, output):
            start, key = forward_starts.pop(name)
            self._finish(f"modules.{name}.forward_s", start)
            self._executed["forward", key].add(name)
            # Output activations are logical bytes, not the complete autograd/HBM footprint.
            size = output.numel() * output.element_size()
            self._activation[name] = max(self._activation.get(name, 0), size)
            self._values.append((f"modules.{name}.output_activation_bytes", size))
            # Attribute newly created autograd nodes to this module, stopping at earlier modules.
            # Timing individual nodes avoids including upstream modules while waiting for parameter grads.
            stack = [output.grad_fn]
            while stack:
                node = stack.pop()
                if node is None or node in seen:
                    continue
                seen.add(node)
                # Parameter AccumulateGrad nodes are reused across micro-batches.
                forward_key = None if hasattr(node, "variable") else key
                handles.append(node.register_prehook(
                    lambda outputs, node=node, key=forward_key: backward_pre(node, key)))
                handles.append(node.register_hook(
                    lambda inputs, outputs, node=node, name=name, key=forward_key: backward_post(name, node, key)))
                stack.extend(previous for previous, _ in node.next_functions)

        def backward_pre(node, forward_key):
            key = self._operation_key("backward")
            if forward_key is not None and key != forward_key:
                raise ValueError("backward micro-batch must match its actual forward graph")
            backwards[node] = self._stamp(), key

        def backward_post(name, node, forward_key):
            start, group = backwards.pop(node)
            self._finish(f"modules.{name}.backward_s", start, group)
            if forward_key is not None:
                self._executed["backward", group].add(name)

        for name in inventory:
            module = self.model.get_submodule(name)
            handles.append(module.register_forward_pre_hook(
                lambda m, i, name=name: forward_pre(name, m, i)))
            handles.append(module.register_forward_hook(
                lambda m, i, o, name=name: forward_post(name, m, i, o)))
        return backwards, forward_starts

    @contextmanager
    def step(self, step_id: int):
        import torch

        _integer("step_id", step_id)
        if self.steps and step_id <= self.steps[-1]["step_id"]:
            raise ValueError("profile step IDs must increase")
        if self._active:
            raise ValueError("profiling steps cannot be nested")
        if self.identity != profile_identity(self.model, **self.identity["parallel"]):
            raise ValueError("model/config/device identity changed")
        inventory = tensor_inventory(self.model, self.optimizer)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        self._pending, self._trace, self._values, self._activation = [], [], [], {}
        self._operation, self._executed = None, {}
        self._saved = {}
        persistent_storage = {t.untyped_storage().data_ptr()
                              for t in (*self.model.parameters(), *self.model.buffers())}

        def pack(tensor):
            # Logical bytes per saved tensor occurrence (aliases may repeat), not physical HBM.
            if tensor.untyped_storage().data_ptr() not in persistent_storage:
                key = self._operation_key("forward")
                name = next(reversed(forward_starts)) if forward_starts else "loss_and_runtime"
                self._saved[key][name] += tensor.numel() * tensor.element_size()
            return tensor.detach()

        handles = []
        self._active = True
        try:
            backwards, forward_starts = self._hooks(inventory, handles)
            wall_start = time.monotonic()
            start = self._stamp()
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                yield
            end = self._stamp()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            wall_s = time.monotonic() - wall_start
            if backwards or forward_starts or self._activation.keys() != inventory.keys():
                raise ValueError("incomplete module backward measurements")
            trace = [dict(row, start_s=self._elapsed(start, left), end_s=self._elapsed(start, right))
                     for row, left, right in self._trace]
            _validate_trace(trace, self.identity["parallel"])
            memory = {"kind": "cuda_hbm" if self.device.type == "cuda" else "logical_tensor_bytes",
                      "modules": tensor_inventory(self.model, self.optimizer),
                      "output_activation_bytes": dict(self._activation),
                      "saved_activation_bytes": {name: max(row[name] for row in self._saved.values())
                                                 for name in (*inventory, "loss_and_runtime")},
                      "peak_allocated_bytes": None, "peak_reserved_bytes": None}
            if self.device.type == "cuda":
                memory.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                              peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
            backward_totals = {}
            for metric, left, right, group in self._pending:
                duration = self._elapsed(left, right)
                if group is None:
                    self.measurements.add(metric, duration)
                else:
                    key = metric, group
                    backward_totals[key] = backward_totals.get(key, 0) + duration
            for (metric, group), duration in backward_totals.items():
                self.measurements.add(metric, duration)
            for metric, value in self._values:
                self.measurements.add(metric, value)
            self.measurements.add("step_time_s", self._elapsed(start, end))
            self.measurements.add("step_wall_time_s", wall_s)
            for name, row in memory["modules"].items():
                for kind, size in row.items():
                    self.measurements.add(f"modules.{name}.{kind}", size)
            for row in self._saved.values():
                for name, size in row.items():
                    self.measurements.add(f"modules.{name}.saved_activation_bytes", size)
            for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
                if memory[key] is not None:
                    self.measurements.add(key, memory[key])
            self.steps.append({"step_id": step_id, "trace": trace, "memory": memory})
        finally:
            for handle in handles:
                handle.remove()
            self._active = False
            self._operation, self._executed = None, {}
            # CUDA events are transient; no activation or gradient tensors are retained.
            self._pending, self._trace, self._values, self._activation = [], [], [], {}
            self._saved = {}

    def add_calibration(self, report: dict) -> None:
        from .transfer_calibration import validate_calibration
        validate_calibration(report)
        if (report["device"] != self.device.type
                or self.identity["device"] not in [r["device_identity"] for r in report["records"]]):
            raise ValueError("calibration device differs from profile device")
        self.calibrations.append(deepcopy(report))
        for record in report["records"]:
            prefix = f"calibration.rank{record['rank']}"
            for duration in record["group_bootstrap_s"]:
                self.measurements.add(f"{prefix}.group_bootstrap_s", duration)
            for row in record["transfers"]:
                edge = f"{prefix}.source{row['source']}.destination{row['destination']}.bytes{row['tensor_bytes']}"
                self.measurements.add(f"{edge}.execution_s", row["execution_time_s"])
                self.measurements.add(f"{edge}.wall_s", row["wall_time_s"])

    def snapshot(self) -> dict:
        return deepcopy({"schema_version": SCHEMA_VERSION, "identity": self.identity,
                         "ema_alpha": self.measurements.alpha, "metrics": self.measurements.metrics,
                         "steps": self.steps, "calibrations": self.calibrations})


def _keys(value, keys, name):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"invalid {name} fields")


def _validate_trace(trace, parallel):
    if not isinstance(trace, list) or not trace:
        raise ValueError("trace must contain actual forward/backward operations")
    pending, completed = set(), set()
    previous_end = 0
    stages = {}
    for row in trace:
        _keys(row, ("kind", "micro_batch", "pipeline", "stage", "phase", "start_s", "end_s"), "trace")
        for name in ("micro_batch", "pipeline", "stage"):
            _integer(name, row[name], 0)
        if row["pipeline"] >= parallel["dp_size"] or row["stage"] >= parallel["pp_size"]:
            raise ValueError("trace differs from parallel configuration")
        for name in ("start_s", "end_s"):
            _finite(name, row[name])
        if row["start_s"] < previous_end or row["end_s"] < row["start_s"]:
            raise ValueError("trace times must be ordered")
        previous_end = row["end_s"]
        key = row["pipeline"], row["stage"], row["micro_batch"]
        if row["phase"] not in ("warmup", "steady", "cooldown"):
            raise ValueError("invalid trace phase")
        if row["kind"] == "forward" and key not in pending | completed:
            pending.add(key)
        elif row["kind"] == "backward" and key in pending:
            pending.remove(key)
            completed.add(key)
        else:
            raise ValueError("duplicate operation or backward without forward")
        stages.setdefault(key[:2], []).append(row)
    if pending:
        raise ValueError("trace contains missing backward operations")
    for (_, stage), operations in stages.items():
        forwards = [r["micro_batch"] for r in operations if r["kind"] == "forward"]
        backwards = [r["micro_batch"] for r in operations if r["kind"] == "backward"]
        queue = build_1f1b_schedule(parallel["pp_size"], len(forwards))[stage]
        expected = [(op.kind, op.phase) for op in queue]
        if forwards != backwards or [(r["kind"], r["phase"]) for r in operations] != expected:
            raise ValueError("trace must follow actual 1F1B warmup/steady/cooldown order")


def _validate_identity(identity):
    _keys(identity, ("model_hash", "config_hash", "module_parameter_bytes", "module_order", "device", "parallel"), "identity")
    for name in ("model_hash", "config_hash"):
        value = identity[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid identity hash")
    module_bytes = identity["module_parameter_bytes"]
    if not isinstance(module_bytes, dict) or not module_bytes:
        raise ValueError("invalid identity module inventory")
    for name, size in module_bytes.items():
        if not isinstance(name, str) or not name:
            raise ValueError("invalid module name")
        _integer("module parameter bytes", size)
    order = identity["module_order"]
    if (not isinstance(order, list) or any(not isinstance(name, str) for name in order)
            or len(order) != len(module_bytes) or set(order) != set(module_bytes)):
        raise ValueError("invalid identity module order")
    _validate_device_identity(identity["device"])
    parallel = identity["parallel"]
    _keys(parallel, ("dp_size", "pp_size", "rank"), "parallel configuration")
    _integer("dp_size", parallel["dp_size"])
    _integer("pp_size", parallel["pp_size"])
    _integer("rank", parallel["rank"], 0)
    if parallel["rank"] >= parallel["dp_size"] * parallel["pp_size"]:
        raise ValueError("invalid parallel rank")


def _validate_device_identity(device):
    if not isinstance(device, dict) or device.get("type") not in ("cpu", "cuda"):
        raise ValueError("invalid device identity")
    cuda = device["type"] == "cuda"
    _keys(device, ("type", "index", "torch", "name", "total_memory_bytes", "capability", "cuda")
          if cuda else ("type", "index", "torch", "name", "system"), "device identity")
    for name in (("torch", "name", "cuda") if cuda else ("torch", "name", "system")):
        if not isinstance(device[name], str) or not device[name]:
            raise ValueError("invalid device identity text")
    if cuda:
        _integer("device index", device["index"], 0)
        _integer("device memory", device["total_memory_bytes"])
        if not isinstance(device["capability"], list) or len(device["capability"]) != 2:
            raise ValueError("invalid CUDA capability")
        for value in device["capability"]:
            _integer("capability", value, 0)
    elif device["index"] is not None:
        raise ValueError("CPU device identity cannot have a GPU index")


def validate_snapshot(payload: dict, expected_identity: dict) -> None:
    _keys(payload, ("schema_version", "identity", "ema_alpha", "metrics", "steps", "calibrations"), "profile")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported profile schema version")
    _validate_identity(payload["identity"])
    if payload["identity"] != expected_identity:
        raise ValueError("profile model/config/device/parallel identity mismatch")
    alpha = Measurements(payload["ema_alpha"]).alpha
    if not isinstance(payload["metrics"], dict):
        raise ValueError("invalid metrics")
    if expected_identity["device"]["type"] == "cpu" and any(
            name in payload["metrics"] for name in ("peak_allocated_bytes", "peak_reserved_bytes")):
        raise ValueError("CPU profiles cannot report measured HBM metrics")
    for name, series in payload["metrics"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("invalid metric name")
        _keys(series, ("samples", "ema"), "metric")
        if not isinstance(series["samples"], list) or not series["samples"]:
            raise ValueError("metric samples must not be empty")
        oracle = Measurements(alpha)
        for sample in series["samples"]:
            oracle.add(name, sample)
        _finite("ema", series["ema"])
        if not math.isclose(series["ema"], oracle.metrics[name]["ema"], rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError("EMA does not match raw samples")
    if not isinstance(payload["steps"], list):
        raise ValueError("invalid steps")
    if payload["steps"] and any(name not in payload["metrics"]
                                or len(payload["metrics"][name]["samples"]) != len(payload["steps"])
                                for name in ("step_time_s", "step_wall_time_s")):
        raise ValueError("missing step timing samples")
    previous_step = 0
    for index, step in enumerate(payload["steps"]):
        _keys(step, ("step_id", "trace", "memory"), "step")
        _integer("step_id", step["step_id"])
        if step["step_id"] <= previous_step:
            raise ValueError("profile step IDs must increase")
        previous_step = step["step_id"]
        parallel = expected_identity["parallel"]
        _validate_trace(step["trace"], parallel)
        if step["trace"][-1]["end_s"] > payload["metrics"]["step_time_s"]["samples"][index]:
            raise ValueError("trace extends beyond measured step time")
        memory = step["memory"]
        _keys(memory, ("kind", "modules", "output_activation_bytes", "saved_activation_bytes",
                       "peak_allocated_bytes", "peak_reserved_bytes"), "memory")
        if (not isinstance(memory["modules"], dict)
                or memory["modules"].keys() != expected_identity["module_parameter_bytes"].keys()):
            raise ValueError("module inventory must match model identity")
        for module, row in memory["modules"].items():
            _keys(row, ("parameter_bytes", "gradient_bytes", "adamw_bytes"), "tensor inventory")
            for name, size in row.items():
                _integer(name, size, 0)
            if row["parameter_bytes"] != expected_identity["module_parameter_bytes"][module]:
                raise ValueError("parameter bytes must match model identity")
        if (not isinstance(memory["output_activation_bytes"], dict)
                or memory["output_activation_bytes"].keys() != memory["modules"].keys()):
            raise ValueError("activation inventory must cover every measured module")
        for size in memory["output_activation_bytes"].values():
            _integer("activation bytes", size, 0)
        if (not isinstance(memory["saved_activation_bytes"], dict)
                or set(memory["saved_activation_bytes"]) != set(memory["modules"]) | {"loss_and_runtime"}):
            raise ValueError("invalid saved activation inventory")
        for size in memory["saved_activation_bytes"].values():
            _integer("saved activation bytes", size, 0)
        cuda = expected_identity["device"]["type"] == "cuda"
        if memory["kind"] != ("cuda_hbm" if cuda else "logical_tensor_bytes"):
            raise ValueError("memory measurement kind differs from device")
        for name in ("peak_allocated_bytes", "peak_reserved_bytes"):
            if cuda:
                _integer(name, memory[name])
            elif memory[name] is not None:
                raise ValueError("CPU profiles cannot report measured HBM")
        if cuda and memory["peak_reserved_bytes"] < memory["peak_allocated_bytes"]:
            raise ValueError("reserved HBM cannot be smaller than allocated HBM")
    if payload["steps"]:
        _validate_module_samples(payload)
    if not isinstance(payload["calibrations"], list):
        raise ValueError("invalid calibrations")
    from .transfer_calibration import validate_calibration
    for report in payload["calibrations"]:
        validate_calibration(report)
        if (report["device"] != expected_identity["device"]["type"]
                or expected_identity["device"] not in [r["device_identity"] for r in report["records"]]):
            raise ValueError("calibration device differs from profile device")


def _validate_module_samples(payload):
    steps, metrics = payload["steps"], payload["metrics"]
    forward_count = sum(r["kind"] == "forward" for step in steps for r in step["trace"])
    modules = payload["identity"]["module_parameter_bytes"]
    sizes = ("parameter_bytes", "gradient_bytes", "adamw_bytes")
    for module in modules:
        for field in (*sizes, "forward_s", "backward_s", "output_activation_bytes", "saved_activation_bytes"):
            name = f"modules.{module}.{field}"
            expected_count = len(steps) if field in sizes else forward_count
            if name not in metrics or len(metrics[name]["samples"]) != expected_count:
                raise ValueError("missing module measurement samples")
            if field.endswith("_bytes"):
                for value in metrics[name]["samples"]:
                    _integer("byte sample", value, 0)
    runtime = "modules.loss_and_runtime.saved_activation_bytes"
    if runtime not in metrics or len(metrics[runtime]["samples"]) != forward_count:
        raise ValueError("missing runtime activation samples")
    for value in metrics[runtime]["samples"]:
        _integer("byte sample", value, 0)
    peaks = ("peak_allocated_bytes", "peak_reserved_bytes") if payload["identity"]["device"]["type"] == "cuda" else ()
    for name in peaks:
        if name not in metrics or len(metrics[name]["samples"]) != len(steps):
            raise ValueError("missing CUDA peak measurement samples")
    offset = 0
    for index, step in enumerate(steps):
        memory = step["memory"]
        count = sum(r["kind"] == "forward" for r in step["trace"])
        positions = {"forward": offset, "backward": offset}
        for row in step["trace"]:
            kind = row["kind"]
            duration = sum(metrics[f"modules.{module}.{kind}_s"]["samples"][positions[kind]]
                           for module in modules)
            if duration > row["end_s"] - row["start_s"] + 1e-12:
                raise ValueError("module timing exceeds actual operation duration")
            positions[kind] += 1
        for name in peaks:
            if metrics[name]["samples"][index] != memory[name]:
                raise ValueError("CUDA peak samples differ from memory inventory")
        for module in modules:
            for field in sizes:
                expected = memory["modules"][module][field]
                if metrics[f"modules.{module}.{field}"]["samples"][index] != expected:
                    raise ValueError("memory samples differ from module inventory")
            for field in ("output_activation_bytes", "saved_activation_bytes"):
                samples = metrics[f"modules.{module}.{field}"]["samples"][offset:offset + count]
                if max(samples) != memory[field][module]:
                    raise ValueError("activation samples differ from module inventory")
        if max(metrics[runtime]["samples"][offset:offset + count]) != memory["saved_activation_bytes"]["loss_and_runtime"]:
            raise ValueError("runtime activation samples differ from inventory")
        offset += count


def export_profile(payload: dict, path: str, *, expected_identity: dict) -> None:
    validate_snapshot(payload, expected_identity)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def load_profile(path: str, *, expected_identity: dict) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_snapshot(payload, expected_identity)
    return payload


def train_profile_step(model, optimizer, state, *, profiler: Profiler | None = None):
    """Actual single-stage 1F1B. Multi-stage runtime supplies its own operation scopes."""
    from .data import make_batch, next_sample_ids
    from .global_loss import micro_batch_loss_sum
    from .step import StepCommit

    if state.global_batch_size != model.config.global_batch_size or len(state.workers) != 1:
        raise ValueError("the local profiling runner requires one worker and the configured global batch")
    if profiler is not None and (profiler.model is not model or profiler.optimizer is not optimizer
                                or profiler.identity["parallel"] != {"dp_size": 1, "pp_size": 1, "rank": 0}):
        raise ValueError("the local runner requires the matching single-stage profiler")
    step_id = state.committed_global_step + 1
    ids = next_sample_ids(state)
    device = next(model.parameters()).device
    losses = []
    queue = build_1f1b_schedule(1, (len(ids) + model.config.micro_batch_size - 1)
                              // model.config.micro_batch_size)[0]
    with profiler.step(step_id) if profiler is not None else nullcontext():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for forward, backward in zip(queue[::2], queue[1::2]):
            offset = forward.micro_batch * model.config.micro_batch_size
            batch = make_batch(ids[offset:offset + model.config.micro_batch_size], model.config, device=device)
            with (profiler.operation(forward.kind, forward.micro_batch, phase=forward.phase)
                  if profiler is not None else nullcontext()):
                loss = micro_batch_loss_sum(model(batch.inputs), batch.targets)
            with (profiler.operation(backward.kind, backward.micro_batch, phase=backward.phase)
                  if profiler is not None else nullcontext()):
                loss.backward()
            losses.append(loss.detach())
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad.div_(len(ids))
        optimizer.step()
        loss_sum = sum(losses).item()  # Wait for queued CUDA updates before acknowledging the step.
    commit = StepCommit(state)
    commit.acknowledge(state.workers[0], step_id)
    return commit.state, loss_sum
