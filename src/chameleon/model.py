"""Splittable, dropout-free Transformer and its complete parameter inventory."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.parametrize import register_parametrization

from .contracts import ModelConfig


@dataclass(frozen=True)
class ParameterInfo:
    name: str
    module_id: str
    shape: tuple[int, ...]
    numel: int
    nbytes: int


def model_module_order(config: ModelConfig) -> tuple[str, ...]:
    return ("embedding", *(f"blocks.{i}" for i in range(config.num_layers)),
            "final_norm", "lm_head")


def validate_stage_layout(config: ModelConfig, stages) -> tuple[tuple[str, ...], ...]:
    stages = tuple(tuple(stage) if isinstance(stage, (tuple, list)) else stage for stage in stages)
    if (not stages or any(not isinstance(stage, tuple) or not stage for stage in stages)
            or tuple(module for stage in stages for module in stage) != model_module_order(config)):
        raise ValueError("stages must partition every model module once in model order")
    return stages


def parameter_inventory(model: nn.Module) -> tuple[ParameterInfo, ...]:
    """Use named parameters so endpoint and newly added modules are included."""
    inventory = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        parts = name.split(".")
        module_id = ".".join(parts[:2]) if parts[0] == "blocks" else parts[0]
        inventory.append(ParameterInfo(name, module_id, tuple(parameter.shape),
                                       parameter.numel(),
                                       parameter.numel() * parameter.element_size()))
    return tuple(inventory)


class _QueryValueBias(nn.Module):
    """A key projection bias cancels in softmax; train only query/value biases."""

    def forward(self, query_bias: torch.Tensor, value_bias: torch.Tensor) -> torch.Tensor:
        return torch.cat((query_bias, torch.zeros_like(query_bias), value_bias))

    def right_inverse(self, bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        query_bias, _, value_bias = bias.chunk(3)
        return query_bias.clone(), value_bias.clone()


class TinyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                config.hidden_size, config.num_heads,
                dim_feedforward=4 * config.hidden_size, dropout=0.0,
                activation="gelu", batch_first=True, norm_first=True,
            )
            for _ in range(config.num_layers)
        )
        for block in self.blocks:
            register_parametrization(block.self_attn, "in_proj_bias", _QueryValueBias())
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.ndim != 2 or token_ids.shape[1] != self.config.sequence_length:
            raise ValueError("token_ids must have the configured sequence length")
        hidden = self.embedding(token_ids)
        mask = torch.ones(self.config.sequence_length, self.config.sequence_length,
                          dtype=torch.bool, device=token_ids.device).triu(1)
        for block in self.blocks:
            hidden = block(hidden, src_mask=mask)
        return self.lm_head(self.final_norm(hidden))


def _to_initial_device(model: nn.Module, device, dtype):
    # Full batches and micro-batches must both use full FP32 matmul precision.
    if dtype == torch.float32 and torch.device(device).type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    return model.to(device=device, dtype=dtype)


def build_initial_model(config: ModelConfig, *, device: str = "cpu",
                        dtype: torch.dtype = torch.float64) -> TinyTransformer:
    """Seed only initial construction; training and topology changes never call this."""
    # Construct on CPU without consuming or reseeding the caller's CPU/CUDA RNGs.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        model = TinyTransformer(config)
    return _to_initial_device(model, device, dtype)


class PipelineStage(nn.Module):
    """Own only the selected initial modules, retaining global parameter names."""

    def __init__(self, model: TinyTransformer, module_ids: tuple[str, ...]):
        super().__init__()
        self.config = model.config
        self.module_ids = module_ids
        for name in module_ids:
            module = model.get_submodule(name)
            if name.startswith("blocks."):
                if not hasattr(self, "blocks"):
                    self.blocks = nn.ModuleDict()
                self.blocks[name.split(".")[1]] = module
            else:
                self.add_module(name, module)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        mask = torch.ones(self.config.sequence_length, self.config.sequence_length,
                          dtype=torch.bool, device=inputs.device).triu(1)
        for name in self.module_ids:
            module = self.get_submodule(name)
            hidden = module(hidden, src_mask=mask) if name.startswith("blocks.") else module(hidden)
        return hidden


def build_initial_stage(config: ModelConfig, module_ids: tuple[str, ...], *,
                        device: str = "cpu", dtype: torch.dtype = torch.float64) -> PipelineStage:
    """Initial startup only; discard unowned CPU modules before moving to the device."""
    stage = PipelineStage(build_initial_model(config), module_ids)
    return _to_initial_device(stage, device, dtype)
