"""Splittable, dropout-free Transformer and its complete parameter inventory."""

from dataclasses import dataclass

import torch
from torch import nn

from .contracts import ModelConfig


@dataclass(frozen=True)
class ParameterInfo:
    name: str
    module_id: str
    shape: tuple[int, ...]
    numel: int
    nbytes: int


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


def build_initial_model(config: ModelConfig, *, device: str = "cpu",
                        dtype: torch.dtype = torch.float64) -> TinyTransformer:
    """Seed only initial construction; training and topology changes never call this."""
    # Construct on CPU without consuming or reseeding the caller's CPU/CUDA RNGs.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.seed)
        model = TinyTransformer(config)
    return model.to(device=device, dtype=dtype)
