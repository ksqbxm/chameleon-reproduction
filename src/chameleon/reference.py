"""Independent, full-batch single-process AdamW oracle, solely for test comparison."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .contracts import ClusterState
from .data import make_batch, next_sample_ids
from .model import TinyTransformer
from .step import StepCommit


def sample_losses(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average valid token losses within each sequence, never across samples."""
    valid = targets.ne(-100)
    counts = valid.sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("each sample must have at least one valid token")
    token_losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none",
                                  ignore_index=-100)
    return token_losses.sum(dim=1) / counts


@dataclass(frozen=True)
class ReferenceStep:
    committed_global_step: int
    sample_ids: tuple[int, ...]
    sample_losses: torch.Tensor
    loss_global_sum: float
    parameters: dict[str, torch.Tensor]
    gradients: dict[str, torch.Tensor]
    optimizer_state: dict[str, dict[str, torch.Tensor]]

    @property
    def global_sample_count(self) -> int:
        return len(self.sample_ids)

    @property
    def loss_global_mean(self) -> float:
        return self.loss_global_sum / self.global_sample_count


class ReferenceTrainer:
    def __init__(self, model: TinyTransformer, state: ClusterState, *,
                 lr: float = 1e-3, weight_decay: float = 0.01):
        if len(state.workers) != 1:
            raise ValueError("reference requires exactly one participant")
        if state.global_batch_size != model.config.global_batch_size:
            raise ValueError("reference global batch size must match the model config")
        self.model = model
        self.commit = StepCommit(state)
        self.optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=lr,
            weight_decay=weight_decay, amsgrad=False,
        )

    def train_step(self) -> ReferenceStep:
        ids = next_sample_ids(self.commit.state)
        batch = make_batch(ids, self.model.config, device=next(self.model.parameters()).device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses = sample_losses(self.model(batch.inputs), batch.targets)
        loss_sum = losses.sum()
        loss_sum.backward()
        parameters = {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        for parameter in parameters.values():
            parameter.grad.div_(len(ids))
        self.optimizer.step()
        self.commit.acknowledge(self.commit.state.workers[0],
                                self.commit.state.committed_global_step + 1)
        return ReferenceStep(
            self.commit.state.committed_global_step, ids, losses.detach().cpu().clone(),
            loss_sum.item(),
            {name: p.detach().cpu().clone() for name, p in parameters.items()},
            {name: p.grad.detach().cpu().clone() for name, p in parameters.items()},
            {name: {key: value.detach().cpu().clone() for key, value in self.optimizer.state[p].items()}
             for name, p in parameters.items()},
        )
