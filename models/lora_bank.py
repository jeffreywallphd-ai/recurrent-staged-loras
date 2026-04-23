"""Step-aware adapter bank for shared or stage-specialized recurrence."""

from __future__ import annotations

from torch import nn
import torch


class LowRankAdapter(nn.Module):
    """Minimal low-rank residual adapter."""

    def __init__(self, hidden_size: int, rank: int, alpha: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.scaling * self.up(self.down(self.dropout(hidden_states)))


class StepAwareLoRABank(nn.Module):
    """Routes adapters by recurrence step index."""

    def __init__(
        self,
        num_steps: int,
        hidden_size: int,
        rank: int,
        alpha: int,
        shared_across_steps: bool,
        enabled: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.shared_across_steps = shared_across_steps
        self.enabled = enabled

        if not enabled:
            self.adapters = nn.ModuleList()
        elif shared_across_steps:
            self.adapters = nn.ModuleList([LowRankAdapter(hidden_size, rank, alpha, dropout=dropout)])
        else:
            self.adapters = nn.ModuleList(
                [LowRankAdapter(hidden_size, rank, alpha, dropout=dropout) for _ in range(num_steps)]
            )

    def get_adapter_for_step(self, step_idx: int) -> LowRankAdapter:
        if not self.enabled:
            raise KeyError("Adapters disabled")
        idx = 0 if self.shared_across_steps else step_idx
        return self.adapters[idx]

    def apply(self, hidden_states: torch.Tensor, step_idx: int) -> torch.Tensor:
        if not self.enabled:
            return hidden_states
        return self.get_adapter_for_step(step_idx)(hidden_states)
