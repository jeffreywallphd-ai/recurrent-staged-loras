"""Recurrent latent refiner block placed between backbone and LM head."""

from __future__ import annotations

from dataclasses import dataclass

from torch import nn
import torch

from .lora_bank import StepAwareLoRABank


@dataclass(slots=True)
class RefinerOutput:
    """Output of latent refinement."""

    refined_hidden_states: torch.Tensor
    per_step_hidden_states: list[torch.Tensor]


class RecurrentLatentRefiner(nn.Module):
    """Applies latent-space refinement with optional recurrence adapters."""

    def __init__(
        self,
        num_steps: int,
        hidden_size: int,
        adapter_bank: StepAwareLoRABank | None = None,
    ) -> None:
        super().__init__()
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.adapter_bank = adapter_bank

        self.w_in = nn.Linear(hidden_size, hidden_size)
        self.w_out = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.step_scale = 0.5

    def step(self, hidden_states: torch.Tensor, step_idx: int) -> torch.Tensor:
        del step_idx
        delta = self.w_out(self.activation(self.w_in(hidden_states)))
        return hidden_states + self.step_scale * delta

    def forward(self, hidden_states: torch.Tensor) -> RefinerOutput:
        current = hidden_states
        history: list[torch.Tensor] = []
        for step_idx in range(self.num_steps):
            current = self.step(current, step_idx=step_idx)
            if self.adapter_bank is not None:
                current = self.adapter_bank.apply(current, step_idx=step_idx)
            history.append(current)
        return RefinerOutput(refined_hidden_states=current, per_step_hidden_states=history)
