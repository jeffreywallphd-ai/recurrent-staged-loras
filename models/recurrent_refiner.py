"""Recurrent latent refiner block placed between backbone and LM head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .lora_bank import StepAwareLoRABank


@dataclass(slots=True)
class RefinerOutput:
    """Output of latent refinement."""

    refined_hidden_states: Any
    per_step_hidden_states: list[Any]


class RecurrentLatentRefiner:
    """Applies latent-space refinement with optional recurrence.

    Supports:
    - no recurrence (single pass)
    - shared recurrence (shared adapter across steps)
    - stage-specialized recurrence (per-step adapters)
    """

    def __init__(self, num_steps: int, adapter_bank: StepAwareLoRABank | None = None) -> None:
        self.num_steps = num_steps
        self.adapter_bank = adapter_bank
        # TODO: initialize core refiner transformation(s).

    def step(self, hidden_states: Any, step_idx: int) -> Any:
        """Apply one latent refinement step."""
        raise NotImplementedError("TODO: implement one refiner step")

    def forward(self, hidden_states: Any) -> RefinerOutput:
        """Run latent refinement for configured number of steps."""
        current = hidden_states
        history: list[Any] = []
        for step_idx in range(self.num_steps):
            current = self.step(current, step_idx=step_idx)
            if self.adapter_bank is not None:
                current = self.adapter_bank.apply(current, step_idx=step_idx)
            history.append(current)
        return RefinerOutput(refined_hidden_states=current, per_step_hidden_states=history)
