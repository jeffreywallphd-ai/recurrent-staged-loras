"""Stage-specialized LoRA-only adapter module.

Applies step-aware low-rank adapters without recurrent latent refinement.
This supports a staged LoRA baseline separate from recurrent refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .lora_bank import StepAwareLoRABank


@dataclass(slots=True)
class StageLoRAOnlyOutput:
    refined_hidden_states: torch.Tensor
    per_step_hidden_states: list[torch.Tensor]


class StageLoRAOnlyAdapter(nn.Module):
    """Applies stage-specialized low-rank adapters without recurrence."""

    def __init__(self, adapter_bank: StepAwareLoRABank) -> None:
        super().__init__()
        self.adapter_bank = adapter_bank
        self._runtime_aligned = False

    def align_to_hidden_states(self, hidden_states: torch.Tensor) -> None:
        param = next(self.adapter_bank.parameters())
        if param.dtype == hidden_states.dtype and param.device == hidden_states.device:
            return
        self.to(device=hidden_states.device, dtype=hidden_states.dtype)
        self._runtime_aligned = True

    def _assert_or_align_runtime(self, hidden_states: torch.Tensor) -> None:
        param = next(self.adapter_bank.parameters())
        if param.dtype == hidden_states.dtype and param.device == hidden_states.device:
            return
        if not self._runtime_aligned:
            self.align_to_hidden_states(hidden_states)
            param = next(self.adapter_bank.parameters())
            if param.dtype == hidden_states.dtype and param.device == hidden_states.device:
                return
        raise RuntimeError(
            "StageLoRAOnlyAdapter dtype/device mismatch: "
            f"hidden_states=({hidden_states.dtype}, {hidden_states.device}) vs "
            f"adapter=({param.dtype}, {param.device})."
        )

    def forward(self, hidden_states: torch.Tensor) -> StageLoRAOnlyOutput:
        self._assert_or_align_runtime(hidden_states)

        per_step: list[torch.Tensor] = []
        for step_idx in range(self.adapter_bank.num_steps):
            adapted = self.adapter_bank.apply(hidden_states, step_idx=step_idx)
            per_step.append(adapted)

        refined = per_step[-1]
        return StageLoRAOnlyOutput(
            refined_hidden_states=refined,
            per_step_hidden_states=per_step,
        )