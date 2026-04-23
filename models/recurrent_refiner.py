"""Recurrent latent-state transformation module.

Inserted between backbone hidden states and LM head projection; optionally
augmented with step-aware adapters for shared/stage-specialized variants.
"""

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
        self._runtime_aligned = False

    def align_to_hidden_states(self, hidden_states: torch.Tensor) -> None:
        """Align refiner/adapters to hidden-state dtype+device once if needed."""
        ref_param = next(self.w_in.parameters())
        if ref_param.dtype == hidden_states.dtype and ref_param.device == hidden_states.device:
            return
        self.to(device=hidden_states.device, dtype=hidden_states.dtype)
        self._runtime_aligned = True

    def _assert_or_align_runtime(self, hidden_states: torch.Tensor) -> None:
        ref_param = next(self.w_in.parameters())
        if ref_param.dtype == hidden_states.dtype and ref_param.device == hidden_states.device:
            return
        if not self._runtime_aligned:
            self.align_to_hidden_states(hidden_states)
            ref_param = next(self.w_in.parameters())
            if ref_param.dtype == hidden_states.dtype and ref_param.device == hidden_states.device:
                return
        raise RuntimeError(
            "Recurrent refiner dtype/device mismatch: "
            f"hidden_states=({hidden_states.dtype}, {hidden_states.device}) vs "
            f"refiner=({ref_param.dtype}, {ref_param.device}). "
            "Likely model-construction alignment was skipped; align refiner/adapters to "
            "the backbone runtime dtype/device in training.config_loader.build_model_from_variant."
        )

    def step(self, hidden_states: torch.Tensor, step_idx: int) -> torch.Tensor:
        """Apply one recurrent latent update step."""
        del step_idx
        delta = self.w_out(self.activation(self.w_in(hidden_states)))
        return hidden_states + self.step_scale * delta

    def forward(self, hidden_states: torch.Tensor) -> RefinerOutput:
        """Run configured recurrent steps and collect per-step hidden states."""
        self._assert_or_align_runtime(hidden_states)
        current = hidden_states
        history: list[torch.Tensor] = []
        for step_idx in range(self.num_steps):
            current = self.step(current, step_idx=step_idx)
            if self.adapter_bank is not None:
                current = self.adapter_bank.apply(current, step_idx=step_idx)
            history.append(current)
        return RefinerOutput(refined_hidden_states=current, per_step_hidden_states=history)
