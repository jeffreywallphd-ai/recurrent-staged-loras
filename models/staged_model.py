"""Top-level model composition used by training/evaluation loops.

Combines frozen backbone, optional recurrent latent refiner, and LM head.
Also exposes per-step hidden-state traces needed for stage-aware supervision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn
import torch

from .config import VariantConfig
from .frozen_base import FrozenBaseCausalLM
from .recurrent_refiner import RecurrentLatentRefiner


@dataclass(slots=True)
class ModelForwardOutput:
    """Unified output for experiment runners."""

    logits: torch.Tensor
    refined_hidden_states: torch.Tensor
    extras: dict[str, Any]


class StagedLatentAdaptationModel(nn.Module):
    """Composes base model, optional refiner, and LM head projection."""

    def __init__(
        self,
        config: VariantConfig,
        base_model: FrozenBaseCausalLM,
        refiner: RecurrentLatentRefiner | None,
    ) -> None:
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.refiner = refiner

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> ModelForwardOutput:
        """Run backbone -> optional refiner -> LM head for one batch."""
        base_out = self.base_model.forward_backbone(input_ids=input_ids, attention_mask=attention_mask)
        if self.refiner is None:
            refined = base_out.hidden_states
            per_step: list[torch.Tensor] = []
        else:
            refiner_out = self.refiner(base_out.hidden_states)
            refined = refiner_out.refined_hidden_states
            per_step = refiner_out.per_step_hidden_states

        logits = self.base_model.project_to_logits(refined)
        return ModelForwardOutput(
            logits=logits,
            refined_hidden_states=refined,
            extras={
                "per_step": per_step,
                "base_metadata": base_out.metadata,
            },
        )
