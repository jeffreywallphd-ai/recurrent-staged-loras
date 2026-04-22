"""Top-level composition for staged latent adaptation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import VariantConfig
from .frozen_base import FrozenBaseCausalLM
from .recurrent_refiner import RecurrentLatentRefiner


@dataclass(slots=True)
class ModelForwardOutput:
    """Unified output for experiment runners."""

    logits: Any
    refined_hidden_states: Any
    extras: dict[str, Any]


class StagedLatentAdaptationModel:
    """Composes base model, latent refiner, and logits projection.

    Planned execution path:
    frozen base LM -> latent refiner (optional recurrent) -> LM head
    """

    def __init__(
        self,
        config: VariantConfig,
        base_model: FrozenBaseCausalLM,
        refiner: RecurrentLatentRefiner | None,
    ) -> None:
        self.config = config
        self.base_model = base_model
        self.refiner = refiner

    def forward(self, input_ids: Any, attention_mask: Any | None = None) -> ModelForwardOutput:
        """Forward pass used by training and evaluation entrypoints."""
        base_out = self.base_model.forward_backbone(input_ids=input_ids, attention_mask=attention_mask)
        if self.refiner is None:
            refined = base_out.hidden_states
            per_step = []
        else:
            refiner_out = self.refiner.forward(base_out.hidden_states)
            refined = refiner_out.refined_hidden_states
            per_step = refiner_out.per_step_hidden_states

        logits = self.base_model.forward_lm_head(refined)
        return ModelForwardOutput(logits=logits, refined_hidden_states=refined, extras={"per_step": per_step})
