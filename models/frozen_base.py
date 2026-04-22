"""Frozen base causal language model wrapper.

This interface exposes the final hidden states from a base LM while keeping the
base weights frozen by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class BaseForwardOutput:
    """Outputs produced by the base model wrapper."""

    hidden_states: Any
    attention_mask: Any
    metadata: dict[str, Any]


class FrozenBaseCausalLM:
    """Thin wrapper around a causal LM used as a fixed backbone.

    Responsibilities:
    - Load and hold the base model.
    - Freeze parameters when configured.
    - Expose final hidden states for latent refinement.
    - Expose/read LM head for final token logits.
    """

    def __init__(self, model_name: str, freeze_base: bool = True) -> None:
        self.model_name = model_name
        self.freeze_base = freeze_base
        self.model = None
        self.lm_head = None
        # TODO: load tokenizer/model and optionally freeze parameters.

    def forward_backbone(self, input_ids: Any, attention_mask: Any | None = None) -> BaseForwardOutput:
        """Run the base backbone and return final hidden states.

        Args:
            input_ids: Token IDs tensor-like object.
            attention_mask: Optional mask tensor-like object.
        """
        raise NotImplementedError("TODO: implement backbone forward pass")

    def forward_lm_head(self, refined_hidden_states: Any) -> Any:
        """Project refined hidden states into vocabulary logits."""
        raise NotImplementedError("TODO: implement LM head forward pass")
