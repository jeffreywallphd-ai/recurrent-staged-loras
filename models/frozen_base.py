"""Frozen base causal language model wrapper.

Supports a narrow Hugging Face causal LM path when `transformers` is available.
For local smoke tests and scaffold configs (`example/...` model names), it uses a
small internal fallback model with deterministic parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any


@dataclass(slots=True)
class BaseForwardOutput:
    """Outputs produced by the base model wrapper."""

    hidden_states: list[list[list[float]]]
    attention_mask: list[list[int]] | None
    metadata: dict[str, Any]


class FrozenBaseCausalLM:
    """Thin wrapper around a causal LM used as a fixed backbone."""

    def __init__(self, model_name: str, freeze_base: bool = True, trust_remote_code: bool = False) -> None:
        self.model_name = model_name
        self.freeze_base = freeze_base
        self.trust_remote_code = trust_remote_code
        self.backend = "internal"
        self.vocab_size = 256
        self.hidden_size = 16
        self._embeddings: list[list[float]] = []
        self._proj: list[list[float]] = []
        self._lm_head: list[list[float]] = []
        self._hf_model: Any | None = None
        self._build_model()

    def _build_model(self) -> None:
        if not self.model_name.startswith("example/"):
            try:
                from transformers import AutoModelForCausalLM  # type: ignore
            except Exception as exc:  # pragma: no cover - exercised only when non-example names are used.
                raise RuntimeError(
                    "transformers is required for non-example model names. "
                    "Use example/* in smoke tests or install transformers."
                ) from exc

            self._hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
            )
            config = getattr(self._hf_model, "config", None)
            self.hidden_size = int(getattr(config, "hidden_size", self.hidden_size))
            self.vocab_size = int(getattr(config, "vocab_size", self.vocab_size))
            if self.freeze_base:
                for param in self._hf_model.parameters():
                    param.requires_grad = False
            self.backend = "huggingface"
            return

        self._build_internal_model()

    def _build_internal_model(self) -> None:
        seed = sum(ord(c) for c in self.model_name) % (2**31)
        rng = random.Random(seed)
        self._embeddings = [
            [rng.uniform(-0.1, 0.1) for _ in range(self.hidden_size)] for _ in range(self.vocab_size)
        ]
        self._proj = [[rng.uniform(-0.05, 0.05) for _ in range(self.hidden_size)] for _ in range(self.hidden_size)]
        self._lm_head = [[rng.uniform(-0.08, 0.08) for _ in range(self.vocab_size)] for _ in range(self.hidden_size)]

    def forward_backbone(
        self, input_ids: list[list[int]], attention_mask: list[list[int]] | None = None
    ) -> BaseForwardOutput:
        """Run the base backbone and return final hidden states."""
        if self._hf_model is not None:
            out = self._hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            return BaseForwardOutput(
                hidden_states=out.hidden_states[-1],
                attention_mask=attention_mask,
                metadata={"backend": self.backend, "frozen": self.freeze_base},
            )

        hidden_states: list[list[list[float]]] = []
        for batch_seq in input_ids:
            seq_out: list[list[float]] = []
            for token_id in batch_seq:
                embed = self._embeddings[token_id % self.vocab_size]
                projected = _matvec(embed, self._proj)
                seq_out.append([math.tanh(v + e) for v, e in zip(projected, embed)])
            hidden_states.append(seq_out)

        return BaseForwardOutput(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            metadata={"backend": self.backend, "frozen": self.freeze_base},
        )

    def forward_lm_head(self, refined_hidden_states: list[list[list[float]]]) -> list[list[list[float]]]:
        """Project refined hidden states into vocabulary logits."""
        if self._hf_model is not None:
            return self._hf_model.lm_head(refined_hidden_states)

        logits: list[list[list[float]]] = []
        for seq in refined_hidden_states:
            seq_logits: list[list[float]] = []
            for token_hidden in seq:
                seq_logits.append(_matvec(token_hidden, self._lm_head))
            logits.append(seq_logits)
        return logits


def _matvec(vector: list[float], matrix: list[list[float]]) -> list[float]:
    out_dim = len(matrix[0])
    out = [0.0 for _ in range(out_dim)]
    for in_idx, in_val in enumerate(vector):
        row = matrix[in_idx]
        for out_idx in range(out_dim):
            out[out_idx] += in_val * row[out_idx]
    return out
