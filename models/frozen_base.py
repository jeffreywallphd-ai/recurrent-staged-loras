"""Frozen base causal language model wrapper.

Supports a Hugging Face causal LM backend for non-``example/*`` model names and
an internal tiny torch model backend for local experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(slots=True)
class BaseForwardOutput:
    """Outputs produced by the base model wrapper."""

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None
    metadata: dict[str, Any]


class LoRALinear(nn.Module):
    """Minimal low-rank adapter on top of a frozen linear projection."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)

        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.lora_b(self.lora_a(self.dropout(x)))


class HiddenStateLoRA(nn.Module):
    """Fallback LoRA transform applied to final hidden states."""

    def __init__(self, hidden_size: int, rank: int, alpha: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.scaling = float(alpha) / float(rank)
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(hidden_size, rank, bias=False)
        self.lora_b = nn.Linear(rank, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.scaling * self.lora_b(self.lora_a(self.dropout(hidden_states)))


class TinyInternalCausalLM(nn.Module):
    """Tiny deterministic causal-LM-like module for local trainability tests."""

    def __init__(self, vocab_size: int = 256, hidden_size: int = 32) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def backbone(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        q = self.q_proj(x)
        v = self.v_proj(x)
        return self.out_proj(self.activation(q + v))


class FrozenBaseCausalLM(nn.Module):
    """Wrapper around a causal LM backbone used for baseline experiments."""

    def __init__(self, model_name: str, freeze_base: bool = True, trust_remote_code: bool = False) -> None:
        super().__init__()
        self.model_name = model_name
        self.freeze_base = freeze_base
        self.trust_remote_code = trust_remote_code

        self.backend = "internal"
        self.vocab_size = 256
        self.hidden_size = 32

        self.internal_model: TinyInternalCausalLM | None = None
        self.hf_model: Any | None = None
        self.hf_lora: HiddenStateLoRA | None = None
        self.standard_lora_enabled = False
        self.standard_lora_scope = "disabled"

        self._build_model()

    def _build_model(self) -> None:
        if self.model_name.startswith("example/"):
            torch.manual_seed(sum(ord(c) for c in self.model_name) % 100_000)
            self.internal_model = TinyInternalCausalLM(vocab_size=self.vocab_size, hidden_size=self.hidden_size)
            if self.freeze_base:
                for param in self.internal_model.parameters():
                    param.requires_grad = False
            return

        try:
            from transformers import AutoModelForCausalLM  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers is required for non-example model names. "
                "Use example/* in local runs or install transformers."
            ) from exc

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code,
        )
        config = getattr(self.hf_model, "config", None)
        self.hidden_size = int(getattr(config, "hidden_size", self.hidden_size))
        self.vocab_size = int(getattr(config, "vocab_size", self.vocab_size))
        self.backend = "huggingface"

        if self.freeze_base:
            for param in self.hf_model.parameters():
                param.requires_grad = False

    def enable_standard_lora(
        self,
        rank: int,
        alpha: int,
        dropout: float,
        target_modules: list[str],
    ) -> None:
        """Enable standard-LoRA path.

        Internal backend: wraps selected tiny-model target linear modules.
        Hugging Face backend: applies a small hidden-state LoRA residual as a
        first-pass fallback (documented limitation).
        """
        if self.standard_lora_enabled:
            return

        if self.internal_model is not None:
            replaced: list[str] = []
            targets = target_modules or ["q_proj", "v_proj"]
            for module_name in targets:
                module = getattr(self.internal_model, module_name, None)
                if isinstance(module, nn.Linear):
                    setattr(
                        self.internal_model,
                        module_name,
                        LoRALinear(base=module, rank=rank, alpha=alpha, dropout=dropout),
                    )
                    replaced.append(module_name)
            self.standard_lora_scope = "internal:" + ",".join(sorted(replaced)) if replaced else "internal:none"
        else:
            self.hf_lora = HiddenStateLoRA(
                hidden_size=self.hidden_size,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            self.standard_lora_scope = "hf_hidden_state_fallback"

        self.standard_lora_enabled = True

    def forward_backbone(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> BaseForwardOutput:
        """Run base backbone and return final hidden states."""
        if self.hf_model is not None:
            out = self.hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = out.hidden_states[-1]
        else:
            if self.internal_model is None:
                raise RuntimeError("Internal model backend was not initialized")
            hidden_states = self.internal_model.backbone(input_ids)

        if self.hf_lora is not None:
            hidden_states = self.hf_lora(hidden_states)

        return BaseForwardOutput(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            metadata={
                "backend": self.backend,
                "frozen": self.freeze_base,
                "standard_lora_enabled": self.standard_lora_enabled,
                "standard_lora_scope": self.standard_lora_scope,
            },
        )

    def forward_lm_head(self, refined_hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states into token logits."""
        if self.hf_model is not None:
            return self.hf_model.lm_head(refined_hidden_states)
        if self.internal_model is None:
            raise RuntimeError("Internal model backend was not initialized")
        return self.internal_model.lm_head(refined_hidden_states)
