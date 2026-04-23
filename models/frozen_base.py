"""Backbone wrapper for frozen-base experiments with optional PEFT LoRA.

Provides a unified interface for:
- real Hugging Face CausalLM backends (study path),
- tiny internal deterministic backend (tests only).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .config import BaseModelConfig


@dataclass(slots=True)
class BaseForwardOutput:
    hidden_states: torch.Tensor
    attention_mask: torch.Tensor | None
    metadata: dict[str, Any]


class TinyInternalCausalLM(nn.Module):
    """Tiny deterministic fallback reserved for unit tests only."""

    def __init__(self, vocab_size: int = 256, hidden_size: int = 64) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def backbone(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        return self.out_proj(torch.tanh(self.q_proj(h) + self.v_proj(h)))


class FrozenBaseCausalLM(nn.Module):
    def __init__(self, config: BaseModelConfig) -> None:
        super().__init__()
        self.config = config
        self.model_name = config.model_name
        self.backend = "huggingface"
        self.vocab_size = 256
        self.hidden_size = 64
        self.internal_model: TinyInternalCausalLM | None = None
        self.hf_model: Any | None = None
        self.standard_lora_enabled = False
        self.standard_lora_scope = "disabled"
        self._build_model()

    def _torch_dtype(self, name: str) -> torch.dtype:
        mapping = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        return mapping.get(name, torch.bfloat16)

    def _build_model(self) -> None:
        # Explicitly restricted tiny path only for tests.
        if self.model_name.startswith("test/"):
            self.backend = "internal"
            self.internal_model = TinyInternalCausalLM(vocab_size=self.vocab_size, hidden_size=self.hidden_size)
            # Keep test backend behavior aligned with HF runtimes where hidden states
            # often come out as bf16/fp16 while new modules default to fp32.
            self.internal_model = self.internal_model.to(dtype=self._torch_dtype(self.config.dtype))
            if self.config.freeze_base:
                for p in self.internal_model.parameters():
                    p.requires_grad = False
            return

        from transformers import AutoModelForCausalLM  # type: ignore

        quantization_config = None
        if self.config.load_in_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self._torch_dtype(self.config.bnb_4bit_compute_dtype),
            )

        kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
            "torch_dtype": self._torch_dtype(self.config.dtype),
        }
        if self.config.device_map is not None:
            kwargs["device_map"] = self.config.device_map
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config

        self.hf_model = AutoModelForCausalLM.from_pretrained(self.model_name, **kwargs)
        if self.config.gradient_checkpointing:
            self.hf_model.gradient_checkpointing_enable()
        cfg = getattr(self.hf_model, "config", None)
        self.hidden_size = int(getattr(cfg, "hidden_size", self.hidden_size))
        self.vocab_size = int(getattr(cfg, "vocab_size", self.vocab_size))

        if self.config.freeze_base:
            for p in self.hf_model.parameters():
                p.requires_grad = False

    def enable_standard_lora(self, rank: int, alpha: int, dropout: float, target_modules: list[str]) -> None:
        """Attach PEFT LoRA adapters to the HF backbone.

        Raises:
            RuntimeError: For test backend or uninitialized HF models.
        """
        if self.standard_lora_enabled:
            return
        if self.internal_model is not None:
            raise RuntimeError("Internal test backend does not support real standard LoRA.")
        if self.hf_model is None:
            raise RuntimeError("HF model not initialized")

        from peft import LoraConfig, get_peft_model  # type: ignore

        resolved_targets = target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        peft_cfg = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=resolved_targets,
            task_type="CAUSAL_LM",
            bias="none",
        )
        self.hf_model = get_peft_model(self.hf_model, peft_cfg)
        self.standard_lora_enabled = True
        self.standard_lora_scope = "hf_target_modules:" + ",".join(resolved_targets)

    def forward_backbone(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> BaseForwardOutput:
        if self.internal_model is not None:
            hidden_states = self.internal_model.backbone(input_ids)
        else:
            if self.hf_model is None:
                raise RuntimeError("HF backend missing")
            out = self.hf_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden_states = out.hidden_states[-1]

        return BaseForwardOutput(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            metadata={
                "backend": self.backend,
                "frozen": self.config.freeze_base,
                "standard_lora_enabled": self.standard_lora_enabled,
                "standard_lora_scope": self.standard_lora_scope,
                "architecture_type": self.config.architecture_type,
                "model_name": self.config.model_name,
            },
        )

    def forward_lm_head(self, refined_hidden_states: torch.Tensor) -> torch.Tensor:
        if self.internal_model is not None:
            return self.internal_model.lm_head(refined_hidden_states)
        if self.hf_model is None:
            raise RuntimeError("HF backend missing")
        return self.hf_model.lm_head(refined_hidden_states)

    def runtime_dtype_device(self) -> tuple[torch.dtype, torch.device]:
        """Return dtype/device used by backbone/LM-head parameters."""
        if self.internal_model is not None:
            param = next(self.internal_model.parameters())
            return param.dtype, param.device
        if self.hf_model is None:
            raise RuntimeError("HF backend missing")
        param = next(self.hf_model.parameters())
        return param.dtype, param.device

    def lm_head_dtype_device(self) -> tuple[torch.dtype, torch.device]:
        """Return dtype/device expected by LM head projection."""
        if self.internal_model is not None:
            param = next(self.internal_model.lm_head.parameters())
            return param.dtype, param.device
        if self.hf_model is None:
            raise RuntimeError("HF backend missing")
        param = next(self.hf_model.lm_head.parameters())
        return param.dtype, param.device

    @property
    def input_device(self) -> torch.device:
        """Best-effort device where token indices should be placed.

        DataLoader tensors are created on CPU by default. For Hugging Face models
        loaded with `device_map="auto"`, embeddings can live on a specific shard
        device, so inputs must target that embedding/input module device.
        """
        if self.internal_model is not None:
            return self.internal_model.embed_tokens.weight.device
        if self.hf_model is None:
            raise RuntimeError("HF backend missing")

        get_input_embeddings = getattr(self.hf_model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            embeddings = get_input_embeddings()
            if embeddings is not None and hasattr(embeddings, "weight"):
                return embeddings.weight.device

        # Fallback for models that do not expose input embeddings.
        return next(self.hf_model.parameters()).device
