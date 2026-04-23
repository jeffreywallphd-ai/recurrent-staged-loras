"""Typed configuration objects for real-model staged adaptation experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RecurrenceMode = Literal["none", "latent_only", "shared", "stage_specialized"]
AdapterSharing = Literal["none", "shared", "per_step"]
ArchitectureType = Literal["dense", "moe"]


@dataclass(slots=True)
class BaseModelConfig:
    model_name: str
    tokenizer_name: str | None = None
    freeze_base: bool = True
    trust_remote_code: bool = False
    dtype: str = "bfloat16"
    device_map: str | None = "auto"
    max_seq_length: int = 4096
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    attn_implementation: str | None = None
    gradient_checkpointing: bool = False
    architecture_type: ArchitectureType = "dense"


@dataclass(slots=True)
class AdapterConfig:
    enabled: bool = False
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RefinerConfig:
    enabled: bool = False
    num_steps: int = 1
    hidden_size: int = 0
    recurrence_mode: RecurrenceMode = "none"
    adapter_sharing: AdapterSharing = "none"


@dataclass(slots=True)
class VariantConfig:
    name: str
    base: BaseModelConfig
    standard_lora: AdapterConfig = field(default_factory=AdapterConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    refiner_adapter: AdapterConfig = field(default_factory=AdapterConfig)
    trainable_modules: list[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.base.architecture_type not in {"dense", "moe"}:
            raise ValueError("model.architecture_type must be one of {'dense','moe'}")
        if not self.refiner.enabled:
            if self.refiner.recurrence_mode != "none":
                raise ValueError("refiner.recurrence_mode must be 'none' when refiner is disabled")
            if self.refiner.adapter_sharing != "none":
                raise ValueError("refiner.adapter_sharing must be 'none' when refiner is disabled")
            if self.refiner.num_steps != 1:
                raise ValueError("refiner.num_steps must be 1 when refiner is disabled")
            return
        if self.refiner.num_steps < 1:
            raise ValueError("refiner.num_steps must be >= 1 when enabled")
        if self.refiner.recurrence_mode == "none":
            raise ValueError("enabled refiner requires non-'none' recurrence_mode")
        if self.refiner.recurrence_mode == "latent_only":
            if self.refiner.adapter_sharing != "none":
                raise ValueError("latent_only requires adapter_sharing='none'")
            if self.refiner_adapter.enabled:
                raise ValueError("latent_only must not enable refiner adapters")
        if self.refiner.recurrence_mode == "shared" and self.refiner.adapter_sharing != "shared":
            raise ValueError("shared recurrence mode requires adapter_sharing='shared'")
        if self.refiner.recurrence_mode == "stage_specialized" and self.refiner.adapter_sharing != "per_step":
            raise ValueError("stage_specialized recurrence mode requires adapter_sharing='per_step'")


def _parse_adapter_config(raw: dict[str, Any] | None) -> AdapterConfig:
    raw = raw or {}
    return AdapterConfig(
        enabled=bool(raw.get("enabled", False)),
        rank=int(raw.get("rank", 8)),
        alpha=int(raw.get("alpha", 16)),
        dropout=float(raw.get("dropout", 0.0)),
        target_modules=list(raw.get("target_modules", [])),
    )


def parse_variant_config(raw: dict[str, Any]) -> VariantConfig:
    model = raw["model"]
    ref_raw = model.get("latent_refiner", {})
    variant = VariantConfig(
        name=str(raw["baseline"]),
        base=BaseModelConfig(
            model_name=str(model["name"]),
            tokenizer_name=model.get("tokenizer_name"),
            freeze_base=bool(model.get("frozen_base", True)),
            trust_remote_code=bool(model.get("trust_remote_code", False)),
            dtype=str(model.get("dtype", "bfloat16")),
            device_map=model.get("device_map", "auto"),
            max_seq_length=int(model.get("max_seq_length", 4096)),
            load_in_4bit=bool(model.get("load_in_4bit", False)),
            bnb_4bit_compute_dtype=str(model.get("bnb_4bit_compute_dtype", "bfloat16")),
            attn_implementation=model.get("attn_implementation"),
            gradient_checkpointing=bool(model.get("gradient_checkpointing", False)),
            architecture_type=str(model.get("architecture_type", "dense")),
        ),
        standard_lora=_parse_adapter_config(model.get("standard_lora")),
        refiner=RefinerConfig(
            enabled=bool(ref_raw.get("enabled", False)),
            num_steps=int(ref_raw.get("num_recurrent_steps", 1)),
            hidden_size=int(ref_raw.get("hidden_size", 0)),
            recurrence_mode=str(ref_raw.get("recurrence_mode", "none")),
            adapter_sharing=str(ref_raw.get("adapter_sharing", "none")),
        ),
        refiner_adapter=_parse_adapter_config(ref_raw.get("adapter")),
        trainable_modules=list(model.get("trainable_modules", [])),
    )
    variant.validate()
    return variant
