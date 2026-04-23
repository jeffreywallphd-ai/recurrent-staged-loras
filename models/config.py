"""Typed configuration objects for model variants.

These structures intentionally stay lightweight. They define the knobs needed
for baseline selection and model wiring while deferring full training features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


RecurrenceMode = Literal["none", "latent_only", "shared", "stage_specialized"]
AdapterSharing = Literal["none", "shared", "per_step"]


@dataclass(slots=True)
class BaseModelConfig:
    """Configuration for the frozen base causal language model wrapper."""

    model_name: str
    freeze_base: bool = True
    trust_remote_code: bool = False


@dataclass(slots=True)
class AdapterConfig:
    """Configuration for standard LoRA and step-aware refiner adapters."""

    enabled: bool = False
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RefinerConfig:
    """Configuration for recurrent latent refinement."""

    enabled: bool = False
    num_steps: int = 1
    hidden_size: int = 0
    recurrence_mode: RecurrenceMode = "none"
    adapter_sharing: AdapterSharing = "none"


@dataclass(slots=True)
class VariantConfig:
    """Top-level model variant definition used by experiments and training."""

    name: str
    base: BaseModelConfig
    standard_lora: AdapterConfig = field(default_factory=AdapterConfig)
    refiner: RefinerConfig = field(default_factory=RefinerConfig)
    refiner_adapter: AdapterConfig = field(default_factory=AdapterConfig)
    trainable_modules: list[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate high-level variant consistency."""
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
            raise ValueError("enabled refiner requires recurrence_mode in {'latent_only','shared','stage_specialized'}")

        if self.refiner.recurrence_mode == "latent_only":
            if self.refiner.adapter_sharing != "none":
                raise ValueError("latent_only mode requires adapter_sharing='none'")
            if self.refiner_adapter.enabled:
                raise ValueError("latent_only mode must not enable refiner adapters")

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
    """Parse loaded JSON config into validated typed configuration."""
    model = raw["model"]
    ref_raw = model.get("latent_refiner", {})

    variant = VariantConfig(
        name=str(raw["baseline"]),
        base=BaseModelConfig(
            model_name=str(model["name"]),
            freeze_base=bool(model.get("frozen_base", True)),
            trust_remote_code=bool(model.get("trust_remote_code", False)),
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
