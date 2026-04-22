"""Typed configuration objects for model variants.

These structures intentionally stay lightweight. They define the knobs needed
for baseline selection and model wiring while deferring implementation details.
"""

from dataclasses import dataclass, field
from typing import Literal


RecurrenceMode = Literal["none", "shared", "stage_specialized"]


@dataclass(slots=True)
class BaseModelConfig:
    """Configuration for the frozen base causal language model wrapper."""

    model_name: str
    freeze_base: bool = True
    trust_remote_code: bool = False


@dataclass(slots=True)
class AdapterConfig:
    """Configuration for standard LoRA and step-aware LoRA adapters."""

    enabled: bool = False
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RefinerConfig:
    """Configuration for the recurrent latent refiner.

    recurrence_mode controls whether recurrence is off, shared across steps, or
    stage-specialized with different adapters per step.
    """

    enabled: bool = False
    num_steps: int = 1
    hidden_size: int = 0
    recurrence_mode: RecurrenceMode = "none"


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
        """Validate high-level variant consistency.

        TODO: extend with stricter checks as implementation details are added.
        """
        if self.refiner.enabled and self.refiner.num_steps < 1:
            raise ValueError("refiner.num_steps must be >= 1 when enabled")
        if self.refiner.recurrence_mode == "none" and self.refiner.num_steps != 1:
            raise ValueError("non-recurrent mode expects exactly one step")
