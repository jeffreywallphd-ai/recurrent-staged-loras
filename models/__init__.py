"""Core model interfaces for latent refiner baseline comparisons."""

from .config import AdapterConfig, BaseModelConfig, RefinerConfig, VariantConfig
from .frozen_base import FrozenBaseCausalLM
from .lora_bank import StepAwareLoRABank
from .recurrent_refiner import RecurrentLatentRefiner
from .staged_model import StagedLatentAdaptationModel

__all__ = [
    "AdapterConfig",
    "BaseModelConfig",
    "RefinerConfig",
    "VariantConfig",
    "FrozenBaseCausalLM",
    "StepAwareLoRABank",
    "RecurrentLatentRefiner",
    "StagedLatentAdaptationModel",
]
