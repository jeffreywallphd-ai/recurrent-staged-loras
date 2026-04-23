"""Experiment configuration loading and model-building helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from models.config import VariantConfig, parse_variant_config
from models.frozen_base import FrozenBaseCausalLM
from models.lora_bank import StepAwareLoRABank
from models.recurrent_refiner import RecurrentLatentRefiner
from models.staged_model import StagedLatentAdaptationModel


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON experiment config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_variant_config(path: str | Path) -> VariantConfig:
    """Load and parse experiment config into typed variant config."""
    return parse_variant_config(load_experiment_config(path))


def build_model_from_variant(variant: VariantConfig) -> StagedLatentAdaptationModel:
    """Construct the staged model path from typed variant config."""
    base_model = FrozenBaseCausalLM(
        model_name=variant.base.model_name,
        freeze_base=variant.base.freeze_base,
        trust_remote_code=variant.base.trust_remote_code,
    )

    if variant.standard_lora.enabled:
        base_model.enable_standard_lora(
            rank=variant.standard_lora.rank,
            alpha=variant.standard_lora.alpha,
            dropout=variant.standard_lora.dropout,
            target_modules=variant.standard_lora.target_modules,
        )

    refiner = None
    if variant.refiner.enabled:
        hidden_size = variant.refiner.hidden_size or base_model.hidden_size

        adapter_bank = None
        if variant.refiner_adapter.enabled:
            adapter_bank = StepAwareLoRABank(
                num_steps=variant.refiner.num_steps,
                hidden_size=hidden_size,
                rank=variant.refiner_adapter.rank,
                alpha=variant.refiner_adapter.alpha,
                shared_across_steps=(variant.refiner.adapter_sharing == "shared"),
                enabled=True,
                dropout=variant.refiner_adapter.dropout,
            )

        refiner = RecurrentLatentRefiner(
            num_steps=variant.refiner.num_steps,
            hidden_size=hidden_size,
            adapter_bank=adapter_bank,
        )

    model = StagedLatentAdaptationModel(config=variant, base_model=base_model, refiner=refiner)
    return model
