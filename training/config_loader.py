"""Experiment configuration loading and model/training-build helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

from models.config import VariantConfig, parse_variant_config
from models.frozen_base import FrozenBaseCausalLM
from models.lora_bank import StepAwareLoRABank
from models.recurrent_refiner import RecurrentLatentRefiner
from models.staged_model import StagedLatentAdaptationModel

SHARED_DATASET_DEFAULTS: dict[str, Any] = {
    "name": "synthetic_integer_sequences",
    "settings": {
        "num_examples": 64,
        "sequence_length": 12,
        "eval_fraction": 0.25,
        "seed": 0,
    },
}

SHARED_OUTPUT_DEFAULTS: dict[str, Any] = {"dir": "outputs/default"}


def _merge_runtime_defaults(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_raw = raw.get("dataset", {})
    dataset_name = str(dataset_raw.get("name", SHARED_DATASET_DEFAULTS["name"]))
    settings = dict(SHARED_DATASET_DEFAULTS["settings"])
    settings.update(dict(dataset_raw.get("settings", {})))

    output_raw = dict(SHARED_OUTPUT_DEFAULTS)
    output_raw.update(dict(raw.get("output", {})))
    return {"name": dataset_name, "settings": settings}, output_raw


@dataclass(slots=True)
class TrainingConfig:
    """Minimal runtime training settings loaded from experiment JSON."""

    batch_size: int = 4
    num_epochs: int = 1
    max_steps: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    seed: int = 0
    eval_interval_steps: int = 4
    checkpoint_interval_steps: int = 4


@dataclass(slots=True)
class RuntimeConfig:
    """Top-level runtime config consumed by training entrypoint."""

    baseline: str
    variant: VariantConfig
    training: TrainingConfig
    dataset: dict[str, Any]
    output: dict[str, Any]
    raw: dict[str, Any]

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "variant": asdict(self.variant),
            "training": asdict(self.training),
            "dataset": self.dataset,
            "output": self.output,
            "raw": self.raw,
        }


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON experiment config."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_variant_config(path: str | Path) -> VariantConfig:
    """Load and parse experiment config into typed variant config."""
    return parse_variant_config(load_experiment_config(path))


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    """Load runtime settings (variant + training/data/output sections)."""
    raw = load_experiment_config(path)
    training_raw = raw.get("training", {})
    dataset_resolved, output_resolved = _merge_runtime_defaults(raw)

    training = TrainingConfig(
        batch_size=int(training_raw.get("batch_size", 4)),
        num_epochs=int(training_raw.get("num_epochs", 1)),
        max_steps=int(training_raw.get("max_steps", 8)),
        learning_rate=float(training_raw.get("learning_rate", 1e-3)),
        weight_decay=float(training_raw.get("weight_decay", 0.0)),
        seed=int(training_raw.get("seed", 0)),
        eval_interval_steps=int(training_raw.get("eval_interval_steps", 4)),
        checkpoint_interval_steps=int(training_raw.get("checkpoint_interval_steps", 4)),
    )

    return RuntimeConfig(
        baseline=str(raw["baseline"]),
        variant=parse_variant_config(raw),
        training=training,
        dataset=dataset_resolved,
        output={"dir": str(output_resolved["dir"])},
        raw=raw,
    )


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
