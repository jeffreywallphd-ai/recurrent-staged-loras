"""Runtime configuration normalization and model-construction helpers.

This module translates raw experiment JSON into validated typed configs and
enforces cross-field constraints that protect study correctness (dataset
support, compute-control modes, ablation compatibility).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json

from models.config import VariantConfig, parse_variant_config
from models.frozen_base import FrozenBaseCausalLM
from models.lora_bank import StepAwareLoRABank
from models.recurrent_refiner import RecurrentLatentRefiner
from models.staged_model import StagedLatentAdaptationModel

SHARED_DATASET_DEFAULTS: dict[str, Any] = {
    "name": "metamath_qa",
    "settings": {
        "subset_size": 25_000,
        "eval_fraction": 0.1,
        "seed": 0,
        "cache_dir": "./.cache/hf_datasets",
        "split": "train",
    },
    "external_evaluations": [],
}

SHARED_OUTPUT_DEFAULTS: dict[str, Any] = {"dir": "outputs/default"}
SUPPORTED_COMPUTE_CONTROL_MODES = {"effective_forward_passes", "wall_time", "tokens"}
SUPPORTED_PRIMARY_DATASETS = {"metamath_qa", "test_synthetic_stage_dataset"}
SUPPORTED_EXTERNAL_EVAL_DATASETS = {"gsm8k", "math", "svamp"}
TOKENIZER_REQUIRED_DATASETS = {"metamath_qa", "gsm8k", "math", "svamp"}
EXTERNAL_EVAL_DEFAULTS: dict[str, Any] = {"split": "test", "subset_size": 0, "seed": 0}


def _merge_runtime_defaults(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Merge user-provided runtime payload with shared dataset/output defaults."""
    dataset_raw = raw.get("dataset", {})
    dataset_name = str(dataset_raw.get("name", SHARED_DATASET_DEFAULTS["name"]))
    settings = dict(SHARED_DATASET_DEFAULTS["settings"])
    settings.update(dict(dataset_raw.get("settings", {})))
    external_evaluations = list(dataset_raw.get("external_evaluations", SHARED_DATASET_DEFAULTS["external_evaluations"]))

    output_raw = dict(SHARED_OUTPUT_DEFAULTS)
    output_raw.update(dict(raw.get("output", {})))
    return {"name": dataset_name, "settings": settings, "external_evaluations": external_evaluations}, output_raw


@dataclass(slots=True)
class ComputeControlConfig:
    enabled: bool = False
    mode: str = "effective_forward_passes"
    max_wall_time_seconds: float | None = None
    max_tokens: int | None = None


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 1
    num_epochs: int = 1
    max_steps: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    seed: int = 0
    eval_interval_steps: int = 100
    checkpoint_interval_steps: int = 100
    eval_enabled: bool = True
    deterministic: bool = False
    compute_control: ComputeControlConfig = field(default_factory=ComputeControlConfig)


@dataclass(slots=True)
class PublishConfig:
    enabled: bool = False
    hub_model_repo: str | None = None
    hub_dataset_repo: str | None = None
    private: bool = True
    commit_message: str | None = None
    include_checkpoint: bool = True
    include_metrics: bool = True
    include_dataset_partitions: bool = True


@dataclass(slots=True)
class RuntimeConfig:
    baseline: str
    variant: VariantConfig
    training: TrainingConfig
    publish: PublishConfig
    dataset: dict[str, Any]
    output: dict[str, Any]
    raw: dict[str, Any]

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "variant": asdict(self.variant),
            "training": asdict(self.training),
            "publish": asdict(self.publish),
            "dataset": self.dataset,
            "output": self.output,
            "raw": self.raw,
        }


def load_experiment_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fp:
        return json.load(fp)


def load_variant_config(path: str | Path) -> VariantConfig:
    return parse_variant_config(load_experiment_config(path))


def load_runtime_config_from_raw(raw: dict[str, Any]) -> RuntimeConfig:
    """Parse and validate a runtime config dictionary.

    Raises:
        ValueError: On unsupported datasets/modes or invalid ablation requests.
    """
    training_raw = raw.get("training", {})
    dataset_resolved, output_resolved = _merge_runtime_defaults(raw)
    compute_raw = dict(training_raw.get("compute_control", {}))
    compute = ComputeControlConfig(
        enabled=bool(compute_raw.get("enabled", False)),
        mode=str(compute_raw.get("mode", "effective_forward_passes")),
        max_wall_time_seconds=(float(compute_raw["max_wall_time_seconds"]) if "max_wall_time_seconds" in compute_raw else None),
        max_tokens=(int(compute_raw["max_tokens"]) if "max_tokens" in compute_raw else None),
    )
    if compute.mode not in SUPPORTED_COMPUTE_CONTROL_MODES:
        raise ValueError(f"Unsupported compute_control mode '{compute.mode}'. Expected one of {sorted(SUPPORTED_COMPUTE_CONTROL_MODES)}")
    if compute.enabled and compute.mode == "wall_time" and (compute.max_wall_time_seconds is None or compute.max_wall_time_seconds <= 0):
        raise ValueError("compute_control.mode=wall_time requires positive training.compute_control.max_wall_time_seconds")
    if compute.enabled and compute.mode == "tokens" and (compute.max_tokens is None or compute.max_tokens <= 0):
        raise ValueError("compute_control.mode=tokens requires positive training.compute_control.max_tokens")

    training = TrainingConfig(
        batch_size=int(training_raw.get("batch_size", 1)),
        num_epochs=int(training_raw.get("num_epochs", 1)),
        max_steps=int(training_raw.get("max_steps", 100)),
        learning_rate=float(training_raw.get("learning_rate", 2e-4)),
        weight_decay=float(training_raw.get("weight_decay", 0.0)),
        seed=int(training_raw.get("seed", 0)),
        eval_interval_steps=int(training_raw.get("eval_interval_steps", 100)),
        checkpoint_interval_steps=int(training_raw.get("checkpoint_interval_steps", 100)),
        eval_enabled=bool(training_raw.get("eval_enabled", True)),
        deterministic=bool(training_raw.get("deterministic", False)),
        compute_control=compute,
    )
    publish_raw = dict(raw.get("publish", {}))
    forbidden_publish_keys = [k for k in publish_raw if any(token in k.lower() for token in ("token", "secret", "api_key", "apikey", "password"))]
    if forbidden_publish_keys:
        raise ValueError(
            "publish config must not contain credentials/secrets. "
            f"Remove keys: {', '.join(sorted(forbidden_publish_keys))}"
        )
    publish = PublishConfig(
        enabled=bool(publish_raw.get("enabled", False)),
        hub_model_repo=(str(publish_raw["hub_model_repo"]) if publish_raw.get("hub_model_repo") else None),
        hub_dataset_repo=(str(publish_raw["hub_dataset_repo"]) if publish_raw.get("hub_dataset_repo") else None),
        private=bool(publish_raw.get("private", True)),
        commit_message=(str(publish_raw["commit_message"]) if publish_raw.get("commit_message") else None),
        include_checkpoint=bool(publish_raw.get("include_checkpoint", True)),
        include_metrics=bool(publish_raw.get("include_metrics", True)),
        include_dataset_partitions=bool(publish_raw.get("include_dataset_partitions", True)),
    )
    if publish.enabled and not publish.hub_model_repo and not publish.hub_dataset_repo:
        raise ValueError("publish.enabled=true requires publish.hub_model_repo and/or publish.hub_dataset_repo")

    dataset_name = str(dataset_resolved["name"]).strip().lower()
    if dataset_name not in SUPPORTED_PRIMARY_DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported primary datasets: {sorted(SUPPORTED_PRIMARY_DATASETS)}")
    dataset_resolved["name"] = dataset_name

    # Normalize external eval entries so downstream code can assume complete
    # names/splits/seeds/subset-size fields for identity accounting.
    normalized_external_evals: list[dict[str, Any]] = []
    for item in dataset_resolved.get("external_evaluations", []):
        if not isinstance(item, dict):
            raise ValueError("dataset.external_evaluations entries must be objects")
        external_name = str(item.get("name", "")).strip().lower()
        if not external_name:
            raise ValueError("dataset.external_evaluations entry missing non-empty 'name'")
        if external_name not in SUPPORTED_EXTERNAL_EVAL_DATASETS:
            raise ValueError(
                f"Unsupported external evaluation dataset '{external_name}'. "
                f"Supported external datasets: {sorted(SUPPORTED_EXTERNAL_EVAL_DATASETS)}"
            )
        split = str(item.get("split", EXTERNAL_EVAL_DEFAULTS["split"])).strip()
        if not split:
            raise ValueError("dataset.external_evaluations entry requires non-empty 'split' when provided")
        subset_size = int(item.get("subset_size", EXTERNAL_EVAL_DEFAULTS["subset_size"]))
        if subset_size < 0:
            raise ValueError("dataset.external_evaluations subset_size must be >= 0")
        seed = int(item.get("seed", EXTERNAL_EVAL_DEFAULTS["seed"]))
        normalized_external_evals.append({
            "name": external_name,
            "split": split,
            "subset_size": subset_size,
            "seed": seed,
        })
    dataset_resolved["external_evaluations"] = normalized_external_evals

    variant = parse_variant_config(raw)
    # Fail-fast ablation checks prevent generating silently invalid sweeps.
    ablation = raw.get("ablation", {})
    if isinstance(ablation, dict):
        if ablation.get("lora_rank") is not None and not (variant.standard_lora.enabled or variant.refiner_adapter.enabled):
            raise ValueError("lora_rank ablation requested but no active adapter found")
        if ablation.get("recurrent_steps") is not None and not variant.refiner.enabled:
            raise ValueError("recurrent_steps ablation requested but latent_refiner.enabled is false")

    return RuntimeConfig(
        baseline=str(raw["baseline"]),
        variant=variant,
        training=training,
        publish=publish,
        dataset=dataset_resolved,
        output={"dir": str(output_resolved["dir"])},
        raw=raw,
    )


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    return load_runtime_config_from_raw(load_experiment_config(path))


def build_model_from_variant(variant: VariantConfig) -> StagedLatentAdaptationModel:
    """Construct concrete model modules from a validated variant config."""
    base_model = FrozenBaseCausalLM(config=variant.base)

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
        # HF backbones frequently emit bf16/fp16 hidden states while newly
        # constructed recurrent modules default to fp32; align once at build time
        # so recurrent matmuls do not hit dtype/device mismatch errors.
        runtime_dtype, runtime_device = base_model.runtime_dtype_device()
        refiner = refiner.to(device=runtime_device, dtype=runtime_dtype)

    return StagedLatentAdaptationModel(config=variant, base_model=base_model, refiner=refiner)
