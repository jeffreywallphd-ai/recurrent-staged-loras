"""Training/evaluation orchestration for a single configured run.

This module is the bridge between validated runtime config objects and the
artifact surface consumed by downstream aggregation/statistics scripts. It owns:

- assembling model + datasets + optimizer,
- applying compute-control constraints to the training loop,
- enforcing required metric and dataset-identity invariants,
- writing run artifacts (`metrics.json`, `metadata.json`, diagnostics files).

Scientific invariant: run-level metrics are only considered pairable for
confirmatory analysis when dataset identity fields are present and stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path
import json
import random
import hashlib

import torch
from torch.utils.data import DataLoader
from functools import partial

from data.dataset import build_external_eval_dataset, build_train_eval_datasets, collate_token_sequences
from models.staged_model import StagedLatentAdaptationModel
from training.config_loader import RuntimeConfig, build_model_from_variant
from training.latent_cache import LATENT_CACHE_STATUS
from training.loop import evaluate, run_training
from training.run_metadata import RunMetadata
from training.model_validation import validate_model_checkpoint
from publish.huggingface_export import publish_run_directory


@dataclass(slots=True)
class TrainingComponents:
    runtime: RuntimeConfig
    model: StagedLatentAdaptationModel
    train_loader: DataLoader[dict[str, torch.Tensor]]
    eval_loader: DataLoader[dict[str, torch.Tensor]]
    external_eval_loaders: dict[str, DataLoader[dict[str, torch.Tensor]]]
    optimizer: torch.optim.Optimizer | None
    trainable_params: int
    preprocessing_summary: dict[str, Any]
    tokenizer: Any | None


@dataclass(slots=True)
class TrainResult:
    final_train_loss: float
    final_eval_loss: float
    best_eval_loss: float
    trainable_params: int
    total_params: int
    global_steps: int
    epochs_completed: int
    backend: str
    output_dir: Path
    checkpoint_path: Path


def _set_seed(seed: int, deterministic: bool) -> None:
    """Seed Python/Torch RNG state for reproducible sampling and initialization."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def _count_trainable_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _count_total_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def _meta_report_for_model(model: StagedLatentAdaptationModel) -> list[dict[str, str]]:
    return model.base_model.meta_parameter_report()


def _validate_model_loading_for_training(*, runtime: RuntimeConfig, model: StagedLatentAdaptationModel) -> None:
    base_cfg = runtime.variant.base
    lm_head_meta = model.base_model.meta_parameter_report(module_path="lm_head")
    meta_rows = _meta_report_for_model(model)

    if lm_head_meta:
        details = ", ".join(f"{item['name']} ({item['module']})" for item in lm_head_meta[:12])
        raise ValueError(
            "Model loading is incompatible with staged training: LM head parameters are on the meta device, "
            "but staged forward calls lm_head directly. "
            f"LM head meta parameters: {details}. "
            "Use a smaller model, set model.model_loading.mode='full_gpu', or disable strict checking only if you "
            "replace direct lm_head submodule calls."
        )

    if base_cfg.model_loading_require_no_meta_for_training and meta_rows:
        sample = ", ".join(f"{item['name']} ({item['module']})" for item in meta_rows[:20])
        raise ValueError(
            "Training aborted before first batch: parameters are still on the meta device. "
            f"Found {len(meta_rows)} meta parameters. Examples: {sample}. "
            "Set model.model_loading.require_no_meta_for_training=false to allow meta/offloaded loading, or "
            "switch to model.model_loading.mode='full_gpu' / a smaller model for materialized training."
        )


def _validate_trainable_gradients(model: StagedLatentAdaptationModel) -> None:
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    expected_trainable = (
        bool(model.config.refiner.enabled)
        or bool(model.config.standard_lora.enabled)
        or bool(model.config.refiner_adapter.enabled)
    )
    if expected_trainable and not trainable:
        raise ValueError("No trainable parameters found for an adaptation-enabled configuration.")
    if model.config.refiner.enabled and not any(name.startswith("refiner.") for name, _ in trainable):
        raise ValueError("Refiner is enabled but no refiner parameters are marked trainable.")


def _log_model_loading_diagnostics(*, runtime: RuntimeConfig, model: StagedLatentAdaptationModel) -> None:
    base_cfg = runtime.variant.base
    base_summary = model.base_model.parameter_device_summary()
    trainable_devices: dict[str, int] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        module = name.split(".", 1)[0]
        key = f"{module}:{param.device}"
        trainable_devices[key] = trainable_devices.get(key, 0) + 1
    meta_rows = _meta_report_for_model(model)
    print(
        "[device] "
        f"loading_mode={base_cfg.model_loading_mode} "
        f"allow_offload={base_cfg.model_loading_allow_offload} "
        f"require_no_meta_for_training={base_cfg.model_loading_require_no_meta_for_training} "
        f"device_map={base_cfg.device_map}"
    )
    print(
        "[device] "
        f"input_embeddings={base_summary['input_embeddings']} "
        f"transformer_body={base_summary['transformer_body']} "
        f"lm_head={base_summary['lm_head']} "
        f"trainable_modules={trainable_devices if trainable_devices else 'none'} "
        f"meta_params={len(meta_rows)}"
    )


def _stable_hash(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _token_count(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.sum().item())
    return 0


def _example_partition_row(example: dict[str, Any], *, split: str, index: int) -> dict[str, Any]:
    source_signature = str(example.get("source_signature", ""))
    answer_text = str(example.get("answer_text", ""))
    answer_text_normalized = str(example.get("answer_text_normalized", ""))
    stage_meta = {
        "stage1_tokens": _token_count(example.get("stage1_mask")),
        "stage2_tokens": _token_count(example.get("stage2_mask")),
        "stage3_tokens": _token_count(example.get("stage3_mask")),
        "answer_tokens": _token_count(example.get("answer_mask")),
    }
    payload = {
        "split": split,
        "index": index,
        "source_signature": source_signature,
        "answer_text": answer_text,
        "answer_text_normalized": answer_text_normalized,
        "stage_token_counts": stage_meta,
    }
    payload["sample_id"] = f"{split}-{index}"
    payload["sample_hash"] = _stable_hash(payload)
    return payload


def _write_dataset_partitions_artifact(*, out_dir: Path, components: TrainingComponents, runtime: RuntimeConfig) -> Path:
    train_rows = [_example_partition_row(dict(ex), split="train", index=i) for i, ex in enumerate(components.train_loader.dataset)]
    eval_rows = [_example_partition_row(dict(ex), split="eval", index=i) for i, ex in enumerate(components.eval_loader.dataset)]
    payload = {
        "source_dataset_name": runtime.dataset["name"],
        "source_split": runtime.dataset.get("settings", {}).get("split", "train"),
        "seed": runtime.training.seed,
        "subset_size": runtime.dataset.get("settings", {}).get("subset_size"),
        "preprocessing_settings": dict(runtime.dataset.get("settings", {})),
        "preprocessing_summary": dict(components.preprocessing_summary),
        "train_sample_count": len(train_rows),
        "eval_sample_count": len(eval_rows),
        "train": train_rows,
        "eval": eval_rows,
    }
    path = out_dir / "dataset_partitions.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


REQUIRED_STUDY_METRICS = [
    "final_answer_accuracy",
    "final_answer_exact_match",
    "normalized_numeric_answer_accuracy",
    "stage_2_token_accuracy",
    "stage_3_token_accuracy",
]

EXTERNAL_DATASET_IDENTITY_FIELDS = [
    "dataset_name",
    "dataset_type",
    "dataset_split",
    "dataset_seed",
    "dataset_subset_size",
    "dataset_eval_fraction",
    "dataset_fingerprint",
    "train_sample_ids_hash",
    "eval_sample_ids_hash",
]


def _require_metric(summary: dict[str, Any], key: str) -> Any:
    """Require a metric needed by the study schema.

    Raises:
        ValueError: If the metric is absent/None, because confirmatory reporting
            requires complete primary outcomes.
    """
    if key not in summary or summary[key] is None:
        raise ValueError(f"Training summary missing required study metric '{key}'")
    return summary[key]


def _to_metrics_payload(eval_result: Any) -> dict[str, Any]:
    return {
        "eval_loss": float(eval_result.loss),
        "stage_2_token_accuracy": eval_result.stage_2_token_accuracy,
        "stage_3_token_accuracy": eval_result.stage_3_token_accuracy,
        "final_answer_accuracy": eval_result.final_answer_accuracy,
        "final_answer_exact_match": eval_result.final_answer_exact_match,
        "final_answer_normalized_match": eval_result.final_answer_normalized_match,
        "normalized_numeric_answer_accuracy": eval_result.normalized_numeric_answer_accuracy,
        "symbolic_answer_accuracy": eval_result.symbolic_answer_accuracy,
        "answer_eval_string_count": int(eval_result.answer_eval_string_count),
        "answer_eval_numeric_count": int(eval_result.answer_eval_numeric_count),
    }


def _extract_external_identity(summary: dict[str, Any], *, external_name: str) -> dict[str, Any]:
    """Validate and extract external-dataset identity fields from preprocessing."""
    missing_fields = [field for field in EXTERNAL_DATASET_IDENTITY_FIELDS if summary.get(field) in (None, "")]
    if missing_fields:
        raise ValueError(
            f"External dataset '{external_name}' preprocessing summary missing required identity fields: "
            + ", ".join(missing_fields)
        )
    return {field: summary.get(field) for field in EXTERNAL_DATASET_IDENTITY_FIELDS}


def build_training_components(runtime: RuntimeConfig) -> TrainingComponents:
    """Build all runtime objects required to execute one run.

    Inputs:
        runtime: Parsed/validated runtime configuration.
    Outputs:
        TrainingComponents bundle with model, loaders, optimizer, summaries.
    Side effects:
        Initializes tokenizer/model weights and dataset objects.
    Failure modes:
        Raises ValueError on invalid external dataset specs or missing tokenizer.
    """
    _set_seed(runtime.training.seed, runtime.training.deterministic)
    model = build_model_from_variant(runtime.variant)
    _validate_trainable_gradients(model)
    _log_model_loading_diagnostics(runtime=runtime, model=model)
    _validate_model_loading_for_training(runtime=runtime, model=model)
    model.train()

    dataset_name = str(runtime.dataset["name"]).strip().lower()
    external_names = [str(item.get("name", "")).strip().lower() for item in runtime.dataset.get("external_evaluations", []) if isinstance(item, dict)]
    # Tokenizer-backed datasets require text decoding for answer-span scoring and
    # deterministic char->token span construction.
    requires_tokenizer = dataset_name in {"metamath_qa"} or any(name in {"gsm8k", "math", "svamp"} for name in external_names)
    tokenizer = None
    if requires_tokenizer:
        from transformers import AutoTokenizer  # type: ignore

        tok_name = runtime.variant.base.tokenizer_name or runtime.variant.base.model_name
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=runtime.variant.base.trust_remote_code)
        except Exception as exc:  # pragma: no cover - backend/network dependent
            raise ValueError(f"Failed to initialize tokenizer '{tok_name}' required by configured datasets") from exc
    if requires_tokenizer and tokenizer is None:
        raise ValueError("Tokenizer is required by configured primary/external datasets but was not initialized")

    dataset_settings = dict(runtime.dataset.get("settings", {}))
    dataset_settings["seed"] = runtime.training.seed
    dataset_settings["max_seq_length"] = runtime.variant.base.max_seq_length

    bundle = build_train_eval_datasets(
        name=dataset_name,
        settings=dataset_settings,
        vocab_size=model.base_model.vocab_size,
        tokenizer=tokenizer,
    )

    pad_token_id = 0
    if tokenizer is not None:
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_token_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0)

    collate_fn = partial(collate_token_sequences, pad_token_id=pad_token_id)
    train_loader = DataLoader(bundle.train, batch_size=runtime.training.batch_size, shuffle=False, collate_fn=collate_fn)
    eval_loader = DataLoader(bundle.eval, batch_size=runtime.training.batch_size, shuffle=False, collate_fn=collate_fn)

    external_eval_loaders: dict[str, DataLoader[dict[str, torch.Tensor]]] = {}
    external_summaries: dict[str, dict[str, Any]] = {}
    for item in runtime.dataset.get("external_evaluations", []):
        if not isinstance(item, dict):
            raise ValueError("dataset.external_evaluations entries must be objects")
        external_name = str(item.get("name", "")).strip().lower()
        if not external_name:
            raise ValueError("dataset.external_evaluations entry missing non-empty 'name'")
        settings = dict(dataset_settings)
        settings.update({k: v for k, v in item.items() if k != "name"})
        ext_bundle = build_external_eval_dataset(name=external_name, settings=settings, tokenizer=tokenizer)
        external_eval_loaders[external_name] = DataLoader(ext_bundle.eval, batch_size=runtime.training.batch_size, shuffle=False, collate_fn=collate_fn)
        external_summaries[external_name] = ext_bundle.preprocessing_summary

    trainable_params = _count_trainable_params(model)
    trainable_tensors = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    if trainable_tensors:
        optimizer = torch.optim.AdamW(trainable_tensors, lr=runtime.training.learning_rate, weight_decay=runtime.training.weight_decay)

    preprocessing_summary: dict[str, Any] = dict(bundle.preprocessing_summary)
    if external_summaries:
        preprocessing_summary["external_evaluations"] = external_summaries

    return TrainingComponents(runtime=runtime, model=model, train_loader=train_loader, eval_loader=eval_loader, external_eval_loaders=external_eval_loaders, optimizer=optimizer, trainable_params=trainable_params, preprocessing_summary=preprocessing_summary, tokenizer=tokenizer)


def run_training_loop(*, components: TrainingComponents, run_name: str, config_name: str = "unknown") -> TrainResult:
    """Execute training/evaluation and emit run-level artifacts.

    This function is the canonical writer for `metrics.json`; downstream tools
    assume field names and identity semantics defined here.
    """
    runtime = components.runtime
    out_dir = Path(runtime.output["dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = RunMetadata(
        run_name=run_name,
        baseline=runtime.baseline,
        dataset_name=runtime.dataset["name"],
        model_name=runtime.variant.base.model_name,
        output_dir=str(out_dir),
    )
    metadata.write(path=out_dir / "metadata.json")
    (out_dir / "config.json").write_text(json.dumps(runtime.to_serializable_dict(), indent=2), encoding="utf-8")
    (out_dir / "dataset_preprocessing_summary.json").write_text(json.dumps(components.preprocessing_summary, indent=2), encoding="utf-8")
    _write_dataset_partitions_artifact(out_dir=out_dir, components=components, runtime=runtime)
    base_checkpoint_path = out_dir / "base_checkpoint.pt"
    torch.save({"model_state_dict": components.model.state_dict()}, base_checkpoint_path)

    base_max_steps = int(runtime.training.max_steps)
    recurrence_steps = int(runtime.variant.refiner.num_steps if runtime.variant.refiner.enabled else 1)
    effective_forward_passes = recurrence_steps
    compute = runtime.training.compute_control
    adjusted_max_steps = base_max_steps
    max_train_tokens: int | None = None
    max_wall_time_seconds: float | None = None
    if compute.enabled:
        # Compute-control equalizes selected budget surfaces, not optimization
        # dynamics. In particular, effective_forward_pass control rescales
        # optimizer-step budget but does not make training trajectories identical.
        if compute.mode == "effective_forward_passes":
            adjusted_max_steps = max(1, int(base_max_steps / max(effective_forward_passes, 1)))
        elif compute.mode == "tokens":
            if compute.max_tokens is None:
                raise ValueError("compute_control.tokens requested but max_tokens is not configured")
            max_train_tokens = int(compute.max_tokens)
        elif compute.mode == "wall_time":
            if compute.max_wall_time_seconds is None:
                raise ValueError("compute_control.wall_time requested but max_wall_time_seconds is not configured")
            max_wall_time_seconds = float(compute.max_wall_time_seconds)
        else:
            raise ValueError(f"Unsupported compute control mode '{compute.mode}'")

    training_summary = run_training(
        model=components.model,
        train_loader=components.train_loader,
        eval_loader=components.eval_loader,
        optimizer=components.optimizer,
        num_epochs=runtime.training.num_epochs,
        max_steps=adjusted_max_steps,
        eval_interval_steps=runtime.training.eval_interval_steps,
        eval_enabled=runtime.training.eval_enabled,
        tokenizer=components.tokenizer,
        max_train_tokens=max_train_tokens,
        max_wall_time_seconds=max_wall_time_seconds,
    )

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save({"step": int(training_summary["global_steps"]), "model_state_dict": components.model.state_dict()}, checkpoint_path)
    if runtime.validation.enabled:
        validation_result = validate_model_checkpoint(
            base_checkpoint=base_checkpoint_path,
            trained_checkpoint=checkpoint_path,
            output_dir=out_dir,
            runtime_config=runtime.to_serializable_dict(),
            validation_cfg=runtime.validation,
        )
        print(f"[validation] report='{validation_result.report_path}' status={'PASS' if validation_result.passed else 'FAIL'}")
        if not validation_result.passed and runtime.validation.blocking:
            missing = "; ".join(validation_result.missing_required_items)
            raise ValueError(f"Model validation failed: {missing}. Report: {validation_result.report_path}")

    total_params = _count_total_params(components.model)
    # These outcomes are required for confirmatory/report tables; fail fast when
    # any are missing to avoid silently publishing partial rows.
    required_metric_values = {name: _require_metric(training_summary, name) for name in REQUIRED_STUDY_METRICS}
    dataset_summary = dict(components.preprocessing_summary)
    global_steps_completed = int(training_summary["global_steps"])
    effective_optimizer_steps = global_steps_completed
    tokens_seen_train = int(training_summary["tokens_seen_train"])
    tokens_per_optimizer_step = (float(tokens_seen_train) / float(effective_optimizer_steps)) if effective_optimizer_steps > 0 else 0.0

    metrics = {
        "run_name": run_name,
        "config_name": config_name,
        "baseline_name": runtime.baseline,
        "baseline_family": runtime.raw.get("baseline_family", runtime.baseline),
        "run_scope": runtime.raw.get("run_scope", "confirmatory"),
        "dataset_name": runtime.dataset["name"],
        "dataset_train_examples": len(components.train_loader.dataset),
        "dataset_eval_examples": len(components.eval_loader.dataset),
        "dataset_type": str(dataset_summary.get("dataset_type", "primary")),
        "dataset_split": dataset_summary.get("dataset_split"),
        "dataset_seed": dataset_summary.get("dataset_seed"),
        "dataset_subset_size": dataset_summary.get("dataset_subset_size"),
        "dataset_eval_fraction": dataset_summary.get("dataset_eval_fraction"),
        "dataset_fingerprint": dataset_summary.get("dataset_fingerprint"),
        "train_sample_ids_hash": dataset_summary.get("train_sample_ids_hash"),
        "eval_sample_ids_hash": dataset_summary.get("eval_sample_ids_hash"),
        "seed": runtime.training.seed,
        "final_train_loss": float(training_summary["train_loss"]),
        "final_eval_loss": float(training_summary["eval_loss"]),
        "best_eval_loss": float(training_summary["best_eval_loss"]),
        "eval_perplexity": float(training_summary["eval_perplexity"]),
        "train_perplexity": float(training_summary["train_perplexity"]),
        "wall_time_seconds_total": float(training_summary["wall_time_seconds_total"]),
        "wall_time_seconds_train": float(training_summary["wall_time_seconds_train"]),
        "wall_time_seconds_eval": float(training_summary["wall_time_seconds_eval"]),
        "tokens_seen_train": tokens_seen_train,
        "tokens_seen_eval": int(training_summary["tokens_seen_eval"]),
        "tokens_per_second_train": float(training_summary["tokens_per_second_train"]),
        "tokens_per_second_eval": float(training_summary["tokens_per_second_eval"]),
        "steps_per_second": float(training_summary["steps_per_second"]),
        "seconds_per_step": float(training_summary["seconds_per_step"]),
        "trainable_params": int(components.trainable_params),
        "total_params": int(total_params),
        "trainable_param_fraction": float(components.trainable_params / total_params) if total_params > 0 else 0.0,
        "architecture_type": runtime.variant.base.architecture_type,
        "model_name": runtime.variant.base.model_name,
        "recurrence_steps": recurrence_steps,
        "effective_forward_passes_per_example": effective_forward_passes,
        "compute_control_enabled": bool(compute.enabled),
        "compute_control_mode": compute.mode,
        "adjusted_max_steps": int(adjusted_max_steps),
        "effective_optimizer_steps": effective_optimizer_steps,
        "tokens_per_optimizer_step": tokens_per_optimizer_step,
        "global_steps_completed": global_steps_completed,
        "epochs_completed": int(training_summary["epochs_completed"]),
        "backend": components.model.base_model.backend,
        "latent_cache": LATENT_CACHE_STATUS,
        "final_answer_accuracy": required_metric_values["final_answer_accuracy"],
        "final_answer_exact_match": required_metric_values["final_answer_exact_match"],
        "final_answer_normalized_match": training_summary.get("final_answer_normalized_match"),
        "answer_span_normalized_accuracy": training_summary.get("final_answer_accuracy"),
        "answer_span_exact_match": training_summary.get("final_answer_exact_match"),
        "answer_span_normalized_match": training_summary.get("final_answer_normalized_match"),
        "stage_3_token_accuracy": required_metric_values["stage_3_token_accuracy"],
        "stage_2_token_accuracy": required_metric_values["stage_2_token_accuracy"],
        "normalized_numeric_answer_accuracy": required_metric_values["normalized_numeric_answer_accuracy"],
        "symbolic_answer_accuracy": training_summary.get("symbolic_answer_accuracy"),
        "answer_span_numeric_accuracy": training_summary.get("normalized_numeric_answer_accuracy"),
        "answer_eval_string_count": int(training_summary.get("answer_eval_string_count", 0)),
        "answer_eval_numeric_count": int(training_summary.get("answer_eval_numeric_count", 0)),
        "answer_eval_skipped_no_stage3": int(training_summary.get("answer_eval_skipped_no_stage3", 0)),
        "answer_eval_skipped_no_answer_span": int(training_summary.get("answer_eval_skipped_no_answer_span", 0)),
        "answer_eval_skipped_missing_answer_text": int(training_summary.get("answer_eval_skipped_missing_answer_text", 0)),
        "answer_eval_skipped_missing_numeric_target": int(training_summary.get("answer_eval_skipped_missing_numeric_target", 0)),
        "answer_eval_normalized_match_count": int(training_summary.get("answer_eval_normalized_match_count", 0)),
        "answer_eval_exact_match_count": int(training_summary.get("answer_eval_exact_match_count", 0)),
        "answer_eval_numeric_match_count": int(training_summary.get("answer_eval_numeric_match_count", 0)),
        "answer_eval_multi_value_target_count": int(training_summary.get("answer_eval_multi_value_target_count", 0)),
        "answer_eval_numeric_pred_value_count": int(training_summary.get("answer_eval_numeric_pred_value_count", 0)),
        "answer_eval_numeric_target_value_count": int(training_summary.get("answer_eval_numeric_target_value_count", 0)),
        "answer_eval_numeric_value_match_count": int(training_summary.get("answer_eval_numeric_value_match_count", 0)),
        "answer_eval_multi_value_exact_set_match_count": int(training_summary.get("answer_eval_multi_value_exact_set_match_count", 0)),
        "answer_eval_multi_value_partial_match_count": int(training_summary.get("answer_eval_multi_value_partial_match_count", 0)),
        "answer_eval_multi_value_unmatched_count": int(training_summary.get("answer_eval_multi_value_unmatched_count", 0)),
        "answer_eval_string_match_numeric_miss_count": int(training_summary.get("answer_eval_string_match_numeric_miss_count", 0)),
        "answer_eval_normalized_only_count": int(training_summary.get("answer_eval_normalized_only_count", 0)),
        "answer_eval_skipped_ambiguous_numeric": int(training_summary.get("answer_eval_skipped_ambiguous_numeric", 0)),
        "symbolic_eval_attempt_count": int(training_summary.get("symbolic_eval_attempt_count", 0)),
        "symbolic_eval_success_count": int(training_summary.get("symbolic_eval_success_count", 0)),
        "symbolic_eval_failure_count": int(training_summary.get("symbolic_eval_failure_count", 0)),
        "answer_eval_symbolic_match_count": int(training_summary.get("answer_eval_symbolic_match_count", 0)),
        "answer_eval_numeric_abs_tolerance": float(training_summary.get("answer_eval_numeric_abs_tolerance", 0.0)),
        "answer_eval_numeric_multi_value_rule": training_summary.get("answer_eval_numeric_multi_value_rule", "strict_set"),
        "answer_eval_answer_length_histogram": training_summary.get("answer_eval_answer_length_histogram", {}),
        "ablation_recurrent_steps": runtime.raw.get("ablation", {}).get("recurrent_steps") if isinstance(runtime.raw.get("ablation", {}), dict) else None,
        "ablation_lora_rank": runtime.raw.get("ablation", {}).get("lora_rank") if isinstance(runtime.raw.get("ablation", {}), dict) else None,
    }

    required_dataset_identity_fields = [
        "dataset_fingerprint",
        "train_sample_ids_hash",
        "eval_sample_ids_hash",
        "dataset_split",
        "dataset_seed",
        "dataset_subset_size",
        "dataset_eval_fraction",
    ]
    # Dataset identity is mandatory because confirmatory pairing later enforces
    # fingerprint/hash equality across compared seeds.
    missing_dataset_fields = [f for f in required_dataset_identity_fields if metrics.get(f) in (None, "")]
    if missing_dataset_fields:
        raise ValueError("Dataset preprocessing summary missing required identity fields: " + ", ".join(missing_dataset_fields))

    external_eval_metrics: dict[str, dict[str, Any]] = {}
    external_preproc = components.preprocessing_summary.get("external_evaluations", {})
    if external_preproc not in ({}, None) and not isinstance(external_preproc, dict):
        raise ValueError("Preprocessing summary external_evaluations must be a mapping when present")
    for ds_name, ds_loader in components.external_eval_loaders.items():
        ds_result = evaluate(model=components.model, dataloader=ds_loader, tokenizer=components.tokenizer)
        ds_summary = external_preproc.get(ds_name, {}) if isinstance(external_preproc, dict) else {}
        if not isinstance(ds_summary, dict):
            raise ValueError(f"External preprocessing summary for dataset '{ds_name}' must be a mapping")
        # External eval is written as a self-contained descriptive payload with
        # its own identity fields so analyses can avoid inheriting primary IDs.
        external_eval_metrics[ds_name] = {
            **_extract_external_identity(ds_summary, external_name=ds_name),
            **_to_metrics_payload(ds_result),
        }
    if external_eval_metrics:
        metrics["external_eval"] = external_eval_metrics

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    answer_eval_diagnostics = {
        "string_answer_scored": metrics["answer_eval_string_count"],
        "numeric_answer_scored": metrics["answer_eval_numeric_count"],
        "skipped_no_stage3": metrics["answer_eval_skipped_no_stage3"],
        "skipped_no_answer_span": metrics["answer_eval_skipped_no_answer_span"],
        "skipped_missing_answer_text": metrics["answer_eval_skipped_missing_answer_text"],
        "skipped_missing_numeric_target": metrics["answer_eval_skipped_missing_numeric_target"],
        "strict_exact_match_count": metrics["answer_eval_exact_match_count"],
        "normalized_string_match_count": metrics["answer_eval_normalized_match_count"],
        "numeric_match_count": metrics["answer_eval_numeric_match_count"],
        "multi_value_answer_count": metrics["answer_eval_multi_value_target_count"],
        "numeric_predicted_value_count": metrics["answer_eval_numeric_pred_value_count"],
        "numeric_target_value_count": metrics["answer_eval_numeric_target_value_count"],
        "numeric_value_match_count": metrics["answer_eval_numeric_value_match_count"],
        "multi_value_exact_set_match_count": metrics["answer_eval_multi_value_exact_set_match_count"],
        "multi_value_partial_match_count": metrics["answer_eval_multi_value_partial_match_count"],
        "multi_value_unmatched_count": metrics["answer_eval_multi_value_unmatched_count"],
        "answer_length_distribution": metrics["answer_eval_answer_length_histogram"],
        "failure_mode_normalized_match_but_not_exact": metrics["answer_eval_normalized_only_count"],
        "failure_mode_numeric_miss_but_string_match": metrics["answer_eval_string_match_numeric_miss_count"],
        "skipped_ambiguous_numeric": metrics["answer_eval_skipped_ambiguous_numeric"],
        "numeric_abs_tolerance": metrics["answer_eval_numeric_abs_tolerance"],
        "symbolic_eval_attempt_count": metrics["symbolic_eval_attempt_count"],
        "symbolic_eval_success_count": metrics["symbolic_eval_success_count"],
        "symbolic_eval_failure_count": metrics["symbolic_eval_failure_count"],
        "symbolic_match_count": metrics["answer_eval_symbolic_match_count"],
        "symbolic_answer_accuracy": metrics["symbolic_answer_accuracy"],
        "numeric_multi_value_rule": metrics["answer_eval_numeric_multi_value_rule"],
        "notes": "Answer metrics decode only tokens in answer_mask/final_answer_mask (answer span, excluding the literal 'Final Answer:' header). stage_3_token_accuracy still uses the full stage3_mask section. Symbolic equivalence is attempted only for expression-like answers; parse failures are counted explicitly.",
    }
    if external_eval_metrics:
        answer_eval_diagnostics["external_eval"] = external_eval_metrics
    (out_dir / "answer_eval_diagnostics.json").write_text(json.dumps(answer_eval_diagnostics, indent=2), encoding="utf-8")
    if runtime.publish.enabled:
        publish_run_directory(run_dir=out_dir, runtime=runtime, publish_cfg=runtime.publish)

    return TrainResult(
        final_train_loss=metrics["final_train_loss"],
        final_eval_loss=metrics["final_eval_loss"],
        best_eval_loss=metrics["best_eval_loss"],
        trainable_params=components.trainable_params,
        total_params=total_params,
        global_steps=metrics["global_steps_completed"],
        epochs_completed=metrics["epochs_completed"],
        backend=metrics["backend"],
        output_dir=out_dir,
        checkpoint_path=checkpoint_path,
    )
