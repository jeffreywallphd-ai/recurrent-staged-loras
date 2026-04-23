"""Reusable training orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path
import json
import random

import torch
from torch.utils.data import DataLoader
from functools import partial

from data.dataset import build_train_eval_datasets, collate_token_sequences
from models.staged_model import StagedLatentAdaptationModel
from training.config_loader import RuntimeConfig, build_model_from_variant
from training.latent_cache import LATENT_CACHE_STATUS
from training.loop import run_training
from training.run_metadata import RunMetadata


@dataclass(slots=True)
class TrainingComponents:
    runtime: RuntimeConfig
    model: StagedLatentAdaptationModel
    train_loader: DataLoader[dict[str, torch.Tensor]]
    eval_loader: DataLoader[dict[str, torch.Tensor]]
    optimizer: torch.optim.Optimizer | None
    trainable_params: int
    preprocessing_summary: dict[str, int]
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


def build_training_components(runtime: RuntimeConfig) -> TrainingComponents:
    _set_seed(runtime.training.seed, runtime.training.deterministic)
    model = build_model_from_variant(runtime.variant)
    model.train()

    tokenizer = None
    if runtime.dataset["name"] == "metamath_qa":
        from transformers import AutoTokenizer  # type: ignore

        tok_name = runtime.variant.base.tokenizer_name or runtime.variant.base.model_name
        tokenizer = AutoTokenizer.from_pretrained(tok_name, trust_remote_code=runtime.variant.base.trust_remote_code)

    dataset_settings = dict(runtime.dataset.get("settings", {}))
    dataset_settings["seed"] = runtime.training.seed
    dataset_settings["max_seq_length"] = runtime.variant.base.max_seq_length

    bundle = build_train_eval_datasets(
        name=runtime.dataset["name"],
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

    trainable_params = _count_trainable_params(model)
    trainable_tensors = [p for p in model.parameters() if p.requires_grad]
    optimizer = None
    if trainable_tensors:
        optimizer = torch.optim.AdamW(trainable_tensors, lr=runtime.training.learning_rate, weight_decay=runtime.training.weight_decay)

    return TrainingComponents(runtime=runtime, model=model, train_loader=train_loader, eval_loader=eval_loader, optimizer=optimizer, trainable_params=trainable_params, preprocessing_summary=bundle.preprocessing_summary, tokenizer=tokenizer)


def run_training_loop(*, components: TrainingComponents, run_name: str, config_name: str = "unknown") -> TrainResult:
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

    training_summary = run_training(
        model=components.model,
        train_loader=components.train_loader,
        eval_loader=components.eval_loader,
        optimizer=components.optimizer,
        num_epochs=runtime.training.num_epochs,
        max_steps=runtime.training.max_steps,
        eval_interval_steps=runtime.training.eval_interval_steps,
        eval_enabled=runtime.training.eval_enabled,
        tokenizer=components.tokenizer,
    )

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save({"step": int(training_summary["global_steps"]), "model_state_dict": components.model.state_dict()}, checkpoint_path)

    total_params = _count_total_params(components.model)
    recurrence_steps = int(runtime.variant.refiner.num_steps if runtime.variant.refiner.enabled else 1)
    metrics = {
        "run_name": run_name,
        "config_name": config_name,
        "baseline_name": runtime.baseline,
        "dataset_name": runtime.dataset["name"],
        "dataset_train_examples": len(components.train_loader.dataset),
        "dataset_eval_examples": len(components.eval_loader.dataset),
        "seed": runtime.training.seed,
        "final_train_loss": float(training_summary["train_loss"]),
        "final_eval_loss": float(training_summary["eval_loss"]),
        "best_eval_loss": float(training_summary["best_eval_loss"]),
        "eval_perplexity": float(training_summary["eval_perplexity"]),
        "train_perplexity": float(training_summary["train_perplexity"]),
        "wall_time_seconds_total": float(training_summary["wall_time_seconds_total"]),
        "wall_time_seconds_train": float(training_summary["wall_time_seconds_train"]),
        "wall_time_seconds_eval": float(training_summary["wall_time_seconds_eval"]),
        "tokens_seen_train": int(training_summary["tokens_seen_train"]),
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
        "effective_forward_passes_per_example": recurrence_steps,
        "global_steps_completed": int(training_summary["global_steps"]),
        "epochs_completed": int(training_summary["epochs_completed"]),
        "backend": components.model.base_model.backend,
        "latent_cache": LATENT_CACHE_STATUS,
        "final_answer_accuracy": training_summary.get("final_answer_accuracy"),
        "final_answer_exact_match": training_summary.get("final_answer_exact_match"),
        "stage_3_token_accuracy": training_summary.get("stage_3_token_accuracy"),
        "stage_2_token_accuracy": training_summary.get("stage_2_token_accuracy"),
        "normalized_numeric_answer_accuracy": training_summary.get("normalized_numeric_answer_accuracy"),
        "answer_eval_string_count": int(training_summary.get("answer_eval_string_count", 0)),
        "answer_eval_numeric_count": int(training_summary.get("answer_eval_numeric_count", 0)),
        "answer_eval_skipped_no_stage3": int(training_summary.get("answer_eval_skipped_no_stage3", 0)),
        "answer_eval_skipped_missing_answer_text": int(training_summary.get("answer_eval_skipped_missing_answer_text", 0)),
        "answer_eval_skipped_missing_numeric_target": int(training_summary.get("answer_eval_skipped_missing_numeric_target", 0)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    answer_eval_diagnostics = {
        "string_answer_scored": metrics["answer_eval_string_count"],
        "numeric_answer_scored": metrics["answer_eval_numeric_count"],
        "skipped_no_stage3": metrics["answer_eval_skipped_no_stage3"],
        "skipped_missing_answer_text": metrics["answer_eval_skipped_missing_answer_text"],
        "skipped_missing_numeric_target": metrics["answer_eval_skipped_missing_numeric_target"],
        "notes": "String metrics use normalized decoded stage-3 answers. Exact match uses strict raw string equality; numeric accuracy uses normalized floating-point extraction.",
    }
    (out_dir / "answer_eval_diagnostics.json").write_text(json.dumps(answer_eval_diagnostics, indent=2), encoding="utf-8")

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
