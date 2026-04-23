"""Reusable training orchestration helpers.

The CLI entrypoint should parse arguments and call these functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import random

import torch
from torch.utils.data import DataLoader

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
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def _count_trainable_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _count_total_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def build_training_components(runtime: RuntimeConfig) -> TrainingComponents:
    _set_seed(runtime.training.seed, runtime.training.deterministic)

    model = build_model_from_variant(runtime.variant)
    model.train()

    dataset_settings = dict(runtime.dataset.get("settings", {}))
    dataset_settings["seed"] = runtime.training.seed
    bundle = build_train_eval_datasets(
        name=runtime.dataset["name"],
        settings=dataset_settings,
        vocab_size=model.base_model.vocab_size,
    )

    train_loader = DataLoader(
        bundle.train,
        batch_size=runtime.training.batch_size,
        shuffle=False,
        collate_fn=collate_token_sequences,
    )
    eval_loader = DataLoader(
        bundle.eval,
        batch_size=runtime.training.batch_size,
        shuffle=False,
        collate_fn=collate_token_sequences,
    )

    trainable_params = _count_trainable_params(model)
    trainable_tensors = [param for param in model.parameters() if param.requires_grad]
    if trainable_params > 0 and not trainable_tensors:
        raise RuntimeError("Expected trainable tensors when trainable_params > 0")

    optimizer = None
    if trainable_tensors:
        optimizer = torch.optim.AdamW(
            trainable_tensors,
            lr=runtime.training.learning_rate,
            weight_decay=runtime.training.weight_decay,
        )

    return TrainingComponents(
        runtime=runtime,
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        optimizer=optimizer,
        trainable_params=trainable_params,
    )


def run_evaluation(*, components: TrainingComponents, global_step: int, epoch_index: int = 0) -> float:
    from training.loop import evaluate

    return evaluate(
        model=components.model,
        dataloader=components.eval_loader,
        global_step=global_step,
        epoch_index=epoch_index,
    )


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
    metadata_path = out_dir / "metadata.json"
    metadata.write(path=metadata_path)

    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(runtime.to_serializable_dict(), indent=2), encoding="utf-8")

    training_summary = run_training(
        model=components.model,
        train_loader=components.train_loader,
        eval_loader=components.eval_loader,
        optimizer=components.optimizer,
        num_epochs=runtime.training.num_epochs,
        max_steps=runtime.training.max_steps,
        eval_interval_steps=runtime.training.eval_interval_steps,
        eval_enabled=runtime.training.eval_enabled,
    )

    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(
        {
            "step": int(training_summary["global_steps"]),
            "model_state_dict": components.model.state_dict(),
            "optimizer_state_dict": components.optimizer.state_dict() if components.optimizer is not None else None,
        },
        checkpoint_path,
    )

    total_params = _count_total_params(components.model)
    trainable_param_fraction = (
        float(components.trainable_params / total_params) if total_params > 0 else 0.0
    )
    dataset_settings = dict(components.runtime.dataset.get("settings", {}))
    metrics = {
        "run_name": run_name,
        "config_name": config_name,
        "baseline_name": runtime.baseline,
        "dataset_name": runtime.dataset["name"],
        "dataset_mode": dataset_settings.get("mode"),
        "dataset_train_examples": len(components.train_loader.dataset),
        "dataset_eval_examples": len(components.eval_loader.dataset),
        "batch_size": runtime.training.batch_size,
        "learning_rate": runtime.training.learning_rate,
        "weight_decay": runtime.training.weight_decay,
        "seed": runtime.training.seed,
        "deterministic": runtime.training.deterministic,
        "final_train_loss": float(training_summary["train_loss"]),
        "final_eval_loss": float(training_summary["eval_loss"]),
        "best_eval_loss": float(training_summary["best_eval_loss"]),
        "train_loss": float(training_summary["train_loss"]),
        "eval_loss": float(training_summary["eval_loss"]),
        "global_steps_completed": int(training_summary["global_steps"]),
        "epochs_completed": int(training_summary["epochs_completed"]),
        "num_steps": int(training_summary["global_steps"]),
        "num_epochs": int(training_summary["epochs_completed"]),
        "tokens_seen_train": int(training_summary["tokens_seen_train"]),
        "tokens_seen_eval": int(training_summary["tokens_seen_eval"]),
        "tokens_per_second_train": float(training_summary["tokens_per_second_train"]),
        "tokens_per_second_eval": float(training_summary["tokens_per_second_eval"]),
        "wall_time_seconds_total": float(training_summary["wall_time_seconds_total"]),
        "wall_time_seconds_train": float(training_summary["wall_time_seconds_train"]),
        "wall_time_seconds_eval": float(training_summary["wall_time_seconds_eval"]),
        "seconds_per_step": float(training_summary["seconds_per_step"]),
        "steps_per_second": float(training_summary["steps_per_second"]),
        "backend": components.model.base_model.backend,
        "trainable_params": components.trainable_params,
        "total_params": total_params,
        "trainable_param_fraction": trainable_param_fraction,
        "latent_cache": LATENT_CACHE_STATUS,
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

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
