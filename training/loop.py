"""Minimal reusable training loop for baseline comparisons."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import build_train_eval_datasets, collate_integer_sequences
from models.staged_model import StagedLatentAdaptationModel
from .config_loader import RuntimeConfig, build_model_from_variant
from .latent_cache import maybe_load_latent_cache, maybe_write_latent_cache
from .run_metadata import RunMetadata


@dataclass(slots=True)
class TrainResult:
    final_train_loss: float
    final_eval_loss: float
    trainable_params: int
    global_steps: int
    backend: str
    output_dir: Path
    checkpoint_paths: list[Path]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _loss_for_batch(model: StagedLatentAdaptationModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = batch["input_ids"][:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def _evaluate(model: StagedLatentAdaptationModel, dataloader: DataLoader[dict[str, torch.Tensor]]) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            losses.append(_loss_for_batch(model, batch).item())
    model.train()
    return float(sum(losses) / len(losses)) if losses else float("nan")


def _count_trainable_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def run_training(runtime: RuntimeConfig, run_name: str) -> TrainResult:
    _set_seed(runtime.training.seed)

    model = build_model_from_variant(runtime.variant)
    model.train()

    bundle = build_train_eval_datasets(
        name=runtime.dataset["name"],
        settings=runtime.dataset.get("settings", {}),
        vocab_size=model.base_model.vocab_size,
    )

    train_loader = DataLoader(
        bundle.train,
        batch_size=runtime.training.batch_size,
        shuffle=False,
        collate_fn=collate_integer_sequences,
    )
    eval_loader = DataLoader(
        bundle.eval,
        batch_size=runtime.training.batch_size,
        shuffle=False,
        collate_fn=collate_integer_sequences,
    )

    trainable_params = _count_trainable_params(model)
    optimizer = None
    if trainable_params > 0:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=runtime.training.learning_rate,
            weight_decay=runtime.training.weight_decay,
        )

    out_dir = Path(runtime.output["dir"]) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Latent-cache hook remains placeholder: check for cache and emit marker files.
    maybe_load_latent_cache(out_dir / "latent_cache", split="train")

    metadata = RunMetadata(
        run_name=run_name,
        baseline=runtime.baseline,
        dataset_name=runtime.dataset["name"],
        model_name=runtime.variant.base.model_name,
        output_dir=str(out_dir),
    )
    metadata.write()

    (out_dir / "parsed_config.json").write_text(
        json.dumps(runtime.to_serializable_dict(), indent=2),
        encoding="utf-8",
    )

    global_steps = 0
    train_loss = float("nan")
    eval_loss = _evaluate(model, eval_loader)
    checkpoints: list[Path] = []

    for _epoch in range(runtime.training.num_epochs):
        for batch in train_loader:
            if global_steps >= runtime.training.max_steps:
                break
            global_steps += 1

            loss = _loss_for_batch(model, batch)
            train_loss = float(loss.item())
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            if global_steps % runtime.training.eval_interval_steps == 0:
                eval_loss = _evaluate(model, eval_loader)

            if global_steps % runtime.training.checkpoint_interval_steps == 0:
                checkpoint_path = out_dir / f"checkpoint_step_{global_steps}.pt"
                torch.save(
                    {
                        "step": global_steps,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                    },
                    checkpoint_path,
                )
                checkpoints.append(checkpoint_path)
        if global_steps >= runtime.training.max_steps:
            break

    final_ckpt = out_dir / "model_final.pt"
    torch.save(
        {
            "step": global_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        },
        final_ckpt,
    )
    checkpoints.append(final_ckpt)

    metrics = {
        "baseline": runtime.baseline,
        "backend": model.base_model.backend,
        "global_steps": global_steps,
        "trainable_params": trainable_params,
        "final_train_loss": train_loss,
        "final_eval_loss": eval_loss,
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    maybe_write_latent_cache(out_dir / "latent_cache", split="train", payload={"status": "placeholder"})

    return TrainResult(
        final_train_loss=train_loss,
        final_eval_loss=eval_loss,
        trainable_params=trainable_params,
        global_steps=global_steps,
        backend=model.base_model.backend,
        output_dir=out_dir,
        checkpoint_paths=checkpoints,
    )

