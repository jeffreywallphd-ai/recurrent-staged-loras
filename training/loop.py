"""Reusable training-loop primitives for baseline comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.staged_model import StagedLatentAdaptationModel


@dataclass(slots=True)
class EpochResult:
    epoch_index: int
    steps_completed: int
    average_loss: float
    tokens_seen: int
    wall_time_seconds: float


def loss_for_batch(model: StagedLatentAdaptationModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def train_epoch(
    *,
    model: StagedLatentAdaptationModel,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    max_steps: int,
    global_step_start: int,
    epoch_index: int,
    eval_loader: DataLoader[dict[str, torch.Tensor]] | None,
    eval_interval_steps: int,
    eval_enabled: bool,
) -> tuple[EpochResult, float | None]:
    """Run one training epoch (possibly truncated by max_steps)."""
    model.train()
    global_step = global_step_start
    loss_values: list[float] = []
    latest_eval_loss: float | None = None
    tokens_seen = 0
    epoch_start = perf_counter()

    for batch in dataloader:
        if global_step >= max_steps:
            break

        tokens_seen += int(batch["labels"][:, 1:].numel())
        loss = loss_for_batch(model, batch)
        loss_values.append(float(loss.item()))
        global_step += 1

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print(f"[train] epoch={epoch_index} step={global_step} loss={loss_values[-1]:.6f}")

        if eval_enabled and eval_loader is not None and eval_interval_steps > 0 and global_step % eval_interval_steps == 0:
            latest_eval_loss = evaluate(
                model=model,
                dataloader=eval_loader,
                global_step=global_step,
                epoch_index=epoch_index,
            )

    average_loss = float(sum(loss_values) / len(loss_values)) if loss_values else float("nan")
    return (
        EpochResult(
            epoch_index=epoch_index,
            steps_completed=global_step - global_step_start,
            average_loss=average_loss,
            tokens_seen=tokens_seen,
            wall_time_seconds=float(perf_counter() - epoch_start),
        ),
        latest_eval_loss,
    )


def evaluate(
    *,
    model: StagedLatentAdaptationModel,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    global_step: int,
    epoch_index: int,
) -> float:
    """Compute mean next-token loss on eval data."""
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            losses.append(float(loss_for_batch(model, batch).item()))
    model.train()
    eval_loss = float(sum(losses) / len(losses)) if losses else float("nan")
    print(f"[eval] epoch={epoch_index} step={global_step} loss={eval_loss:.6f}")
    return eval_loss


def run_training(
    *,
    model: StagedLatentAdaptationModel,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    eval_loader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    num_epochs: int,
    max_steps: int,
    eval_interval_steps: int,
    eval_enabled: bool,
) -> dict[str, float | int]:
    """Execute a full training run from prepared components."""
    run_start = perf_counter()
    global_step = 0
    train_loss = float("nan")
    eval_loss = float("nan")
    best_eval_loss = float("nan")
    eval_losses: list[float] = []
    epochs_completed = 0
    tokens_seen_train = 0
    train_wall_time_seconds = 0.0
    eval_wall_time_seconds = 0.0
    tokens_seen_eval = 0

    for epoch_idx in range(1, num_epochs + 1):
        epoch_result, interval_eval_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            max_steps=max_steps,
            global_step_start=global_step,
            epoch_index=epoch_idx,
            eval_loader=eval_loader,
            eval_interval_steps=eval_interval_steps,
            eval_enabled=eval_enabled,
        )
        global_step += epoch_result.steps_completed
        tokens_seen_train += epoch_result.tokens_seen
        train_wall_time_seconds += epoch_result.wall_time_seconds
        epochs_completed = epoch_idx
        train_loss = epoch_result.average_loss
        if interval_eval_loss is not None:
            eval_loss = interval_eval_loss
            eval_losses.append(interval_eval_loss)

        if global_step >= max_steps:
            break

    if eval_enabled:
        eval_tokens = sum(int(batch["labels"][:, 1:].numel()) for batch in eval_loader)
        eval_start = perf_counter()
        eval_loss = evaluate(model=model, dataloader=eval_loader, global_step=global_step, epoch_index=epochs_completed)
        eval_wall_time_seconds = float(perf_counter() - eval_start)
        tokens_seen_eval = eval_tokens
        eval_losses.append(eval_loss)

    if eval_losses:
        best_eval_loss = float(min(eval_losses))

    wall_time_total = float(perf_counter() - run_start)
    steps_per_second = float(global_step / train_wall_time_seconds) if train_wall_time_seconds > 0 else 0.0
    seconds_per_step = float(train_wall_time_seconds / global_step) if global_step > 0 else 0.0
    tokens_per_second_train = float(tokens_seen_train / train_wall_time_seconds) if train_wall_time_seconds > 0 else 0.0
    tokens_per_second_eval = float(tokens_seen_eval / eval_wall_time_seconds) if eval_wall_time_seconds > 0 else 0.0

    return {
        "global_steps": global_step,
        "epochs_completed": epochs_completed,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "best_eval_loss": best_eval_loss,
        "tokens_seen_train": tokens_seen_train,
        "tokens_seen_eval": tokens_seen_eval,
        "wall_time_seconds_total": wall_time_total,
        "wall_time_seconds_train": train_wall_time_seconds,
        "wall_time_seconds_eval": eval_wall_time_seconds,
        "tokens_per_second_train": tokens_per_second_train,
        "tokens_per_second_eval": tokens_per_second_eval,
        "seconds_per_step": seconds_per_step,
        "steps_per_second": steps_per_second,
    }
