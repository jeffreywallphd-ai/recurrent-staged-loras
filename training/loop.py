"""Reusable training-loop primitives for baseline comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.staged_model import StagedLatentAdaptationModel


@dataclass(slots=True)
class EpochResult:
    epoch_index: int
    steps_completed: int
    average_loss: float


def loss_for_batch(model: StagedLatentAdaptationModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = batch["input_ids"][:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def train_epoch(
    *,
    model: StagedLatentAdaptationModel,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer | None,
    max_steps: int,
    global_step_start: int,
    epoch_index: int,
) -> EpochResult:
    """Run one training epoch (possibly truncated by max_steps)."""
    model.train()
    global_step = global_step_start
    loss_values: list[float] = []

    for batch in dataloader:
        if global_step >= max_steps:
            break

        loss = loss_for_batch(model, batch)
        loss_values.append(float(loss.item()))
        global_step += 1

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print(f"[train] epoch={epoch_index} step={global_step} loss={loss_values[-1]:.6f}")

    average_loss = float(sum(loss_values) / len(loss_values)) if loss_values else float("nan")
    return EpochResult(epoch_index=epoch_index, steps_completed=global_step - global_step_start, average_loss=average_loss)


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
) -> dict[str, float | int]:
    """Execute a full training run from prepared components."""
    global_step = 0
    train_loss = float("nan")
    eval_loss = evaluate(model=model, dataloader=eval_loader, global_step=global_step, epoch_index=0)

    for epoch_idx in range(1, num_epochs + 1):
        epoch_result = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            max_steps=max_steps,
            global_step_start=global_step,
            epoch_index=epoch_idx,
        )
        global_step += epoch_result.steps_completed
        train_loss = epoch_result.average_loss

        if global_step % eval_interval_steps == 0 or global_step >= max_steps:
            eval_loss = evaluate(model=model, dataloader=eval_loader, global_step=global_step, epoch_index=epoch_idx)

        if global_step >= max_steps:
            break

    return {
        "global_steps": global_step,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
    }
