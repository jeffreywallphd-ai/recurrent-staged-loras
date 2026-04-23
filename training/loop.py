"""Reusable training-loop primitives for baseline comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.staged_model import StagedLatentAdaptationModel


@dataclass(slots=True)
class EvalResult:
    loss: float
    wall_time_seconds: float
    tokens_seen: int
    next_token_accuracy: float
    top_5_accuracy: float
    target_token_accuracy: float | None
    target_sequence_exact_match: float | None


@dataclass(slots=True)
class EpochResult:
    epoch_index: int
    steps_completed: int
    average_loss: float
    tokens_seen: int
    train_wall_time_seconds: float
    interval_evals: list[EvalResult]


def _flatten_next_token_targets(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["labels"][:, 1:]
    return labels, labels.reshape(-1)


def loss_for_batch(model: StagedLatentAdaptationModel, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    _, labels_flat = _flatten_next_token_targets(batch)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels_flat)


def _safe_perplexity(loss: float) -> float:
    if not isfinite(loss):
        return float("nan")
    return float(torch.exp(torch.tensor(min(loss, 20.0), dtype=torch.float64)).item())


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
) -> EpochResult:
    """Run one training epoch (possibly truncated by max_steps)."""
    model.train()
    global_step = global_step_start
    loss_values: list[float] = []
    tokens_seen = 0
    train_wall_time_seconds = 0.0
    interval_evals: list[EvalResult] = []

    for batch in dataloader:
        if global_step >= max_steps:
            break

        step_start = perf_counter()
        tokens_seen += int(batch["labels"][:, 1:].numel())
        loss = loss_for_batch(model, batch)
        loss_values.append(float(loss.item()))
        global_step += 1

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        train_wall_time_seconds += float(perf_counter() - step_start)

        print(f"[train] epoch={epoch_index} step={global_step} loss={loss_values[-1]:.6f}")

        if eval_enabled and eval_loader is not None and eval_interval_steps > 0 and global_step % eval_interval_steps == 0:
            interval_evals.append(
                evaluate(
                    model=model,
                    dataloader=eval_loader,
                    global_step=global_step,
                    epoch_index=epoch_index,
                )
            )

    average_loss = float(sum(loss_values) / len(loss_values)) if loss_values else float("nan")
    return EpochResult(
        epoch_index=epoch_index,
        steps_completed=global_step - global_step_start,
        average_loss=average_loss,
        tokens_seen=tokens_seen,
        train_wall_time_seconds=train_wall_time_seconds,
        interval_evals=interval_evals,
    )


def evaluate(
    *,
    model: StagedLatentAdaptationModel,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    global_step: int,
    epoch_index: int,
) -> EvalResult:
    """Compute mean next-token metrics on eval data."""
    model.eval()
    eval_start = perf_counter()
    loss_sum = 0.0
    num_batches = 0
    next_token_correct = 0
    top_5_correct = 0
    tokens_seen = 0
    target_tokens_seen = 0
    target_tokens_correct = 0
    target_sequences_total = 0
    target_sequences_exact_match = 0

    with torch.no_grad():
        for batch in dataloader:
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            logits = out.logits[:, :-1, :]
            labels_matrix, labels_flat = _flatten_next_token_targets(batch)

            loss_sum += float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels_flat).item())
            num_batches += 1

            predictions = logits.argmax(dim=-1)
            matches = predictions.eq(labels_matrix)
            batch_tokens_seen = int(labels_flat.numel())
            tokens_seen += batch_tokens_seen
            next_token_correct += int(matches.sum().item())

            k = min(5, logits.shape[-1])
            top_k = logits.topk(k=k, dim=-1).indices
            top_5_correct += int(top_k.eq(labels_matrix.unsqueeze(-1)).any(dim=-1).sum().item())

            if "target_mask" in batch:
                target_mask = batch["target_mask"][:, 1:].bool()
                target_tokens = int(target_mask.sum().item())
                if target_tokens > 0:
                    target_tokens_seen += target_tokens
                    target_tokens_correct += int(matches[target_mask].sum().item())
                    sample_has_target = target_mask.any(dim=1)
                    per_sample_all_correct = (~target_mask | matches).all(dim=1)
                    target_sequences_total += int(sample_has_target.sum().item())
                    target_sequences_exact_match += int((sample_has_target & per_sample_all_correct).sum().item())

    model.train()
    eval_loss = float(loss_sum / num_batches) if num_batches else float("nan")
    next_token_accuracy = float(next_token_correct / tokens_seen) if tokens_seen > 0 else float("nan")
    top_5_accuracy = float(top_5_correct / tokens_seen) if tokens_seen > 0 else float("nan")
    target_token_accuracy = (
        float(target_tokens_correct / target_tokens_seen) if target_tokens_seen > 0 else None
    )
    target_sequence_exact_match = (
        float(target_sequences_exact_match / target_sequences_total) if target_sequences_total > 0 else None
    )

    print(
        f"[eval] epoch={epoch_index} step={global_step} "
        f"loss={eval_loss:.6f} next_token_acc={next_token_accuracy:.6f} top5_acc={top_5_accuracy:.6f}"
    )
    return EvalResult(
        loss=eval_loss,
        wall_time_seconds=float(perf_counter() - eval_start),
        tokens_seen=tokens_seen,
        next_token_accuracy=next_token_accuracy,
        top_5_accuracy=top_5_accuracy,
        target_token_accuracy=target_token_accuracy,
        target_sequence_exact_match=target_sequence_exact_match,
    )


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
    eval_results: list[EvalResult] = []
    epochs_completed = 0
    tokens_seen_train = 0
    train_wall_time_seconds = 0.0

    for epoch_idx in range(1, num_epochs + 1):
        epoch_result = train_epoch(
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
        train_wall_time_seconds += epoch_result.train_wall_time_seconds
        epochs_completed = epoch_idx
        train_loss = epoch_result.average_loss
        eval_results.extend(epoch_result.interval_evals)

        if global_step >= max_steps:
            break

    if eval_enabled:
        eval_results.append(
            evaluate(model=model, dataloader=eval_loader, global_step=global_step, epoch_index=epochs_completed)
        )

    if eval_results:
        eval_loss = eval_results[-1].loss
        best_eval_loss = float(min(result.loss for result in eval_results))

    eval_wall_time_seconds = float(sum(result.wall_time_seconds for result in eval_results))
    tokens_seen_eval = int(sum(result.tokens_seen for result in eval_results))
    wall_time_total = float(perf_counter() - run_start)

    steps_per_second = float(global_step / train_wall_time_seconds) if train_wall_time_seconds > 0 else 0.0
    seconds_per_step = float(train_wall_time_seconds / global_step) if global_step > 0 else 0.0
    tokens_per_second_train = float(tokens_seen_train / train_wall_time_seconds) if train_wall_time_seconds > 0 else 0.0
    tokens_per_second_eval = float(tokens_seen_eval / eval_wall_time_seconds) if eval_wall_time_seconds > 0 else 0.0

    last_eval = eval_results[-1] if eval_results else None
    output: dict[str, float | int] = {
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
        "eval_perplexity": _safe_perplexity(eval_loss),
        "train_perplexity": _safe_perplexity(train_loss),
    }
    if last_eval is not None:
        output["eval_next_token_accuracy"] = last_eval.next_token_accuracy
        output["eval_top_5_accuracy"] = last_eval.top_5_accuracy
        if last_eval.target_token_accuracy is not None:
            output["eval_target_token_accuracy"] = last_eval.target_token_accuracy
        if last_eval.target_sequence_exact_match is not None:
            output["eval_target_sequence_exact_match"] = last_eval.target_sequence_exact_match
    return output
