"""Reusable training-loop primitives for baseline comparisons."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.staged_model import StagedLatentAdaptationModel
from training.answer_eval import NUMERIC_ABS_TOL, NUMERIC_MULTI_VALUE_RULE, normalize_answer_text, numeric_match


@dataclass(slots=True)
class EvalResult:
    loss: float
    wall_time_seconds: float
    tokens_seen: int
    stage_2_token_accuracy: float | None
    stage_3_token_accuracy: float | None
    final_answer_accuracy: float | None
    final_answer_exact_match: float | None
    final_answer_normalized_match: float | None
    normalized_numeric_answer_accuracy: float | None
    answer_eval_string_count: int
    answer_eval_numeric_count: int
    answer_eval_skipped_no_stage3: int
    answer_eval_skipped_no_answer_span: int
    answer_eval_skipped_missing_answer_text: int
    answer_eval_skipped_missing_numeric_target: int
    answer_eval_normalized_match_count: int
    answer_eval_exact_match_count: int
    answer_eval_numeric_match_count: int
    answer_eval_multi_value_target_count: int
    answer_eval_numeric_pred_value_count: int
    answer_eval_numeric_target_value_count: int
    answer_eval_numeric_value_match_count: int
    answer_eval_string_match_numeric_miss_count: int
    answer_eval_normalized_only_count: int
    answer_eval_skipped_ambiguous_numeric: int
    answer_eval_length_histogram: dict[str, int]


def _safe_perplexity(loss: float) -> float:
    if not isfinite(loss):
        return float("nan")
    return float(torch.exp(torch.tensor(min(loss, 20.0), dtype=torch.float64)).item())


def _masked_ce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    vocab = logits.shape[-1]
    per_tok = F.cross_entropy(logits.reshape(-1, vocab), labels.reshape(-1), reduction="none").reshape_as(labels)
    denom = mask.sum().clamp(min=1)
    return (per_tok * mask.float()).sum() / denom


def _decode_answer_tokens(token_ids: torch.Tensor, tokenizer: Any | None) -> str:
    ids = [int(x) for x in token_ids.tolist()]
    if tokenizer is not None:
        return str(tokenizer.decode(ids, skip_special_tokens=True)).strip()
    return " ".join(str(x) for x in ids).strip()


def loss_for_batch(model: StagedLatentAdaptationModel, batch: dict[str, torch.Tensor | list[str]]) -> torch.Tensor:
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    assert isinstance(input_ids, torch.Tensor) and isinstance(attention_mask, torch.Tensor)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    labels = batch["labels"][:, 1:]
    assert isinstance(labels, torch.Tensor)

    if model.config.refiner.enabled and out.extras["per_step"]:
        step_hidden_states = out.extras["per_step"]
        logits_steps = [model.base_model.forward_lm_head(h)[:, :-1, :] for h in step_hidden_states]

        stage1 = batch["stage1_mask"][:, 1:]
        stage2 = batch["stage2_mask"][:, 1:]
        stage3 = batch["stage3_mask"][:, 1:]
        assert isinstance(stage1, torch.Tensor) and isinstance(stage2, torch.Tensor) and isinstance(stage3, torch.Tensor)

        stage_masks = [stage1, stage2, stage3]
        losses: list[torch.Tensor] = []
        for idx, step_logits in enumerate(logits_steps):
            target_mask = stage_masks[min(idx, 2)]
            losses.append(_masked_ce(step_logits, labels, target_mask))
        return torch.stack(losses).mean()

    logits = out.logits[:, :-1, :]
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def train_epoch(
    *,
    model: StagedLatentAdaptationModel,
    dataloader: DataLoader[dict[str, torch.Tensor | list[str]]],
    optimizer: torch.optim.Optimizer | None,
    max_steps: int,
    global_step_start: int,
    eval_enabled: bool,
    eval_interval_steps: int,
    eval_loader: DataLoader[dict[str, torch.Tensor | list[str]]],
    tokenizer: Any | None,
) -> tuple[float, int, float, int, list[EvalResult]]:
    model.train()
    step = global_step_start
    losses: list[float] = []
    tokens_seen = 0
    wall = 0.0
    interval_results: list[EvalResult] = []

    for batch in dataloader:
        if step >= max_steps:
            break
        start = perf_counter()
        labels = batch["labels"][:, 1:]
        assert isinstance(labels, torch.Tensor)
        tokens_seen += int(labels.ne(-100).sum().item())
        loss = loss_for_batch(model, batch)
        losses.append(float(loss.item()))
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        step += 1
        wall += perf_counter() - start

        if eval_enabled and eval_interval_steps > 0 and (step % eval_interval_steps == 0):
            interval_results.append(evaluate(model=model, dataloader=eval_loader, tokenizer=tokenizer))

    avg = float(sum(losses) / len(losses)) if losses else float("nan")
    return avg, step - global_step_start, wall, tokens_seen, interval_results


def evaluate(*, model: StagedLatentAdaptationModel, dataloader: DataLoader[dict[str, torch.Tensor | list[str]]], tokenizer: Any | None = None) -> EvalResult:
    model.eval()
    start = perf_counter()
    losses: list[float] = []
    tokens_seen = 0

    s2_correct = s2_total = 0
    s3_correct = s3_total = 0

    normalized_correct = normalized_total = 0
    exact_correct = exact_total = 0
    normalized_match_correct = normalized_match_total = 0
    numeric_correct = numeric_total = 0
    numeric_match_count = 0
    multi_value_target_count = 0
    numeric_pred_value_count = 0
    numeric_target_value_count = 0
    numeric_value_match_count = 0
    string_match_numeric_miss_count = 0
    normalized_only_count = 0
    skipped_ambiguous_numeric = 0
    answer_length_bins: Counter[str] = Counter()

    skipped_no_stage3 = 0
    skipped_no_answer_span = 0
    skipped_missing_answer_text = 0
    skipped_missing_numeric_target = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"][:, 1:]
            stage2_mask = batch["stage2_mask"][:, 1:]
            stage3_mask = batch["stage3_mask"][:, 1:]
            answer_mask = batch["answer_mask"][:, 1:]
            assert isinstance(input_ids, torch.Tensor)
            assert isinstance(attention_mask, torch.Tensor)
            assert isinstance(labels, torch.Tensor)
            assert isinstance(stage2_mask, torch.Tensor)
            assert isinstance(stage3_mask, torch.Tensor)
            assert isinstance(answer_mask, torch.Tensor)

            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[:, :-1, :]
            pred = logits.argmax(dim=-1)

            losses.append(float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)).item()))
            tokens_seen += int(labels.ne(-100).sum().item())

            if int(stage2_mask.sum().item()) > 0:
                s2_total += int(stage2_mask.sum().item())
                s2_correct += int((pred.eq(labels) & stage2_mask).sum().item())
            if int(stage3_mask.sum().item()) > 0:
                s3_total += int(stage3_mask.sum().item())
                s3_correct += int((pred.eq(labels) & stage3_mask).sum().item())

            target_texts = batch.get("answer_text", [])
            target_texts_normalized = batch.get("answer_text_normalized", [])
            for i in range(labels.shape[0]):
                sample_mask = stage3_mask[i]
                if int(sample_mask.sum().item()) == 0:
                    skipped_no_stage3 += 1
                    continue

                sample_answer_mask = answer_mask[i]
                if int(sample_answer_mask.sum().item()) == 0:
                    skipped_no_answer_span += 1
                    continue

                pred_answer = _decode_answer_tokens(pred[i][sample_answer_mask], tokenizer)
                pred_norm = normalize_answer_text(pred_answer)
                answer_len = len(pred_answer.strip())
                if answer_len <= 4:
                    answer_length_bins["0-4"] += 1
                elif answer_len <= 16:
                    answer_length_bins["5-16"] += 1
                elif answer_len <= 64:
                    answer_length_bins["17-64"] += 1
                else:
                    answer_length_bins["65+"] += 1

                gold_raw = str(target_texts[i]).strip() if i < len(target_texts) else ""
                gold_norm = str(target_texts_normalized[i]).strip() if i < len(target_texts_normalized) else ""
                if gold_norm:
                    normalized_total += 1
                    if pred_norm == gold_norm:
                        normalized_correct += 1
                else:
                    skipped_missing_answer_text += 1

                if gold_raw:
                    exact_total += 1
                    if pred_answer.strip() == gold_raw:
                        exact_correct += 1
                    normalized_match_total += 1
                    gold_norm_eval = normalize_answer_text(gold_raw)
                    if pred_norm == gold_norm_eval:
                        normalized_match_correct += 1
                    if pred_norm == gold_norm_eval and pred_answer.strip() != gold_raw:
                        normalized_only_count += 1

                if not gold_raw:
                    skipped_missing_numeric_target += 1
                    continue

                numeric_total += 1
                num_result = numeric_match(pred_answer, gold_raw)
                if num_result.skipped:
                    skipped_missing_numeric_target += 1
                    skipped_ambiguous_numeric += 1
                    continue
                numeric_pred_value_count += num_result.predicted_count
                numeric_target_value_count += num_result.target_count
                numeric_value_match_count += num_result.match_count
                if num_result.is_multi_value_target:
                    multi_value_target_count += 1
                if num_result.is_match:
                    numeric_correct += 1
                    numeric_match_count += 1
                if pred_norm == normalize_answer_text(gold_raw) and not num_result.is_match:
                    string_match_numeric_miss_count += 1

    model.train()
    loss = float(sum(losses) / len(losses)) if losses else float("nan")
    stage2_acc = float(s2_correct / s2_total) if s2_total > 0 else None
    stage3_acc = float(s3_correct / s3_total) if s3_total > 0 else None
    final_acc = float(normalized_correct / normalized_total) if normalized_total > 0 else None
    exact = float(exact_correct / exact_total) if exact_total > 0 else None
    normalized_match = float(normalized_match_correct / normalized_match_total) if normalized_match_total > 0 else None
    numeric_acc = float(numeric_correct / numeric_total) if numeric_total > 0 else None
    return EvalResult(
        loss=loss,
        wall_time_seconds=float(perf_counter() - start),
        tokens_seen=tokens_seen,
        stage_2_token_accuracy=stage2_acc,
        stage_3_token_accuracy=stage3_acc,
        final_answer_accuracy=final_acc,
        final_answer_exact_match=exact,
        final_answer_normalized_match=normalized_match,
        normalized_numeric_answer_accuracy=numeric_acc,
        answer_eval_string_count=normalized_total,
        answer_eval_numeric_count=numeric_total,
        answer_eval_skipped_no_stage3=skipped_no_stage3,
        answer_eval_skipped_no_answer_span=skipped_no_answer_span,
        answer_eval_skipped_missing_answer_text=skipped_missing_answer_text,
        answer_eval_skipped_missing_numeric_target=skipped_missing_numeric_target,
        answer_eval_normalized_match_count=normalized_match_correct,
        answer_eval_exact_match_count=exact_correct,
        answer_eval_numeric_match_count=numeric_match_count,
        answer_eval_multi_value_target_count=multi_value_target_count,
        answer_eval_numeric_pred_value_count=numeric_pred_value_count,
        answer_eval_numeric_target_value_count=numeric_target_value_count,
        answer_eval_numeric_value_match_count=numeric_value_match_count,
        answer_eval_string_match_numeric_miss_count=string_match_numeric_miss_count,
        answer_eval_normalized_only_count=normalized_only_count,
        answer_eval_skipped_ambiguous_numeric=skipped_ambiguous_numeric,
        answer_eval_length_histogram=dict(answer_length_bins),
    )


def run_training(
    *,
    model: StagedLatentAdaptationModel,
    train_loader: DataLoader[dict[str, torch.Tensor | list[str]]],
    eval_loader: DataLoader[dict[str, torch.Tensor | list[str]]],
    optimizer: torch.optim.Optimizer | None,
    num_epochs: int,
    max_steps: int,
    eval_interval_steps: int,
    eval_enabled: bool,
    tokenizer: Any | None = None,
) -> dict[str, float | int]:
    run_start = perf_counter()
    global_steps = 0
    epochs_completed = 0
    train_loss = float("nan")
    tokens_train = 0
    wall_train = 0.0
    eval_results: list[EvalResult] = []

    for _ in range(num_epochs):
        train_loss, done, wall, tokens, interval_evals = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            max_steps=max_steps,
            global_step_start=global_steps,
            eval_enabled=eval_enabled,
            eval_interval_steps=eval_interval_steps,
            eval_loader=eval_loader,
            tokenizer=tokenizer,
        )
        global_steps += done
        tokens_train += tokens
        wall_train += wall
        epochs_completed += 1
        eval_results.extend(interval_evals)
        if global_steps >= max_steps:
            break

    if eval_enabled:
        needs_final_eval = not eval_results or (eval_interval_steps <= 0 or global_steps % eval_interval_steps != 0)
        if needs_final_eval:
            eval_results.append(evaluate(model=model, dataloader=eval_loader, tokenizer=tokenizer))

    last_eval = eval_results[-1]
    best_eval_loss = min(x.loss for x in eval_results)
    wall_eval = sum(x.wall_time_seconds for x in eval_results)
    tokens_eval = sum(x.tokens_seen for x in eval_results)
    wall_total = perf_counter() - run_start

    return {
        "global_steps": global_steps,
        "epochs_completed": epochs_completed,
        "train_loss": train_loss,
        "eval_loss": last_eval.loss,
        "best_eval_loss": float(best_eval_loss),
        "tokens_seen_train": int(tokens_train),
        "tokens_seen_eval": int(tokens_eval),
        "wall_time_seconds_total": float(wall_total),
        "wall_time_seconds_train": float(wall_train),
        "wall_time_seconds_eval": float(wall_eval),
        "tokens_per_second_train": float(tokens_train / wall_train) if wall_train > 0 else 0.0,
        "tokens_per_second_eval": float(tokens_eval / wall_eval) if wall_eval > 0 else 0.0,
        "seconds_per_step": float(wall_train / global_steps) if global_steps > 0 else 0.0,
        "steps_per_second": float(global_steps / wall_train) if wall_train > 0 else 0.0,
        "eval_perplexity": _safe_perplexity(last_eval.loss),
        "train_perplexity": _safe_perplexity(train_loss),
        "stage_2_token_accuracy": last_eval.stage_2_token_accuracy,
        "stage_3_token_accuracy": last_eval.stage_3_token_accuracy,
        "final_answer_accuracy": last_eval.final_answer_accuracy,
        "final_answer_exact_match": last_eval.final_answer_exact_match,
        "final_answer_normalized_match": last_eval.final_answer_normalized_match,
        "normalized_numeric_answer_accuracy": last_eval.normalized_numeric_answer_accuracy,
        "answer_eval_string_count": int(last_eval.answer_eval_string_count),
        "answer_eval_numeric_count": int(last_eval.answer_eval_numeric_count),
        "answer_eval_skipped_no_stage3": int(last_eval.answer_eval_skipped_no_stage3),
        "answer_eval_skipped_no_answer_span": int(last_eval.answer_eval_skipped_no_answer_span),
        "answer_eval_skipped_missing_answer_text": int(last_eval.answer_eval_skipped_missing_answer_text),
        "answer_eval_skipped_missing_numeric_target": int(last_eval.answer_eval_skipped_missing_numeric_target),
        "answer_eval_normalized_match_count": int(last_eval.answer_eval_normalized_match_count),
        "answer_eval_exact_match_count": int(last_eval.answer_eval_exact_match_count),
        "answer_eval_numeric_match_count": int(last_eval.answer_eval_numeric_match_count),
        "answer_eval_multi_value_target_count": int(last_eval.answer_eval_multi_value_target_count),
        "answer_eval_numeric_pred_value_count": int(last_eval.answer_eval_numeric_pred_value_count),
        "answer_eval_numeric_target_value_count": int(last_eval.answer_eval_numeric_target_value_count),
        "answer_eval_numeric_value_match_count": int(last_eval.answer_eval_numeric_value_match_count),
        "answer_eval_string_match_numeric_miss_count": int(last_eval.answer_eval_string_match_numeric_miss_count),
        "answer_eval_normalized_only_count": int(last_eval.answer_eval_normalized_only_count),
        "answer_eval_skipped_ambiguous_numeric": int(last_eval.answer_eval_skipped_ambiguous_numeric),
        "answer_eval_numeric_abs_tolerance": float(NUMERIC_ABS_TOL),
        "answer_eval_numeric_multi_value_rule": NUMERIC_MULTI_VALUE_RULE,
        "answer_eval_answer_length_histogram": dict(last_eval.answer_eval_length_histogram),
    }
