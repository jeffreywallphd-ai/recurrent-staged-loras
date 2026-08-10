"""Core optimization/evaluation loop primitives used by `training.engine`.

Responsibilities:
- compute stage-aware training loss,
- run bounded training epochs with optional compute ceilings,
- compute answer-span-aware metrics for reportable outputs.

Invariant: answer-level metrics are computed on `answer_mask` spans (not full
stage-3 text), while token-level stage accuracy keeps stage-mask semantics.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
from dataclasses import dataclass
from math import isfinite
from time import perf_counter
from typing import Any
#from xml.parsers.expat import model

#from accelerate import optimizer
#from accelerate import optimizer
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.staged_model import StagedLatentAdaptationModel
from training.answer_eval import (
    NUMERIC_ABS_TOL,
    NUMERIC_MULTI_VALUE_RULE,
    normalize_answer_text,
    numeric_match,
    symbolic_equivalence_match,
)


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
    symbolic_answer_accuracy: float | None
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
    answer_eval_multi_value_exact_set_match_count: int
    answer_eval_multi_value_partial_match_count: int
    answer_eval_multi_value_unmatched_count: int
    answer_eval_string_match_numeric_miss_count: int
    answer_eval_normalized_only_count: int
    answer_eval_skipped_ambiguous_numeric: int
    answer_eval_symbolic_attempt_count: int
    answer_eval_symbolic_success_count: int
    answer_eval_symbolic_failure_count: int
    answer_eval_symbolic_match_count: int
    answer_eval_length_histogram: dict[str, int]
    answer_eval_failures: list[dict[str, Any]]


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


def _sample_key(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
    """Create the same deterministic sample key used by the teacher cache."""
    input_bytes = (
        input_ids.detach()
        .to(dtype=torch.int32, device="cpu")
        .contiguous()
        .numpy()
        .tobytes()
    )
    mask_bytes = (
        attention_mask.detach()
        .to(dtype=torch.int8, device="cpu")
        .contiguous()
        .numpy()
        .tobytes()
    )
    digest = sha256()
    digest.update(input_bytes)
    digest.update(mask_bytes)
    return digest.hexdigest()


def loss_for_batch(
    model: StagedLatentAdaptationModel,
    batch: dict[str, torch.Tensor | list[str]],
    *,
    distillation_targets: dict[str, dict[str, Any]] | None = None,
    distillation_weight: float = 0.0,
    distillation_loss_type: str = "mse",
    distillation_stats: dict[str, float | int] | None = None,
) -> torch.Tensor:
    """Compute one batch loss with optional offline hidden-state distillation.

    For recurrent models, per-step logits are aligned to stage masks (stage1-3).
    When distillation is enabled, the final recurrent hidden state is matched to
    cached Standard-LoRA teacher states at final-answer token positions.
    """
    device = next(model.parameters()).device

    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)

    out = model(input_ids=input_ids, attention_mask=attention_mask)

    labels = batch["labels"][:, 1:]
    assert isinstance(labels, torch.Tensor)

    task_loss: torch.Tensor
    step_hidden_states: list[torch.Tensor] = []

    if model.config.refiner.enabled and out.extras["per_step"]:
        step_hidden_states = out.extras["per_step"]
        logits_steps = [
            model.base_model.forward_lm_head(hidden)[:, :-1, :]
            for hidden in step_hidden_states
        ]

        stage1 = batch["stage1_mask"][:, 1:]
        stage2 = batch["stage2_mask"][:, 1:]
        stage3 = batch["stage3_mask"][:, 1:]
        assert isinstance(stage1, torch.Tensor)
        assert isinstance(stage2, torch.Tensor)
        assert isinstance(stage3, torch.Tensor)

        stage_masks = [stage1, stage2, stage3]
        stage_losses: list[torch.Tensor] = []
        for idx, step_logits in enumerate(logits_steps):
            target_mask = stage_masks[min(idx, 2)]
            stage_losses.append(_masked_ce(step_logits, labels, target_mask))
        task_loss = torch.stack(stage_losses).mean()
    else:
        logits = out.logits[:, :-1, :]
        task_loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        )

    if (
        not distillation_targets
        or distillation_weight <= 0.0
        or not step_hidden_states
    ):
        return task_loss

    answer_mask = batch["answer_mask"][:, 1:].bool()
    assert isinstance(answer_mask, torch.Tensor)

    final_student_hidden = step_hidden_states[-1][:, :-1, :]
    distill_losses: list[torch.Tensor] = []

    for index in range(input_ids.shape[0]):
        if distillation_stats is not None:
            distillation_stats["lookups"] = int(distillation_stats.get("lookups", 0)) + 1

        key = _sample_key(input_ids[index], attention_mask[index])
        cached = distillation_targets.get(key)
        if cached is None:
            if distillation_stats is not None:
                distillation_stats["misses"] = int(distillation_stats.get("misses", 0)) + 1
            continue

        valid_positions = answer_mask[index]
        student_answer_hidden = final_student_hidden[index][valid_positions]
        teacher_answer_hidden = cached.get("teacher_answer_hidden")

        if not isinstance(teacher_answer_hidden, torch.Tensor):
            if distillation_stats is not None:
                distillation_stats["invalid_targets"] = int(
                    distillation_stats.get("invalid_targets", 0)
                ) + 1
            continue
        if student_answer_hidden.shape[0] != teacher_answer_hidden.shape[0]:
            raise ValueError(
                "Teacher/student answer-token count mismatch for cached sample: "
                f"student={student_answer_hidden.shape[0]}, "
                f"teacher={teacher_answer_hidden.shape[0]}"
            )

        teacher_answer_hidden = teacher_answer_hidden.to(
            device=student_answer_hidden.device,
            dtype=student_answer_hidden.dtype,
        )
        if distillation_loss_type == "mse":
            sample_distillation_loss = F.mse_loss(
                student_answer_hidden,
                teacher_answer_hidden,
            )
        elif distillation_loss_type == "cosine":
            # Compare representation direction rather than exact coordinates.
            # Compute in float32 for numerical stability when the model uses bf16/fp16.
            student_float = student_answer_hidden.float()
            teacher_float = teacher_answer_hidden.float()

            token_cosine_similarity = F.cosine_similarity(
                student_float,
                teacher_float,
                dim=-1,
                eps=1e-8,
            )
            sample_distillation_loss = (1.0 - token_cosine_similarity).mean()
        else:
            raise ValueError(
                "Unsupported distillation loss type "
                f"'{distillation_loss_type}'. Expected 'mse' or 'cosine'."
            )
        distill_losses.append(sample_distillation_loss)

        if distillation_stats is not None:
            distillation_stats["matches"] = int(
                distillation_stats.get("matches", 0)
            ) + 1
            distillation_stats["matched_answer_tokens"] = int(
                distillation_stats.get("matched_answer_tokens", 0)
            ) + int(student_answer_hidden.shape[0])
            distillation_stats["distillation_loss_sum"] = float(
                distillation_stats.get("distillation_loss_sum", 0.0)
            ) + float(sample_distillation_loss.detach().item())
            distillation_stats["distillation_loss_count"] = int(
                distillation_stats.get("distillation_loss_count", 0)
            ) + 1

    if not distill_losses:
        return task_loss

    distillation_loss = torch.stack(distill_losses).mean()

    if distillation_stats is not None:
        debug_count = int(distillation_stats.get("debug_print_count", 0))

        if debug_count < 10:
            weighted_distillation = float(distillation_weight) * distillation_loss
            total_loss = task_loss + weighted_distillation

            print(
                "[distill-debug] "
                f"type={distillation_loss_type} "
                f"task={task_loss.detach().item():.6f} "
                f"raw_distill={distillation_loss.detach().item():.6f} "
                f"weighted_distill={weighted_distillation.detach().item():.6f} "
                f"total={total_loss.detach().item():.6f}"
            )

            distillation_stats["debug_print_count"] = debug_count + 1

    return task_loss + float(distillation_weight) * distillation_loss


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
    max_train_tokens: int | None = None,
    max_wall_time_seconds: float | None = None,
    lr_scheduler: Any | None = None,
    gradient_accumulation_steps: int = 1,
    distillation_targets: dict[str, dict[str, Any]] | None = None,
    distillation_weight: float = 0.0,
    distillation_loss_type: str = "mse",
    distillation_stats: dict[str, float | int] | None = None,
) -> tuple[float, int, float, int, list[EvalResult]]:
    """Run one epoch (or partial epoch) with optional eval intervals.

    Returns:
        (avg_train_loss, steps_done, wall_time_seconds, train_tokens_seen, interval_eval_results)
    """
    model.train()
    step = global_step_start
    losses: list[float] = []
    tokens_seen = 0
    wall = 0.0
    interval_results: list[EvalResult] = []
    accumulation_steps = max(1, int(gradient_accumulation_steps))
    micro_step = 0
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    for batch in dataloader:
        if step >= max_steps:
            break
        if max_train_tokens is not None and tokens_seen >= max_train_tokens:
            break
        if max_wall_time_seconds is not None and wall >= max_wall_time_seconds:
            break
        start = perf_counter()
        labels = batch["labels"][:, 1:]
        assert isinstance(labels, torch.Tensor)
        tokens_seen += int(labels.ne(-100).sum().item())
        loss = loss_for_batch(
            model,
            batch,
            distillation_targets=distillation_targets,
            distillation_weight=distillation_weight,
            distillation_loss_type=distillation_loss_type,
            distillation_stats=distillation_stats,
        )
        losses.append(float(loss.item()))
        if optimizer is not None:
            (loss / accumulation_steps).backward()
            micro_step += 1

            should_step = micro_step % accumulation_steps == 0 or (step + 1) >= max_steps
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if lr_scheduler is not None:
                    lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        step += 1
        wall += perf_counter() - start

        if eval_enabled and eval_interval_steps > 0 and (step % eval_interval_steps == 0):
            interval_results.append(evaluate(model=model, dataloader=eval_loader, tokenizer=tokenizer))

    avg = float(sum(losses) / len(losses)) if losses else float("nan")
    return avg, step - global_step_start, wall, tokens_seen, interval_results


def evaluate(*, model: StagedLatentAdaptationModel, dataloader: DataLoader[dict[str, torch.Tensor | list[str]]], tokenizer: Any | None = None) -> EvalResult:
    """Evaluate model on one dataloader and compute study metrics.

    Failure modes:
        Assumes masks and labels are present with expected keys/shapes.
    """
    was_training = model.training
    if was_training:
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
    multi_value_exact_set_match_count = 0
    multi_value_partial_match_count = 0
    multi_value_unmatched_count = 0
    numeric_pred_value_count = 0
    numeric_target_value_count = 0
    numeric_value_match_count = 0
    string_match_numeric_miss_count = 0
    normalized_only_count = 0
    skipped_ambiguous_numeric = 0
    symbolic_attempt_count = 0
    symbolic_success_count = 0
    symbolic_failure_count = 0
    symbolic_match_count = 0
    symbolic_total = 0
    answer_length_bins: Counter[str] = Counter()

    answer_eval_failures: list[dict[str, Any]] = []
    max_answer_eval_failures = 50

    skipped_no_stage3 = 0
    skipped_no_answer_span = 0
    skipped_missing_answer_text = 0
    skipped_missing_numeric_target = 0

    with torch.no_grad():
        for batch in dataloader:
            device = next(model.parameters()).device

            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

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

            eval_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
            )

            if torch.isnan(eval_loss) or torch.isinf(eval_loss):
                print("[nan-debug] eval loss invalid")
                print("[nan-debug] logits has nan:", torch.isnan(logits).any().item())
                print("[nan-debug] logits has inf:", torch.isinf(logits).any().item())
                print("[nan-debug] logits min:", logits.nan_to_num().min().item())
                print("[nan-debug] logits max:", logits.nan_to_num().max().item())
                print("[nan-debug] labels min:", labels.min().item())
                print("[nan-debug] labels max:", labels.max().item())
                print("[nan-debug] input_ids min:", input_ids.min().item())
                print("[nan-debug] input_ids max:", input_ids.max().item())

            losses.append(float(eval_loss.item()))
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

                # Final-answer metrics intentionally score only the answer span,
                # excluding literal headers like "Final Answer:".
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
                gold_norm_eval = normalize_answer_text(gold_raw) if gold_raw else gold_norm

                if gold_norm:
                    normalized_total += 1
                    if pred_norm == gold_norm:
                        normalized_correct += 1
                    elif len(answer_eval_failures) < max_answer_eval_failures:
                        answer_eval_failures.append(
                            {
                                "predicted_answer": pred_answer,
                                "target_answer": gold_raw,
                                "predicted_normalized": pred_norm,
                                "target_normalized": gold_norm,
                                "failure_type": "normalized_mismatch",
                            }
                        )
                else:
                    skipped_missing_answer_text += 1

                if gold_raw:
                    exact_total += 1
                    if pred_answer.strip() == gold_raw:
                        exact_correct += 1

                    normalized_match_total += 1
                    if pred_norm == gold_norm_eval:
                        normalized_match_correct += 1

                    if pred_norm == gold_norm_eval and pred_answer.strip() != gold_raw:
                        normalized_only_count += 1

                    symbolic_result = symbolic_equivalence_match(pred_answer, gold_raw)
                    if symbolic_result.attempted:
                        symbolic_attempt_count += 1
                        if symbolic_result.parse_success:
                            symbolic_success_count += 1
                            symbolic_total += 1
                            if symbolic_result.is_match:
                                symbolic_match_count += 1
                        else:
                            symbolic_failure_count += 1

                # Numeric scoring is only valid when a numeric target is present;
                # missing/ambiguous targets are tracked as explicit skips.
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
                    if num_result.multi_value_status == "exact_set_match":
                        multi_value_exact_set_match_count += 1
                    elif num_result.multi_value_status == "partial_overlap":
                        multi_value_partial_match_count += 1
                    elif num_result.multi_value_status == "unmatched":
                        multi_value_unmatched_count += 1

                if num_result.is_match:
                    numeric_correct += 1
                    numeric_match_count += 1
                elif len(answer_eval_failures) < max_answer_eval_failures and pred_norm == gold_norm_eval:
                    # This catches cases that normalize as strings but fail numeric scoring.
                    answer_eval_failures.append(
                        {
                            "predicted_answer": pred_answer,
                            "target_answer": gold_raw,
                            "predicted_normalized": pred_norm,
                            "target_normalized": gold_norm_eval,
                            "failure_type": "numeric_miss_but_string_match",
                            "numeric_predicted_count": num_result.predicted_count,
                            "numeric_target_count": num_result.target_count,
                            "numeric_value_match_count": num_result.match_count,
                            "multi_value_status": num_result.multi_value_status,
                        }
                    )

                if pred_norm == gold_norm_eval and not num_result.is_match:
                    string_match_numeric_miss_count += 1

    if was_training:
        model.train()

    loss = float(sum(losses) / len(losses)) if losses else float("nan")
    stage2_acc = float(s2_correct / s2_total) if s2_total > 0 else None
    stage3_acc = float(s3_correct / s3_total) if s3_total > 0 else None
    final_acc = float(normalized_correct / normalized_total) if normalized_total > 0 else None
    exact = float(exact_correct / exact_total) if exact_total > 0 else None
    normalized_match = float(normalized_match_correct / normalized_match_total) if normalized_match_total > 0 else None
    numeric_acc = float(numeric_correct / numeric_total) if numeric_total > 0 else None
    symbolic_acc = float(symbolic_match_count / symbolic_total) if symbolic_total > 0 else None

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
        symbolic_answer_accuracy=symbolic_acc,
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
        answer_eval_multi_value_exact_set_match_count=multi_value_exact_set_match_count,
        answer_eval_multi_value_partial_match_count=multi_value_partial_match_count,
        answer_eval_multi_value_unmatched_count=multi_value_unmatched_count,
        answer_eval_string_match_numeric_miss_count=string_match_numeric_miss_count,
        answer_eval_normalized_only_count=normalized_only_count,
        answer_eval_skipped_ambiguous_numeric=skipped_ambiguous_numeric,
        answer_eval_symbolic_attempt_count=symbolic_attempt_count,
        answer_eval_symbolic_success_count=symbolic_success_count,
        answer_eval_symbolic_failure_count=symbolic_failure_count,
        answer_eval_symbolic_match_count=symbolic_match_count,
        answer_eval_length_histogram=dict(answer_length_bins),
        answer_eval_failures=answer_eval_failures,
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
    max_train_tokens: int | None = None,
    max_wall_time_seconds: float | None = None,
    lr_scheduler: Any | None = None,
    gradient_accumulation_steps: int = 1,
    distillation_targets: dict[str, dict[str, Any]] | None = None,
    distillation_weight: float = 0.0,
    distillation_loss_type: str = "mse",
) -> dict[str, Any]:
    """Run multi-epoch training with step/token/time stopping criteria.

    Side effects:
        Mutates model parameters when optimizer is provided.
    """
    run_start = perf_counter()
    global_steps = 0
    epochs_completed = 0
    train_loss = float("nan")
    tokens_train = 0
    wall_train = 0.0
    eval_results: list[EvalResult] = []

    train_loss_history: list[float] = []
    eval_loss_history: list[float] = []

    distillation_stats: dict[str, float | int] = {
        "lookups": 0,
        "matches": 0,
        "misses": 0,
        "invalid_targets": 0,
        "matched_answer_tokens": 0,
        "distillation_loss_sum": 0.0,
        "distillation_loss_count": 0,
        "debug_print_count": 0,
    }

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
            max_train_tokens=max_train_tokens,
            max_wall_time_seconds=max_wall_time_seconds,
            lr_scheduler=lr_scheduler,
            gradient_accumulation_steps=gradient_accumulation_steps,
            distillation_targets=distillation_targets,
            distillation_weight=distillation_weight,
            distillation_loss_type=distillation_loss_type,
            distillation_stats=distillation_stats,
        )
        global_steps += done
        tokens_train += tokens
        wall_train += wall
        epochs_completed += 1
        eval_results.extend(interval_evals)

        train_loss_history.append(float(train_loss))

        for ev in interval_evals:
             eval_loss_history.append(float(ev.loss))

        if global_steps >= max_steps:
            break
        if max_train_tokens is not None and tokens_train >= max_train_tokens:
            break
        if max_wall_time_seconds is not None and wall_train >= max_wall_time_seconds:
            break

    if eval_enabled:
        # Guarantee a terminal eval snapshot unless the final optimizer step
        # already produced one at an exact interval boundary.
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
        "train_loss_history": train_loss_history,
        "eval_loss_history": eval_loss_history,
        "stage_2_token_accuracy": last_eval.stage_2_token_accuracy,
        "stage_3_token_accuracy": last_eval.stage_3_token_accuracy,
        "final_answer_accuracy": last_eval.final_answer_accuracy,
        "final_answer_exact_match": last_eval.final_answer_exact_match,
        "final_answer_normalized_match": last_eval.final_answer_normalized_match,
        "normalized_numeric_answer_accuracy": last_eval.normalized_numeric_answer_accuracy,
        "symbolic_answer_accuracy": last_eval.symbolic_answer_accuracy,
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
        "answer_eval_multi_value_exact_set_match_count": int(last_eval.answer_eval_multi_value_exact_set_match_count),
        "answer_eval_multi_value_partial_match_count": int(last_eval.answer_eval_multi_value_partial_match_count),
        "answer_eval_multi_value_unmatched_count": int(last_eval.answer_eval_multi_value_unmatched_count),
        "answer_eval_string_match_numeric_miss_count": int(last_eval.answer_eval_string_match_numeric_miss_count),
        "answer_eval_normalized_only_count": int(last_eval.answer_eval_normalized_only_count),
        "answer_eval_skipped_ambiguous_numeric": int(last_eval.answer_eval_skipped_ambiguous_numeric),
        "symbolic_eval_attempt_count": int(last_eval.answer_eval_symbolic_attempt_count),
        "symbolic_eval_success_count": int(last_eval.answer_eval_symbolic_success_count),
        "symbolic_eval_failure_count": int(last_eval.answer_eval_symbolic_failure_count),
        "answer_eval_symbolic_match_count": int(last_eval.answer_eval_symbolic_match_count),
        "answer_eval_numeric_abs_tolerance": float(NUMERIC_ABS_TOL),
        "answer_eval_numeric_multi_value_rule": NUMERIC_MULTI_VALUE_RULE,
        "answer_eval_answer_length_histogram": dict(last_eval.answer_eval_length_histogram),
        "answer_eval_failures": list(last_eval.answer_eval_failures),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "distillation_enabled": bool(distillation_targets and distillation_weight > 0.0),
        "distillation_weight": float(distillation_weight),
        "distillation_loss_type": str(distillation_loss_type),
        "distillation_cached_examples": int(len(distillation_targets or {})),
        "distillation_lookups": int(distillation_stats["lookups"]),
        "distillation_matches": int(distillation_stats["matches"]),
        "distillation_misses": int(distillation_stats["misses"]),
        "distillation_invalid_targets": int(distillation_stats["invalid_targets"]),
        "distillation_matched_answer_tokens": int(
            distillation_stats["matched_answer_tokens"]
        ),
        "distillation_match_rate": (
            float(distillation_stats["matches"])
            / float(distillation_stats["lookups"])
            if int(distillation_stats["lookups"]) > 0
            else 0.0
        ),
        "distillation_mean_hidden_loss": (
            float(distillation_stats["distillation_loss_sum"])
            / float(distillation_stats["distillation_loss_count"])
            if int(distillation_stats["distillation_loss_count"]) > 0
            else None
        ),
        # Backward-compatible metric name. It is meaningful only when
        # distillation_loss_type == "mse".
        "distillation_mean_hidden_mse": (
            float(distillation_stats["distillation_loss_sum"])
            / float(distillation_stats["distillation_loss_count"])
            if (
                distillation_loss_type == "mse"
                and int(distillation_stats["distillation_loss_count"]) > 0
            )
            else None
        ),
    }