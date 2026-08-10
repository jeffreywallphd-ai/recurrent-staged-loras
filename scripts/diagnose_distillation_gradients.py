"""Diagnose gradient agreement between task loss and hidden-state distillation.

Measures:
- task loss
- raw hidden-state distillation loss
- task gradient norm
- distillation gradient norm
- weighted distillation gradient norm
- gradient cosine similarity

Interpretation:
    cosine near +1  -> gradients strongly agree
    cosine near  0  -> gradients are mostly unrelated
    cosine below  0 -> gradients conflict
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

import torch
import torch.nn.functional as F

from training.config_loader import load_runtime_config
from training.engine import build_training_components


def sample_key(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> str:
    """Create the same deterministic key used by the distillation cache."""

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


def masked_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Same masked cross-entropy used by training.loop."""

    vocab = logits.shape[-1]

    per_token_loss = F.cross_entropy(
        logits.reshape(-1, vocab),
        labels.reshape(-1),
        reduction="none",
    ).reshape_as(labels)

    denominator = mask.sum().clamp(min=1)

    return (
        per_token_loss * mask.float()
    ).sum() / denominator


def gradient_statistics(
    task_grads: tuple[torch.Tensor | None, ...],
    distill_grads: tuple[torch.Tensor | None, ...],
    distillation_weight: float,
) -> dict[str, float]:
    """Calculate gradient norms and cosine similarity without flattening."""

    dot_product = torch.tensor(0.0, dtype=torch.float64)
    task_norm_sq = torch.tensor(0.0, dtype=torch.float64)
    distill_norm_sq = torch.tensor(0.0, dtype=torch.float64)

    used_tensors = 0

    for task_grad, distill_grad in zip(
        task_grads,
        distill_grads,
        strict=True,
    ):
        if task_grad is None or distill_grad is None:
            continue

        task_float = task_grad.detach().float()
        distill_float = distill_grad.detach().float()

        dot_product += (
            task_float * distill_float
        ).sum().double().cpu()

        task_norm_sq += (
            task_float.square()
        ).sum().double().cpu()

        distill_norm_sq += (
            distill_float.square()
        ).sum().double().cpu()

        used_tensors += 1

    task_norm = float(
        torch.sqrt(task_norm_sq).item()
    )

    raw_distill_norm = float(
        torch.sqrt(distill_norm_sq).item()
    )

    weighted_distill_norm = (
        abs(float(distillation_weight))
        * raw_distill_norm
    )

    denominator = task_norm * raw_distill_norm

    if denominator > 0:
        cosine = float(
            dot_product.item() / denominator
        )
    else:
        cosine = float("nan")

    if task_norm > 0:
        weighted_to_task_ratio = (
            weighted_distill_norm / task_norm
        )
    else:
        weighted_to_task_ratio = float("nan")

    return {
        "task_grad_norm": task_norm,
        "raw_distill_grad_norm": raw_distill_norm,
        "weighted_distill_grad_norm": weighted_distill_norm,
        "weighted_distill_to_task_grad_ratio": weighted_to_task_ratio,
        "cosine_similarity": cosine,
        "used_parameter_tensors": used_tensors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure gradient agreement between task loss "
            "and hidden-state distillation loss."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Distillation experiment config.",
    )

    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of training batches to inspect.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_batches < 1:
        raise ValueError(
            "--num_batches must be at least 1."
        )

    print(f"Loading config: {args.config}")

    runtime = load_runtime_config(
        args.config
    )

    training_raw = runtime.raw.get(
        "training",
        {},
    )

    distillation_config = training_raw.get(
        "distillation",
        {},
    )

    if not isinstance(
        distillation_config,
        dict,
    ):
        raise ValueError(
            "training.distillation must be a dictionary."
        )

    if not distillation_config.get(
        "enabled",
        False,
    ):
        raise ValueError(
            "Distillation must be enabled in the config."
        )

    distillation_weight = float(
        distillation_config.get(
            "weight",
            0.0,
        )
    )

    cache_value = distillation_config.get(
        "cache_path"
    )

    if not cache_value:
        raise ValueError(
            "No distillation cache_path found."
        )

    cache_path = Path(
        str(cache_value)
    )

    if not cache_path.exists():
        raise FileNotFoundError(
            f"Cache does not exist: {cache_path}"
        )

    print(
        f"Distillation weight: "
        f"{distillation_weight}"
    )

    print(
        f"Loading cache: "
        f"{cache_path}"
    )

    payload = torch.load(
        cache_path,
        map_location="cpu",
        weights_only=False,
    )

    targets = payload.get(
        "targets"
    )

    if not isinstance(
        targets,
        dict,
    ):
        raise ValueError(
            "Cache does not contain a targets dictionary."
        )

    print(
        f"Cached targets: "
        f"{len(targets)}"
    )

    print(
        "Building model and dataset..."
    )

    components = build_training_components(
        runtime
    )

    model = components.model
    model.train()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    print(
        f"Trainable tensors: "
        f"{len(trainable_parameters)}"
    )

    cosine_values: list[float] = []
    gradient_ratios: list[float] = []

    matched_batches = 0
    attempted_batches = 0

    print()
    print("=" * 78)
    print("DISTILLATION GRADIENT DIAGNOSTIC")
    print("=" * 78)

    for batch_number, batch in enumerate(
        components.train_loader,
        start=1,
    ):
        if matched_batches >= args.num_batches:
            break

        attempted_batches += 1

        device = next(
            model.parameters()
        ).device

        batch = {
            key: (
                value.to(device)
                if isinstance(
                    value,
                    torch.Tensor,
                )
                else value
            )
            for key, value in batch.items()
        }

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        labels = batch["labels"][:, 1:]

        stage1_mask = (
            batch["stage1_mask"][:, 1:]
        )

        stage2_mask = (
            batch["stage2_mask"][:, 1:]
        )

        stage3_mask = (
            batch["stage3_mask"][:, 1:]
        )

        answer_mask = (
            batch["answer_mask"][:, 1:]
            .bool()
        )

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        step_hidden_states = (
            output.extras["per_step"]
        )

        if not step_hidden_states:
            raise RuntimeError(
                "The configured model did not "
                "produce recurrent step hidden states."
            )

        # -----------------------------------------------------
        # Normal recurrent task loss
        # -----------------------------------------------------

        logits_steps = [
            model.base_model.forward_lm_head(
                hidden
            )[:, :-1, :]
            for hidden in step_hidden_states
        ]

        stage_masks = [
            stage1_mask,
            stage2_mask,
            stage3_mask,
        ]

        stage_losses = []

        for index, step_logits in enumerate(
            logits_steps
        ):
            target_mask = stage_masks[
                min(index, 2)
            ]

            stage_losses.append(
                masked_ce(
                    step_logits,
                    labels,
                    target_mask,
                )
            )

        task_loss = torch.stack(
            stage_losses
        ).mean()

        # -----------------------------------------------------
        # Hidden-state distillation loss
        # -----------------------------------------------------

        final_student_hidden = (
            step_hidden_states[-1][
                :,
                :-1,
                :
            ]
        )

        distillation_losses = []

        for sample_index in range(
            input_ids.shape[0]
        ):
            key = sample_key(
                input_ids[sample_index],
                attention_mask[sample_index],
            )

            cached = targets.get(
                key
            )

            if cached is None:
                continue

            valid_positions = (
                answer_mask[sample_index]
            )

            student_answer_hidden = (
                final_student_hidden[
                    sample_index
                ][valid_positions]
            )

            teacher_answer_hidden = (
                cached.get(
                    "teacher_answer_hidden"
                )
            )

            if not isinstance(
                teacher_answer_hidden,
                torch.Tensor,
            ):
                continue

            if (
                student_answer_hidden.shape[0]
                != teacher_answer_hidden.shape[0]
            ):
                raise ValueError(
                    "Teacher/student answer-token "
                    "count mismatch."
                )

            teacher_answer_hidden = (
                teacher_answer_hidden.to(
                    device=student_answer_hidden.device,
                    dtype=student_answer_hidden.dtype,
                )
            )

            distillation_losses.append(
                F.mse_loss(
                    student_answer_hidden,
                    teacher_answer_hidden,
                )
            )

        if not distillation_losses:
            print(
                f"[skip] batch={batch_number} "
                "no matching teacher target"
            )
            continue

        distillation_loss = torch.stack(
            distillation_losses
        ).mean()

        # -----------------------------------------------------
        # Calculate gradients independently
        # -----------------------------------------------------

        task_grads = torch.autograd.grad(
            task_loss,
            trainable_parameters,
            retain_graph=True,
            allow_unused=True,
        )

        distill_grads = torch.autograd.grad(
            distillation_loss,
            trainable_parameters,
            retain_graph=False,
            allow_unused=True,
        )

        statistics = gradient_statistics(
            task_grads,
            distill_grads,
            distillation_weight,
        )

        cosine = statistics[
            "cosine_similarity"
        ]

        ratio = statistics[
            "weighted_distill_to_task_grad_ratio"
        ]

        cosine_values.append(
            cosine
        )

        gradient_ratios.append(
            ratio
        )

        matched_batches += 1

        weighted_loss = (
            distillation_weight
            * float(
                distillation_loss
                .detach()
                .item()
            )
        )

        print(
            f"[{matched_batches:02d}/"
            f"{args.num_batches:02d}] "
            f"task={task_loss.detach().item():.6f} "
            f"raw_distill="
            f"{distillation_loss.detach().item():.6f} "
            f"weighted_loss="
            f"{weighted_loss:.6f}"
        )

        print(
            "       "
            f"task_grad="
            f"{statistics['task_grad_norm']:.6f} "
            f"raw_distill_grad="
            f"{statistics['raw_distill_grad_norm']:.6f} "
            f"weighted_distill_grad="
            f"{statistics['weighted_distill_grad_norm']:.6f}"
        )

        print(
            "       "
            f"weighted/task_grad_ratio="
            f"{ratio:.6f} "
            f"cosine="
            f"{cosine:.6f}"
        )

        # Clear references before next forward pass.
        del output
        del task_grads
        del distill_grads

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not cosine_values:
        raise RuntimeError(
            "No batches with matching teacher "
            "targets were diagnosed."
        )

    mean_cosine = (
        sum(cosine_values)
        / len(cosine_values)
    )

    mean_ratio = (
        sum(gradient_ratios)
        / len(gradient_ratios)
    )

    negative_count = sum(
        value < 0
        for value in cosine_values
    )

    print()
    print("=" * 78)
    print("GRADIENT DIAGNOSTIC SUMMARY")
    print("=" * 78)

    print(
        f"Batches analyzed              : "
        f"{len(cosine_values)}"
    )

    print(
        f"Mean cosine similarity        : "
        f"{mean_cosine:.6f}"
    )

    print(
        f"Negative cosine batches       : "
        f"{negative_count}/"
        f"{len(cosine_values)}"
    )

    print(
        f"Mean weighted/task grad ratio : "
        f"{mean_ratio:.6f}"
    )

    print()

    if mean_cosine < 0:
        print(
            "Interpretation: task and distillation "
            "gradients conflict on average."
        )
    elif mean_cosine < 0.2:
        print(
            "Interpretation: task and distillation "
            "gradients are weakly aligned / mostly unrelated."
        )
    else:
        print(
            "Interpretation: task and distillation "
            "gradients show positive alignment."
        )


if __name__ == "__main__":
    main()