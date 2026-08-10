"""Cache Standard LoRA hidden-state targets for offline distillation.

Only hidden states corresponding to final-answer tokens are stored.

Each example is identified by a stable hash of its input IDs and attention
mask so the student training loop can retrieve the correct teacher target.

The cache is seed-specific because different dataset seeds may select
different MetaMathQA training examples.
"""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import torch

from training.config_loader import load_runtime_config
from training.engine import build_training_components


def sample_key(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> str:
    """Create a deterministic identifier for one tokenized example."""

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


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Cache Standard LoRA answer-token hidden states "
            "for offline distillation."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=(
            "Training/dataset seed whose examples should be cached. "
            "Usually one of 11, 22, or 33."
        ),
    )

    parser.add_argument(
        "--teacher-config",
        default="experiments/configs/standard_lora.json",
        help="Standard LoRA teacher configuration.",
    )

    parser.add_argument(
        "--teacher-checkpoint",
        required=True,
        help=(
            "Standard LoRA checkpoint corresponding to the requested seed."
        ),
    )

    parser.add_argument(
        "--max-examples",
        type=int,
        default=1000,
        help="Maximum number of training examples to cache.",
    )

    parser.add_argument(
        "--output-dir",
        default="outputs/distillation_cache",
        help="Directory where cache and metadata files are written.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_examples <= 0:
        raise ValueError("--max-examples must be greater than zero.")

    teacher_config = Path(args.teacher_config)
    teacher_checkpoint = Path(args.teacher_checkpoint)
    output_dir = Path(args.output_dir)

    if not teacher_config.exists():
        raise FileNotFoundError(
            f"Missing teacher config: {teacher_config}"
        )

    if not teacher_checkpoint.exists():
        raise FileNotFoundError(
            f"Missing teacher checkpoint: {teacher_checkpoint}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / (
        f"standard_lora_answer_hidden_train_"
        f"seed{args.seed}_{args.max_examples}.pt"
    )

    metadata_path = output_dir / (
        f"standard_lora_answer_hidden_train_"
        f"seed{args.seed}_{args.max_examples}_metadata.json"
    )

    # ---------------------------------------------------------
    # Load teacher configuration
    # ---------------------------------------------------------

    print("[distill-cache] Loading teacher config...")

    runtime = load_runtime_config(
        teacher_config
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Make the teacher dataset use the requested seed.
    #
    # run_all_experiments.py performs this same kind of override
    # for experiment runs. We must do it here BEFORE building
    # training components so the cached examples match the
    # student's dataset for this seed.
    # ---------------------------------------------------------

    runtime.training.seed = args.seed

    if "settings" not in runtime.dataset:
        raise KeyError(
            "Runtime dataset configuration does not contain 'settings'."
        )

    runtime.dataset["settings"]["seed"] = args.seed

    print(
        f"[distill-cache] Seed: {args.seed}"
    )

    print(
        "[distill-cache] Dataset seed: "
        f"{runtime.dataset['settings']['seed']}"
    )

    print(
        "[distill-cache] Training seed: "
        f"{runtime.training.seed}"
    )

    # ---------------------------------------------------------
    # Build teacher + matching dataset
    # ---------------------------------------------------------

    print(
        "[distill-cache] Building Standard LoRA teacher "
        "and training dataset..."
    )

    components = build_training_components(
        runtime
    )

    teacher = components.model

    # ---------------------------------------------------------
    # Load the trained Standard LoRA teacher
    # ---------------------------------------------------------

    print(
        f"[distill-cache] Loading teacher checkpoint: "
        f"{teacher_checkpoint}"
    )

    checkpoint = torch.load(
        teacher_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Teacher checkpoint must be a dictionary."
        )

    state_dict = checkpoint.get(
        "model_state_dict"
    )

    if not isinstance(state_dict, dict):
        raise ValueError(
            "Teacher checkpoint is missing model_state_dict."
        )

    load_result = teacher.load_state_dict(
        state_dict,
        strict=True,
    )

    print(
        f"[distill-cache] Teacher checkpoint loaded: "
        f"{load_result}"
    )

    teacher.requires_grad_(False)
    teacher.eval()

    # ---------------------------------------------------------
    # Cache teacher hidden states
    # ---------------------------------------------------------

    targets: dict[
        str,
        dict[str, torch.Tensor | int],
    ] = {}

    examples_cached = 0
    answer_tokens_cached = 0

    print(
        "[distill-cache] Beginning teacher-target caching..."
    )

    with torch.no_grad():

        for batch in components.train_loader:

            device = next(
                teacher.parameters()
            ).device

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            answer_mask = batch[
                "answer_mask"
            ].to(device)

            output = teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # Hidden state at position t predicts token t+1.
            shifted_teacher_hidden = (
                output.refined_hidden_states[
                    :,
                    :-1,
                    :
                ]
            )

            shifted_answer_mask = (
                answer_mask[
                    :,
                    1:
                ]
                .bool()
            )

            batch_size = input_ids.shape[0]

            for index in range(batch_size):

                if examples_cached >= args.max_examples:
                    break

                valid_answer_positions = (
                    shifted_answer_mask[index]
                )

                answer_token_count = int(
                    valid_answer_positions
                    .sum()
                    .item()
                )

                if answer_token_count == 0:
                    continue

                key = sample_key(
                    input_ids[index],
                    attention_mask[index],
                )

                teacher_answer_hidden = (
                    shifted_teacher_hidden[index][
                        valid_answer_positions
                    ]
                )

                # Store as bf16 to keep the cache relatively small.
                teacher_answer_hidden = (
                    teacher_answer_hidden
                    .detach()
                    .to(
                        dtype=torch.bfloat16,
                        device="cpu",
                    )
                    .contiguous()
                )

                targets[key] = {
                    "teacher_answer_hidden":
                        teacher_answer_hidden,
                    "answer_token_count":
                        answer_token_count,
                }

                examples_cached += 1
                answer_tokens_cached += (
                    answer_token_count
                )

                if examples_cached % 50 == 0:
                    print(
                        "[distill-cache] "
                        f"cached "
                        f"{examples_cached}/"
                        f"{args.max_examples} "
                        "examples"
                    )

            if examples_cached >= args.max_examples:
                break

    if examples_cached == 0:
        raise RuntimeError(
            "No teacher targets were cached."
        )

    # ---------------------------------------------------------
    # Save cache
    # ---------------------------------------------------------

    payload = {
        "version": 2,
        "seed": args.seed,
        "teacher_checkpoint_step":
            checkpoint.get("step"),
        "max_examples_requested":
            args.max_examples,
        "examples_cached":
            examples_cached,
        "answer_tokens_cached":
            answer_tokens_cached,
        "targets":
            targets,
    }

    torch.save(
        payload,
        output_path,
    )

    # ---------------------------------------------------------
    # Save human-readable metadata
    # ---------------------------------------------------------

    metadata = {
        "version": 2,
        "seed": args.seed,
        "training_seed":
            runtime.training.seed,
        "dataset_seed":
            runtime.dataset["settings"]["seed"],
        "teacher_config":
            str(teacher_config),
        "teacher_checkpoint":
            str(teacher_checkpoint),
        "teacher_checkpoint_step":
            checkpoint.get("step"),
        "examples_cached":
            examples_cached,
        "answer_tokens_cached":
            answer_tokens_cached,
        "hidden_dtype":
            "bfloat16",
        "scope": (
            "training examples, "
            "final-answer token positions only"
        ),
        "sample_identity": (
            "SHA256(input_ids + attention_mask)"
        ),
        "output_path":
            str(output_path),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("DISTILLATION CACHE COMPLETE")
    print("=" * 70)

    print(
        f"Seed                 : "
        f"{args.seed}"
    )

    print(
        f"Teacher checkpoint   : "
        f"{teacher_checkpoint}"
    )

    print(
        f"Examples cached      : "
        f"{examples_cached}"
    )

    print(
        f"Answer tokens cached : "
        f"{answer_tokens_cached}"
    )

    print(
        f"Cache saved to       : "
        f"{output_path}"
    )

    print(
        f"Metadata saved to    : "
        f"{metadata_path}"
    )


if __name__ == "__main__":
    main()