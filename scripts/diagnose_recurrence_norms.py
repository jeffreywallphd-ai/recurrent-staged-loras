from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from training.config_loader import (
    build_model_from_variant,
    load_runtime_config,
)


def build_prompt(question: str) -> str:
    """Match the prompt used in the GSM8K generation evaluation."""
    return f"Problem:\n{question.strip()}\n\nReasoning:\n"


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    """
    Calculate stable summary statistics for a hidden-state tensor.

    Values are calculated in float32 to avoid bfloat16 precision issues.
    """
    value = tensor.detach().float()

    return {
        "rms": float(torch.sqrt(torch.mean(value.square())).item()),
        "mean_abs": float(value.abs().mean().item()),
        "max_abs": float(value.abs().max().item()),
        "l2_mean_per_token": float(
            torch.linalg.vector_norm(value, dim=-1).mean().item()
        ),
    }


def delta_stats(
    before: torch.Tensor,
    after: torch.Tensor,
) -> dict[str, float]:
    """Measure the size of a transformation relative to its input."""
    before_f = before.detach().float()
    after_f = after.detach().float()
    delta = after_f - before_f

    before_rms = torch.sqrt(torch.mean(before_f.square()))
    delta_rms = torch.sqrt(torch.mean(delta.square()))

    relative_rms = (
        float((delta_rms / before_rms).item())
        if float(before_rms.item()) > 0
        else 0.0
    )

    return {
        "delta_rms": float(delta_rms.item()),
        "delta_mean_abs": float(delta.abs().mean().item()),
        "delta_max_abs": float(delta.abs().max().item()),
        "relative_delta_rms": relative_rms,
    }


@torch.no_grad()
def diagnose_example(
    model: Any,
    tokenizer: Any,
    question: str,
    dataset_index: int,
) -> dict[str, Any]:
    """Run one prompt through the backbone and each recurrence stage."""

    prompt = build_prompt(question)

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    device = next(model.parameters()).device

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    # Frozen Qwen backbone output before recurrence.
    base_output = model.base_model.forward_backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    current = base_output.hidden_states

    if model.refiner is None:
        raise RuntimeError(
            "This model does not contain a recurrent refiner."
        )

    refiner = model.refiner
    refiner._assert_or_align_runtime(current)

    record: dict[str, Any] = {
        "dataset_index": dataset_index,
        "question": question,
        "prompt_tokens": int(input_ids.shape[1]),
        "step_scale": float(refiner.step_scale),
        "num_steps": int(refiner.num_steps),
        "base_hidden": tensor_stats(current),
        "steps": [],
    }

    for step_index in range(refiner.num_steps):
        before_step = current

        # Core recurrent transformation:
        # h <- h + step_scale * delta
        after_core = refiner.step(
            before_step,
            step_idx=step_index,
        )

        core_change = delta_stats(
            before_step,
            after_core,
        )

        adapter_change: dict[str, float] | None = None

        if refiner.adapter_bank is not None:
            after_adapter = refiner.adapter_bank.apply(
                after_core,
                step_idx=step_index,
            )

            adapter_change = delta_stats(
                after_core,
                after_adapter,
            )

            current = after_adapter
        else:
            current = after_core

        total_change = delta_stats(
            before_step,
            current,
        )

        record["steps"].append(
            {
                "step": step_index + 1,
                "before": tensor_stats(before_step),
                "after_core_refiner": tensor_stats(after_core),
                "core_change": core_change,
                "adapter_change": adapter_change,
                "after_complete_step": tensor_stats(current),
                "total_step_change": total_change,
            }
        )

    record["final_hidden"] = tensor_stats(current)

    base_rms = record["base_hidden"]["rms"]
    final_rms = record["final_hidden"]["rms"]

    record["final_to_base_rms_ratio"] = (
        final_rms / base_rms
        if base_rms > 0
        else 0.0
    )

    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose hidden-state and adapter norm changes "
            "inside a recurrent-staged model."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Recurrent experiment configuration.",
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="checkpoint.pt containing model_state_dict.",
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=100,
        help="First GSM8K test-set index to inspect.",
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=20,
        help="Number of GSM8K prompts to inspect.",
    )

    parser.add_argument(
        "--output",
        default="outputs/diagnostics/recurrence_norms.json",
        help="Path for the diagnostic JSON.",
    )

    args = parser.parse_args()

    if args.start_index < 0:
        raise ValueError("--start_index must be at least 0.")

    if args.num_examples < 1:
        raise ValueError("--num_examples must be at least 1.")

    print(f"Loading config: {args.config}")
    runtime = load_runtime_config(args.config)

    tokenizer_name = (
        runtime.variant.base.tokenizer_name
        or runtime.variant.base.model_name
    )

    print(f"Loading tokenizer: {tokenizer_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=runtime.variant.base.trust_remote_code,
    )

    if (
        tokenizer.pad_token_id is None
        and tokenizer.eos_token_id is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    print("Building model...")
    model = build_model_from_variant(runtime.variant)

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a dictionary.")

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    load_result = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    print("Checkpoint loaded.")
    print(load_result)

    model.eval()

    print("Loading GSM8K test set...")

    dataset = load_dataset(
        "gsm8k",
        "main",
        split="test",
    )

    if args.start_index >= len(dataset):
        raise ValueError(
            f"--start_index {args.start_index} is outside "
            f"the dataset of size {len(dataset)}."
        )

    end_index = min(
        args.start_index + args.num_examples,
        len(dataset),
    )

    records: list[dict[str, Any]] = []

    for dataset_index in range(args.start_index, end_index):
        row = dataset[dataset_index]
        question = str(row["question"])

        result = diagnose_example(
            model=model,
            tokenizer=tokenizer,
            question=question,
            dataset_index=dataset_index,
        )

        records.append(result)

        step_text = " | ".join(
            (
                f"step{step['step']}: "
                f"core={step['core_change']['relative_delta_rms']:.4f}, "
                f"adapter="
                f"{step['adapter_change']['relative_delta_rms']:.4f}, "
                f"hidden_rms="
                f"{step['after_complete_step']['rms']:.4f}"
            )
            for step in result["steps"]
            if step["adapter_change"] is not None
        )

        print(
            f"[{len(records):03d}/{end_index - args.start_index}] "
            f"index={dataset_index} "
            f"base_rms={result['base_hidden']['rms']:.4f} | "
            f"{step_text} | "
            f"final/base="
            f"{result['final_to_base_rms_ratio']:.4f}"
        )

    # Aggregate the diagnostic values across prompts.
    aggregate_steps: list[dict[str, float | int]] = []

    num_steps = int(records[0]["num_steps"])

    for step_index in range(num_steps):
        core_relative = [
            record["steps"][step_index]["core_change"][
                "relative_delta_rms"
            ]
            for record in records
        ]

        adapter_relative = [
            record["steps"][step_index]["adapter_change"][
                "relative_delta_rms"
            ]
            for record in records
            if record["steps"][step_index]["adapter_change"]
            is not None
        ]

        hidden_rms = [
            record["steps"][step_index]["after_complete_step"]["rms"]
            for record in records
        ]

        aggregate_steps.append(
            {
                "step": step_index + 1,
                "mean_core_relative_delta_rms": (
                    sum(core_relative) / len(core_relative)
                ),
                "mean_adapter_relative_delta_rms": (
                    sum(adapter_relative) / len(adapter_relative)
                    if adapter_relative
                    else 0.0
                ),
                "mean_hidden_rms_after_step": (
                    sum(hidden_rms) / len(hidden_rms)
                ),
            }
        )

    final_ratios = [
        record["final_to_base_rms_ratio"]
        for record in records
    ]

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "model_name": runtime.variant.base.model_name,
        "baseline": runtime.baseline,
        "dataset": "gsm8k",
        "dataset_split": "test",
        "start_index": args.start_index,
        "end_index_exclusive": end_index,
        "num_examples": len(records),
        "step_scale": float(model.refiner.step_scale),
        "num_recurrent_steps": int(model.refiner.num_steps),
        "mean_final_to_base_rms_ratio": (
            sum(final_ratios) / len(final_ratios)
        ),
        "aggregate_steps": aggregate_steps,
        "examples": records,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("RECURRENCE HIDDEN-STATE DIAGNOSTIC")
    print("=" * 72)
    print(f"Examples                  : {len(records)}")
    print(
        f"Dataset index range       : "
        f"{args.start_index} to {end_index - 1}"
    )
    print(
        f"Mean final/base RMS ratio : "
        f"{summary['mean_final_to_base_rms_ratio']:.6f}"
    )

    for step in aggregate_steps:
        print(
            f"Step {step['step']} | "
            f"core relative change="
            f"{step['mean_core_relative_delta_rms']:.6f} | "
            f"adapter relative change="
            f"{step['mean_adapter_relative_delta_rms']:.6f} | "
            f"hidden RMS="
            f"{step['mean_hidden_rms_after_step']:.6f}"
        )

    print(f"Saved to                  : {output_path}")


if __name__ == "__main__":
    main()