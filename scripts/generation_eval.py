from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from training.config_loader import (
    build_model_from_variant,
    load_runtime_config,
)


def build_prompt(question: str) -> str:
    """
    Match the beginning of the MetaMathQA training format
    without providing the gold reasoning or answer.
    """
    return f"Problem:\n{question.strip()}\n\nReasoning:\n"


def extract_gsm8k_gold(answer: str) -> str:
    """Extract GSM8K's final answer after ####."""
    if "####" in answer:
        return answer.split("####")[-1].strip()

    return answer.strip()


def extract_final_number(text: str) -> str | None:
    """
    Extract the FIRST numeric answer following the model's FIRST
    'Final Answer:' marker.

    This avoids accidentally scoring numbers from extra text that
    the model may generate after already giving its answer.

    If no 'Final Answer:' marker exists, fall back to the final
    numeric value appearing in the generated text.
    """
    number_pattern = r"-?\$?\d[\d,]*(?:\.\d+)?"

    if "Final Answer:" in text:
        candidate = text.split("Final Answer:", 1)[1]

        matches = re.findall(
            number_pattern,
            candidate,
        )

        if not matches:
            return None

        value = matches[0]

    else:
        matches = re.findall(
            number_pattern,
            text,
        )

        if not matches:
            return None

        value = matches[-1]

    return (
        value.replace("$", "")
        .replace(",", "")
        .strip()
    )


def normalize_number(value: str | None) -> str | None:
    """
    Normalize numeric strings so values such as
    42, 42.0, and 42.000 compare consistently.
    """
    if value is None:
        return None

    try:
        number = float(value)

        if number.is_integer():
            return str(int(number))

        return (
            f"{number:.10f}"
            .rstrip("0")
            .rstrip(".")
        )

    except ValueError:
        return value.strip()


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> str:
    """
    Perform greedy autoregressive generation through
    StagedLatentAdaptationModel.

    Calling the custom model wrapper instead of the underlying
    Hugging Face .generate() method ensures recurrent refinement
    layers are included when evaluating recurrent architectures.
    """

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )

    # Use the device on which the model parameters reside.
    device = next(model.parameters()).device

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    prompt_length = input_ids.shape[1]

    eos_token_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Greedy decoding: choose the most likely next token.
        next_token = (
            output.logits[:, -1, :]
            .argmax(dim=-1, keepdim=True)
        )

        input_ids = torch.cat(
            [input_ids, next_token],
            dim=1,
        )

        next_attention = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        attention_mask = torch.cat(
            [attention_mask, next_attention],
            dim=1,
        )

        # Stop if the model generates EOS.
        if (
            eos_token_id is not None
            and int(next_token.item()) == int(eos_token_id)
        ):
            break

    generated_ids = input_ids[
        0,
        prompt_length:
    ]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generation-based GSM8K evaluation "
            "for recurrent-staged-loras."
        )
    )

    # ---------------------------------------------------------
    # Arguments
    # ---------------------------------------------------------

    parser.add_argument(
        "--config",
        required=True,
        help="Experiment config used to construct the model.",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Optional checkpoint.pt containing model_state_dict. "
            "Leave this out when evaluating the untouched base model."
        ),
    )

    parser.add_argument(
        "--ablation",
        choices=(
            "full",
            "core_only",
            "adapters_only",
        ),
        default="full",
        help=(
            "Evaluation-time recurrent ablation. "
            "'full' uses the complete trained model; "
            "'core_only' disables the per-step adapters; "
            "'adapters_only' sets the recurrent core step scale to zero."
        ),
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=100,
        help="Number of GSM8K test examples to evaluate.",
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help=(
            "Zero-based index of the first GSM8K test example to evaluate. "
            "For example, --start_index 100 --num_examples 200 evaluates "
            "examples 100 through 299."
        ),
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of tokens generated per problem.",
    )

    parser.add_argument(
        "--output",
        default="outputs/generation_eval/results.json",
        help="Where to save evaluation results.",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load experiment configuration
    # ---------------------------------------------------------

    print(f"Loading config: {args.config}")

    runtime = load_runtime_config(
        args.config
    )

    # ---------------------------------------------------------
    # Tokenizer
    # ---------------------------------------------------------

    tokenizer_name = (
        runtime.variant.base.tokenizer_name
        or runtime.variant.base.model_name
    )

    print(
        f"Loading tokenizer: {tokenizer_name}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=(
            runtime.variant.base.trust_remote_code
        ),
    )

    if (
        tokenizer.pad_token_id is None
        and tokenizer.eos_token_id is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------
    # Build experiment architecture
    # ---------------------------------------------------------

    print("Building model...")

    model = build_model_from_variant(
        runtime.variant
    )

    # ---------------------------------------------------------
    # Restore trained weights when a checkpoint is supplied
    # ---------------------------------------------------------

    if args.checkpoint:

        checkpoint_path = Path(
            args.checkpoint
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint does not exist: "
                f"{checkpoint_path}"
            )

        print(
            f"Loading checkpoint: "
            f"{checkpoint_path}"
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise TypeError(
                "Checkpoint must be a dictionary."
            )

        if "model_state_dict" not in checkpoint:
            raise KeyError(
                "Checkpoint does not contain "
                "'model_state_dict'."
            )

        load_result = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        print("Checkpoint loaded.")
        print(load_result)

    else:

        print("No checkpoint supplied.")
        print(
            "Evaluating the pretrained/base "
            "model directly."
        )

    # ---------------------------------------------------------
    # Optional evaluation-time recurrent ablation
    # ---------------------------------------------------------

    if args.ablation != "full":
        refiner = getattr(model, "refiner", None)

        if refiner is None:
            raise RuntimeError(
                f"Ablation '{args.ablation}' requires a model "
                "with an enabled recurrent refiner."
            )

        if args.ablation == "core_only":
            adapter_bank = getattr(refiner, "adapter_bank", None)

            if adapter_bank is None:
                print(
                    "Warning: core_only requested, but this model "
                    "does not contain a recurrent adapter bank."
                )
            else:
                adapter_bank.enabled = False

            print(
                "Ablation active: CORE ONLY "
                "(per-step adapters disabled)."
            )

        elif args.ablation == "adapters_only":
            original_step_scale = float(refiner.step_scale)
            refiner.step_scale = 0.0

            print(
                "Ablation active: ADAPTERS ONLY "
                "(recurrent core disabled; "
                f"step_scale {original_step_scale} -> 0.0)."
            )

    else:
        print("Ablation active: FULL MODEL.")

    # Evaluation mode for both checkpointed and base models.
    model.eval()

    # ---------------------------------------------------------
    # Load GSM8K
    # ---------------------------------------------------------

    print("Loading GSM8K test set...")

    dataset = load_dataset(
        "gsm8k",
        "main",
        split="test",
    )

    if args.start_index < 0:
        raise ValueError("--start_index must be zero or greater.")

    if args.num_examples <= 0:
        raise ValueError("--num_examples must be greater than zero.")

    dataset_size = len(dataset)

    if args.start_index >= dataset_size:
        raise ValueError(
            f"--start_index {args.start_index} is outside the GSM8K test set "
            f"of size {dataset_size}."
        )

    end_index = min(
        args.start_index + args.num_examples,
        dataset_size,
    )

    num_examples = end_index - args.start_index

    correct = 0
    evaluated = 0

    results: list[dict] = []

    # ---------------------------------------------------------
    # Generation evaluation
    # ---------------------------------------------------------

    for local_index, dataset_index in enumerate(
        range(args.start_index, end_index)
    ):

        row = dataset[dataset_index]

        question = str(
            row["question"]
        )

        gold_answer = extract_gsm8k_gold(
            str(row["answer"])
        )

        prompt = build_prompt(
            question
        )

        generated_text = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )

        predicted_number = extract_final_number(
            generated_text
        )

        gold_number = extract_final_number(
            gold_answer
        )

        predicted_normalized = normalize_number(
            predicted_number
        )

        gold_normalized = normalize_number(
            gold_number
        )

        is_correct = (
            predicted_normalized is not None
            and gold_normalized is not None
            and predicted_normalized
            == gold_normalized
        )

        if is_correct:
            correct += 1

        evaluated += 1

        result = {
            "index": dataset_index,
            "evaluation_position": local_index,
            "question": question,
            "prompt": prompt,
            "gold_answer": gold_answer,
            "generated_text": generated_text,
            "predicted_number": predicted_normalized,
            "gold_number": gold_normalized,
            "correct": is_correct,
        }

        results.append(
            result
        )

        running_accuracy = (
            correct / evaluated
        )

        print(
            f"[{evaluated:03d}/{num_examples}] "
            f"correct={is_correct} "
            f"pred={predicted_normalized} "
            f"gold={gold_normalized} "
            f"running_accuracy="
            f"{running_accuracy:.4f}"
        )

    # ---------------------------------------------------------
    # Final metrics
    # ---------------------------------------------------------

    final_accuracy = (
        correct / evaluated
        if evaluated
        else 0.0
    )

    summary = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "model_name": runtime.variant.base.model_name,
        "baseline": runtime.baseline,
        "ablation": args.ablation,
        "dataset": "gsm8k",
        "dataset_split": "test",
        "evaluation_type": "free_generation",
        "decoding": "greedy",
        "requested_num_examples": args.num_examples,
        "num_examples": evaluated,
        "start_index": args.start_index,
        "end_index_exclusive": end_index,
        "max_new_tokens": args.max_new_tokens,
        "correct": correct,
        "accuracy": final_accuracy,
        "accuracy_percent": (
            final_accuracy * 100
        ),
        "results": results,
    }

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print()
    print("=" * 60)
    print(
        "GSM8K GENERATION EVALUATION"
    )
    print("=" * 60)

    print(
        f"Model               : "
        f"{runtime.variant.base.model_name}"
    )

    print(
        f"Baseline            : "
        f"{runtime.baseline}"
    )

    print(
        f"Checkpoint          : "
        f"{args.checkpoint or 'None (base model)'}"
    )

    print(
        f"Ablation mode       : "
        f"{args.ablation}"
    )

    print(
        f"Questions evaluated : "
        f"{evaluated}"
    )

    print(
        f"Dataset index range : "
        f"{args.start_index} to {end_index - 1}"
    )

    print(
        f"Correct answers     : "
        f"{correct}"
    )

    print(
        f"Accuracy            : "
        f"{final_accuracy:.4f}"
    )

    print(
        f"Accuracy (%)        : "
        f"{final_accuracy * 100:.2f}%"
    )

    print(
        f"Results saved to    : "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()