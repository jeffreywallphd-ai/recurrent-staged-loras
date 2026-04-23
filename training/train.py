"""Training entrypoint for lightweight trainability smoke checks."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .baseline_selector import select_baseline
from .config_loader import build_model_from_variant, load_experiment_config
from .run_metadata import RunMetadata
from models.config import parse_variant_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent adaptation baseline experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    parser.add_argument("--run-name", required=True, help="Human-readable run name")
    return parser.parse_args()


def _synthetic_batch(batch_size: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    flat = torch.arange(batch_size * seq_len, dtype=torch.long) % vocab_size
    return flat.reshape(batch_size, seq_len)


def _count_trainable_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _count_grad_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad and param.grad is not None)


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    baseline = select_baseline(cfg)
    variant = parse_variant_config(cfg)

    model = build_model_from_variant(variant)
    model.train()

    input_ids = _synthetic_batch(batch_size=2, seq_len=6, vocab_size=model.base_model.vocab_size)
    attention_mask = torch.ones_like(input_ids)

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = output.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    loss.backward()

    grad_params = _count_grad_params(model)
    if grad_params == 0:
        raise RuntimeError("No gradients found on trainable parameters during smoke train step")

    metadata = RunMetadata(
        run_name=args.run_name,
        baseline=baseline,
        dataset_name=cfg["dataset"]["name"],
        model_name=cfg["model"]["name"],
        output_dir=cfg["output"]["dir"],
    )
    metadata_path = metadata.write()

    print(
        "[ok] "
        f"Baseline='{baseline}' backend='{model.base_model.backend}' "
        f"logits_shape={tuple(output.logits.shape)} "
        f"trainable_params={_count_trainable_params(model)} "
        f"grad_params={grad_params} "
        f"loss={loss.item():.4f} "
        f"metadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
