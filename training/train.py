"""Training entrypoint scaffold for recurrent latent adaptation experiments."""

from __future__ import annotations

import argparse

from .baseline_selector import select_baseline
from .config_loader import build_model_from_variant, load_experiment_config
from .run_metadata import RunMetadata
from models.config import parse_variant_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent adaptation baseline experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    parser.add_argument("--run-name", required=True, help="Human-readable run name")
    return parser.parse_args()


def _synthetic_batch(batch_size: int, seq_len: int, vocab_size: int) -> list[list[int]]:
    return [[(batch_idx * seq_len + tok_idx) % vocab_size for tok_idx in range(seq_len)] for batch_idx in range(batch_size)]


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    baseline = select_baseline(cfg)
    variant = parse_variant_config(cfg)

    model = build_model_from_variant(variant)
    input_ids = _synthetic_batch(batch_size=2, seq_len=4, vocab_size=model.base_model.vocab_size)
    attention_mask = [[1 for _ in seq] for seq in input_ids]
    output = model.forward(input_ids=input_ids, attention_mask=attention_mask)

    metadata = RunMetadata(
        run_name=args.run_name,
        baseline=baseline,
        dataset_name=cfg["dataset"]["name"],
        model_name=cfg["model"]["name"],
        output_dir=cfg["output"]["dir"],
    )
    metadata_path = metadata.write()

    logits_shape = (len(output.logits), len(output.logits[0]), len(output.logits[0][0]))
    print(
        "[ok] "
        f"Baseline '{baseline}' built and smoke-forwarded successfully. "
        f"Logits shape={logits_shape}. Metadata: {metadata_path}"
    )


if __name__ == "__main__":
    main()
