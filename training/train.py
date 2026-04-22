"""Training entrypoint scaffold for recurrent latent adaptation experiments."""

from __future__ import annotations

import argparse

from .baseline_selector import select_baseline
from .config_loader import load_experiment_config
from .run_metadata import RunMetadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent adaptation baseline experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    parser.add_argument("--run-name", required=True, help="Human-readable run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    baseline = select_baseline(cfg)

    metadata = RunMetadata(
        run_name=args.run_name,
        baseline=baseline,
        dataset_name=cfg["dataset"]["name"],
        model_name=cfg["model"]["name"],
        output_dir=cfg["output"]["dir"],
    )
    metadata_path = metadata.write()

    # TODO: wire model construction, data pipeline, optimizer, and trainer loop.
    print(f"[placeholder] Prepared baseline '{baseline}'. Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
