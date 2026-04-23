"""Training entrypoint for minimal end-to-end baseline runs."""

from __future__ import annotations

import argparse

from .baseline_selector import select_baseline
from .config_loader import load_runtime_config
from .loop import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latent adaptation baseline experiment")
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    parser.add_argument("--run-name", required=True, help="Human-readable run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = load_runtime_config(args.config)
    baseline = select_baseline(runtime.raw)
    result = run_training(runtime, run_name=args.run_name)

    print(
        "[ok] "
        f"baseline='{baseline}' "
        f"backend='{result.backend}' "
        f"trainable_params={result.trainable_params} "
        f"final_train_loss={result.final_train_loss:.4f} "
        f"final_eval_loss={result.final_eval_loss:.4f} "
        f"output_dir='{result.output_dir}'"
    )


if __name__ == "__main__":
    main()
