"""Utility to compare outcome and compute metrics across runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DISPLAY_COLUMNS = [
    "baseline",
    "config_name",
    "dataset_name",
    "dataset_mode",
    "final_eval_loss",
    "best_eval_loss",
    "eval_perplexity",
    "eval_next_token_accuracy",
    "eval_top_5_accuracy",
    "eval_target_token_accuracy",
    "eval_target_sequence_exact_match",
    "trainable_params",
    "trainable_param_fraction",
    "wall_time_seconds_total",
    "wall_time_seconds_train",
    "wall_time_seconds_eval",
    "tokens_per_second_train",
    "tokens_per_second_eval",
    "steps_per_second",
    "path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare metrics from multiple runs")
    parser.add_argument("metrics", nargs="+", help="Paths to metrics.json files")
    return parser.parse_args()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    args = parse_args()
    rows = []
    for raw_path in args.metrics:
        path = Path(raw_path)
        metric = _load(path)
        rows.append(
            {
                "baseline": metric.get("baseline_name", "unknown"),
                "config_name": metric.get("config_name", "unknown"),
                "dataset_name": metric.get("dataset_name", "unknown"),
                "dataset_mode": metric.get("dataset_mode", "n/a"),
                "final_eval_loss": metric.get("final_eval_loss", metric.get("eval_loss")),
                "best_eval_loss": metric.get("best_eval_loss"),
                "eval_perplexity": metric.get("eval_perplexity"),
                "eval_next_token_accuracy": metric.get("eval_next_token_accuracy"),
                "eval_top_5_accuracy": metric.get("eval_top_5_accuracy"),
                "eval_target_token_accuracy": metric.get("eval_target_token_accuracy"),
                "eval_target_sequence_exact_match": metric.get("eval_target_sequence_exact_match"),
                "trainable_params": metric.get("trainable_params"),
                "trainable_param_fraction": metric.get("trainable_param_fraction"),
                "wall_time_seconds_total": metric.get("wall_time_seconds_total"),
                "wall_time_seconds_train": metric.get("wall_time_seconds_train"),
                "wall_time_seconds_eval": metric.get("wall_time_seconds_eval"),
                "tokens_per_second_train": metric.get("tokens_per_second_train"),
                "tokens_per_second_eval": metric.get("tokens_per_second_eval"),
                "steps_per_second": metric.get("steps_per_second"),
                "path": str(path),
            }
        )

    widths = {h: max(len(h), *(len(_fmt(r[h])) for r in rows)) for h in DISPLAY_COLUMNS}

    def line(parts: list[str]) -> str:
        return " | ".join(part.ljust(widths[h]) for part, h in zip(parts, DISPLAY_COLUMNS, strict=True))

    print(line(DISPLAY_COLUMNS))
    print("-+-".join("-" * widths[h] for h in DISPLAY_COLUMNS))
    for row in rows:
        print(line([_fmt(row[h]) for h in DISPLAY_COLUMNS]))


if __name__ == "__main__":
    main()
