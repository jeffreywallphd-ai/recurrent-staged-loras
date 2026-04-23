"""Utility to compare outcome and compute metrics across runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.metrics_schema import RUN_METRICS_FIELDS


DISPLAY_COLUMNS = [
    "baseline_name",
    "config_name",
    "dataset_name",
    "architecture_type",
    "model_name",
    "final_eval_loss",
    "best_eval_loss",
    "eval_perplexity",
    "stage_2_token_accuracy",
    "stage_3_token_accuracy",
    "final_answer_accuracy",
    "final_answer_exact_match",
    "normalized_numeric_answer_accuracy",
    "trainable_params",
    "trainable_param_fraction",
    "wall_time_seconds_total",
    "tokens_per_second_train",
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
        rows.append({**{k: metric.get(k) for k in RUN_METRICS_FIELDS}, "path": str(path)})

    widths = {h: max(len(h), *(len(_fmt(r.get(h))) for r in rows)) for h in DISPLAY_COLUMNS}

    def line(parts: list[str]) -> str:
        return " | ".join(part.ljust(widths[h]) for part, h in zip(parts, DISPLAY_COLUMNS, strict=True))

    print(line(DISPLAY_COLUMNS))
    print("-+-".join("-" * widths[h] for h in DISPLAY_COLUMNS))
    for row in rows:
        print(line([_fmt(row.get(h)) for h in DISPLAY_COLUMNS]))


if __name__ == "__main__":
    main()
