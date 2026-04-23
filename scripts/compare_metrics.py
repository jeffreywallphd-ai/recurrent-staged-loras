"""Utility to compare outcome and compute metrics across runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare metrics from multiple runs")
    parser.add_argument("metrics", nargs="+", help="Paths to metrics.json files")
    return parser.parse_args()


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object) -> str:
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
                "final_train_loss": metric.get("final_train_loss", metric.get("train_loss", "n/a")),
                "final_eval_loss": metric.get("final_eval_loss", metric.get("eval_loss", "n/a")),
                "global_steps_completed": metric.get("global_steps_completed", metric.get("num_steps", "n/a")),
                "trainable_params": metric.get("trainable_params", "n/a"),
                "total_params": metric.get("total_params", "n/a"),
                "trainable_param_fraction": metric.get("trainable_param_fraction", "n/a"),
                "wall_time_seconds_total": metric.get("wall_time_seconds_total", "n/a"),
                "tokens_per_second_train": metric.get("tokens_per_second_train", "n/a"),
                "steps_per_second": metric.get("steps_per_second", "n/a"),
                "path": str(path),
            }
        )

    headers = [
        "baseline",
        "final_train_loss",
        "final_eval_loss",
        "global_steps_completed",
        "trainable_params",
        "total_params",
        "trainable_param_fraction",
        "wall_time_seconds_total",
        "tokens_per_second_train",
        "steps_per_second",
        "path",
    ]
    widths = {h: max(len(h), *(len(_fmt(r[h])) for r in rows)) for h in headers}

    def line(parts: list[str]) -> str:
        return " | ".join(part.ljust(widths[h]) for part, h in zip(parts, headers, strict=True))

    print(line(headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(line([_fmt(row[h]) for h in headers]))


if __name__ == "__main__":
    main()
