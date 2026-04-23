"""Minimal utility to compare metrics.json files."""

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
                "train_loss": metric.get("train_loss", "n/a"),
                "eval_loss": metric.get("eval_loss", "n/a"),
                "num_steps": metric.get("num_steps", "n/a"),
                "num_epochs": metric.get("num_epochs", "n/a"),
                "path": str(path),
            }
        )

    headers = ["baseline", "train_loss", "eval_loss", "num_steps", "num_epochs", "path"]
    widths = {h: max(len(h), *(len(_fmt(r[h])) for r in rows)) for h in headers}

    def line(parts: list[str]) -> str:
        return " | ".join(part.ljust(widths[h]) for part, h in zip(parts, headers, strict=True))

    print(line(headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(line([_fmt(row[h]) for h in headers]))


if __name__ == "__main__":
    main()
