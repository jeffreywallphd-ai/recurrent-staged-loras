"""Run multiple experiment configs sequentially and aggregate metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from training.config_loader import load_runtime_config
from training.engine import build_training_components, run_training_loop
from training.metrics_schema import AGG_GROUP_BY_FIELDS, AGGREGATE_METRICS, RUN_METRICS_FIELDS


SUMMARY_FIELDS = [
    "row_type",
    *RUN_METRICS_FIELDS,
    "num_runs",
    "metric_name",
    "metric_mean",
    "metric_std",
]


def _collect_config_paths(configs: list[str], config_dir: str | None) -> list[Path]:
    paths = [Path(c) for c in configs]
    if config_dir:
        paths.extend(sorted(Path(config_dir).glob("*.json")))
    if not paths:
        raise ValueError("No config files provided")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all experiment configs sequentially")
    parser.add_argument("--configs", nargs="*", default=[])
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    return parser.parse_args()


def _agg(rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for k in AGG_GROUP_BY_FIELDS:
        out[k] = rows[0].get(k)
    out["num_runs"] = len(rows)
    stats: dict[str, dict[str, float]] = {}
    for metric in AGGREGATE_METRICS:
        vals = [float(r[metric]) for r in rows if r.get(metric) is not None]
        if vals:
            stats[metric] = {
                "mean": mean(vals),
                "std": stdev(vals) if len(vals) > 1 else 0.0,
            }
    out["metrics"] = stats
    return out


def main() -> None:
    args = parse_args()
    config_paths = _collect_config_paths(args.configs, args.config_dir)

    runs: list[dict[str, object]] = []
    for config_path in config_paths:
        for seed in args.seeds:
            runtime = load_runtime_config(config_path)
            runtime.training.seed = seed
            runtime.dataset["settings"]["seed"] = seed
            runtime.output["dir"] = str(Path("outputs") / runtime.baseline)
            run_name = f"{config_path.stem}_seed{seed}"
            result = run_training_loop(
                components=build_training_components(runtime),
                run_name=run_name,
                config_name=config_path.name,
            )
            metrics = json.loads((result.output_dir / "metrics.json").read_text(encoding="utf-8"))
            runs.append(metrics)
            print(f"[ok] {run_name}")

    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in runs:
        key = (str(row["baseline_name"]), str(row["architecture_type"]), str(row["model_name"]))
        grouped.setdefault(key, []).append(row)
    aggregates = [_agg(rows) for rows in grouped.values()]

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps({"runs": runs, "aggregates": aggregates}, indent=2), encoding="utf-8")
    (output_dir / "aggregates.json").write_text(json.dumps(aggregates, indent=2), encoding="utf-8")

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        for r in runs:
            row = {"row_type": "run"}
            row.update({k: r.get(k) for k in RUN_METRICS_FIELDS})
            w.writerow(row)
        for agg in aggregates:
            for metric_name, stat in dict(agg.get("metrics", {})).items():
                row = {
                    "row_type": "aggregate",
                    **{k: agg.get(k) for k in AGG_GROUP_BY_FIELDS},
                    "num_runs": agg.get("num_runs"),
                    "metric_name": metric_name,
                    "metric_mean": stat.get("mean"),
                    "metric_std": stat.get("std"),
                }
                w.writerow(row)


if __name__ == "__main__":
    main()
