"""Run multiple experiment configs sequentially and aggregate metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from training.config_loader import load_runtime_config
from training.engine import build_training_components, run_training_loop
from training.metrics_schema import AGG_GROUP_BY_FIELDS, AGGREGATE_METRICS, REPORT_TABLE_FIELDS, RUN_METRICS_FIELDS


SUMMARY_FIELDS = [
    "row_type",
    *RUN_METRICS_FIELDS,
    "num_runs",
    "metric_name",
    "metric_mean",
    "metric_std",
]


def _is_pilot(path: Path) -> bool:
    return path.stem.endswith("_pilot")


def _collect_config_paths(configs: list[str], config_dir: str | None) -> list[Path]:
    paths = [Path(c) for c in configs]
    if config_dir:
        paths.extend(sorted(Path(config_dir).glob("*.json")))
    if not paths:
        raise ValueError("No config files provided")
    unique = sorted({p.resolve() for p in paths})
    return [Path(p) for p in unique]


def _filter_config_paths(paths: list[Path], preset_scope: str) -> list[Path]:
    if preset_scope == "all":
        return paths
    if preset_scope == "pilot":
        return [p for p in paths if _is_pilot(p)]
    return [p for p in paths if not _is_pilot(p)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all experiment configs sequentially")
    parser.add_argument("--configs", nargs="*", default=[])
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--preset-scope", choices=["study", "pilot", "all"], default="study")
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


def _group_key(row: dict[str, object]) -> tuple[str, ...]:
    return tuple(str(row.get(key, "")) for key in AGG_GROUP_BY_FIELDS)


def _write_report_table(*, output_dir: Path, runs: list[dict[str, object]], aggregates: list[dict[str, object]]) -> None:
    report_rows: list[dict[str, object]] = []
    for run in runs:
        report_rows.append({
            "row_type": "run",
            "run_name": run.get("run_name"),
            "config_name": run.get("config_name"),
            "baseline_name": run.get("baseline_name"),
            "dataset_name": run.get("dataset_name"),
            "seed": run.get("seed"),
            "architecture_type": run.get("architecture_type"),
            "model_name": run.get("model_name"),
            "final_eval_loss": run.get("final_eval_loss"),
            "eval_perplexity": run.get("eval_perplexity"),
            "final_answer_accuracy": run.get("final_answer_accuracy"),
            "final_answer_exact_match": run.get("final_answer_exact_match"),
            "normalized_numeric_answer_accuracy": run.get("normalized_numeric_answer_accuracy"),
            "trainable_param_fraction": run.get("trainable_param_fraction"),
            "wall_time_seconds_total": run.get("wall_time_seconds_total"),
            "tokens_per_second_train": run.get("tokens_per_second_train"),
        })
    for agg in aggregates:
        for metric_name, stat in dict(agg.get("metrics", {})).items():
            report_rows.append({
                "row_type": "aggregate",
                "config_name": agg.get("config_name"),
                "baseline_name": agg.get("baseline_name"),
                "dataset_name": agg.get("dataset_name"),
                "architecture_type": agg.get("architecture_type"),
                "model_name": agg.get("model_name"),
                "num_runs": agg.get("num_runs"),
                "metric_name": metric_name,
                "metric_mean": stat.get("mean"),
                "metric_std": stat.get("std"),
            })

    with (output_dir / "report_table.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=REPORT_TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)


def main() -> None:
    args = parse_args()
    raw_config_paths = _collect_config_paths(args.configs, args.config_dir)
    config_paths = _filter_config_paths(raw_config_paths, args.preset_scope)
    if not config_paths:
        raise ValueError(f"No configs selected after applying preset scope '{args.preset_scope}'.")

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

    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for row in runs:
        grouped.setdefault(_group_key(row), []).append(row)
    aggregates = [_agg(rows) for rows in grouped.values()]

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "preset_scope": args.preset_scope,
                "config_paths": [str(p) for p in config_paths],
                "runs": runs,
                "aggregates": aggregates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
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

    _write_report_table(output_dir=output_dir, runs=runs, aggregates=aggregates)


if __name__ == "__main__":
    main()
