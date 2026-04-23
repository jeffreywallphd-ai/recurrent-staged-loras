"""Utility to compare outcome and compute metrics across run and aggregate artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.metrics_schema import AGG_GROUP_BY_FIELDS, RUN_METRICS_FIELDS


DISPLAY_COLUMNS = [
    "baseline_name",
    "architecture_type",
    "model_name",
    "config_name",
    "final_eval_loss",
    "eval_perplexity",
    "stage_2_token_accuracy",
    "stage_3_token_accuracy",
    "final_answer_accuracy",
    "final_answer_exact_match",
    "final_answer_normalized_match",
    "symbolic_answer_accuracy",
    "normalized_numeric_answer_accuracy",
    "trainable_param_fraction",
    "wall_time_seconds_total",
    "tokens_per_second_train",
    "dataset",
    "report_tier",
    "path",
]

AGG_DISPLAY_COLUMNS = [
    *AGG_GROUP_BY_FIELDS,
    "metric_name",
    "metric_mean",
    "metric_std",
    "num_runs",
    "source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare metrics from multiple runs")
    parser.add_argument("metrics", nargs="*", help="Paths to run-level metrics.json files")
    parser.add_argument("--aggregates", nargs="*", default=[], help="Paths to aggregate artifact JSON files")
    parser.add_argument("--view", choices=["all", "runs", "aggregates"], default="all")
    parser.add_argument("--dataset", default=None, help="Filter by dataset namespace (primary or external dataset name)")
    parser.add_argument("--compute-mode", default=None, help="Filter by compute_control_mode")
    parser.add_argument("--ablation-recurrent-steps", type=int, default=None)
    parser.add_argument("--ablation-lora-rank", type=int, default=None)
    return parser.parse_args()


def _load(path: Path) -> dict[str, object] | list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _print_table(rows: list[dict[str, object]], headers: list[str]) -> None:
    if not rows:
        return
    widths = {h: max(len(h), *(len(_fmt(r.get(h))) for r in rows)) for h in headers}

    def line(parts: list[str]) -> str:
        return " | ".join(part.ljust(widths[h]) for part, h in zip(parts, headers, strict=True))

    print(line(headers))
    print("-+-".join("-" * widths[h] for h in headers))
    for row in rows:
        print(line([_fmt(row.get(h)) for h in headers]))


def _flatten_aggregates(payload: object, source: str) -> list[dict[str, object]]:
    if isinstance(payload, dict) and "aggregates" in payload:
        payload = payload["aggregates"]
    if not isinstance(payload, list):
        return []
    flat: list[dict[str, object]] = []
    for agg in payload:
        if not isinstance(agg, dict):
            continue
        metrics = agg.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        for metric_name, stat in metrics.items():
            if not isinstance(stat, dict):
                continue
            flat.append(
                {
                    **{k: agg.get(k) for k in AGG_GROUP_BY_FIELDS},
                    "metric_name": metric_name,
                    "metric_mean": stat.get("mean"),
                    "metric_std": stat.get("std"),
                    "num_runs": agg.get("num_runs"),
                    "source": source,
                }
            )
    return flat


def main() -> None:
    args = parse_args()

    run_rows = []
    for raw_path in args.metrics:
        path = Path(raw_path)
        metric = _load(path)
        if isinstance(metric, dict):
            if args.compute_mode and str(metric.get("compute_control_mode")) != args.compute_mode:
                continue
            if args.ablation_recurrent_steps is not None and int(metric.get("ablation_recurrent_steps") or -1) != args.ablation_recurrent_steps:
                continue
            if args.ablation_lora_rank is not None and int(metric.get("ablation_lora_rank") or -1) != args.ablation_lora_rank:
                continue
            dataset_filter = args.dataset
            base_row = {**{k: metric.get(k) for k in RUN_METRICS_FIELDS}, "dataset": "primary", "report_tier": "primary_run", "path": str(path)}
            external = dict(metric.get("external_eval", {}))
            if dataset_filter in (None, "primary"):
                run_rows.append(base_row)
            if dataset_filter not in (None, "primary"):
                payload = external.get(dataset_filter)
                if isinstance(payload, dict):
                    run_rows.append(
                        {
                            **base_row,
                            "dataset": dataset_filter,
                            "report_tier": "external_eval",
                            "final_eval_loss": payload.get("eval_loss"),
                            "stage_2_token_accuracy": payload.get("stage_2_token_accuracy"),
                            "stage_3_token_accuracy": payload.get("stage_3_token_accuracy"),
                            "final_answer_accuracy": payload.get("final_answer_accuracy"),
                            "final_answer_exact_match": payload.get("final_answer_exact_match"),
                            "normalized_numeric_answer_accuracy": payload.get("normalized_numeric_answer_accuracy"),
                        }
                    )

    if run_rows and args.view in {"all", "runs"}:
        _print_table(run_rows, DISPLAY_COLUMNS)

    aggregate_rows: list[dict[str, object]] = []
    for raw_path in args.aggregates:
        path = Path(raw_path)
        payload = _load(path)
        aggregate_rows.extend(_flatten_aggregates(payload, source=str(path)))

    if aggregate_rows and args.view in {"all", "aggregates"}:
        print()
        print("Paper-style aggregate mean/std summary")
        _print_table(aggregate_rows, AGG_DISPLAY_COLUMNS)


if __name__ == "__main__":
    main()
