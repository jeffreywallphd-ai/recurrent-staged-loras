"""Run multiple experiment configs sequentially and aggregate metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from training.config_loader import load_runtime_config
from training.engine import build_training_components, run_training_loop


SUMMARY_FIELDS = [
    "baseline",
    "run_name",
    "config_name",
    "dataset_name",
    "dataset_mode",
    "config_path",
    "metrics_path",
    "output_dir",
    "final_train_loss",
    "final_eval_loss",
    "best_eval_loss",
    "eval_perplexity",
    "eval_next_token_accuracy",
    "eval_top_5_accuracy",
    "eval_target_token_accuracy",
    "eval_target_sequence_exact_match",
    "global_steps_completed",
    "epochs_completed",
    "trainable_params",
    "total_params",
    "trainable_param_fraction",
    "wall_time_seconds_total",
    "wall_time_seconds_train",
    "wall_time_seconds_eval",
    "tokens_seen_train",
    "tokens_seen_eval",
    "tokens_per_second_train",
    "tokens_per_second_eval",
    "steps_per_second",
    "seconds_per_step",
]


def _collect_config_paths(configs: list[str], config_dir: str | None) -> list[Path]:
    paths = [Path(c) for c in configs]
    if config_dir:
        paths.extend(sorted(Path(config_dir).glob("*.json")))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            deduped.append(p)
    if not deduped:
        raise ValueError("No config files provided. Use --configs and/or --config-dir.")
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all experiment configs sequentially")
    parser.add_argument("--configs", nargs="*", default=[], help="Explicit config paths")
    parser.add_argument("--config-dir", default=None, help="Directory containing *.json experiment configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = _collect_config_paths(args.configs, args.config_dir)

    runs: list[dict[str, object]] = []
    for config_path in config_paths:
        runtime = load_runtime_config(config_path)
        runtime.output["dir"] = str(Path("outputs") / runtime.baseline)
        run_name = config_path.stem

        components = build_training_components(runtime)
        result = run_training_loop(components=components, run_name=run_name, config_name=config_path.name)

        metrics_path = result.output_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        run_row = {
            "baseline": runtime.baseline,
            "run_name": run_name,
            "config_name": config_path.name,
            "dataset_name": metrics.get("dataset_name"),
            "dataset_mode": metrics.get("dataset_mode"),
            "config_path": str(config_path),
            "metrics_path": str(metrics_path),
            "output_dir": str(result.output_dir),
        }
        for field in SUMMARY_FIELDS:
            if field not in run_row and field in metrics:
                run_row[field] = metrics[field]
        runs.append({**run_row, "metrics": metrics})
        print(f"[ok] baseline={runtime.baseline} run={run_name} metrics={metrics_path}")

    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary = {"runs": runs}
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    summary_csv_path = output_dir / "summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for run in runs:
            writer.writerow({key: run.get(key) for key in SUMMARY_FIELDS})

    print(f"[ok] summary={summary_path}")
    print(f"[ok] summary_csv={summary_csv_path}")


if __name__ == "__main__":
    main()
