"""Run multiple experiment configs sequentially and aggregate metrics."""

from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from training.config_loader import load_experiment_config, load_runtime_config_from_raw
from training.engine import build_training_components, run_training_loop
from training.metrics_schema import AGG_GROUP_BY_FIELDS, AGGREGATE_METRICS, REPORT_TABLE_FIELDS, RUN_METRICS_FIELDS




AGGREGATION_CONTROL_FIELDS = [
    "baseline_name",
    "architecture_type",
    "model_name",
    "dataset_name",
    "config_name",
    "compute_control_enabled",
    "compute_control_mode",
    "recurrence_steps",
    "ablation_recurrent_steps",
    "ablation_lora_rank",
    "run_scope",
]


def _normalize_group_value(value: object) -> object:
    return None if value == "" else value

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


def _matches_config_family(path: Path, family: str) -> bool:
    stem = path.stem
    if family == "all":
        return True
    if family == "confirmatory":
        return all(token not in stem for token in ("_pilot", "_debug", "_external_eval", "_compute_controlled", "_ablation"))
    if family == "pilot":
        return stem.endswith("_pilot")
    if family == "debug":
        return "_debug" in stem
    if family == "external_eval":
        return "_external_eval" in stem
    if family == "compute_controlled":
        return "_compute_controlled" in stem
    if family == "ablation":
        return "_ablation" in stem
    raise ValueError(f"Unknown config family filter '{family}'")


def _filter_by_config_family(paths: list[Path], family: str) -> list[Path]:
    return [p for p in paths if _matches_config_family(p, family)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all experiment configs sequentially")
    parser.add_argument("--configs", nargs="*", default=[])
    parser.add_argument("--config-dir", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=[11, 22, 33])
    parser.add_argument("--preset-scope", choices=["study", "pilot", "all"], default="study")
    parser.add_argument("--run-scope", choices=["confirmatory", "ablation", "all"], default="confirmatory")
    parser.add_argument(
        "--config-family",
        choices=["all", "confirmatory", "pilot", "debug", "external_eval", "compute_controlled", "ablation"],
        default="all",
        help="Optional suffix-based config selector for experiments/configs naming families.",
    )
    return parser.parse_args()


def _agg(rows: list[dict[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for k in AGG_GROUP_BY_FIELDS:
        out[k] = rows[0].get(k)
    for field in AGGREGATION_CONTROL_FIELDS:
        values = {_normalize_group_value(r.get(field)) for r in rows}
        if len(values) != 1:
            raise ValueError(f"Invalid aggregate group: heterogeneous '{field}' values detected: {sorted(values, key=str)}")
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


def _has_ablation_payload(raw: dict[str, Any]) -> bool:
    ablations = raw.get("ablations", {})
    if not isinstance(ablations, dict):
        raise ValueError("ablations must be an object when provided")
    return ("recurrent_steps" in ablations) or ("lora_rank" in ablations)


def _apply_rank_ablation(derived: dict[str, Any], rank: int) -> None:
    model = derived.setdefault("model", {})
    standard_lora = model.setdefault("standard_lora", {})
    latent_refiner = model.setdefault("latent_refiner", {})
    refiner_adapter = latent_refiner.setdefault("adapter", {})
    standard_enabled = bool(standard_lora.get("enabled", False))
    refiner_adapter_enabled = bool(refiner_adapter.get("enabled", False))
    if not standard_enabled and not refiner_adapter_enabled:
        raise ValueError("lora_rank ablation requested but no active adapter found")
    if standard_enabled:
        standard_lora["rank"] = int(rank)
    if refiner_adapter_enabled:
        refiner_adapter["rank"] = int(rank)


def _apply_recurrent_step_ablation(derived: dict[str, Any], recurrent_steps: int) -> None:
    latent_refiner = derived.setdefault("model", {}).setdefault("latent_refiner", {})
    if not bool(latent_refiner.get("enabled", False)):
        raise ValueError("recurrent_steps ablation requested but latent_refiner.enabled is false")
    latent_refiner["num_recurrent_steps"] = int(recurrent_steps)


def _build_ablation_runs(config_path: Path, run_scope: str) -> list[tuple[str, dict[str, Any], str]]:
    raw = load_experiment_config(config_path)
    has_ablation = _has_ablation_payload(raw)
    if run_scope == "confirmatory":
        if has_ablation:
            return []
        confirmatory = json.loads(json.dumps(raw))
        confirmatory["run_scope"] = "confirmatory"
        confirmatory["baseline_family"] = str(raw["baseline"])
        return [(config_path.name, confirmatory, "confirmatory")]

    if run_scope == "ablation" and not has_ablation:
        return []

    ablations = dict(raw.get("ablations", {}))
    has_recurrent_axis = "recurrent_steps" in ablations
    has_rank_axis = "lora_rank" in ablations
    recurrent_steps = list(ablations.get("recurrent_steps", [])) if has_recurrent_axis else []
    lora_ranks = list(ablations.get("lora_rank", [])) if has_rank_axis else []
    if not recurrent_steps and not lora_ranks:
        if has_recurrent_axis:
            raise ValueError("ablations.recurrent_steps was provided but empty; supply at least one integer step")
        if has_rank_axis:
            raise ValueError("ablations.lora_rank was provided but empty; supply at least one integer rank")
        only = json.loads(json.dumps(raw))
        only["run_scope"] = "confirmatory"
        only["baseline_family"] = str(raw["baseline"])
        return [(config_path.name, only, "confirmatory")]

    if _is_pilot(config_path):
        raise ValueError("Ablations must not be attached to pilot configs to avoid silent mixing with confirmatory presets")

    rec_values = recurrent_steps if has_recurrent_axis else [None]
    rank_values = lora_ranks if has_rank_axis else [None]
    expanded: list[tuple[str, dict[str, Any], str]] = []
    for rec, rank in product(rec_values, rank_values):
        derived = json.loads(json.dumps(raw))
        name_parts: list[str] = []
        if rec is not None:
            _apply_recurrent_step_ablation(derived, int(rec))
            name_parts.append(f"r{int(rec)}")
        if rank is not None:
            _apply_rank_ablation(derived, int(rank))
            name_parts.append(f"rank{int(rank)}")
        suffix = "_".join(name_parts)
        derived_baseline = f"{raw['baseline']}_{suffix}"
        derived["baseline"] = derived_baseline
        derived["baseline_family"] = str(raw["baseline"])
        derived["run_scope"] = "ablation"
        derived["ablation"] = {
            "recurrent_steps": (int(rec) if rec is not None else None),
            "lora_rank": (int(rank) if rank is not None else None),
        }
        derived_name = f"{config_path.stem}__{suffix}.json"
        expanded.append((derived_name, derived, "ablation"))
    return expanded


def _write_report_table(*, output_dir: Path, runs: list[dict[str, object]], aggregates: list[dict[str, object]]) -> None:
    report_rows: list[dict[str, object]] = []
    for run in runs:
        report_rows.append({
            "row_type": "run",
            "report_tier": "primary_run",
            "run_name": run.get("run_name"),
            "config_name": run.get("config_name"),
            "baseline_name": run.get("baseline_name"),
            "baseline_family": run.get("baseline_family") or run.get("baseline_name"),
            "dataset_name": run.get("dataset_name"),
            "dataset_type": run.get("dataset_type", "primary"),
            "dataset_scope": "primary",
            "dataset_split": run.get("dataset_split"),
            "dataset_seed": run.get("dataset_seed"),
            "dataset_subset_size": run.get("dataset_subset_size"),
            "dataset_eval_fraction": run.get("dataset_eval_fraction"),
            "dataset_fingerprint": run.get("dataset_fingerprint"),
            "train_sample_ids_hash": run.get("train_sample_ids_hash"),
            "eval_sample_ids_hash": run.get("eval_sample_ids_hash"),
            "dataset": "primary",
            "run_scope": run.get("run_scope"),
            "seed": run.get("seed"),
            "architecture_type": run.get("architecture_type"),
            "model_name": run.get("model_name"),
            "compute_control_enabled": run.get("compute_control_enabled"),
            "compute_control_mode": run.get("compute_control_mode"),
            "effective_optimizer_steps": run.get("effective_optimizer_steps"),
            "tokens_per_optimizer_step": run.get("tokens_per_optimizer_step"),
            "ablation_recurrent_steps": run.get("ablation_recurrent_steps"),
            "ablation_lora_rank": run.get("ablation_lora_rank"),
            "final_eval_loss": run.get("final_eval_loss"),
            "eval_perplexity": run.get("eval_perplexity"),
            "stage_2_token_accuracy": run.get("stage_2_token_accuracy"),
            "stage_3_token_accuracy": run.get("stage_3_token_accuracy"),
            "final_answer_accuracy": run.get("final_answer_accuracy"),
            "final_answer_exact_match": run.get("final_answer_exact_match"),
            "normalized_numeric_answer_accuracy": run.get("normalized_numeric_answer_accuracy"),
            "trainable_param_fraction": run.get("trainable_param_fraction"),
            "wall_time_seconds_total": run.get("wall_time_seconds_total"),
            "tokens_per_second_train": run.get("tokens_per_second_train"),
        })
        for external_name, payload in dict(run.get("external_eval", {})).items():
            if not isinstance(payload, dict):
                continue
            report_rows.append({
                "row_type": "run",
                "report_tier": "external_eval",
                "run_name": run.get("run_name"),
                "config_name": run.get("config_name"),
                "baseline_name": run.get("baseline_name"),
                "baseline_family": run.get("baseline_family") or run.get("baseline_name"),
                "dataset_name": payload.get("dataset_name", external_name),
                "dataset_type": payload.get("dataset_type", "external"),
                "dataset_scope": "external",
                "dataset_split": payload.get("dataset_split"),
                "dataset_seed": payload.get("dataset_seed"),
                "dataset_subset_size": payload.get("dataset_subset_size"),
                "dataset_eval_fraction": payload.get("dataset_eval_fraction"),
                "dataset_fingerprint": payload.get("dataset_fingerprint"),
                "train_sample_ids_hash": payload.get("train_sample_ids_hash"),
                "eval_sample_ids_hash": payload.get("eval_sample_ids_hash"),
                "dataset": external_name,
                "run_scope": run.get("run_scope"),
                "seed": run.get("seed"),
                "architecture_type": run.get("architecture_type"),
                "model_name": run.get("model_name"),
                "compute_control_enabled": run.get("compute_control_enabled"),
                "compute_control_mode": run.get("compute_control_mode"),
                "effective_optimizer_steps": run.get("effective_optimizer_steps"),
                "tokens_per_optimizer_step": run.get("tokens_per_optimizer_step"),
                "ablation_recurrent_steps": run.get("ablation_recurrent_steps"),
                "ablation_lora_rank": run.get("ablation_lora_rank"),
                "final_eval_loss": payload.get("eval_loss"),
                "stage_2_token_accuracy": payload.get("stage_2_token_accuracy"),
                "stage_3_token_accuracy": payload.get("stage_3_token_accuracy"),
                "final_answer_accuracy": payload.get("final_answer_accuracy"),
                "final_answer_exact_match": payload.get("final_answer_exact_match"),
                "normalized_numeric_answer_accuracy": payload.get("normalized_numeric_answer_accuracy"),
            })

    for agg in aggregates:
        metrics = dict(agg.get("metrics", {}))
        report_rows.append({
            "row_type": "aggregate",
            "report_tier": "primary_aggregate",
            "config_name": agg.get("config_name"),
            "baseline_name": agg.get("baseline_name"),
            "baseline_family": agg.get("baseline_family") or agg.get("baseline_name"),
            "dataset_name": agg.get("dataset_name"),
            "dataset_type": agg.get("dataset_type", "primary"),
            "dataset_scope": "primary",
            "dataset_split": None,
            "dataset_seed": None,
            "dataset_subset_size": None,
            "dataset_eval_fraction": None,
            "dataset_fingerprint": None,
            "train_sample_ids_hash": None,
            "eval_sample_ids_hash": None,
            "dataset": "primary",
            "run_scope": agg.get("run_scope"),
            "architecture_type": agg.get("architecture_type"),
            "model_name": agg.get("model_name"),
            "compute_control_enabled": agg.get("compute_control_enabled"),
            "compute_control_mode": agg.get("compute_control_mode"),
            "effective_optimizer_steps": None,
            "tokens_per_optimizer_step": None,
            "ablation_recurrent_steps": agg.get("ablation_recurrent_steps"),
            "ablation_lora_rank": agg.get("ablation_lora_rank"),
            "num_runs": agg.get("num_runs"),
            "final_eval_loss": dict(metrics.get("final_eval_loss", {})).get("mean"),
            "final_eval_loss_std": dict(metrics.get("final_eval_loss", {})).get("std"),
            "eval_perplexity": dict(metrics.get("eval_perplexity", {})).get("mean"),
            "eval_perplexity_std": dict(metrics.get("eval_perplexity", {})).get("std"),
            "stage_2_token_accuracy": dict(metrics.get("stage_2_token_accuracy", {})).get("mean"),
            "stage_2_token_accuracy_std": dict(metrics.get("stage_2_token_accuracy", {})).get("std"),
            "stage_3_token_accuracy": dict(metrics.get("stage_3_token_accuracy", {})).get("mean"),
            "stage_3_token_accuracy_std": dict(metrics.get("stage_3_token_accuracy", {})).get("std"),
            "final_answer_accuracy": dict(metrics.get("final_answer_accuracy", {})).get("mean"),
            "final_answer_accuracy_std": dict(metrics.get("final_answer_accuracy", {})).get("std"),
            "final_answer_exact_match": dict(metrics.get("final_answer_exact_match", {})).get("mean"),
            "final_answer_exact_match_std": dict(metrics.get("final_answer_exact_match", {})).get("std"),
            "normalized_numeric_answer_accuracy": dict(metrics.get("normalized_numeric_answer_accuracy", {})).get("mean"),
            "normalized_numeric_answer_accuracy_std": dict(metrics.get("normalized_numeric_answer_accuracy", {})).get("std"),
            "trainable_param_fraction": dict(metrics.get("trainable_param_fraction", {})).get("mean"),
            "trainable_param_fraction_std": dict(metrics.get("trainable_param_fraction", {})).get("std"),
            "wall_time_seconds_total": dict(metrics.get("wall_time_seconds_total", {})).get("mean"),
            "wall_time_seconds_total_std": dict(metrics.get("wall_time_seconds_total", {})).get("std"),
            "tokens_per_second_train": dict(metrics.get("tokens_per_second_train", {})).get("mean"),
            "tokens_per_second_train_std": dict(metrics.get("tokens_per_second_train", {})).get("std"),
        })

    with (output_dir / "report_table.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=REPORT_TABLE_FIELDS)
        writer.writeheader()
        writer.writerows(report_rows)


def main() -> None:
    args = parse_args()
    raw_config_paths = _collect_config_paths(args.configs, args.config_dir)
    config_paths = _filter_config_paths(raw_config_paths, args.preset_scope)
    config_paths = _filter_by_config_family(config_paths, args.config_family)
    if not config_paths:
        raise ValueError(
            f"No configs selected after applying preset scope '{args.preset_scope}' "
            f"and config family '{args.config_family}'."
        )

    runs: list[dict[str, object]] = []
    for config_path in config_paths:
        variants = _build_ablation_runs(config_path, args.run_scope)
        for derived_name, raw_cfg, derived_scope in variants:
            for seed in args.seeds:
                runtime = load_runtime_config_from_raw(raw_cfg)
                if args.run_scope == "confirmatory" and derived_scope != "confirmatory":
                    raise ValueError("confirmatory run-scope cannot execute ablation-derived configs")
                runtime.training.seed = seed
                runtime.dataset["settings"]["seed"] = seed
                runtime.output["dir"] = str(Path("outputs") / runtime.baseline)
                run_name = f"{Path(derived_name).stem}_seed{seed}"
                result = run_training_loop(
                    components=build_training_components(runtime),
                    run_name=run_name,
                    config_name=derived_name,
                )
                metrics = json.loads((result.output_dir / "metrics.json").read_text(encoding="utf-8"))
                metrics.setdefault("dataset_type", "primary")
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
                "run_scope": args.run_scope,
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
