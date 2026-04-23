"""Confirmatory/descriptive statistics for experiment outputs.

Pipeline role:
- validates run identity + metric completeness assumptions,
- builds planned baseline contrasts,
- runs paired inferential tests for confirmatory primary outcomes,
- emits descriptive secondary/efficiency summaries and report artifacts.

Key invariant: paired confirmatory comparisons require matching dataset identity
(`dataset_fingerprint` and `eval_sample_ids_hash`) for each paired seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from analysis.analysis_schema import (
    ALPHA,
    CONFIRMATORY_FWER_METHOD,
    EFFICIENCY_OUTCOMES,
    PRIMARY_CONFIRMATORY_OUTCOMES,
    REQUIRED_ID_COLUMNS,
    SECONDARY_OUTCOMES,
    build_confirmatory_contrasts,
)
from analysis.reporting import write_markdown_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run statistical analysis on outputs summary artifacts")
    parser.add_argument("--input", default="outputs/summary.json", help="Path to outputs/summary.json or outputs/summary.csv")
    parser.add_argument("--output-dir", default="outputs", help="Output directory for statistical artifacts")
    parser.add_argument("--allow-unpaired", action="store_true", help="Allow unpaired fallback when seed pairing fails")
    parser.add_argument("--dataset-scope", choices=["primary", "external", "all"], default="primary")
    parser.add_argument("--compute-controlled-only", action="store_true")
    parser.add_argument("--ablation-only", action="store_true")
    parser.add_argument("--allow-ablations-in-analysis", action="store_true")
    parser.add_argument("--allow-pilot-runs-in-analysis", action="store_true")
    parser.add_argument("--allow-external-in-confirmatory", action="store_true")
    return parser.parse_args()


def _load_runs(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "runs" in payload:
            runs = payload["runs"]
        elif isinstance(payload, list):
            runs = payload
        else:
            raise ValueError("JSON input must be a summary object with `runs` or a list of run rows")
        if not isinstance(runs, list):
            raise ValueError("Invalid runs payload in JSON input")
        return [dict(r) for r in runs]

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            rows = [dict(r) for r in reader]
        run_rows = [r for r in rows if r.get("row_type", "run") == "run"]
        return run_rows

    raise ValueError("Unsupported input format; expected .json or .csv")


def _coerce_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validate_runs(runs: list[dict[str, object]]) -> None:
    """Validate required identifier and metric availability constraints."""
    for idx, row in enumerate(runs):
        for required in REQUIRED_ID_COLUMNS:
            if row.get(required) in (None, ""):
                raise ValueError(f"Run row {idx} missing required identifier field: {required}")
    required_metrics = PRIMARY_CONFIRMATORY_OUTCOMES + SECONDARY_OUTCOMES + EFFICIENCY_OUTCOMES
    missing = [m for m in required_metrics if all(_coerce_float(r.get(m)) is None for r in runs)]
    if missing:
        raise ValueError(f"Required metric(s) missing from all runs: {', '.join(missing)}")


def _is_ablation_row(row: dict[str, object]) -> bool:
    return row.get("ablation_recurrent_steps") not in (None, "") or row.get("ablation_lora_rank") not in (None, "")


def _is_pilot_row(row: dict[str, object]) -> bool:
    config_name = str(row.get("config_name", ""))
    return config_name.endswith("_pilot.json")


def _require_confirmatory_purity(
    runs: list[dict[str, object]],
    *,
    allow_ablations_in_analysis: bool,
    allow_pilot_runs_in_analysis: bool,
    allow_external_in_confirmatory: bool,
) -> None:
    """Reject non-confirmatory rows unless explicit overrides are provided."""
    if not allow_ablations_in_analysis:
        ablation_runs = [r for r in runs if _is_ablation_row(r)]
        if ablation_runs:
            raise ValueError(
                "Confirmatory analysis received ablation-derived runs. "
                "Set --allow-ablations-in-analysis to include them explicitly."
            )
    if not allow_pilot_runs_in_analysis:
        pilot_runs = [r for r in runs if _is_pilot_row(r)]
        if pilot_runs:
            raise ValueError(
                "Confirmatory analysis received pilot runs (*_pilot.json). "
                "Set --allow-pilot-runs-in-analysis to include them explicitly."
            )
    if not allow_external_in_confirmatory:
        external = [r for r in runs if str(r.get("dataset_type", "primary")) == "external"]
        if external:
            raise ValueError(
                "Confirmatory analysis received external-evaluation rows. "
                "Use descriptive external mode (e.g., --dataset-scope external) or set --allow-external-in-confirmatory."
            )


ConditionSeedKey = tuple[str, str, str, str, str, int]
ConditionFamilyKey = tuple[str, str, str, str]
EXTERNAL_IDENTITY_FIELDS = [
    "dataset_name",
    "dataset_type",
    "dataset_split",
    "dataset_seed",
    "dataset_subset_size",
    "dataset_eval_fraction",
    "dataset_fingerprint",
    "train_sample_ids_hash",
    "eval_sample_ids_hash",
]


def _group_runs(runs: list[dict[str, object]]) -> dict[ConditionSeedKey, dict[str, object]]:
    grouped: dict[ConditionSeedKey, dict[str, object]] = {}
    for row in runs:
        key = (
            str(row["architecture_type"]),
            str(row["baseline_name"]),
            str(row["model_name"]),
            str(row["dataset_name"]),
            str(row["config_name"]),
            int(row["seed"]),
        )
        if key in grouped:
            raise ValueError(
                "Duplicate run for condition+seed key detected: "
                f"architecture_type={key[0]}, baseline_name={key[1]}, model_name={key[2]}, "
                f"dataset_name={key[3]}, config_name={key[4]}, seed={key[5]}"
            )
        grouped[key] = row
    return grouped


def _extract_condition_families(
    grouped: dict[ConditionSeedKey, dict[str, object]], architecture_type: str, baseline_name: str
) -> dict[tuple[str, str], dict[int, dict[str, object]]]:
    families: dict[tuple[str, str], dict[int, dict[str, object]]] = {}
    for (arch, baseline, model, dataset, config, seed), row in grouped.items():
        if arch != architecture_type or baseline != baseline_name:
            continue
        family_key = (model, dataset)
        families.setdefault(family_key, {})[seed] = row
    return families


def _require_homogeneous_family(
    *,
    grouped: dict[ConditionSeedKey, dict[str, object]],
    architecture_type: str,
    baseline_a: str,
    baseline_b: str,
) -> tuple[ConditionFamilyKey, dict[int, dict[str, object]], dict[int, dict[str, object]]]:
    """Require exactly one comparable (model,dataset) family per baseline."""
    families_a = _extract_condition_families(grouped, architecture_type, baseline_a)
    families_b = _extract_condition_families(grouped, architecture_type, baseline_b)

    if not families_a or not families_b:
        raise ValueError(f"Missing planned contrast groups for {architecture_type}: {baseline_a} vs {baseline_b}")

    if len(families_a) != 1 or len(families_b) != 1:
        raise ValueError(
            "Heterogeneous comparison group detected for planned confirmatory contrast "
            f"{architecture_type}: {baseline_a} vs {baseline_b}. "
            f"Expected one (model_name, dataset_name) family per baseline, got "
            f"{sorted(families_a)} for {baseline_a} and {sorted(families_b)} for {baseline_b}."
        )

    family_a = next(iter(families_a))
    family_b = next(iter(families_b))
    if family_a != family_b:
        mismatch_parts: list[str] = []
        if family_a[0] != family_b[0]:
            mismatch_parts.append(f"model_name differs ({family_a[0]} vs {family_b[0]})")
        if family_a[1] != family_b[1]:
            mismatch_parts.append(f"dataset_name differs ({family_a[1]} vs {family_b[1]})")
        mismatch_text = "; ".join(mismatch_parts) if mismatch_parts else "unknown family mismatch"
        raise ValueError(
            "Non-homogeneous planned contrast detected: confirmatory contrasts require matching "
            "architecture_type, model_name, and dataset_name across baselines. "
            f"Invalid contrast {architecture_type}: {baseline_a} vs {baseline_b} because {mismatch_text}."
        )

    model_name, dataset_name = family_a
    return (
        (architecture_type, baseline_a, model_name, dataset_name),
        families_a[family_a],
        families_b[family_b],
    )


def _get_stats_backend() -> Any:
    try:
        from scipy import stats
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "scipy is required for confirmatory inference (Wilcoxon signed-rank and paired t-test sensitivity). "
            "Install scipy to run analysis/statistical_analysis.py."
        ) from exc
    return stats


def _wilcoxon_signed_rank(differences: list[float]) -> float:
    non_zero = [d for d in differences if abs(d) > 0]
    if not non_zero:
        return 1.0
    stats = _get_stats_backend()
    result = stats.wilcoxon(non_zero, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(result.pvalue)


def _paired_t_pvalue(differences: list[float]) -> float | None:
    if len(differences) < 2:
        return None
    stats = _get_stats_backend()
    zeros = [0.0] * len(differences)
    result = stats.ttest_rel(differences, zeros, alternative="two-sided")
    return float(result.pvalue)


def _effect_size(differences: list[float]) -> float | None:
    if len(differences) < 2:
        return None
    mu = mean(differences)
    sd = math.sqrt(sum((d - mu) ** 2 for d in differences) / (len(differences) - 1))
    if sd == 0:
        return None
    return mu / sd


def _paired_mean_difference_bootstrap_ci(
    differences: list[float], *, alpha: float = ALPHA, n_resamples: int = 5000, seed: int = 0
) -> tuple[float | None, float | None]:
    if not differences:
        return (None, None)
    rng = random.Random(seed)
    n = len(differences)
    resampled_means = [mean(differences[rng.randrange(n)] for _ in range(n)) for _ in range(n_resamples)]
    resampled_means.sort()
    lower_idx = int((alpha / 2) * (n_resamples - 1))
    upper_idx = int((1 - alpha / 2) * (n_resamples - 1))
    return (float(resampled_means[lower_idx]), float(resampled_means[upper_idx]))


def _holm_adjust(rows: list[dict[str, object]], alpha: float = ALPHA) -> None:
    eligible = [
        (i, float(r["raw_p_value"]))
        for i, r in enumerate(rows)
        if r.get("analysis_tier") == "confirmatory" and r.get("raw_p_value") is not None
    ]
    if not eligible:
        return
    eligible.sort(key=lambda x: x[1])
    m = len(eligible)
    adjusted = [1.0] * m
    for k, (_idx, p) in enumerate(eligible):
        adjusted[k] = min(1.0, (m - k) * p)
    for k in range(1, m):
        adjusted[k] = max(adjusted[k], adjusted[k - 1])
    for k, (orig_idx, _p) in enumerate(eligible):
        rows[orig_idx]["holm_adjusted_p_value"] = adjusted[k]
        rows[orig_idx]["reject_after_holm"] = adjusted[k] <= alpha


def _compare_metric(
    *,
    grouped: dict[ConditionSeedKey, dict[str, object]],
    architecture_type: str,
    baseline_a: str,
    baseline_b: str,
    metric_name: str,
    allow_unpaired: bool,
    primary: bool,
) -> dict[str, object]:
    """Compare one metric for one planned contrast.

    Returns a row compatible with confirmatory/secondary output artifacts.
    """
    (family, rows_a, rows_b) = _require_homogeneous_family(
        grouped=grouped,
        architecture_type=architecture_type,
        baseline_a=baseline_a,
        baseline_b=baseline_b,
    )
    _, _, model_name, dataset_name = family
    overlap = sorted(set(rows_a) & set(rows_b))
    config_name_a = str(next(iter(rows_a.values())).get("config_name", ""))
    config_name_b = str(next(iter(rows_b.values())).get("config_name", ""))

    if not overlap:
        if primary and not allow_unpaired:
            raise ValueError(
                f"No overlapping seeds for confirmatory contrast {architecture_type}: {baseline_a} vs {baseline_b} "
                f"(model={model_name}, dataset={dataset_name}, config_a={config_name_a}, config_b={config_name_b}). "
                "Re-run with aligned seeds or pass --allow-unpaired to explicitly downgrade to descriptive."
            )
        all_a = sorted(rows_a)
        all_b = sorted(rows_b)
        values_a = [_coerce_float(rows_a[s].get(metric_name)) for s in all_a]
        values_b = [_coerce_float(rows_b[s].get(metric_name)) for s in all_b]
        values_a = [v for v in values_a if v is not None]
        values_b = [v for v in values_b if v is not None]
        if not values_a or not values_b:
            raise ValueError(f"Metric {metric_name} entirely missing for {architecture_type}: {baseline_a} vs {baseline_b}")
        diff = mean(values_a) - mean(values_b)
        return {
            "architecture_type": architecture_type,
            "model_name": model_name,
            "dataset_name": dataset_name,
            "config_name_a": config_name_a,
            "config_name_b": config_name_b,
            "metric_name": metric_name,
            "baseline_a": baseline_a,
            "baseline_b": baseline_b,
            "analysis_tier": "descriptive_downgraded" if primary else "descriptive",
            "downgraded_to_descriptive": bool(primary),
            "pairing_used": "unpaired",
            "seeds_included": [],
            "n_pairs": 0,
            "test_name": "descriptive_unpaired_difference",
            "raw_p_value": None,
            "holm_adjusted_p_value": None,
            "reject_after_holm": None,
            "mean_difference": diff,
            "median_difference": diff,
            "mean_difference_ci_low": None,
            "mean_difference_ci_high": None,
            "effect_size": None,
            "direction_of_effect": "positive" if diff > 0 else ("negative" if diff < 0 else "neutral"),
            "paired_ttest_p_value": None,
            "notes": "No overlapping seeds; treated as descriptive-only unpaired summary and excluded from Holm correction.",
        }

    paired_values: list[tuple[int, float, float]] = []
    for seed in overlap:
        fp_a = rows_a[seed].get("dataset_fingerprint")
        fp_b = rows_b[seed].get("dataset_fingerprint")
        eval_hash_a = rows_a[seed].get("eval_sample_ids_hash")
        eval_hash_b = rows_b[seed].get("eval_sample_ids_hash")
        # Dataset identity must match across paired seeds. Without this, paired
        # inferential tests would compare different evaluation cohorts.
        if fp_a != fp_b or eval_hash_a != eval_hash_b:
            scope = "external" if (rows_a[seed].get("dataset_scope") == "external" or rows_b[seed].get("dataset_scope") == "external") else "primary"
            raise ValueError(
                f"Paired runs are not comparable in {scope} scope: dataset identity mismatch for "
                f"{architecture_type}: {baseline_a} vs {baseline_b}, seed={seed}. "
                f"dataset_fingerprint ({fp_a} vs {fp_b}), eval_sample_ids_hash ({eval_hash_a} vs {eval_hash_b})."
            )
        va = _coerce_float(rows_a[seed].get(metric_name))
        vb = _coerce_float(rows_b[seed].get(metric_name))
        if va is None or vb is None:
            raise ValueError(
                "Confirmatory comparison has partial metric missingness for paired seed "
                f"{seed} on metric '{metric_name}' ({architecture_type}: {baseline_a} vs {baseline_b}). "
                "Do not drop incomplete rows from confirmatory inference."
            )
        paired_values.append((seed, va, vb))

    if not paired_values:
        raise ValueError(f"Metric {metric_name} missing on all overlapping seeds for {architecture_type}: {baseline_a} vs {baseline_b}")

    seeds = [s for s, _, _ in paired_values]
    differences = [va - vb for _, va, vb in paired_values]
    wilcoxon_p = _wilcoxon_signed_rank(differences)

    mean_diff = mean(differences)
    median_diff = median(differences)
    ci_low, ci_high = _paired_mean_difference_bootstrap_ci(differences)
    effect = _effect_size(differences)
    direction = "positive" if mean_diff > 0 else ("negative" if mean_diff < 0 else "neutral")

    return {
        "architecture_type": architecture_type,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "config_name_a": config_name_a,
        "config_name_b": config_name_b,
        "metric_name": metric_name,
        "baseline_a": baseline_a,
        "baseline_b": baseline_b,
        "analysis_tier": "confirmatory" if primary else "descriptive",
        "downgraded_to_descriptive": False,
        "pairing_used": "paired_by_seed",
        "seeds_included": seeds,
        "n_pairs": len(seeds),
        "test_name": "wilcoxon_signed_rank_scipy",
        "raw_p_value": wilcoxon_p,
        "mean_difference": mean_diff,
        "median_difference": median_diff,
        "mean_difference_ci_low": ci_low,
        "mean_difference_ci_high": ci_high,
        "effect_size": effect,
        "direction_of_effect": direction,
        "paired_ttest_p_value": _paired_t_pvalue(differences),
        "notes": "Primary inference uses scipy.stats.wilcoxon (two-sided, zero_method='wilcox'); scipy.stats.ttest_rel is sensitivity-only.",
    }


def run_analysis(
    *,
    input_path: Path,
    output_dir: Path,
    allow_unpaired: bool,
    dataset_scope: str = "primary",
    compute_controlled_only: bool = False,
    ablation_only: bool = False,
    allow_ablations_in_analysis: bool = False,
    allow_pilot_runs_in_analysis: bool = False,
    allow_external_in_confirmatory: bool = False,
) -> dict[str, object]:
    """Run the full statistical pipeline and write analysis artifacts."""
    def _require_external_identity_fields(*, row: dict[str, object], row_label: str) -> None:
        missing = [field for field in EXTERNAL_IDENTITY_FIELDS if row.get(field) in (None, "")]
        if missing:
            raise ValueError(f"External analysis row '{row_label}' missing required external identity fields: {', '.join(missing)}")

    source_runs = _load_runs(input_path)
    runs = [dict(r) for r in source_runs]
    if dataset_scope in {"external", "all"}:
        external_rows: list[dict[str, object]] = []
        for row in source_runs:
            external_eval = row.get("external_eval", {})
            if not isinstance(external_eval, dict):
                continue
            for external_dataset_name, payload in external_eval.items():
                if not isinstance(payload, dict):
                    continue
                missing_payload_fields = [field for field in EXTERNAL_IDENTITY_FIELDS if payload.get(field) in (None, "")]
                if missing_payload_fields:
                    raise ValueError(
                        "External evaluation payload missing required identity fields for "
                        f"dataset '{external_dataset_name}' in run '{row.get('run_name', '<unknown>')}': "
                        + ", ".join(missing_payload_fields)
                    )
                flattened = dict(row)
                for field in EXTERNAL_IDENTITY_FIELDS:
                    # External-analysis rows explicitly replace identity fields
                    # with external payload values to avoid primary-dataset bleed.
                    flattened[field] = payload.get(field)
                flattened["dataset_scope"] = "external"
                flattened["analysis_tier_default"] = "descriptive"
                flattened["final_eval_loss"] = payload.get("eval_loss")
                for metric_name in (
                    "stage_2_token_accuracy",
                    "stage_3_token_accuracy",
                    "final_answer_accuracy",
                    "final_answer_exact_match",
                    "normalized_numeric_answer_accuracy",
                ):
                    flattened[metric_name] = payload.get(metric_name)
                _require_external_identity_fields(
                    row=flattened,
                    row_label=f"{row.get('run_name', '<unknown>')}::{external_dataset_name}",
                )
                external_rows.append(flattened)
        runs.extend(external_rows)
    if dataset_scope == "primary":
        runs = [r for r in runs if r.get("dataset_scope") != "external"]
    elif dataset_scope == "external":
        runs = [r for r in runs if r.get("dataset_scope") == "external"]
    if compute_controlled_only:
        runs = [r for r in runs if bool(r.get("compute_control_enabled"))]
    if ablation_only:
        runs = [r for r in runs if (r.get("ablation_recurrent_steps") not in (None, "") or r.get("ablation_lora_rank") not in (None, ""))]
    for row in runs:
        row.setdefault("dataset_type", "primary")
    _validate_runs(runs)
    grouped = _group_runs(runs)
    primary_rows = [r for r in runs if r.get("dataset_scope") != "external"]
    external_rows = [r for r in runs if r.get("dataset_scope") == "external"]
    if dataset_scope in {"external", "all"}:
        for row in external_rows:
            _require_external_identity_fields(
                row=row,
                row_label=f"{row.get('run_name', '<unknown>')}::{row.get('dataset_name', '<unknown>')}",
            )
    primary_grouped = _group_runs(primary_rows)
    external_grouped = _group_runs(external_rows)

    contrasts = build_confirmatory_contrasts()
    confirmatory_rows: list[dict[str, object]] = []
    confirmatory_enabled = dataset_scope in {"primary", "all"}
    if confirmatory_enabled:
        # Confirmatory inference is primary-dataset scoped by default; external
        # datasets remain descriptive unless explicitly opted in.
        _require_confirmatory_purity(
            primary_rows,
            allow_ablations_in_analysis=allow_ablations_in_analysis,
            allow_pilot_runs_in_analysis=allow_pilot_runs_in_analysis,
            allow_external_in_confirmatory=allow_external_in_confirmatory,
        )
        for contrast in contrasts:
            for metric in PRIMARY_CONFIRMATORY_OUTCOMES:
                confirmatory_rows.append(
                    _compare_metric(
                        grouped=primary_grouped,
                        architecture_type=contrast.architecture_type,
                        baseline_a=contrast.baseline_a,
                        baseline_b=contrast.baseline_b,
                        metric_name=metric,
                        allow_unpaired=allow_unpaired,
                        primary=True,
                    )
                )

    _holm_adjust(confirmatory_rows)

    secondary_rows: list[dict[str, object]] = []
    efficiency_rows: list[dict[str, object]] = []
    descriptive_grouped_sets: list[dict[ConditionSeedKey, dict[str, object]]] = []
    if dataset_scope in {"primary", "all"}:
        descriptive_grouped_sets.append(primary_grouped)
    if dataset_scope in {"external", "all"}:
        descriptive_grouped_sets.append(external_grouped)

    for grouped_set in descriptive_grouped_sets:
        if not grouped_set:
            continue
        for contrast in contrasts:
            for metric in SECONDARY_OUTCOMES:
                secondary_rows.append(
                    _compare_metric(
                        grouped=grouped_set,
                        architecture_type=contrast.architecture_type,
                        baseline_a=contrast.baseline_a,
                        baseline_b=contrast.baseline_b,
                        metric_name=metric,
                        allow_unpaired=allow_unpaired,
                        primary=False,
                    )
                )
            for metric in EFFICIENCY_OUTCOMES:
                efficiency_rows.append(
                    _compare_metric(
                        grouped=grouped_set,
                        architecture_type=contrast.architecture_type,
                        baseline_a=contrast.baseline_a,
                        baseline_b=contrast.baseline_b,
                        metric_name=metric,
                        allow_unpaired=allow_unpaired,
                        primary=False,
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    confirmatory_json = output_dir / "statistical_analysis_confirmatory.json"
    secondary_json = output_dir / "statistical_analysis_secondary.json"
    efficiency_json = output_dir / "statistical_analysis_efficiency.json"
    confirmatory_csv = output_dir / "statistical_analysis_confirmatory.csv"
    report_path = output_dir / "statistical_analysis_report.md"
    metadata_path = output_dir / "statistical_analysis_metadata.json"

    confirmatory_json.write_text(json.dumps(confirmatory_rows, indent=2), encoding="utf-8")
    secondary_json.write_text(json.dumps(secondary_rows, indent=2), encoding="utf-8")
    efficiency_json.write_text(json.dumps(efficiency_rows, indent=2), encoding="utf-8")

    with confirmatory_csv.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = [
            "architecture_type",
            "model_name",
            "dataset_name",
            "config_name_a",
            "config_name_b",
            "metric_name",
            "baseline_a",
            "baseline_b",
            "analysis_tier",
            "downgraded_to_descriptive",
            "pairing_used",
            "seeds_included",
            "n_pairs",
            "test_name",
            "raw_p_value",
            "holm_adjusted_p_value",
            "reject_after_holm",
            "mean_difference",
            "median_difference",
            "mean_difference_ci_low",
            "mean_difference_ci_high",
            "effect_size",
            "direction_of_effect",
            "paired_ttest_p_value",
            "notes",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in confirmatory_rows:
            writer.writerow({**row, "seeds_included": ",".join(str(s) for s in row["seeds_included"])})

    write_markdown_report(
        output_path=report_path,
        confirmatory_rows=confirmatory_rows,
        secondary_rows=secondary_rows,
        efficiency_rows=efficiency_rows,
    )

    metadata = {
        "primary_confirmatory_outcomes": PRIMARY_CONFIRMATORY_OUTCOMES,
        "planned_contrasts": [asdict(c) for c in contrasts],
        "family_wise_error_method": CONFIRMATORY_FWER_METHOD,
        "alpha": ALPHA,
        "statistical_test_backend": {
            "library": "scipy",
            "confirmatory_test": "scipy.stats.wilcoxon (two-sided, zero_method='wilcox', method='auto')",
            "sensitivity_test": "scipy.stats.ttest_rel (two-sided)",
            "ci_method": "bootstrap percentile CI for paired mean difference (5000 resamples, RNG seed=0)",
        },
        "allow_unpaired": allow_unpaired,
        "allow_ablations_in_analysis": allow_ablations_in_analysis,
        "allow_pilot_runs_in_analysis": allow_pilot_runs_in_analysis,
        "allow_external_in_confirmatory": allow_external_in_confirmatory,
        "dataset_scope": dataset_scope,
        "confirmatory_enabled": confirmatory_enabled,
        "any_rows_downgraded_to_descriptive": any(bool(r.get("downgraded_to_descriptive")) for r in confirmatory_rows),
        "any_pairing_failures": any(r.get("pairing_used") == "unpaired" for r in confirmatory_rows),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "input": str(input_path),
        "n_runs": len(runs),
        "confirmatory_rows": len(confirmatory_rows),
        "secondary_rows": len(secondary_rows),
        "efficiency_rows": len(efficiency_rows),
        "confirmatory_contrasts": [asdict(c) for c in contrasts],
        "outputs": {
            "confirmatory_json": str(confirmatory_json),
            "confirmatory_csv": str(confirmatory_csv),
            "secondary_json": str(secondary_json),
            "efficiency_json": str(efficiency_json),
            "report_md": str(report_path),
            "metadata_json": str(metadata_path),
        },
    }


def main() -> None:
    args = parse_args()
    result = run_analysis(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        allow_unpaired=args.allow_unpaired,
        dataset_scope=args.dataset_scope,
        compute_controlled_only=args.compute_controlled_only,
        ablation_only=args.ablation_only,
        allow_ablations_in_analysis=args.allow_ablations_in_analysis,
        allow_pilot_runs_in_analysis=args.allow_pilot_runs_in_analysis,
        allow_external_in_confirmatory=args.allow_external_in_confirmatory,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
