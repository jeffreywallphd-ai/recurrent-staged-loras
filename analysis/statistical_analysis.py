"""Per-metric confirmatory and descriptive statistical analysis pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

from analysis.analysis_schema import (
    ALPHA,
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
    for idx, row in enumerate(runs):
        for required in REQUIRED_ID_COLUMNS:
            if row.get(required) in (None, ""):
                raise ValueError(f"Run row {idx} missing required identifier field: {required}")
    required_metrics = PRIMARY_CONFIRMATORY_OUTCOMES + SECONDARY_OUTCOMES + EFFICIENCY_OUTCOMES
    missing = [m for m in required_metrics if all(_coerce_float(r.get(m)) is None for r in runs)]
    if missing:
        raise ValueError(f"Required metric(s) missing from all runs: {', '.join(missing)}")


def _group_runs(runs: list[dict[str, object]]) -> dict[tuple[str, str], dict[int, dict[str, object]]]:
    grouped: dict[tuple[str, str], dict[int, dict[str, object]]] = {}
    for row in runs:
        arch = str(row["architecture_type"])
        baseline = str(row["baseline_name"])
        seed = int(row["seed"])
        grouped.setdefault((arch, baseline), {})[seed] = row
    return grouped


def _wilcoxon_signed_rank(differences: list[float]) -> float:
    non_zero = [d for d in differences if abs(d) > 0]
    n = len(non_zero)
    if n == 0:
        return 1.0
    abs_vals = sorted((abs(d), i) for i, d in enumerate(non_zero))
    ranks = [0.0] * n
    i = 0
    rank = 1
    while i < n:
        j = i
        while j + 1 < n and abs_vals[j + 1][0] == abs_vals[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i)) / 2.0
        for k in range(i, j + 1):
            ranks[abs_vals[k][1]] = avg_rank
        rank += j - i + 1
        i = j + 1

    w_plus = sum(r for r, d in zip(ranks, non_zero, strict=True) if d > 0)
    w_minus = sum(r for r, d in zip(ranks, non_zero, strict=True) if d < 0)
    w = min(w_plus, w_minus)

    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w == 0:
        return 1.0
    z = (w - mean_w) / math.sqrt(var_w)
    p = 2.0 * (1.0 - _normal_cdf(abs(z)))
    return max(0.0, min(1.0, p))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _paired_t_pvalue(differences: list[float]) -> float | None:
    n = len(differences)
    if n < 2:
        return None
    mu = mean(differences)
    var = sum((d - mu) ** 2 for d in differences) / (n - 1)
    if var == 0:
        return 1.0
    t_stat = mu / math.sqrt(var / n)
    p_norm_approx = 2.0 * (1.0 - _normal_cdf(abs(t_stat)))
    return max(0.0, min(1.0, p_norm_approx))


def _effect_size(differences: list[float]) -> float | None:
    if len(differences) < 2:
        return None
    mu = mean(differences)
    sd = math.sqrt(sum((d - mu) ** 2 for d in differences) / (len(differences) - 1))
    if sd == 0:
        return None
    return mu / sd


def _holm_adjust(rows: list[dict[str, object]], alpha: float = ALPHA) -> None:
    indexed = [(i, float(r["raw_p_value"])) for i, r in enumerate(rows)]
    indexed.sort(key=lambda x: x[1])
    m = len(indexed)
    adjusted = [1.0] * m
    for k, (_idx, p) in enumerate(indexed):
        adjusted[k] = min(1.0, (m - k) * p)
    for k in range(1, m):
        adjusted[k] = max(adjusted[k], adjusted[k - 1])
    for k, (orig_idx, _p) in enumerate(indexed):
        rows[orig_idx]["holm_adjusted_p_value"] = adjusted[k]
        rows[orig_idx]["reject_after_holm"] = adjusted[k] <= alpha


def _compare_metric(
    *,
    grouped: dict[tuple[str, str], dict[int, dict[str, object]]],
    architecture_type: str,
    baseline_a: str,
    baseline_b: str,
    metric_name: str,
    allow_unpaired: bool,
    primary: bool,
) -> dict[str, object]:
    key_a = (architecture_type, baseline_a)
    key_b = (architecture_type, baseline_b)
    if key_a not in grouped or key_b not in grouped:
        raise ValueError(f"Missing planned contrast groups for {architecture_type}: {baseline_a} vs {baseline_b}")

    rows_a = grouped[key_a]
    rows_b = grouped[key_b]
    overlap = sorted(set(rows_a) & set(rows_b))

    if not overlap:
        if primary and not allow_unpaired:
            raise ValueError(
                f"No overlapping seeds for confirmatory contrast {architecture_type}: {baseline_a} vs {baseline_b}. "
                "Re-run with aligned seeds or pass --allow-unpaired to explicitly downgrade."
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
            "metric_name": metric_name,
            "baseline_a": baseline_a,
            "baseline_b": baseline_b,
            "pairing_used": "unpaired",
            "seeds_included": [],
            "n_pairs": 0,
            "test_name": "descriptive_unpaired_difference",
            "raw_p_value": 1.0,
            "mean_difference": diff,
            "median_difference": diff,
            "effect_size": None,
            "direction_of_effect": "positive" if diff > 0 else ("negative" if diff < 0 else "neutral"),
            "paired_ttest_p_value": None,
            "notes": "No overlapping seeds; unpaired downgrade used.",
        }

    paired_values: list[tuple[int, float, float]] = []
    for seed in overlap:
        va = _coerce_float(rows_a[seed].get(metric_name))
        vb = _coerce_float(rows_b[seed].get(metric_name))
        if va is None or vb is None:
            continue
        paired_values.append((seed, va, vb))

    if not paired_values:
        raise ValueError(f"Metric {metric_name} missing on all overlapping seeds for {architecture_type}: {baseline_a} vs {baseline_b}")

    seeds = [s for s, _, _ in paired_values]
    differences = [va - vb for _, va, vb in paired_values]
    wilcoxon_p = _wilcoxon_signed_rank(differences)

    mean_diff = mean(differences)
    median_diff = median(differences)
    effect = _effect_size(differences)
    direction = "positive" if mean_diff > 0 else ("negative" if mean_diff < 0 else "neutral")

    return {
        "architecture_type": architecture_type,
        "metric_name": metric_name,
        "baseline_a": baseline_a,
        "baseline_b": baseline_b,
        "pairing_used": "paired_by_seed",
        "seeds_included": seeds,
        "n_pairs": len(seeds),
        "test_name": "wilcoxon_signed_rank",
        "raw_p_value": wilcoxon_p,
        "mean_difference": mean_diff,
        "median_difference": median_diff,
        "effect_size": effect,
        "direction_of_effect": direction,
        "paired_ttest_p_value": _paired_t_pvalue(differences),
        "notes": "Primary inference uses Wilcoxon signed-rank; paired t-test is sensitivity only.",
    }


def run_analysis(*, input_path: Path, output_dir: Path, allow_unpaired: bool) -> dict[str, object]:
    runs = _load_runs(input_path)
    _validate_runs(runs)
    grouped = _group_runs(runs)

    contrasts = build_confirmatory_contrasts()
    confirmatory_rows: list[dict[str, object]] = []
    for contrast in contrasts:
        for metric in PRIMARY_CONFIRMATORY_OUTCOMES:
            confirmatory_rows.append(
                _compare_metric(
                    grouped=grouped,
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
    for contrast in contrasts:
        for metric in SECONDARY_OUTCOMES:
            secondary_rows.append(
                _compare_metric(
                    grouped=grouped,
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
                    grouped=grouped,
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

    confirmatory_json.write_text(json.dumps(confirmatory_rows, indent=2), encoding="utf-8")
    secondary_json.write_text(json.dumps(secondary_rows, indent=2), encoding="utf-8")
    efficiency_json.write_text(json.dumps(efficiency_rows, indent=2), encoding="utf-8")

    with confirmatory_csv.open("w", encoding="utf-8", newline="") as fp:
        fieldnames = [
            "architecture_type",
            "metric_name",
            "baseline_a",
            "baseline_b",
            "pairing_used",
            "seeds_included",
            "n_pairs",
            "test_name",
            "raw_p_value",
            "holm_adjusted_p_value",
            "reject_after_holm",
            "mean_difference",
            "median_difference",
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
        },
    }


def main() -> None:
    args = parse_args()
    result = run_analysis(input_path=Path(args.input), output_dir=Path(args.output_dir), allow_unpaired=args.allow_unpaired)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
