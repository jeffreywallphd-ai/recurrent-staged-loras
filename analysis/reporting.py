"""Reporting helpers for statistical analysis outputs."""

from __future__ import annotations

from pathlib import Path

from analysis.analysis_schema import CONFIRMATORY_FWER_METHOD, PRIMARY_CONFIRMATORY_OUTCOMES


def _fmt_p(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g}"


def write_markdown_report(
    *,
    output_path: Path,
    confirmatory_rows: list[dict[str, object]],
    secondary_rows: list[dict[str, object]],
    efficiency_rows: list[dict[str, object]],
) -> None:
    lines: list[str] = []
    lines.append("# Statistical Analysis Report")
    lines.append("")
    lines.append("## Confirmatory analysis plan")
    lines.append("")
    lines.append("- Primary outcomes: " + ", ".join(f"`{m}`" for m in PRIMARY_CONFIRMATORY_OUTCOMES))
    lines.append("- Planned contrasts: `stage_specialized_recurrence` vs each comparator (`standard_lora`, `shared_recurrence`, `latent_refiner_only`) within each architecture (`dense`, `moe`).")
    lines.append(f"- Family-wise error control: `{CONFIRMATORY_FWER_METHOD}` correction across the full confirmatory family of primary-outcome tests.")
    lines.append("- Pairing strategy: paired-by-seed when overlap exists; explicit downgrade notes if unpaired analysis is used.")
    lines.append("")

    lines.append("## Corrected confirmatory results")
    lines.append("")
    header = "| architecture | metric | contrast | pairing | n_pairs | raw_p | holm_p | reject_after_holm | mean_diff | median_diff | effect_size |"
    sep = "|---|---|---|---|---:|---:|---:|---|---:|---:|---:|"
    lines.extend([header, sep])
    for row in confirmatory_rows:
        lines.append(
            "| {architecture} | {metric} | {a} - {b} | {pairing} | {n} | {raw_p} | {holm_p} | {reject} | {mean:.6g} | {median:.6g} | {effect} |".format(
                architecture=row["architecture_type"],
                metric=row["metric_name"],
                a=row["baseline_a"],
                b=row["baseline_b"],
                pairing=row["pairing_used"],
                n=row["n_pairs"],
                raw_p=_fmt_p(row.get("raw_p_value")),
                holm_p=_fmt_p(row.get("holm_adjusted_p_value")),
                reject="yes" if row.get("reject_after_holm") else "no",
                mean=float(row["mean_difference"]),
                median=float(row["median_difference"]),
                effect="n/a" if row.get("effect_size") is None else f"{float(row['effect_size']):.6g}",
            )
        )
    lines.append("")

    lines.append("## Main claim summary (table-driven)")
    lines.append("")
    for architecture in ["dense", "moe"]:
        lines.append(f"### {architecture}")
        lines.append("")
        subset = [r for r in confirmatory_rows if r["architecture_type"] == architecture]
        for comparator in ["standard_lora", "shared_recurrence", "latent_refiner_only"]:
            comp_rows = [r for r in subset if r["baseline_b"] == comparator]
            wins = sum(1 for r in comp_rows if r.get("reject_after_holm") and float(r["mean_difference"]) > 0)
            total = len(comp_rows)
            lines.append(f"- `stage_specialized_recurrence` vs `{comparator}`: {wins}/{total} primary outcomes improved with Holm-adjusted significance.")
        lines.append("")

    lines.append("## Secondary outcomes (descriptive)")
    lines.append("")
    lines.append("Secondary outcomes are analyzed separately and are not included in confirmatory Holm correction.")
    for row in secondary_rows:
        lines.append(
            f"- {row['architecture_type']} | {row['metric_name']} | {row['baseline_a']} - {row['baseline_b']} | mean_diff={float(row['mean_difference']):.6g}, median_diff={float(row['median_difference']):.6g}."
        )
    lines.append("")

    lines.append("## Efficiency outcomes (descriptive)")
    lines.append("")
    lines.append("Efficiency/comparative outcomes are descriptive by default and not treated as confirmatory tests.")
    for row in efficiency_rows:
        lines.append(
            f"- {row['architecture_type']} | {row['metric_name']} | {row['baseline_a']} - {row['baseline_b']} | mean_diff={float(row['mean_difference']):.6g}, median_diff={float(row['median_difference']):.6g}."
        )
    lines.append("")

    lines.append("## Pairing and limitations")
    lines.append("")
    lines.append("- Seed overlap is required for paired confirmatory tests; rows include explicit pairing metadata and notes.")
    lines.append("- Wilcoxon signed-rank is used as the primary confirmatory test; paired t-test p-values are sensitivity checks only.")
    lines.append("- Mixed metric types are intentionally handled per metric rather than through MANOVA/PERMANOVA.")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
