import json
from pathlib import Path

import pytest

from analysis.analysis_schema import PRIMARY_CONFIRMATORY_OUTCOMES
from analysis.statistical_analysis import run_analysis


def _make_run(*, arch: str, baseline: str, seed: int, offset: float) -> dict[str, object]:
    return {
        "run_name": f"{arch}_{baseline}_{seed}",
        "config_name": f"{arch}_{baseline}.json",
        "baseline_name": baseline,
        "dataset_name": "metamath_qa",
        "seed": seed,
        "architecture_type": arch,
        "final_answer_accuracy": 0.6 + offset,
        "final_answer_exact_match": 0.55 + offset,
        "normalized_numeric_answer_accuracy": 0.58 + offset,
        "final_eval_loss": 1.0 - offset,
        "eval_perplexity": 3.0 - offset,
        "stage_2_token_accuracy": 0.7 + offset,
        "stage_3_token_accuracy": 0.65 + offset,
        "wall_time_seconds_total": 100.0 - (offset * 10),
        "tokens_per_second_train": 200.0 + (offset * 10),
        "trainable_param_fraction": 0.02,
        "effective_forward_passes_per_example": 3.0,
    }


def _write_summary(tmp_path: Path, runs: list[dict[str, object]]) -> Path:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"runs": runs, "aggregates": []}))
    return path


def test_grouping_pairing_and_outputs(tmp_path: Path) -> None:
    runs = []
    for arch in ["dense", "moe"]:
        for seed in [11, 22, 33]:
            runs.append(_make_run(arch=arch, baseline="stage_specialized_recurrence", seed=seed, offset=0.2))
            runs.append(_make_run(arch=arch, baseline="standard_lora", seed=seed, offset=0.0))
            runs.append(_make_run(arch=arch, baseline="shared_recurrence", seed=seed, offset=0.05))
            runs.append(_make_run(arch=arch, baseline="latent_refiner_only", seed=seed, offset=0.1))

    summary = _write_summary(tmp_path, runs)
    out_dir = tmp_path / "outputs"
    result = run_analysis(input_path=summary, output_dir=out_dir, allow_unpaired=False)

    assert result["confirmatory_rows"] == 18
    confirmatory = json.loads((out_dir / "statistical_analysis_confirmatory.json").read_text())
    assert all(row["pairing_used"] == "paired_by_seed" for row in confirmatory)
    assert all(row["n_pairs"] == 3 for row in confirmatory)
    assert {tuple(row["seeds_included"]) for row in confirmatory} == {(11, 22, 33)}

    assert (out_dir / "statistical_analysis_secondary.json").exists()
    assert (out_dir / "statistical_analysis_efficiency.json").exists()


def test_holm_applied_across_confirmatory_family(tmp_path: Path) -> None:
    runs = []
    for arch in ["dense", "moe"]:
        for seed in [1, 2, 3]:
            runs.append(_make_run(arch=arch, baseline="stage_specialized_recurrence", seed=seed, offset=0.0))
            runs.append(_make_run(arch=arch, baseline="standard_lora", seed=seed, offset=0.0))
            runs.append(_make_run(arch=arch, baseline="shared_recurrence", seed=seed, offset=0.0))
            runs.append(_make_run(arch=arch, baseline="latent_refiner_only", seed=seed, offset=0.0))

    summary = _write_summary(tmp_path, runs)
    out_dir = tmp_path / "outputs"
    run_analysis(input_path=summary, output_dir=out_dir, allow_unpaired=False)
    rows = json.loads((out_dir / "statistical_analysis_confirmatory.json").read_text())
    assert len(rows) == 2 * 3 * len(PRIMARY_CONFIRMATORY_OUTCOMES)
    assert all("holm_adjusted_p_value" in row for row in rows)


def test_missing_required_metric_raises_clear_error(tmp_path: Path) -> None:
    runs = [_make_run(arch="dense", baseline="stage_specialized_recurrence", seed=11, offset=0.2)]
    for r in runs:
        r.pop("final_answer_accuracy")
    summary = _write_summary(tmp_path, runs)

    with pytest.raises(ValueError, match=r"Required metric\(s\) missing"):
        run_analysis(input_path=summary, output_dir=tmp_path / "outputs", allow_unpaired=False)


def test_pairing_failure_fails_loudly_without_unpaired_flag(tmp_path: Path) -> None:
    runs = []
    for seed in [11, 22, 33]:
        runs.append(_make_run(arch="dense", baseline="stage_specialized_recurrence", seed=seed, offset=0.2))
    for seed in [44, 55, 66]:
        runs.append(_make_run(arch="dense", baseline="standard_lora", seed=seed, offset=0.0))
    # Include remaining required groups for contrast validation.
    for seed in [11, 22, 33]:
        runs.append(_make_run(arch="dense", baseline="shared_recurrence", seed=seed, offset=0.05))
        runs.append(_make_run(arch="dense", baseline="latent_refiner_only", seed=seed, offset=0.1))
        runs.append(_make_run(arch="moe", baseline="stage_specialized_recurrence", seed=seed, offset=0.2))
        runs.append(_make_run(arch="moe", baseline="standard_lora", seed=seed, offset=0.0))
        runs.append(_make_run(arch="moe", baseline="shared_recurrence", seed=seed, offset=0.05))
        runs.append(_make_run(arch="moe", baseline="latent_refiner_only", seed=seed, offset=0.1))

    summary = _write_summary(tmp_path, runs)
    with pytest.raises(ValueError, match="No overlapping seeds"):
        run_analysis(input_path=summary, output_dir=tmp_path / "outputs", allow_unpaired=False)


def test_markdown_report_contains_key_sections(tmp_path: Path) -> None:
    runs = []
    for arch in ["dense", "moe"]:
        for seed in [11, 22, 33]:
            runs.append(_make_run(arch=arch, baseline="stage_specialized_recurrence", seed=seed, offset=0.2))
            runs.append(_make_run(arch=arch, baseline="standard_lora", seed=seed, offset=0.0))
            runs.append(_make_run(arch=arch, baseline="shared_recurrence", seed=seed, offset=0.05))
            runs.append(_make_run(arch=arch, baseline="latent_refiner_only", seed=seed, offset=0.1))
    summary = _write_summary(tmp_path, runs)
    out_dir = tmp_path / "outputs"
    run_analysis(input_path=summary, output_dir=out_dir, allow_unpaired=False)

    report = (out_dir / "statistical_analysis_report.md").read_text()
    assert "## Corrected confirmatory results" in report
    assert "## Secondary outcomes (descriptive)" in report
    assert "## Efficiency outcomes (descriptive)" in report
