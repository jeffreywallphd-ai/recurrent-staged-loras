import json
from pathlib import Path

import pytest

from analysis.analysis_schema import PRIMARY_CONFIRMATORY_OUTCOMES
from analysis.statistical_analysis import _group_runs, run_analysis




class _FakeResult:
    def __init__(self, pvalue: float) -> None:
        self.pvalue = pvalue


class _FakeStats:
    @staticmethod
    def wilcoxon(values, **_kwargs):
        if all(abs(v) < 1e-12 for v in values):
            return _FakeResult(1.0)
        return _FakeResult(0.01)

    @staticmethod
    def ttest_rel(_a, _b, **_kwargs):
        return _FakeResult(0.02)


@pytest.fixture(autouse=True)
def _patch_stats_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("analysis.statistical_analysis._get_stats_backend", lambda: _FakeStats)

def _make_run(
    *,
    arch: str,
    baseline: str,
    seed: int,
    offset: float,
    config_name: str | None = None,
    dataset_name: str = "metamath_qa",
    model_name: str | None = None,
) -> dict[str, object]:
    return {
        "run_name": f"{arch}_{baseline}_{seed}",
        "config_name": config_name or f"{arch}_study.json",
        "baseline_name": baseline,
        "dataset_name": dataset_name,
        "seed": seed,
        "architecture_type": arch,
        "model_name": model_name or ("Qwen/Qwen3-8B" if arch == "dense" else "allenai/OLMoE-1B-7B-0125-Instruct"),
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
        "compute_control_enabled": True,
        "compute_control_mode": "effective_forward_passes",
        "ablation_recurrent_steps": None,
        "ablation_lora_rank": None,
    }


def _write_summary(tmp_path: Path, runs: list[dict[str, object]]) -> Path:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"runs": runs, "aggregates": []}))
    return path


def _build_complete_matrix(*, seeds: list[int]) -> list[dict[str, object]]:
    runs = []
    for arch in ["dense", "moe"]:
        for seed in seeds:
            runs.append(_make_run(arch=arch, baseline="stage_specialized_recurrence", seed=seed, offset=0.2))
            runs.append(_make_run(arch=arch, baseline="standard_lora", seed=seed, offset=0.0))
            runs.append(_make_run(arch=arch, baseline="shared_recurrence", seed=seed, offset=0.05))
            runs.append(_make_run(arch=arch, baseline="latent_refiner_only", seed=seed, offset=0.1))
    return runs


def test_grouping_key_keeps_config_dataset_model_identity() -> None:
    runs = [
        _make_run(arch="dense", baseline="standard_lora", seed=11, offset=0.0, config_name="dense_standard_lora.json"),
        _make_run(arch="dense", baseline="standard_lora", seed=11, offset=0.0, config_name="dense_standard_lora_pilot.json"),
        _make_run(arch="dense", baseline="standard_lora", seed=11, offset=0.0, dataset_name="other_dataset"),
        _make_run(arch="dense", baseline="standard_lora", seed=11, offset=0.0, model_name="alt/model"),
    ]
    grouped = _group_runs(runs)
    assert len(grouped) == 4


def test_heterogeneous_comparison_groups_raise_clear_error(tmp_path: Path) -> None:
    runs = _build_complete_matrix(seeds=[11, 22, 33])
    runs.append(
        _make_run(
            arch="dense",
            baseline="stage_specialized_recurrence",
            seed=44,
            offset=0.2,
            dataset_name="other_dataset",
        )
    )

    summary = _write_summary(tmp_path, runs)
    with pytest.raises(ValueError, match="Heterogeneous comparison group detected"):
        run_analysis(input_path=summary, output_dir=tmp_path / "outputs", allow_unpaired=False)


def test_confirmatory_allows_different_config_names_with_matched_family(tmp_path: Path) -> None:
    runs = _build_complete_matrix(seeds=[11, 22, 33])
    for row in runs:
        if row["baseline_name"] == "stage_specialized_recurrence":
            row["config_name"] = f"{row['architecture_type']}_stage_specialized_recurrence.json"
        elif row["baseline_name"] == "standard_lora":
            row["config_name"] = f"{row['architecture_type']}_standard_lora.json"

    summary = _write_summary(tmp_path, runs)
    out_dir = tmp_path / "outputs"
    run_analysis(input_path=summary, output_dir=out_dir, allow_unpaired=False)
    confirmatory = json.loads((out_dir / "statistical_analysis_confirmatory.json").read_text())
    row = next(r for r in confirmatory if r["baseline_b"] == "standard_lora" and r["architecture_type"] == "dense")
    assert row["config_name_a"].endswith("stage_specialized_recurrence.json")
    assert row["config_name_b"].endswith("standard_lora.json")


def test_confirmatory_rows_receive_holm_and_ci_fields(tmp_path: Path) -> None:
    runs = _build_complete_matrix(seeds=[11, 22, 33])
    summary = _write_summary(tmp_path, runs)
    out_dir = tmp_path / "outputs"
    result = run_analysis(input_path=summary, output_dir=out_dir, allow_unpaired=False)

    assert result["confirmatory_rows"] == 18
    confirmatory = json.loads((out_dir / "statistical_analysis_confirmatory.json").read_text())
    assert all(row["analysis_tier"] == "confirmatory" for row in confirmatory)
    assert all(row["holm_adjusted_p_value"] is not None for row in confirmatory)
    assert all("config_name_a" in row and "config_name_b" in row for row in confirmatory)
    assert all("mean_difference_ci_low" in row and "mean_difference_ci_high" in row for row in confirmatory)


def test_unpaired_downgraded_rows_excluded_from_holm(tmp_path: Path) -> None:
    runs = []
    for seed in [11, 22, 33]:
        runs.append(_make_run(arch="dense", baseline="stage_specialized_recurrence", seed=seed, offset=0.2))
    for seed in [44, 55, 66]:
        runs.append(_make_run(arch="dense", baseline="standard_lora", seed=seed, offset=0.0))
    for seed in [11, 22, 33]:
        runs.append(_make_run(arch="dense", baseline="shared_recurrence", seed=seed, offset=0.05))
        runs.append(_make_run(arch="dense", baseline="latent_refiner_only", seed=seed, offset=0.1))
        runs.append(_make_run(arch="moe", baseline="stage_specialized_recurrence", seed=seed, offset=0.2))
        runs.append(_make_run(arch="moe", baseline="standard_lora", seed=seed, offset=0.0))
        runs.append(_make_run(arch="moe", baseline="shared_recurrence", seed=seed, offset=0.05))
        runs.append(_make_run(arch="moe", baseline="latent_refiner_only", seed=seed, offset=0.1))

    summary = _write_summary(tmp_path, runs)
    out_dir = tmp_path / "outputs"
    run_analysis(input_path=summary, output_dir=out_dir, allow_unpaired=True)
    rows = json.loads((out_dir / "statistical_analysis_confirmatory.json").read_text())

    downgraded = [r for r in rows if r["pairing_used"] == "unpaired"]
    assert downgraded
    assert all(r["analysis_tier"] == "descriptive_downgraded" for r in downgraded)
    assert all(r["raw_p_value"] is None and r["holm_adjusted_p_value"] is None for r in downgraded)

    confirmatory = [r for r in rows if r["analysis_tier"] == "confirmatory"]
    assert confirmatory
    assert all(r["holm_adjusted_p_value"] is not None for r in confirmatory)


def test_metadata_artifact_written_with_expected_keys(tmp_path: Path) -> None:
    runs = _build_complete_matrix(seeds=[1, 2, 3])
    summary = _write_summary(tmp_path, runs)
    out_dir = tmp_path / "outputs"
    run_analysis(input_path=summary, output_dir=out_dir, allow_unpaired=False)

    metadata = json.loads((out_dir / "statistical_analysis_metadata.json").read_text())
    assert metadata["primary_confirmatory_outcomes"] == PRIMARY_CONFIRMATORY_OUTCOMES
    assert metadata["family_wise_error_method"] == "holm"
    assert "alpha" in metadata
    assert "statistical_test_backend" in metadata
    assert "any_rows_downgraded_to_descriptive" in metadata
    assert "any_pairing_failures" in metadata


def test_markdown_report_contains_key_sections(tmp_path: Path) -> None:
    runs = _build_complete_matrix(seeds=[11, 22, 33])
    summary = _write_summary(tmp_path, runs)
    out_dir = tmp_path / "outputs"
    run_analysis(input_path=summary, output_dir=out_dir, allow_unpaired=False)

    report = (out_dir / "statistical_analysis_report.md").read_text()
    assert "## Corrected confirmatory results" in report
    assert "mean_diff_95ci" in report
    assert "## Secondary outcomes (descriptive)" in report
    assert "## Efficiency outcomes (descriptive)" in report


def test_statistical_filters_support_compute_control_and_ablation(tmp_path: Path) -> None:
    runs = _build_complete_matrix(seeds=[11, 22, 33])
    runs[0]["ablation_recurrent_steps"] = 2
    runs[0]["ablation_lora_rank"] = 8
    summary = _write_summary(tmp_path, runs)
    out_dir = tmp_path / "outputs"
    result = run_analysis(
        input_path=summary,
        output_dir=out_dir,
        allow_unpaired=False,
        compute_controlled_only=True,
        ablation_only=False,
    )
    assert result["confirmatory_rows"] > 0


def test_external_dataset_scope_supports_flattened_analysis(tmp_path: Path) -> None:
    runs = _build_complete_matrix(seeds=[11, 22, 33])
    for row in runs:
        row["external_eval"] = {
            "gsm8k": {
                "eval_loss": 1.2,
                "stage_2_token_accuracy": 0.3,
                "stage_3_token_accuracy": 0.4,
                "final_answer_accuracy": 0.5,
                "final_answer_exact_match": 0.45,
                "normalized_numeric_answer_accuracy": 0.47,
            }
        }
    summary = _write_summary(tmp_path, runs)
    out_dir = tmp_path / "outputs"
    result = run_analysis(input_path=summary, output_dir=out_dir, allow_unpaired=False, dataset_scope="external")
    assert result["confirmatory_rows"] == 0
    secondary = json.loads((out_dir / "statistical_analysis_secondary.json").read_text())
    assert secondary
    assert all(row["analysis_tier"] == "descriptive" for row in secondary)
    assert all(row["dataset_name"] == "gsm8k" for row in secondary)


def test_dataset_scope_all_keeps_primary_confirmatory(tmp_path: Path) -> None:
    runs = _build_complete_matrix(seeds=[11, 22, 33])
    for row in runs:
        row["external_eval"] = {
            "gsm8k": {
                "eval_loss": 1.2,
                "stage_2_token_accuracy": 0.3,
                "stage_3_token_accuracy": 0.4,
                "final_answer_accuracy": 0.5,
                "final_answer_exact_match": 0.45,
                "normalized_numeric_answer_accuracy": 0.47,
            }
        }
    summary = _write_summary(tmp_path, runs)
    out_dir = tmp_path / "outputs"
    run_analysis(input_path=summary, output_dir=out_dir, allow_unpaired=False, dataset_scope="all")
    confirmatory = json.loads((out_dir / "statistical_analysis_confirmatory.json").read_text())
    assert confirmatory
    assert all(row["dataset_name"] == "metamath_qa" for row in confirmatory)
