"""End-to-end tiny training tests for baseline variants."""

from __future__ import annotations

from pathlib import Path
import json
import math
import subprocess
import sys

from training.config_loader import load_runtime_config
from training.engine import build_training_components, run_training_loop


EXPECTED_METRIC_KEYS = {"baseline_name", "train_loss", "eval_loss", "num_steps", "num_epochs"}
EXPECTED_ADDITIONAL_METRIC_KEYS = {
    "run_name",
    "config_name",
    "dataset_name",
    "dataset_mode",
    "dataset_train_examples",
    "dataset_eval_examples",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "seed",
    "deterministic",
    "final_train_loss",
    "final_eval_loss",
    "best_eval_loss",
    "eval_perplexity",
    "train_perplexity",
    "eval_next_token_accuracy",
    "eval_top_5_accuracy",
    "global_steps_completed",
    "epochs_completed",
    "tokens_seen_train",
    "tokens_seen_eval",
    "tokens_per_second_train",
    "tokens_per_second_eval",
    "wall_time_seconds_total",
    "wall_time_seconds_train",
    "wall_time_seconds_eval",
    "seconds_per_step",
    "steps_per_second",
    "trainable_params",
    "total_params",
    "trainable_param_fraction",
}


def _run_with_temp_output(config_name: str, tmp_path: Path) -> tuple[Path, dict[str, object], object]:
    runtime = load_runtime_config(Path("experiments/configs") / config_name)
    runtime.output["dir"] = str(tmp_path / config_name.replace(".json", ""))

    components = build_training_components(runtime)
    result = run_training_loop(components=components, run_name="pytest_run", config_name=config_name)

    out_dir = result.output_dir
    assert out_dir.exists()
    assert (out_dir / "metadata.json").exists()
    assert (out_dir / "config.json").exists()
    assert (out_dir / "metrics.json").exists()
    assert (out_dir / "checkpoint.pt").exists()
    assert result.global_steps == runtime.training.max_steps
    assert result.trainable_params > 0

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert EXPECTED_METRIC_KEYS.issubset(metrics.keys())
    assert EXPECTED_ADDITIONAL_METRIC_KEYS.issubset(metrics.keys())
    assert isinstance(metrics["num_steps"], int)
    assert metrics["num_steps"] == runtime.training.max_steps
    assert math.isfinite(float(metrics["train_loss"]))
    assert math.isfinite(float(metrics["eval_loss"]))
    assert math.isfinite(float(metrics["final_train_loss"]))
    assert math.isfinite(float(metrics["final_eval_loss"]))
    assert math.isfinite(float(metrics["best_eval_loss"]))
    assert math.isfinite(float(metrics["eval_perplexity"]))
    assert math.isfinite(float(metrics["train_perplexity"]))
    assert 0.0 <= float(metrics["eval_next_token_accuracy"]) <= 1.0
    assert 0.0 <= float(metrics["eval_top_5_accuracy"]) <= 1.0
    assert int(metrics["global_steps_completed"]) == runtime.training.max_steps
    assert int(metrics["epochs_completed"]) >= 1
    assert int(metrics["tokens_seen_train"]) > 0
    assert int(metrics["tokens_seen_eval"]) > 0
    assert float(metrics["wall_time_seconds_total"]) >= 0.0
    assert float(metrics["wall_time_seconds_train"]) >= 0.0
    assert float(metrics["wall_time_seconds_eval"]) >= 0.0
    assert math.isfinite(float(metrics["tokens_per_second_train"]))
    assert math.isfinite(float(metrics["tokens_per_second_eval"]))
    assert math.isfinite(float(metrics["seconds_per_step"]))
    assert math.isfinite(float(metrics["steps_per_second"]))
    assert float(metrics["trainable_param_fraction"]) >= 0.0
    assert int(metrics["total_params"]) >= int(metrics["trainable_params"])
    return out_dir, metrics, runtime


def test_standard_lora_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("standard_lora.json", tmp_path)


def test_latent_refiner_only_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("latent_refiner_only.json", tmp_path)


def test_shared_recurrence_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("shared_recurrence.json", tmp_path)


def test_stage_specialized_recurrence_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("stage_specialized_recurrence.json", tmp_path)


def test_reproducibility_same_seed_same_metrics(tmp_path: Path) -> None:
    _, metrics_a, _ = _run_with_temp_output("standard_lora.json", tmp_path / "a")
    _, metrics_b, _ = _run_with_temp_output("standard_lora.json", tmp_path / "b")

    assert metrics_a["num_steps"] == metrics_b["num_steps"]
    assert metrics_a["num_epochs"] == metrics_b["num_epochs"]
    assert math.isclose(float(metrics_a["train_loss"]), float(metrics_b["train_loss"]), rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(float(metrics_a["eval_loss"]), float(metrics_b["eval_loss"]), rel_tol=1e-9, abs_tol=1e-9)


def test_interval_eval_time_and_tokens_are_accumulated(tmp_path: Path) -> None:
    _, metrics, runtime = _run_with_temp_output("standard_lora.json", tmp_path / "interval")
    # Re-run with denser interval eval to validate accumulation behavior.
    runtime.training.eval_interval_steps = 1
    runtime.output["dir"] = str(tmp_path / "interval_dense")
    components = build_training_components(runtime)
    result = run_training_loop(components=components, run_name="pytest_run", config_name="standard_lora.json")
    dense_metrics = json.loads((result.output_dir / "metrics.json").read_text(encoding="utf-8"))

    assert float(dense_metrics["wall_time_seconds_eval"]) >= 0.0
    assert float(dense_metrics["wall_time_seconds_train"]) >= 0.0
    assert float(dense_metrics["wall_time_seconds_total"]) >= float(dense_metrics["wall_time_seconds_train"])
    assert int(dense_metrics["tokens_seen_eval"]) > int(metrics["tokens_seen_eval"])


def test_structured_sequence_metrics_present(tmp_path: Path) -> None:
    runtime = load_runtime_config(Path("experiments/configs/standard_lora.json"))
    runtime.dataset["name"] = "structured_sequence"
    runtime.dataset["settings"] = {
        "num_examples": 12,
        "prefix_length": 4,
        "target_length": 6,
        "eval_fraction": 0.25,
        "seed": runtime.training.seed,
        "mode": "structured_sequence",
    }
    runtime.output["dir"] = str(tmp_path / "structured")
    components = build_training_components(runtime)
    result = run_training_loop(components=components, run_name="pytest_structured", config_name="structured_from_standard")
    metrics = json.loads((result.output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert 0.0 <= float(metrics["eval_target_token_accuracy"]) <= 1.0
    assert 0.0 <= float(metrics["eval_target_sequence_exact_match"]) <= 1.0


def test_multi_run_script_produces_summary_and_outputs() -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/run_all_experiments.py",
            "--configs",
            "experiments/configs/base.json",
            "experiments/configs/standard_lora.json",
        ],
        check=True,
    )

    summary_path = Path("outputs/summary.json")
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "runs" in summary
    assert isinstance(summary["runs"], list)
    baselines = {run["baseline"] for run in summary["runs"]}
    assert "base" in baselines
    assert "standard_lora" in baselines
    assert Path("outputs/summary.csv").exists()
    assert Path("outputs/base/base/metrics.json").exists()
    assert Path("outputs/standard_lora/standard_lora/metrics.json").exists()


def test_multi_run_summary_retains_multiple_runs_for_same_baseline(tmp_path: Path) -> None:
    custom_a = tmp_path / "base_copy_a.json"
    custom_b = tmp_path / "base_copy_b.json"
    source = Path("experiments/configs/base.json").read_text(encoding="utf-8")
    custom_a.write_text(source, encoding="utf-8")
    custom_b.write_text(source, encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "scripts/run_all_experiments.py",
            "--configs",
            str(custom_a),
            str(custom_b),
        ],
        check=True,
    )

    summary = json.loads(Path("outputs/summary.json").read_text(encoding="utf-8"))
    base_runs = [run for run in summary["runs"] if run["baseline"] == "base"]
    run_names = {run["run_name"] for run in base_runs}
    assert "base_copy_a" in run_names
    assert "base_copy_b" in run_names


def test_summary_artifacts_include_richer_metrics() -> None:
    summary = json.loads(Path("outputs/summary.json").read_text(encoding="utf-8"))
    assert summary["runs"]
    run = summary["runs"][0]
    assert "eval_perplexity" in run
    assert "eval_next_token_accuracy" in run
    assert "tokens_per_second_eval" in run


def test_compare_metrics_script_supports_richer_schema(tmp_path: Path) -> None:
    out_dir, _, _ = _run_with_temp_output("standard_lora.json", tmp_path / "compare")
    metrics_path = out_dir / "metrics.json"
    result = subprocess.run(
        [sys.executable, "scripts/compare_metrics.py", str(metrics_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "tokens_per_second_eval" in result.stdout
    assert "eval_perplexity" in result.stdout
    assert "wall_time_seconds_train" in result.stdout
