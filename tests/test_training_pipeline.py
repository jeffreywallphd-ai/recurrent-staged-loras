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


def _run_with_temp_output(config_name: str, tmp_path: Path) -> tuple[Path, dict[str, object]]:
    runtime = load_runtime_config(Path("experiments/configs") / config_name)
    runtime.output["dir"] = str(tmp_path / config_name.replace(".json", ""))

    components = build_training_components(runtime)
    result = run_training_loop(components=components, run_name="pytest_run")

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
    assert isinstance(metrics["num_steps"], int)
    assert metrics["num_steps"] == runtime.training.max_steps
    assert math.isfinite(float(metrics["train_loss"]))
    assert math.isfinite(float(metrics["eval_loss"]))
    return out_dir, metrics


def test_standard_lora_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("standard_lora.json", tmp_path)


def test_latent_refiner_only_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("latent_refiner_only.json", tmp_path)


def test_shared_recurrence_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("shared_recurrence.json", tmp_path)


def test_stage_specialized_recurrence_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("stage_specialized_recurrence.json", tmp_path)


def test_reproducibility_same_seed_same_metrics(tmp_path: Path) -> None:
    _, metrics_a = _run_with_temp_output("standard_lora.json", tmp_path / "a")
    _, metrics_b = _run_with_temp_output("standard_lora.json", tmp_path / "b")

    assert metrics_a["num_steps"] == metrics_b["num_steps"]
    assert metrics_a["num_epochs"] == metrics_b["num_epochs"]
    assert math.isclose(float(metrics_a["train_loss"]), float(metrics_b["train_loss"]), rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(float(metrics_a["eval_loss"]), float(metrics_b["eval_loss"]), rel_tol=1e-9, abs_tol=1e-9)


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

    summary_path = Path("output/summary.json")
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "base" in summary
    assert "standard_lora" in summary
    assert Path("output/base/base/metrics.json").exists()
    assert Path("output/standard_lora/standard_lora/metrics.json").exists()
