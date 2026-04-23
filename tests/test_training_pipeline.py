"""End-to-end tiny training tests for baseline variants."""

from __future__ import annotations

from pathlib import Path

from training.config_loader import load_runtime_config
from training.loop import run_training


def _run_with_temp_output(config_name: str, tmp_path: Path) -> Path:
    runtime = load_runtime_config(Path("experiments/configs") / config_name)
    runtime.output["dir"] = str(tmp_path / config_name.replace(".json", ""))
    result = run_training(runtime, run_name="pytest_run")

    out_dir = result.output_dir
    assert out_dir.exists()
    assert (out_dir / "run_metadata.json").exists()
    assert (out_dir / "parsed_config.json").exists()
    assert (out_dir / "metrics_summary.json").exists()
    assert (out_dir / "model_final.pt").exists()
    assert result.global_steps == runtime.training.max_steps
    assert result.trainable_params > 0
    return out_dir


def test_standard_lora_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("standard_lora.json", tmp_path)


def test_latent_refiner_only_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("latent_refiner_only.json", tmp_path)


def test_shared_recurrence_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("shared_recurrence.json", tmp_path)


def test_stage_specialized_recurrence_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("stage_specialized_recurrence.json", tmp_path)
