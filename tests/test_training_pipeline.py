"""End-to-end tiny training tests for baseline variants."""

from __future__ import annotations

from pathlib import Path
import json
import math

from training.config_loader import load_runtime_config
from training.engine import build_training_components, run_training_loop


def _run_with_temp_output(config_name: str, tmp_path: Path) -> Path:
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
    assert isinstance(metrics["steps"], int)
    assert metrics["steps"] == runtime.training.max_steps
    assert math.isfinite(float(metrics["train_loss"]))
    assert math.isfinite(float(metrics["eval_loss"]))
    return out_dir


def test_standard_lora_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("standard_lora.json", tmp_path)


def test_latent_refiner_only_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("latent_refiner_only.json", tmp_path)


def test_shared_recurrence_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("shared_recurrence.json", tmp_path)


def test_stage_specialized_recurrence_tiny_training_run(tmp_path: Path) -> None:
    _run_with_temp_output("stage_specialized_recurrence.json", tmp_path)
