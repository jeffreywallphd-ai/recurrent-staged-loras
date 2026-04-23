from pathlib import Path
import json
import subprocess
import sys

from models.config import parse_variant_config
from training.config_loader import RuntimeConfig, TrainingConfig
from training.engine import build_training_components, run_training_loop


def _runtime(tmp_path: Path, baseline: str = "stage_specialized_recurrence") -> RuntimeConfig:
    variant = parse_variant_config(
        {
            "baseline": baseline,
            "model": {
                "name": "test/tiny",
                "architecture_type": "dense",
                "standard_lora": {"enabled": baseline == "standard_lora", "target_modules": ["q_proj", "v_proj"]},
                "latent_refiner": {
                    "enabled": baseline in {"latent_refiner_only", "shared_recurrence", "stage_specialized_recurrence"},
                    "num_recurrent_steps": 3 if baseline != "base" else 1,
                    "recurrence_mode": {
                        "base": "none",
                        "standard_lora": "none",
                        "latent_refiner_only": "latent_only",
                        "shared_recurrence": "shared",
                        "stage_specialized_recurrence": "stage_specialized",
                    }[baseline],
                    "adapter_sharing": {
                        "base": "none",
                        "standard_lora": "none",
                        "latent_refiner_only": "none",
                        "shared_recurrence": "shared",
                        "stage_specialized_recurrence": "per_step",
                    }[baseline],
                    "adapter": {"enabled": baseline in {"shared_recurrence", "stage_specialized_recurrence"}},
                },
            },
        }
    )
    return RuntimeConfig(
        baseline=baseline,
        variant=variant,
        training=TrainingConfig(batch_size=2, max_steps=3, num_epochs=1, eval_interval_steps=1, eval_enabled=True),
        dataset={"name": "test_synthetic_stage_dataset", "settings": {"subset_size": 12, "sequence_length": 9, "eval_fraction": 0.25, "seed": 7}},
        output={"dir": str(tmp_path)},
        raw={},
    )


def test_stage_aware_loss_and_metrics_fields_present(tmp_path: Path) -> None:
    rt = _runtime(tmp_path)
    result = run_training_loop(components=build_training_components(rt), run_name="run", config_name="unit")
    metrics = json.loads((result.output_dir / "metrics.json").read_text())
    for key in [
        "final_answer_exact_match",
        "final_answer_accuracy",
        "stage_3_token_accuracy",
        "stage_2_token_accuracy",
        "normalized_numeric_answer_accuracy",
        "architecture_type",
        "recurrence_steps",
        "effective_forward_passes_per_example",
    ]:
        assert key in metrics


def test_multi_seed_summary_includes_aggregates(tmp_path: Path) -> None:
    (tmp_path / "tiny.json").write_text(json.dumps({
        "baseline": "base",
        "model": {"name": "test/tiny", "architecture_type": "dense", "standard_lora": {"enabled": False}, "latent_refiner": {"enabled": False, "num_recurrent_steps": 1, "recurrence_mode": "none", "adapter_sharing": "none"}},
        "dataset": {"name": "test_synthetic_stage_dataset", "settings": {"subset_size": 12, "sequence_length": 9, "eval_fraction": 0.25, "seed": 1}},
        "training": {"batch_size": 2, "num_epochs": 1, "max_steps": 2, "eval_interval_steps": 1, "eval_enabled": True},
        "output": {"dir": str(tmp_path / "out")}
    }))
    subprocess.run(
        [
            sys.executable,
            "scripts/run_all_experiments.py",
            "--configs",
            str(tmp_path / "tiny.json"),
            "--seeds",
            "1",
            "2",
            "3",
        ],
        check=True,
    )
    summary = json.loads(Path("outputs/summary.json").read_text())
    assert "runs" in summary and "aggregates" in summary
    assert summary["aggregates"]
