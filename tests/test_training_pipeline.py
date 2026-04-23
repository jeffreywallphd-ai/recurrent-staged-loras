from pathlib import Path
import json
import subprocess
import sys

import torch
from torch.utils.data import DataLoader

from data.dataset import SequenceDataset, collate_token_sequences
from models.config import parse_variant_config
from training.config_loader import RuntimeConfig, TrainingConfig
from training.engine import build_training_components, run_training_loop
from training.loop import evaluate, run_training
from training.metrics_schema import AGGREGATE_METRICS, RUN_METRICS_FIELDS


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


def test_interval_eval_runs_at_step_cadence() -> None:
    model = build_training_components(_runtime(Path("/tmp"))).model
    examples = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "labels": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "stage1_mask": torch.tensor([1, 0, 0, 0], dtype=torch.bool),
            "stage2_mask": torch.tensor([0, 1, 0, 0], dtype=torch.bool),
            "stage3_mask": torch.tensor([0, 0, 1, 1], dtype=torch.bool),
            "answer_text": "3 4",
            "answer_text_normalized": "3 4",
            "answer_numeric_normalized": "3.0",
        }
        for _ in range(6)
    ]
    ds = SequenceDataset(examples)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=lambda b: collate_token_sequences(b, pad_token_id=0))
    summary = run_training(
        model=model,
        train_loader=loader,
        eval_loader=loader,
        optimizer=None,
        num_epochs=1,
        max_steps=5,
        eval_interval_steps=2,
        eval_enabled=True,
    )
    per_eval_tokens = 3 * len(ds)
    assert summary["tokens_seen_eval"] == per_eval_tokens * 3  # steps 2,4 + final step 5


def test_answer_metrics_are_not_stage3_token_proxy() -> None:
    model = build_training_components(_runtime(Path("/tmp"))).model
    examples = [
        {
            "input_ids": torch.tensor([1, 2, 2, 2], dtype=torch.long),
            "labels": torch.tensor([1, 2, 2, 2], dtype=torch.long),
            "stage1_mask": torch.tensor([1, 0, 0, 0], dtype=torch.bool),
            "stage2_mask": torch.tensor([0, 1, 0, 0], dtype=torch.bool),
            "stage3_mask": torch.tensor([0, 0, 1, 1], dtype=torch.bool),
            "answer_text": "999 999",
            "answer_text_normalized": "999 999",
            "answer_numeric_normalized": "999.0",
        }
    ]
    loader = DataLoader(SequenceDataset(examples), batch_size=1, shuffle=False, collate_fn=lambda b: collate_token_sequences(b, pad_token_id=0))
    eval_result = evaluate(model=model, dataloader=loader, tokenizer=None)
    assert eval_result.stage_3_token_accuracy is not None
    assert eval_result.final_answer_accuracy is not None
    assert eval_result.final_answer_accuracy == 0.0


def test_multi_seed_summary_includes_aggregate_metrics_and_artifacts(tmp_path: Path) -> None:
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
    for metric in AGGREGATE_METRICS:
        assert metric in summary["aggregates"][0]["metrics"]
    assert Path("outputs/aggregates.json").exists()


def test_dataset_preprocessing_summary_artifact_written(tmp_path: Path) -> None:
    rt = _runtime(tmp_path, baseline="base")
    result = run_training_loop(components=build_training_components(rt), run_name="summary", config_name="unit")
    stats_path = result.output_dir / "dataset_preprocessing_summary.json"
    stats = json.loads(stats_path.read_text())
    assert "kept_examples" in stats


def test_metrics_schema_matches_docs() -> None:
    docs = Path("docs/experiments.md").read_text(encoding="utf-8")
    for field in RUN_METRICS_FIELDS:
        assert field in docs
