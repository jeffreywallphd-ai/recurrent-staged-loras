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
from training.metrics_schema import AGGREGATE_METRICS, AGG_GROUP_BY_FIELDS, REPORT_TABLE_FIELDS, RUN_METRICS_FIELDS


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


def _tiny_config(tmp_path: Path, *, name: str, subset_size: int = 12) -> Path:
    path = tmp_path / name
    path.write_text(
        json.dumps(
            {
                "baseline": "base",
                "model": {"name": "test/tiny", "architecture_type": "dense", "standard_lora": {"enabled": False}, "latent_refiner": {"enabled": False, "num_recurrent_steps": 1, "recurrence_mode": "none", "adapter_sharing": "none"}},
                "dataset": {"name": "test_synthetic_stage_dataset", "settings": {"subset_size": subset_size, "sequence_length": 9, "eval_fraction": 0.25, "seed": 1}},
                "training": {"batch_size": 2, "num_epochs": 1, "max_steps": 2, "eval_interval_steps": 1, "eval_enabled": True},
                "output": {"dir": str(tmp_path / "out")},
            }
        )
    )
    return path


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
        "train_perplexity",
        "answer_eval_string_count",
    ]:
        assert key in metrics
    assert (result.output_dir / "answer_eval_diagnostics.json").exists()


def test_interval_eval_runs_at_step_cadence() -> None:
    model = build_training_components(_runtime(Path("/tmp"))).model
    examples = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "labels": torch.tensor([1, 2, 3, 4], dtype=torch.long),
            "stage1_mask": torch.tensor([1, 0, 0, 0], dtype=torch.bool),
            "stage2_mask": torch.tensor([0, 1, 0, 0], dtype=torch.bool),
            "stage3_mask": torch.tensor([0, 0, 1, 1], dtype=torch.bool),
            "final_answer_mask": torch.tensor([0, 0, 1, 1], dtype=torch.bool),
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
            "final_answer_mask": torch.tensor([0, 0, 1, 1], dtype=torch.bool),
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


def test_final_answer_metrics_use_answer_span_not_stage3_header() -> None:
    class _FakeOutput:
        def __init__(self, logits: torch.Tensor) -> None:
            self.logits = logits

    class _FakeModel:
        def __init__(self) -> None:
            self._training = True
            self.config = type("Cfg", (), {"refiner": type("Ref", (), {"enabled": False})()})()

        def eval(self) -> None:
            self._training = False

        def train(self) -> None:
            self._training = True

        def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> _FakeOutput:
            del attention_mask
            batch, seq_len = input_ids.shape
            vocab = 32
            logits = torch.full((batch, seq_len, vocab), -1000.0)
            # Predict header token 0 incorrectly, but answer token 5 correctly.
            logits[:, 0, 0] = 1000.0
            logits[:, 1, 5] = 1000.0
            return _FakeOutput(logits=logits)

    examples = [
        {
            "input_ids": torch.tensor([2, 5, 0], dtype=torch.long),
            "labels": torch.tensor([2, 5, 0], dtype=torch.long),
            "stage1_mask": torch.tensor([1, 0, 0], dtype=torch.bool),
            "stage2_mask": torch.tensor([0, 0, 0], dtype=torch.bool),
            "stage3_mask": torch.tensor([0, 1, 1], dtype=torch.bool),
            "final_answer_mask": torch.tensor([0, 1, 0], dtype=torch.bool),
            "answer_text": "5",
            "answer_text_normalized": "5",
            "answer_numeric_normalized": "5.0",
        }
    ]
    loader = DataLoader(SequenceDataset(examples), batch_size=1, shuffle=False, collate_fn=lambda b: collate_token_sequences(b, pad_token_id=0))
    eval_result = evaluate(model=_FakeModel(), dataloader=loader, tokenizer=None)
    assert eval_result.stage_3_token_accuracy == 0.5
    assert eval_result.final_answer_accuracy == 1.0
    assert eval_result.final_answer_exact_match == 1.0
    assert eval_result.normalized_numeric_answer_accuracy == 1.0


def test_multi_seed_summary_includes_aggregate_metrics_and_artifacts(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path, name="tiny.json")
    subprocess.run(
        [
            sys.executable,
            "scripts/run_all_experiments.py",
            "--configs",
            str(cfg),
            "--seeds",
            "1",
            "2",
            "3",
            "--preset-scope",
            "all",
        ],
        check=True,
    )
    summary = json.loads(Path("outputs/summary.json").read_text())
    assert "runs" in summary and "aggregates" in summary
    assert summary["aggregates"]
    for metric in AGGREGATE_METRICS:
        assert metric in summary["aggregates"][0]["metrics"]
    for key in AGG_GROUP_BY_FIELDS:
        assert key in summary["aggregates"][0]
    assert Path("outputs/aggregates.json").exists()
    assert Path("outputs/report_table.csv").exists()


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
    assert "answer_span_normalized_accuracy" in docs
    assert "answer_eval_skipped_no_answer_span" in docs


def test_pilot_and_study_do_not_collapse_in_aggregate_grouping(tmp_path: Path) -> None:
    study_cfg = _tiny_config(tmp_path, name="standard_lora.json", subset_size=12)
    pilot_cfg = _tiny_config(tmp_path, name="standard_lora_pilot.json", subset_size=8)
    subprocess.run(
        [
            sys.executable,
            "scripts/run_all_experiments.py",
            "--configs",
            str(study_cfg),
            str(pilot_cfg),
            "--seeds",
            "1",
            "--preset-scope",
            "all",
        ],
        check=True,
    )
    summary = json.loads(Path("outputs/summary.json").read_text())
    config_names = {agg["config_name"] for agg in summary["aggregates"]}
    assert "standard_lora.json" in config_names
    assert "standard_lora_pilot.json" in config_names


def test_study_only_selection_excludes_pilot_configs(tmp_path: Path) -> None:
    study_cfg = _tiny_config(tmp_path, name="base.json")
    pilot_cfg = _tiny_config(tmp_path, name="base_pilot.json")
    subprocess.run(
        [
            sys.executable,
            "scripts/run_all_experiments.py",
            "--configs",
            str(study_cfg),
            str(pilot_cfg),
            "--seeds",
            "1",
            "--preset-scope",
            "study",
        ],
        check=True,
    )
    summary = json.loads(Path("outputs/summary.json").read_text())
    assert summary["preset_scope"] == "study"
    assert all(not str(run["config_name"]).endswith("_pilot.json") for run in summary["runs"])


def test_answer_metric_definitions_documented_as_distinct() -> None:
    docs = Path("docs/experiments.md").read_text(encoding="utf-8")
    assert "final_answer_accuracy" in docs
    assert "final_answer_exact_match" in docs
    assert "strict raw-string equality" in docs
    assert "final_answer_mask" in docs


def test_report_table_has_paper_facing_columns(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path, name="report.json")
    subprocess.run(
        [sys.executable, "scripts/run_all_experiments.py", "--configs", str(cfg), "--seeds", "1", "--preset-scope", "all"],
        check=True,
    )
    import csv

    with Path("outputs/report_table.csv").open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))
    assert rows
    assert set(REPORT_TABLE_FIELDS).issubset(set(rows[0].keys()))
    row_types = {r["row_type"] for r in rows}
    assert "run" in row_types and "aggregate" in row_types


def test_aggregates_include_stage_token_metrics(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path, name="agg.json")
    subprocess.run(
        [sys.executable, "scripts/run_all_experiments.py", "--configs", str(cfg), "--seeds", "1", "2", "--preset-scope", "all"],
        check=True,
    )
    aggregates = json.loads(Path("outputs/aggregates.json").read_text())
    metrics = aggregates[0]["metrics"]
    assert "stage_2_token_accuracy" in metrics
    assert "stage_3_token_accuracy" in metrics
