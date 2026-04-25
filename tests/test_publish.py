from pathlib import Path
import json
import sys

import pytest

from training.config_loader import load_runtime_config_from_raw
from training.engine import build_training_components, run_training_loop
from publish.huggingface_export import publish_run_directory


def _base_raw(tmp_path: Path) -> dict:
    return {
        "baseline": "base",
        "model": {
            "name": "test/tiny",
            "architecture_type": "dense",
            "standard_lora": {"enabled": False},
            "latent_refiner": {"enabled": False, "num_recurrent_steps": 1, "recurrence_mode": "none", "adapter_sharing": "none"},
        },
        "dataset": {"name": "test_synthetic_stage_dataset", "settings": {"subset_size": 12, "sequence_length": 9, "eval_fraction": 0.25, "seed": 7}},
        "training": {"batch_size": 2, "num_epochs": 1, "max_steps": 2, "eval_interval_steps": 1, "eval_enabled": True},
        "output": {"dir": str(tmp_path)},
    }


def test_publish_defaults_disabled(tmp_path: Path) -> None:
    runtime = load_runtime_config_from_raw(_base_raw(tmp_path))
    assert runtime.publish.enabled is False
    assert runtime.publish.hub_model_repo is None
    assert runtime.publish.hub_dataset_repo is None
    assert runtime.publish.include_checkpoint is False
    assert runtime.publish.max_shard_size == "4GB"


def test_output_dir_defaults_to_hf_home_generated_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", "D:/huggingface")
    raw = {
        "baseline": "base",
        "model": {
            "name": "test/tiny",
            "architecture_type": "dense",
            "standard_lora": {"enabled": False},
            "latent_refiner": {"enabled": False, "num_recurrent_steps": 1, "recurrence_mode": "none", "adapter_sharing": "none"},
        },
    }
    runtime = load_runtime_config_from_raw(raw)
    assert runtime.output["dir"].replace("\\\\", "/") == "D:/huggingface/generated_models"


def test_output_dir_defaults_to_trimmed_hf_home_generated_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HOME", "  D:/huggingface  ")
    raw = {
        "baseline": "base",
        "model": {
            "name": "test/tiny",
            "architecture_type": "dense",
            "standard_lora": {"enabled": False},
            "latent_refiner": {"enabled": False, "num_recurrent_steps": 1, "recurrence_mode": "none", "adapter_sharing": "none"},
        },
    }
    runtime = load_runtime_config_from_raw(raw)
    assert runtime.output["dir"].replace("\\\\", "/") == "D:/huggingface/generated_models"


def test_output_dir_logging_reports_default_source(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("HF_HOME", "D:/huggingface")
    raw = {
        "baseline": "base",
        "model": {
            "name": "test/tiny",
            "architecture_type": "dense",
            "standard_lora": {"enabled": False},
            "latent_refiner": {"enabled": False, "num_recurrent_steps": 1, "recurrence_mode": "none", "adapter_sharing": "none"},
        },
    }
    load_runtime_config_from_raw(raw)
    captured = capsys.readouterr()
    assert "[output]" in captured.out
    assert "source=default from HF_HOME/generated_models" in captured.out
    assert "resolved_dir='D:/huggingface/generated_models'" in captured.out


def test_output_dir_logging_reports_explicit_override(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("HF_HOME", "D:/huggingface")
    raw = _base_raw(Path("custom/output"))
    raw["output"] = {"dir": "custom/outputs"}
    load_runtime_config_from_raw(raw)
    captured = capsys.readouterr()
    assert "source=runtime.output.dir (explicit override)" in captured.out
    assert "resolved_dir='custom/outputs'" in captured.out


def test_publish_enabled_requires_repo_ids(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["publish"] = {"enabled": True}
    with pytest.raises(ValueError, match="requires"):
        load_runtime_config_from_raw(raw)


def test_publish_config_rejects_token_keys(tmp_path: Path) -> None:
    raw = _base_raw(tmp_path)
    raw["publish"] = {"enabled": False, "hf_token": "secret"}
    with pytest.raises(ValueError, match="must not contain credentials"):
        load_runtime_config_from_raw(raw)


def test_dataset_partitions_artifact_contains_identity_fields(tmp_path: Path) -> None:
    runtime = load_runtime_config_from_raw(_base_raw(tmp_path))
    result = run_training_loop(components=build_training_components(runtime), run_name="publish_identity", config_name="unit")
    payload = json.loads((result.output_dir / "dataset_partitions.json").read_text())
    assert "train" in payload and "eval" in payload
    assert payload["train"] and payload["eval"]
    row = payload["train"][0]
    for key in ("sample_id", "sample_hash", "source_signature", "stage_token_counts", "answer_text", "answer_text_normalized"):
        assert key in row


def test_publish_not_called_unless_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sys.modules.pop("publish.huggingface_export", None)
    runtime = load_runtime_config_from_raw(_base_raw(tmp_path))
    run_training_loop(components=build_training_components(runtime), run_name="no_publish", config_name="unit")
    assert "publish.huggingface_export" not in sys.modules


def test_publish_utility_builds_hf_compatible_payloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _base_raw(tmp_path)
    runtime = load_runtime_config_from_raw(raw)
    runtime.validation.lora_expected = False
    runtime.validation.recurrent_expected = False
    result = run_training_loop(components=build_training_components(runtime), run_name="publish_files", config_name="unit")

    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    uploads: list[tuple[str, str]] = []

    class _FakeApi:
        def create_repo(self, repo_id, private, repo_type, exist_ok):
            uploads.append(("create_repo", f"{repo_type}:{repo_id}:{private}:{exist_ok}"))

        def upload_folder(self, folder_path, repo_id, repo_type, commit_message):
            uploads.append(("upload_folder", f"{repo_type}:{repo_id}:{commit_message}"))
            folder = Path(folder_path)
            if repo_type == "model":
                assert (folder / "README.md").exists()
                assert (folder / "config.json").exists()
                assert any(folder.glob("model*.safetensors"))
                if (folder / "model.safetensors.index.json").exists():
                    index = json.loads((folder / "model.safetensors.index.json").read_text(encoding="utf-8"))
                    for filename in index.get("weight_map", {}).values():
                        assert (folder / filename).exists()
            if repo_type == "dataset":
                assert (folder / "README.md").exists()
                assert (folder / "dataset_partitions.json").exists()

    class _FakeDatasetDict:
        def push_to_hub(self, repo_id, private, commit_message):
            uploads.append(("dataset_push", f"{repo_id}:{private}:{commit_message}"))

    monkeypatch.setattr("publish.huggingface_export.HfApi", lambda: _FakeApi())
    monkeypatch.setattr("publish.huggingface_export._build_datasetdict_payload", lambda _payload: _FakeDatasetDict())

    runtime.publish.enabled = True
    runtime.publish.hub_model_repo = "org/model-repo"
    runtime.publish.hub_dataset_repo = "org/dataset-repo"
    runtime.publish.private = True
    runtime.publish.max_shard_size = "1MB"
    publish_run_directory(run_dir=result.output_dir, runtime=runtime, publish_cfg=runtime.publish)
    actions = [a for a, _ in uploads]
    assert "create_repo" in actions
    assert "upload_folder" in actions
    assert "dataset_push" in actions
    assert (result.output_dir / "model_validation_report.md").exists()
    assert (result.output_dir / "hf_model").exists()
    assert any((result.output_dir / "hf_model").glob("model*.safetensors"))
    assert not (result.output_dir / "hf_model" / "checkpoint.pt").exists()
    assert not (result.output_dir / "checkpoint.pt").exists()


def test_gitignore_includes_secret_and_output_patterns() -> None:
    text = Path(".gitignore").read_text(encoding="utf-8")
    for pattern in (".env", ".env.*", "*.env", ".cache/", "outputs/", "hf_token*", "huggingface_token*"):
        assert pattern in text
