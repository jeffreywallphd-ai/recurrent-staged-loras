from pathlib import Path
import json

import torch

from training.model_validation import ModelValidationConfig, load_checkpoint_state_dict, validate_model_checkpoint


def _runtime_config(*, lora_enabled: bool, recurrent_enabled: bool) -> dict[str, object]:
    return {
        "variant": {
            "base": {"architecture_type": "dense"},
            "standard_lora": {"enabled": lora_enabled},
            "refiner_adapter": {"enabled": False},
            "refiner": {"enabled": recurrent_enabled, "num_steps": 2 if recurrent_enabled else 1},
        }
    }


def _save(path: Path, payload: object) -> Path:
    torch.save(payload, path)
    return path


def test_lora_keys_detected_correctly(tmp_path: Path) -> None:
    base = _save(tmp_path / "base.pt", {"model_state_dict": {"layer.weight": torch.zeros(2, 2)}})
    trained = _save(
        tmp_path / "trained.pt",
        {"model_state_dict": {"layer.weight": torch.zeros(2, 2), "adapter.lora_A.weight": torch.ones(2, 1), "adapter.lora_B.weight": torch.ones(1, 2)}},
    )
    result = validate_model_checkpoint(
        base_checkpoint=base,
        trained_checkpoint=trained,
        output_dir=tmp_path,
        runtime_config=_runtime_config(lora_enabled=True, recurrent_enabled=False),
        validation_cfg=ModelValidationConfig(),
    )
    assert result.passed
    diff = json.loads((tmp_path / "model_validation_diff.json").read_text())
    assert "adapter.lora_A.weight" in diff["lora_keys"]


def test_recurrent_keys_detected_correctly(tmp_path: Path) -> None:
    base = _save(tmp_path / "base.pt", {"state_dict": {"layer.weight": torch.zeros(2, 2)}})
    trained = _save(tmp_path / "trained.pt", {"state_dict": {"layer.weight": torch.zeros(2, 2), "refiner.recurrent_projection.weight": torch.ones(2, 2)}})
    result = validate_model_checkpoint(
        base_checkpoint=base,
        trained_checkpoint=trained,
        output_dir=tmp_path,
        runtime_config=_runtime_config(lora_enabled=False, recurrent_enabled=True),
        validation_cfg=ModelValidationConfig(),
    )
    assert result.passed
    diff = json.loads((tmp_path / "model_validation_diff.json").read_text())
    assert "refiner.recurrent_projection.weight" in diff["recurrent_keys"]


def test_missing_expected_lora_keys_fail_validation(tmp_path: Path) -> None:
    base = _save(tmp_path / "base.pt", {"layer.weight": torch.zeros(2, 2)})
    trained = _save(tmp_path / "trained.pt", {"layer.weight": torch.zeros(2, 2)})
    result = validate_model_checkpoint(
        base_checkpoint=base,
        trained_checkpoint=trained,
        output_dir=tmp_path,
        runtime_config=_runtime_config(lora_enabled=True, recurrent_enabled=False),
        validation_cfg=ModelValidationConfig(),
    )
    assert not result.passed
    assert any("LoRA" in item for item in result.missing_required_items)


def test_missing_expected_recurrent_keys_fail_validation(tmp_path: Path) -> None:
    base = _save(tmp_path / "base.pt", {"layer.weight": torch.zeros(2, 2)})
    trained = _save(tmp_path / "trained.pt", {"layer.weight": torch.zeros(2, 2)})
    result = validate_model_checkpoint(
        base_checkpoint=base,
        trained_checkpoint=trained,
        output_dir=tmp_path,
        runtime_config=_runtime_config(lora_enabled=False, recurrent_enabled=True),
        validation_cfg=ModelValidationConfig(),
    )
    assert not result.passed
    assert any("recurrent" in item.lower() for item in result.missing_required_items)


def test_added_removed_shape_changed_keys_reported(tmp_path: Path) -> None:
    base = _save(tmp_path / "base.pt", {"layer.weight": torch.zeros(2, 2), "removed.bias": torch.zeros(2)})
    trained = _save(tmp_path / "trained.pt", {"layer.weight": torch.zeros(3, 2), "added.weight": torch.zeros(1)})
    result = validate_model_checkpoint(
        base_checkpoint=base,
        trained_checkpoint=trained,
        output_dir=tmp_path,
        runtime_config=_runtime_config(lora_enabled=False, recurrent_enabled=False),
        validation_cfg=ModelValidationConfig(),
    )
    assert result.passed
    diff = json.loads((tmp_path / "model_validation_diff.json").read_text())
    assert "added.weight" in diff["added_keys"]
    assert "removed.bias" in diff["removed_keys"]
    assert any(item["key"] == "layer.weight" for item in diff["shape_changes"])


def test_report_file_written_to_output_folder(tmp_path: Path) -> None:
    base = _save(tmp_path / "base.pt", {"layer.weight": torch.zeros(2, 2)})
    trained = _save(tmp_path / "trained.pt", {"layer.weight": torch.zeros(2, 2)})
    result = validate_model_checkpoint(
        base_checkpoint=base,
        trained_checkpoint=trained,
        output_dir=tmp_path,
        runtime_config=_runtime_config(lora_enabled=False, recurrent_enabled=False),
        validation_cfg=ModelValidationConfig(),
    )
    assert result.report_path.exists()
    assert (tmp_path / "model_validation_report.md").exists()


def test_checkpoint_formats_are_handled(tmp_path: Path) -> None:
    raw_state_path = _save(tmp_path / "raw.pt", {"weight": torch.zeros(1)})
    nested_state_path = _save(tmp_path / "nested.pt", {"state_dict": {"weight": torch.zeros(1)}})
    nested_model_state_path = _save(tmp_path / "nested_model.pt", {"model_state_dict": {"weight": torch.zeros(1)}})

    raw = load_checkpoint_state_dict(raw_state_path)
    nested = load_checkpoint_state_dict(nested_state_path)
    nested_model = load_checkpoint_state_dict(nested_model_state_path)

    assert "weight" in raw
    assert "weight" in nested
    assert "weight" in nested_model
