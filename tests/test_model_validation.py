from pathlib import Path
import json

import torch
from huggingface_hub import save_torch_state_dict
from safetensors.torch import save_file

from training.model_validation import ModelValidationConfig, load_checkpoint_state_dict, validate_model_checkpoint


def _runtime_config(*, lora_enabled: bool, recurrent_enabled: bool, lora_merged_before_save: bool = False) -> dict[str, object]:
    return {
        "variant": {
            "base": {"architecture_type": "dense"},
            "standard_lora": {"enabled": lora_enabled},
            "refiner_adapter": {"enabled": False},
            "refiner": {"enabled": recurrent_enabled, "num_steps": 2 if recurrent_enabled else 1},
        },
        "validation": {"lora_merged_before_save": lora_merged_before_save},
    }


def _runtime_config_with_refiner_adapter_only(*, recurrent_enabled: bool) -> dict[str, object]:
    return {
        "variant": {
            "base": {"architecture_type": "dense"},
            "standard_lora": {"enabled": False},
            "refiner_adapter": {"enabled": True},
            "refiner": {"enabled": recurrent_enabled, "num_steps": 3 if recurrent_enabled else 1},
        },
        "validation": {"lora_merged_before_save": False},
    }


def _save(path: Path, payload: object) -> Path:
    torch.save(payload, path)
    return path


def test_validation_reads_sharded_safetensors_index(tmp_path: Path) -> None:
    base = _save(tmp_path / "base.pt", {"model_state_dict": {"layer.weight": torch.zeros(2, 2)}})
    model_dir = tmp_path / "hf_model"
    model_dir.mkdir()
    save_torch_state_dict(
        {
            "layer.weight": torch.zeros(64, 64),
            "adapter.lora_A.weight": torch.ones(64, 16),
            "adapter.lora_B.weight": torch.ones(16, 64),
        },
        str(model_dir),
        safe_serialization=True,
        max_shard_size="1KB",
        filename_pattern="model{suffix}.safetensors",
    )
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    result = validate_model_checkpoint(
        base_checkpoint=base,
        trained_checkpoint=model_dir,
        output_dir=tmp_path,
        runtime_config=_runtime_config(lora_enabled=True, recurrent_enabled=False),
        validation_cfg=ModelValidationConfig(),
    )
    assert result.passed
    diff = json.loads((tmp_path / "model_validation_diff.json").read_text())
    assert diff["serialization"]["trained"]["serialization_format"] == "sharded safetensors"


def test_validation_detects_missing_shard_file(tmp_path: Path) -> None:
    base = _save(tmp_path / "base.pt", {"state_dict": {"layer.weight": torch.zeros(2, 2)}})
    model_dir = tmp_path / "hf_missing"
    model_dir.mkdir()
    save_torch_state_dict(
        {"layer.weight": torch.zeros(128, 128), "extra.weight": torch.zeros(128, 128)},
        str(model_dir),
        safe_serialization=True,
        max_shard_size="1KB",
        filename_pattern="model{suffix}.safetensors",
    )
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    missing = sorted(model_dir.glob("model-*.safetensors"))[0]
    missing.unlink()

    result = validate_model_checkpoint(
        base_checkpoint=base,
        trained_checkpoint=model_dir,
        output_dir=tmp_path,
        runtime_config=_runtime_config(lora_enabled=False, recurrent_enabled=False),
        validation_cfg=ModelValidationConfig(),
    )
    assert not result.passed
    assert any("missing shard" in item.lower() for item in result.missing_required_items)


def test_lora_and_recurrent_keys_detected_across_shards(tmp_path: Path) -> None:
    base = _save(tmp_path / "base.pt", {"model_state_dict": {"layer.weight": torch.zeros(2, 2)}})
    model_dir = tmp_path / "hf_recurrent_lora"
    model_dir.mkdir()
    save_torch_state_dict(
        {
            "layer.weight": torch.zeros(128, 128),
            "adapter.lora_A.weight": torch.ones(128, 16),
            "adapter.lora_B.weight": torch.ones(16, 128),
            "refiner.recurrent_projection.weight": torch.ones(128, 128),
        },
        str(model_dir),
        safe_serialization=True,
        max_shard_size="1KB",
        filename_pattern="model{suffix}.safetensors",
    )
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    result = validate_model_checkpoint(
        base_checkpoint=base,
        trained_checkpoint=model_dir,
        output_dir=tmp_path,
        runtime_config=_runtime_config(lora_enabled=True, recurrent_enabled=True),
        validation_cfg=ModelValidationConfig(),
    )
    assert result.passed
    diff = json.loads((tmp_path / "model_validation_diff.json").read_text())
    assert any("lora_A" in key for key in diff["lora_keys"])
    assert any("recurrent" in key for key in diff["recurrent_keys"])


def test_validation_accepts_explicitly_merged_lora_metadata(tmp_path: Path) -> None:
    base = _save(tmp_path / "base.pt", {"layer.weight": torch.zeros(2, 2)})
    model_dir = tmp_path / "hf_merged_lora"
    model_dir.mkdir()
    save_file({"layer.weight": torch.zeros(2, 2)}, str(model_dir / "model.safetensors"))
    (model_dir / "config.json").write_text(json.dumps({"lora_merged": True}), encoding="utf-8")

    result = validate_model_checkpoint(
        base_checkpoint=base,
        trained_checkpoint=model_dir,
        output_dir=tmp_path,
        runtime_config=_runtime_config(lora_enabled=True, recurrent_enabled=False),
        validation_cfg=ModelValidationConfig(),
    )
    assert result.passed


def test_validation_fails_when_expected_lora_or_recurrent_missing(tmp_path: Path) -> None:
    base = _save(tmp_path / "base.pt", {"layer.weight": torch.zeros(2, 2)})
    trained = _save(tmp_path / "trained.pt", {"layer.weight": torch.zeros(2, 2)})
    result = validate_model_checkpoint(
        base_checkpoint=base,
        trained_checkpoint=trained,
        output_dir=tmp_path,
        runtime_config=_runtime_config(lora_enabled=True, recurrent_enabled=True),
        validation_cfg=ModelValidationConfig(),
    )
    assert not result.passed
    assert any("lora" in item.lower() for item in result.missing_required_items)
    assert any("recurrent" in item.lower() for item in result.missing_required_items)


def test_validation_uses_any_lora_flag_for_config_check(tmp_path: Path) -> None:
    base = _save(tmp_path / "base.pt", {"model_state_dict": {"layer.weight": torch.zeros(2, 2)}})
    trained = _save(
        tmp_path / "trained.pt",
        {
            "model_state_dict": {
                "layer.weight": torch.zeros(2, 2),
                "refiner.w_in.weight": torch.zeros(2, 2),
                "refiner.w_out.weight": torch.zeros(2, 2),
                "refiner.adapter_bank.adapters.0.down.weight": torch.zeros(1, 2),
                "refiner.adapter_bank.adapters.0.up.weight": torch.zeros(2, 1),
            }
        },
    )
    result = validate_model_checkpoint(
        base_checkpoint=base,
        trained_checkpoint=trained,
        output_dir=tmp_path,
        runtime_config=_runtime_config_with_refiner_adapter_only(recurrent_enabled=True),
        validation_cfg=ModelValidationConfig(),
    )
    assert result.passed
    report_text = (tmp_path / "model_validation_report.md").read_text(encoding="utf-8")
    assert "variant.any_lora.enabled" in report_text


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
    st_path = tmp_path / "single.safetensors"
    save_file({"weight": torch.zeros(1)}, str(st_path))

    raw = load_checkpoint_state_dict(raw_state_path)
    nested = load_checkpoint_state_dict(nested_state_path)
    nested_model = load_checkpoint_state_dict(nested_model_state_path)
    safetensors_state = load_checkpoint_state_dict(st_path)

    assert "weight" in raw
    assert "weight" in nested
    assert "weight" in nested_model
    assert "weight" in safetensors_state
