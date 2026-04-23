from pathlib import Path

from models.config import parse_variant_config
from training.config_loader import load_experiment_config


def test_default_dense_and_moe_models_are_real() -> None:
    dense = load_experiment_config(Path("experiments/configs/base.json"))
    moe = load_experiment_config(Path("experiments/configs/moe_base.json"))

    assert dense["model"]["name"] == "Qwen/Qwen3-8B"
    assert dense["model"]["architecture_type"] == "dense"
    assert moe["model"]["name"] == "allenai/OLMoE-1B-7B-0125-Instruct"
    assert moe["model"]["architecture_type"] == "moe"


def test_default_dataset_is_real_not_synthetic() -> None:
    cfg = load_experiment_config(Path("experiments/configs/standard_lora.json"))
    assert cfg["dataset"]["name"] == "metamath_qa"
    assert int(cfg["dataset"]["settings"]["subset_size"]) >= 25_000


def test_variant_model_fields_parse_full_real_backend_schema() -> None:
    cfg = load_experiment_config(Path("experiments/configs/base.json"))
    variant = parse_variant_config(cfg)
    assert variant.base.tokenizer_name
    assert variant.base.dtype == "bfloat16"
    assert variant.base.device_map == "auto"
    assert variant.base.max_seq_length > 0
    assert variant.base.gradient_checkpointing is True


def test_study_and_pilot_presets_are_separated() -> None:
    study = load_experiment_config(Path("experiments/configs/standard_lora.json"))
    pilot = load_experiment_config(Path("experiments/configs/standard_lora_pilot.json"))
    assert int(study["training"]["max_steps"]) > int(pilot["training"]["max_steps"])
