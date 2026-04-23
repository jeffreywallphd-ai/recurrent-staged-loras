"""Smoke tests for configuration templates and baseline validation."""

from pathlib import Path

from training.baseline_selector import select_baseline
from training.config_loader import build_model_from_variant, load_experiment_config
from models.config import parse_variant_config


def test_baseline_selector_accepts_all_known_baselines() -> None:
    baselines = {
        "base",
        "standard_lora",
        "latent_refiner_only",
        "shared_recurrence",
        "stage_specialized_recurrence",
    }
    for baseline in baselines:
        assert select_baseline({"baseline": baseline}) == baseline


def test_all_baseline_configs_load_and_validate_selection() -> None:
    config_dir = Path("experiments/configs")
    config_names = [
        "base.json",
        "standard_lora.json",
        "latent_refiner_only.json",
        "shared_recurrence.json",
        "stage_specialized_recurrence.json",
    ]

    for config_name in config_names:
        cfg = load_experiment_config(config_dir / config_name)
        assert select_baseline(cfg) == cfg["baseline"]
        variant = parse_variant_config(cfg)
        model = build_model_from_variant(variant)
        if cfg["baseline"] == "standard_lora":
            assert model.base_model.standard_lora_enabled is True
        if cfg["baseline"] == "base":
            assert model.base_model.standard_lora_enabled is False


def test_latent_refiner_only_uses_explicit_non_adapterized_mode() -> None:
    cfg = load_experiment_config(Path("experiments/configs/latent_refiner_only.json"))
    latent_refiner = cfg["model"]["latent_refiner"]

    assert latent_refiner["enabled"] is True
    assert latent_refiner["recurrence_mode"] == "latent_only"
    assert latent_refiner["adapter_sharing"] == "none"
