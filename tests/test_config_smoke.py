"""Smoke tests for configuration templates and baseline validation."""

from pathlib import Path

from training.baseline_selector import select_baseline
from training.config_loader import load_experiment_config


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
