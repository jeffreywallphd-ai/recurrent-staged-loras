from pathlib import Path

from models.config import parse_variant_config
from training.config_loader import load_experiment_config, load_runtime_config


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


def test_all_config_families_load_with_runtime_config() -> None:
    config_dir = Path("experiments/configs")
    families = {
        "confirmatory": sorted(p for p in config_dir.glob("*.json") if all(token not in p.stem for token in ("_pilot", "_debug", "_external_eval", "_compute_controlled", "_ablation"))),
        "pilot": sorted(config_dir.glob("*_pilot.json")),
        "debug": sorted(config_dir.glob("*_debug*.json")),
        "external_eval": sorted(config_dir.glob("*_external_eval.json")),
        "compute_controlled": sorted(config_dir.glob("*_compute_controlled.json")),
        "ablation": sorted(config_dir.glob("*_ablation.json")),
    }
    for family, paths in families.items():
        assert paths, f"expected at least one config for family {family}"
        for path in paths:
            load_runtime_config(path)


def test_robust_study_workflow_matrix_is_shipped_for_dense_and_moe() -> None:
    config_dir = Path("experiments/configs")
    required_baselines = [
        "base",
        "standard_lora",
        "latent_refiner_only",
        "shared_recurrence",
        "stage_specialized_recurrence",
    ]
    family_suffix_by_key = {
        "confirmatory": "",
        "compute_controlled": "_compute_controlled",
        "external_eval": "_external_eval",
        "debug": "_debug",
    }

    for architecture_prefix in ("", "moe_"):
        for baseline in required_baselines:
            for family_key, suffix in family_suffix_by_key.items():
                name = f"{architecture_prefix}{baseline}{suffix}.json"
                path = config_dir / name
                assert path.exists(), f"missing {family_key} config for {architecture_prefix or 'dense_'}{baseline}"
                load_runtime_config(path)

    ablation_compatible_baselines = [
        "latent_refiner_only",
        "shared_recurrence",
        "stage_specialized_recurrence",
        "standard_lora",
    ]
    for architecture_prefix in ("", "moe_"):
        for baseline in ablation_compatible_baselines:
            path = config_dir / f"{architecture_prefix}{baseline}_ablation.json"
            assert path.exists(), f"missing ablation config for {architecture_prefix or 'dense_'}{baseline}"
            load_runtime_config(path)


def test_external_eval_entries_are_normalized() -> None:
    runtime = load_runtime_config(Path("experiments/configs/stage_specialized_recurrence_debug_external_eval.json"))
    external = runtime.dataset["external_evaluations"]
    assert len(external) == 3
    for item in external:
        assert item["split"] == "test"
        assert int(item["subset_size"]) == 3
        assert int(item["seed"]) == 3


def test_tiny_external_subset_debug_path_is_represented_in_shipped_configs() -> None:
    debug_external_configs = [
        Path("experiments/configs/stage_specialized_recurrence_debug_external_eval.json"),
        Path("experiments/configs/moe_stage_specialized_recurrence_debug_external_eval.json"),
    ]
    for path in debug_external_configs:
        runtime = load_runtime_config(path)
        external = {item["name"]: item for item in runtime.dataset["external_evaluations"]}
        assert external["gsm8k"]["subset_size"] == 3
        assert external["math"]["subset_size"] == 3
        assert external["svamp"]["subset_size"] == 3
