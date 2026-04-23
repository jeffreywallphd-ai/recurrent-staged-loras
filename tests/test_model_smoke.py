import torch

from models.config import parse_variant_config
from training.config_loader import build_model_from_variant


def _variant(baseline: str, architecture_type: str = "dense"):
    return parse_variant_config(
        {
            "baseline": baseline,
            "model": {
                "name": "test/tiny",
                "architecture_type": architecture_type,
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


def test_dense_and_moe_defaults_build() -> None:
    dense_model = build_model_from_variant(_variant("base", "dense"))
    moe_model = build_model_from_variant(_variant("base", "moe"))
    assert dense_model.config.base.architecture_type == "dense"
    assert moe_model.config.base.architecture_type == "moe"


def test_stage_aware_forward_path_has_per_step_states() -> None:
    model = build_model_from_variant(_variant("stage_specialized_recurrence"))
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    out = model(input_ids=x, attention_mask=torch.ones_like(x))
    assert len(out.extras["per_step"]) == 3
