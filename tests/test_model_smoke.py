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


def _first_refiner_param(model):
    assert model.refiner is not None
    return next(model.refiner.parameters())


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


def test_bfloat16_backbone_alignment_prevents_refiner_matmul_mismatch() -> None:
    variant = _variant("stage_specialized_recurrence")
    variant.base.dtype = "bfloat16"
    model = build_model_from_variant(variant)
    x = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    out = model(input_ids=x, attention_mask=torch.ones_like(x))
    assert out.refined_hidden_states.dtype == torch.bfloat16
    assert _first_refiner_param(model).dtype == torch.bfloat16


def test_stage_specialized_adapter_bank_matches_hidden_state_dtype_and_device() -> None:
    variant = _variant("stage_specialized_recurrence")
    variant.base.dtype = "bfloat16"
    model = build_model_from_variant(variant)
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = model(input_ids=x, attention_mask=torch.ones_like(x))
    assert out.refined_hidden_states.dtype == torch.bfloat16
    for adapter in model.refiner.adapter_bank.adapters:  # type: ignore[union-attr]
        param = next(adapter.parameters())
        assert param.dtype == torch.bfloat16
        assert param.device == out.refined_hidden_states.device


def test_shared_recurrence_adapter_matches_hidden_state_dtype_and_device() -> None:
    variant = _variant("shared_recurrence")
    variant.base.dtype = "bfloat16"
    model = build_model_from_variant(variant)
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = model(input_ids=x, attention_mask=torch.ones_like(x))
    assert out.refined_hidden_states.dtype == torch.bfloat16
    adapter_param = next(model.refiner.adapter_bank.adapters[0].parameters())  # type: ignore[union-attr]
    assert adapter_param.dtype == torch.bfloat16
    assert adapter_param.device == out.refined_hidden_states.device


def test_latent_refiner_only_matches_hidden_state_dtype_and_device() -> None:
    variant = _variant("latent_refiner_only")
    variant.base.dtype = "bfloat16"
    model = build_model_from_variant(variant)
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = model(input_ids=x, attention_mask=torch.ones_like(x))
    assert out.refined_hidden_states.dtype == torch.bfloat16
    assert _first_refiner_param(model).dtype == torch.bfloat16


def test_float32_backbone_still_works() -> None:
    variant = _variant("shared_recurrence")
    variant.base.dtype = "float32"
    model = build_model_from_variant(variant)
    x = torch.tensor([[1, 2, 3]], dtype=torch.long)
    out = model(input_ids=x, attention_mask=torch.ones_like(x))
    assert out.refined_hidden_states.dtype == torch.float32
