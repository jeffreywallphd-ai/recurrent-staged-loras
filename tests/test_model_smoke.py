"""Model-path smoke tests for trainable baseline variants."""

from pathlib import Path

import torch
import torch.nn.functional as F

from training.config_loader import build_model_from_variant, load_variant_config


def _forward_loss_backward(config_name: str) -> tuple[object, object]:
    variant = load_variant_config(Path("experiments/configs") / config_name)
    model = build_model_from_variant(variant)
    model.train()

    input_ids = torch.tensor([[1, 2, 3, 4], [4, 5, 6, 7]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask)

    assert out.logits.shape == (2, 4, model.base_model.vocab_size)
    assert out.refined_hidden_states.shape[:2] == (2, 4)

    expected_steps = variant.refiner.num_steps if variant.refiner.enabled else 0
    assert len(out.extras["per_step"]) == expected_steps

    loss = F.cross_entropy(out.logits[:, :-1, :].reshape(-1, model.base_model.vocab_size), input_ids[:, 1:].reshape(-1))
    loss.backward()

    grad_param_names = {
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is not None
    }
    assert grad_param_names
    return model, grad_param_names


def test_base_forward_loss_backward_smoke() -> None:
    model, grad_names = _forward_loss_backward("base.json")
    assert model.refiner is None
    assert model.base_model.standard_lora_enabled is False
    assert not any("lora" in name for name in grad_names)


def test_standard_lora_forward_loss_backward_smoke() -> None:
    model, grad_names = _forward_loss_backward("standard_lora.json")
    assert model.refiner is None
    assert model.base_model.standard_lora_enabled is True
    assert any("lora" in name for name in grad_names)


def test_latent_refiner_only_forward_loss_backward_smoke() -> None:
    model, grad_names = _forward_loss_backward("latent_refiner_only.json")
    assert model.refiner is not None
    assert model.base_model.standard_lora_enabled is False
    assert any(name.startswith("refiner.") for name in grad_names)
    assert not any("adapter_bank" in name for name in grad_names)


def test_shared_recurrence_forward_loss_backward_smoke() -> None:
    model, grad_names = _forward_loss_backward("shared_recurrence.json")
    assert model.refiner is not None
    assert model.refiner.adapter_bank is not None
    assert len(model.refiner.adapter_bank.adapters) == 1
    assert any("adapter_bank" in name for name in grad_names)


def test_stage_specialized_recurrence_forward_loss_backward_smoke() -> None:
    model, grad_names = _forward_loss_backward("stage_specialized_recurrence.json")
    assert model.refiner is not None
    assert model.refiner.adapter_bank is not None
    assert len(model.refiner.adapter_bank.adapters) == model.config.refiner.num_steps
    assert any("adapter_bank.adapters.0" in name for name in grad_names)
