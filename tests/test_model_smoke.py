"""Model-path smoke tests for baseline trainability ownership."""

from pathlib import Path

import torch
import torch.nn.functional as F

from training.config_loader import build_model_from_variant, load_variant_config


def _snapshot_params(model: object) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters()}


def _forward_loss_backward(config_name: str) -> tuple[object, set[str], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
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
    before = _snapshot_params(model)

    optimizer = None
    if any(param.requires_grad for param in model.parameters()):
        optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=1e-2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    after = _snapshot_params(model)
    grad_param_names = {name for name, param in model.named_parameters() if param.requires_grad and param.grad is not None}
    return model, grad_param_names, before, after


def _assert_internal_base_is_frozen(model: object) -> None:
    internal = model.base_model.internal_model
    assert internal is not None
    for name, param in internal.named_parameters():
        if "lora_" in name:
            continue
        assert param.requires_grad is False, f"expected frozen base param: {name}"


def _changed_param_names(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> set[str]:
    changed = set()
    for name, previous in before.items():
        current = after[name]
        if not torch.equal(previous, current):
            changed.add(name)
    return changed


def test_base_forward_loss_backward_smoke_and_frozen_ownership() -> None:
    model, grad_names, before, after = _forward_loss_backward("base.json")
    assert model.refiner is None
    assert model.base_model.standard_lora_enabled is False
    _assert_internal_base_is_frozen(model)
    assert not grad_names
    assert not _changed_param_names(before, after)


def test_standard_lora_forward_loss_backward_smoke_and_ownership() -> None:
    model, grad_names, before, after = _forward_loss_backward("standard_lora.json")
    assert model.refiner is None
    assert model.base_model.standard_lora_enabled is True
    _assert_internal_base_is_frozen(model)
    assert grad_names
    assert all("lora_" in name for name in grad_names)

    changed = _changed_param_names(before, after)
    assert changed
    assert all("lora_" in name for name in changed)


def test_latent_refiner_only_forward_loss_backward_smoke_and_ownership() -> None:
    model, grad_names, before, after = _forward_loss_backward("latent_refiner_only.json")
    assert model.refiner is not None
    assert model.base_model.standard_lora_enabled is False
    _assert_internal_base_is_frozen(model)
    assert grad_names
    assert all(name.startswith("refiner.") for name in grad_names)
    assert not any("adapter_bank" in name for name in grad_names)

    changed = _changed_param_names(before, after)
    assert changed
    assert all(name.startswith("refiner.") for name in changed)


def test_shared_recurrence_forward_loss_backward_smoke_and_ownership() -> None:
    model, grad_names, before, after = _forward_loss_backward("shared_recurrence.json")
    assert model.refiner is not None
    assert model.refiner.adapter_bank is not None
    assert len(model.refiner.adapter_bank.adapters) == 1
    _assert_internal_base_is_frozen(model)
    assert any(name.startswith("refiner.") for name in grad_names)
    assert any("adapter_bank.adapters.0" in name for name in grad_names)

    changed = _changed_param_names(before, after)
    assert any(name.startswith("refiner.") for name in changed)
    assert any("adapter_bank.adapters.0" in name for name in changed)


def test_stage_specialized_recurrence_forward_loss_backward_smoke_and_ownership() -> None:
    model, grad_names, before, after = _forward_loss_backward("stage_specialized_recurrence.json")
    assert model.refiner is not None
    assert model.refiner.adapter_bank is not None
    assert len(model.refiner.adapter_bank.adapters) == model.config.refiner.num_steps
    _assert_internal_base_is_frozen(model)
    assert any(name.startswith("refiner.") for name in grad_names)
    assert any("adapter_bank.adapters.0" in name for name in grad_names)
    assert any("adapter_bank.adapters.1" in name for name in grad_names)

    changed = _changed_param_names(before, after)
    assert any(name.startswith("refiner.") for name in changed)
    assert any("adapter_bank.adapters.0" in name for name in changed)
    assert any("adapter_bank.adapters.1" in name for name in changed)
