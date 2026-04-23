"""Model-path smoke tests for scaffolded baselines."""

from pathlib import Path

from training.config_loader import build_model_from_variant, load_variant_config


def _smoke_forward(config_name: str) -> None:
    variant = load_variant_config(Path("experiments/configs") / config_name)
    model = build_model_from_variant(variant)

    input_ids = [[1, 2, 3], [4, 5, 6]]
    attention_mask = [[1, 1, 1], [1, 1, 1]]
    out = model.forward(input_ids=input_ids, attention_mask=attention_mask)

    assert len(out.logits) == 2
    assert len(out.logits[0]) == 3
    assert len(out.logits[0][0]) == model.base_model.vocab_size
    assert len(out.refined_hidden_states) == 2
    assert len(out.refined_hidden_states[0]) == 3

    expected_steps = variant.refiner.num_steps if variant.refiner.enabled else 0
    assert len(out.extras["per_step"]) == expected_steps


def test_base_forward_smoke() -> None:
    _smoke_forward("base.json")


def test_latent_refiner_only_forward_smoke() -> None:
    _smoke_forward("latent_refiner_only.json")


def test_shared_recurrence_forward_smoke() -> None:
    _smoke_forward("shared_recurrence.json")


def test_stage_specialized_recurrence_forward_smoke() -> None:
    _smoke_forward("stage_specialized_recurrence.json")
