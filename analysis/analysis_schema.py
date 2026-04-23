"""Canonical statistical analysis configuration for staged adaptation study."""

from __future__ import annotations

from dataclasses import dataclass


PRIMARY_CONFIRMATORY_OUTCOMES = [
    "final_answer_accuracy",
    "final_answer_exact_match",
    "normalized_numeric_answer_accuracy",
]

SECONDARY_OUTCOMES = [
    "final_eval_loss",
    "eval_perplexity",
    "stage_2_token_accuracy",
    "stage_3_token_accuracy",
]

EFFICIENCY_OUTCOMES = [
    "wall_time_seconds_total",
    "tokens_per_second_train",
    "trainable_param_fraction",
    "effective_forward_passes_per_example",
]

ARCHITECTURES = ["dense", "moe"]

STAGE_SPECIALIZED_BASELINE = "stage_specialized_recurrence"
PLANNED_COMPARATORS = [
    "standard_lora",
    "shared_recurrence",
    "latent_refiner_only",
]


@dataclass(frozen=True)
class PlannedContrast:
    architecture_type: str
    baseline_a: str
    baseline_b: str


def build_confirmatory_contrasts() -> list[PlannedContrast]:
    """Return canonical confirmatory contrasts for all architectures."""
    return [
        PlannedContrast(
            architecture_type=architecture,
            baseline_a=STAGE_SPECIALIZED_BASELINE,
            baseline_b=comparator,
        )
        for architecture in ARCHITECTURES
        for comparator in PLANNED_COMPARATORS
    ]


CONFIRMATORY_FWER_METHOD = "holm"
ALPHA = 0.05

REQUIRED_ID_COLUMNS = ["architecture_type", "baseline_name", "model_name", "dataset_name", "config_name", "seed"]
