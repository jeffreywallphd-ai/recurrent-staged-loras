# Statistical Analysis Plan (Dense vs MoE Staged Adaptation)

## Why MANOVA/PERMANOVA are not the primary analysis

This study combines mixed outcome types (accuracies, losses, perplexity, efficiency) with repeated runs by seed. A single omnibus MANOVA/PERMANOVA would impose assumptions and interpretability tradeoffs that are poorly matched to this setup. Instead, analysis is performed per metric with planned, pre-specified contrasts using run-level observations.

## Confirmatory strategy

Confirmatory claims are restricted to a small set of pre-specified primary outcomes:

- `final_answer_accuracy`
- `final_answer_exact_match`
- `normalized_numeric_answer_accuracy`

Planned confirmatory contrasts are performed within each architecture (`dense`, `moe`):

- `stage_specialized_recurrence` vs `standard_lora`
- `stage_specialized_recurrence` vs `shared_recurrence`
- `stage_specialized_recurrence` vs `latent_refiner_only`

Each run is one observation; seed is the repeated-run unit. Pairing by seed is required for the planned paired analysis whenever overlap exists.

## Family-wise error control

Raw p-values from all confirmatory tests (all primary outcomes × all planned contrasts across architectures) are adjusted using Holm correction. The adjusted p-values and rejection decisions are reported explicitly.

## Secondary and exploratory separation

Secondary and efficiency outcomes are analyzed and reported separately from confirmatory claims:

- Secondary outcomes: `final_eval_loss`, `eval_perplexity`, `stage_2_token_accuracy`, `stage_3_token_accuracy`
- Efficiency outcomes: `wall_time_seconds_total`, `tokens_per_second_train`, `trainable_param_fraction`, `effective_forward_passes_per_example`

These are descriptive by default and are not mixed into the confirmatory family-wise correction pool unless explicitly reconfigured.

## Artifacts

`analysis/statistical_analysis.py` writes:

- `outputs/statistical_analysis_confirmatory.json`
- `outputs/statistical_analysis_confirmatory.csv`
- `outputs/statistical_analysis_secondary.json`
- `outputs/statistical_analysis_efficiency.json`
- `outputs/statistical_analysis_report.md`
