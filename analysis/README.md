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

## Grouping and pairing rules (strict)

Run grouping key is exact condition identity by:

- `architecture_type`
- `baseline_name`
- `model_name`
- `dataset_name`
- `config_name`
- `seed`

Confirmatory comparisons enforce homogeneous condition families before any test:

- each baseline arm in a planned contrast must map to exactly one `(model_name, dataset_name, config_name)` family;
- both arms must share that same family.

If a contrast contains mixed families (for example study + pilot runs or mismatched config names), analysis fails loudly with an explicit error and does not merge rows.

## Confirmatory eligibility and tests

A confirmatory row is eligible for Holm correction only when all are true:

- planned primary outcome;
- planned contrast;
- paired-by-seed overlap exists;
- confirmatory p-value is non-null.

Primary confirmatory test backend:

- `scipy.stats.wilcoxon` (two-sided, `zero_method="wilcox"`, `method="auto"`).

Sensitivity-only test:

- `scipy.stats.ttest_rel` (two-sided paired t-test).

If `scipy` is unavailable, analysis errors clearly; it does not fall back to hand-rolled approximations for confirmatory inference.

## Family-wise error control

Raw p-values from confirmatory-eligible rows only (all primary outcomes × planned contrasts × architectures, excluding downgraded rows) are adjusted using Holm correction. Adjusted p-values and rejection decisions are reported explicitly.

## Unpaired fallback behavior (`--allow-unpaired`)

If no seed overlap exists:

- without `--allow-unpaired`: raise a hard error for primary confirmatory rows;
- with `--allow-unpaired`: produce descriptive downgraded rows only for impacted primary comparisons:
  - `analysis_tier = "descriptive_downgraded"`
  - `raw_p_value = null`
  - `holm_adjusted_p_value = null`
  - `reject_after_holm = null`
  - excluded from Holm correction pool.

## Interval estimates

For paired rows, report a reproducible bootstrap percentile 95% CI for paired mean difference:

- 5000 bootstrap resamples
- fixed RNG seed = 0
- fields: `mean_difference_ci_low`, `mean_difference_ci_high`.

## Secondary and exploratory separation

Secondary and efficiency outcomes are analyzed and reported separately from confirmatory claims:

- Secondary outcomes: `final_eval_loss`, `eval_perplexity`, `stage_2_token_accuracy`, `stage_3_token_accuracy`
- Efficiency outcomes: `wall_time_seconds_total`, `tokens_per_second_train`, `trainable_param_fraction`, `effective_forward_passes_per_example`

These are descriptive by default and are not mixed into the confirmatory family-wise correction pool.

## Artifacts

`analysis/statistical_analysis.py` writes:

- `outputs/statistical_analysis_confirmatory.json`
- `outputs/statistical_analysis_confirmatory.csv`
- `outputs/statistical_analysis_secondary.json`
- `outputs/statistical_analysis_efficiency.json`
- `outputs/statistical_analysis_report.md`
- `outputs/statistical_analysis_metadata.json`
