# Experiment Protocol (Dense vs MoE Staged Adaptation Study)

## Model families

- **Dense default family:** `Qwen/Qwen3-8B`.
- **MoE default family:** `allenai/OLMoE-1B-7B-0125-Instruct`.
- Both families run the same baseline matrix with architecture held explicit via `model.architecture_type` (`dense`/`moe`).

## Baseline matrix

1. `base` (frozen backbone only)
2. `standard_lora` (real PEFT LoRA on target modules)
3. `latent_refiner_only`
4. `shared_recurrence`
5. `stage_specialized_recurrence`

Dense and MoE configs are paired per baseline for controlled cross-architecture comparison.

## Explicit study matrix

See `docs/study_matrix.md` for reportable vs pilot presets, required seeds, and fairness metrics.

## Dataset and staging protocol

Default dataset is `meta-math/MetaMathQA` (`dataset.name = metamath_qa`).
Optional external evaluation datasets can be configured with:

```json
"dataset": {
  "external_evaluations": [
    {"name": "gsm8k", "split": "test"},
    {"name": "math", "split": "test"},
    {"name": "svamp", "split": "test"}
  ]
}
```

External evaluation is opt-in only. When enabled, each dataset is evaluated after primary eval and written under `external_eval.<dataset>.*` in `metrics.json`.
External eval rows are descriptive-only by default, are labeled `report_tier=external_eval`, and are excluded from primary aggregates and confirmatory inference unless an explicit analysis override is set.
Each `external_eval.<dataset>` payload is self-contained: metric values and the external dataset identity fields are stored together (`dataset_name`, `dataset_type=external`, `dataset_split`, `dataset_seed`, `dataset_subset_size`, `dataset_eval_fraction`, `dataset_fingerprint`, `train_sample_ids_hash`, `eval_sample_ids_hash`).

Each sample is converted into a staged sequence:

1. `Stage 1` = `Problem:` prefix + original query text
2. `Stage 2` = `Reasoning:` section extracted from response
3. `Stage 3` = `Final Answer:` section extracted from response

Answer extraction order:
- split on final `####` marker when present,
- else extract last `\\boxed{...}` answer,
- else extract suffix after final `the answer is` marker,
- else mark stage-3 content empty and filter the row.

Token-level stage masks are created from tokenizer offset mappings and made disjoint (`stage1_mask`, `stage2_mask`, `stage3_mask`).
An explicit `answer_mask` (legacy alias: `final_answer_mask`) is also created for answer-only tokens inside Stage 3, excluding the literal `Final Answer:` header.
Rows without stage-3 tokens are filtered.

Per-run preprocessing statistics are written to `dataset_preprocessing_summary.json`, including:
- kept examples,
- filtered examples (empty/problem-response missing, stage-3 missing),
- examples with non-empty stage-2 and stage-3,
- samples with valid answer-only spans,
- samples with numeric answers,
- samples excluded or degraded (filtered + empty answer-span after tokenization).
- deterministic dataset identity artifacts:
  - `dataset_fingerprint`
  - `train_sample_ids_hash`
  - `eval_sample_ids_hash`
  - `dataset_split`, `dataset_seed`, `dataset_subset_size`, `dataset_eval_fraction`

Dataset fingerprint protocol (stable SHA-256 over canonical JSON) includes:
- dataset name
- upstream split
- dataset seed
- preprocessing settings
- selected sample identities hash
- train/eval sample-ID hashes and counts

## Train/eval protocol

- Backbone is frozen in the main study path.
- Stage-aware loss aligns recurrence steps to stage masks.
- `eval_interval_steps` is true **step-based** periodic eval (not epoch-only).
- Final eval runs at the end unless the final step already landed on an interval boundary.
- Eval token/time totals include all interval eval passes plus final eval (when present).

## Answer-span protocol and metrics semantics

- `stage_2_token_accuracy`: token accuracy on `stage2_mask` (reasoning section).
- `stage_3_token_accuracy`: token accuracy on `stage3_mask` (full Stage 3 section, including `Final Answer:` header tokens).
- `final_answer_accuracy`: compatibility metric; normalized string match on decoded `answer_mask` span only.
- `final_answer_exact_match`: strict raw-string equality on decoded answer text (outer whitespace trimmed only).
- `final_answer_normalized_match`: reviewer-facing normalized string metric using layered normalization (case fold, math-format cleanup including `$...$` and `\boxed{...}`, punctuation/spacing cleanup, semantic numeric canonicalization).
- `symbolic_answer_accuracy`: symbolic-equivalence metric for expression-like answers only. The evaluator attempts SymPy parsing/equivalence (`simplify`/`equals`) when either side looks math-expression-like.
- `normalized_numeric_answer_accuracy`: numeric answer metric on `answer_mask` spans. Numeric values are extracted (integers, decimals, scientific notation, signed values, simple fractions). Multi-value evaluation defaults to `strict_set` (set equality under tolerance-aware matching, order-invariant). Alternate rules (`subset`, `any`) are diagnostic-only and must be explicitly selected.
- Reviewer-facing aliases are also written: `answer_span_normalized_accuracy`, `answer_span_exact_match`, `answer_span_normalized_match`, and `answer_span_numeric_accuracy`.

Limitations:
- Symbolic evaluation is conditional: non-expression answers are not attempted, and parse failures are counted explicitly.
- SymPy equivalence is conservative and may not cover all mathematical notations emitted by language models.
- Multi-value numeric scoring is tolerance-based set matching and still does not prove symbolic equivalence of non-numeric structured outputs.

## Canonical run-level metrics schema (`metrics.json`)

Official fields:
`run_name`, `config_name`, `baseline_name`, `baseline_family`, `run_scope`, `dataset_name`, `dataset_type`, `dataset_split`, `dataset_seed`, `dataset_subset_size`, `dataset_eval_fraction`, `dataset_fingerprint`, `train_sample_ids_hash`, `eval_sample_ids_hash`, `dataset_train_examples`, `dataset_eval_examples`, `seed`, `architecture_type`, `model_name`, `final_train_loss`, `final_eval_loss`, `best_eval_loss`, `eval_perplexity`, `train_perplexity`, `stage_2_token_accuracy`, `stage_3_token_accuracy`, `final_answer_accuracy`, `final_answer_exact_match`, `final_answer_normalized_match`, `symbolic_answer_accuracy`, `normalized_numeric_answer_accuracy`, `answer_span_normalized_accuracy`, `answer_span_exact_match`, `answer_span_normalized_match`, `answer_span_numeric_accuracy`, `answer_eval_string_count`, `answer_eval_numeric_count`, `answer_eval_skipped_no_stage3`, `answer_eval_skipped_no_answer_span`, `answer_eval_skipped_missing_answer_text`, `answer_eval_skipped_missing_numeric_target`, `answer_eval_normalized_match_count`, `answer_eval_exact_match_count`, `answer_eval_numeric_match_count`, `answer_eval_multi_value_target_count`, `answer_eval_numeric_pred_value_count`, `answer_eval_numeric_target_value_count`, `answer_eval_numeric_value_match_count`, `answer_eval_multi_value_exact_set_match_count`, `answer_eval_multi_value_partial_match_count`, `answer_eval_multi_value_unmatched_count`, `answer_eval_string_match_numeric_miss_count`, `answer_eval_normalized_only_count`, `answer_eval_skipped_ambiguous_numeric`, `symbolic_eval_attempt_count`, `symbolic_eval_success_count`, `symbolic_eval_failure_count`, `answer_eval_symbolic_match_count`, `answer_eval_numeric_abs_tolerance`, `answer_eval_numeric_multi_value_rule`, `answer_eval_answer_length_histogram`, `wall_time_seconds_total`, `wall_time_seconds_train`, `wall_time_seconds_eval`, `tokens_seen_train`, `tokens_seen_eval`, `tokens_per_second_train`, `tokens_per_second_eval`, `steps_per_second`, `seconds_per_step`, `trainable_params`, `total_params`, `trainable_param_fraction`, `recurrence_steps`, `effective_forward_passes_per_example`, `compute_control_enabled`, `compute_control_mode`, `adjusted_max_steps`, `effective_optimizer_steps`, `tokens_per_optimizer_step`, `ablation_recurrent_steps`, `ablation_lora_rank`, `external_eval`, `global_steps_completed`, `epochs_completed`, `backend`, `latent_cache`.

Each run also writes `answer_eval_diagnostics.json` with answer-scoring counts and skip reasons.

## Compute fairness protocol

Comparisons report:
- trainable parameter fraction,
- recurrence steps / effective forward passes per example,
- tokens/sec and total wall-clock time.

These are used together with quality metrics to interpret performance vs compute budget.

### Compute-controlled comparison (opt-in)

Config:

```json
"training": {
  "compute_control": {
    "enabled": true,
    "mode": "effective_forward_passes"
  }
}
```

Supported modes:
- `effective_forward_passes`: adjusts max training steps by recurrence factor.
- `tokens`: enforces `training.compute_control.max_tokens`.
- `wall_time`: enforces `training.compute_control.max_wall_time_seconds`.

Run outputs now include: `compute_control_enabled`, `compute_control_mode`, `effective_forward_passes_per_example`, `adjusted_max_steps`, `effective_optimizer_steps`, `tokens_per_optimizer_step`.
Important interpretation: `effective_forward_passes_per_example` equalizes forward-pass budget only; it does **not** equalize optimizer dynamics. Token and wall-time modes alter stopping criteria differently and should not be interpreted as identical optimization trajectories.

### Ablation design (opt-in)

Config:

```json
"ablations": {
  "recurrent_steps": [1, 2, 3, 4],
  "lora_rank": [4, 8, 16]
}
```

`scripts/run_all_experiments.py` expands only requested ablation dimensions with strict compatibility checks:
- `ablations.recurrent_steps` is valid only when `model.latent_refiner.enabled=true`.
- `ablations.lora_rank` is routed to active adapters only:
  - `model.standard_lora.rank` when `standard_lora.enabled=true`
  - `model.latent_refiner.adapter.rank` when `latent_refiner.adapter.enabled=true`
  - both when both adapters are active
  - hard error when neither adapter is active (`lora_rank ablation requested but no active adapter found`).
- one-dimensional ablations are fully supported (`_r{steps}` recurrence-only, `_rank{rank}` rank-only), and two-dimensional ablations use `_r{steps}_rank{rank}`.
- unused ablation axes remain `null` in `ablation_recurrent_steps` / `ablation_lora_rank`.

Run-scope controls:
- `--run-scope confirmatory` (default): excludes configs with ablations.
- `--run-scope ablation`: runs only ablation-derived configs.
- `--run-scope all`: includes both.

Derived run names append only the requested ablation suffixes and metrics include `ablation_recurrent_steps` and `ablation_lora_rank`.
Naming is explicit: `baseline_name` keeps full derived name, while `baseline_family` is the original pre-ablation baseline.

## Multi-seed aggregation protocol

`python scripts/run_all_experiments.py --config-dir experiments/configs --seeds 11 22 33 --preset-scope study` writes:
- per-run `metrics.json` files,
- `outputs/summary.json` with `runs` and grouped `aggregates`,
- `outputs/aggregates.json` (aggregate-only artifact),
- `outputs/summary.csv` with both run rows and aggregate rows,
- `outputs/report_table.csv` with report-ready run and aggregate records.

Aggregates are grouped by:
- `baseline_name`
- `architecture_type`
- `model_name`
- `dataset_name`
- `config_name`
- `compute_control_enabled`
- `compute_control_mode`
- `recurrence_steps`
- `ablation_recurrent_steps`
- `ablation_lora_rank`
- `run_scope`

Aggregation is fail-fast: if any required control field is heterogeneous inside a candidate group, aggregation errors instead of averaging incompatible runs.

For each group, mean/std are reported for:
- `final_eval_loss`
- `eval_perplexity`
- `stage_2_token_accuracy`
- `stage_3_token_accuracy`
- `final_answer_accuracy`
- `final_answer_exact_match`
- `symbolic_answer_accuracy`
- `normalized_numeric_answer_accuracy`
- `wall_time_seconds_total`
- `tokens_per_second_train`
- `trainable_param_fraction`

## Presets: study vs pilot

- **Study presets:** base config names (for reportable runs).
- **Pilot presets:** `*_pilot.json` variants for smoke checks and quick iteration.
- Default orchestration mode is reportable study runs only: `--preset-scope study`.

## Which config should I use?

Use explicit suffix-based families in `experiments/configs`:

- **Confirmatory (reportable):** `*.json` with no family suffixes.
- **Pilot (non-reportable):** `*_pilot.json`.
- **Compute-controlled (reportable/descriptive depending analysis plan):** `*_compute_controlled.json`.
- **External-eval checks (descriptive by default):** `*_external_eval.json`.
- **Ablations (non-confirmatory unless explicitly opted into analysis):** `*_ablation.json`.
- **Debug/smoke (non-reportable):** `*_debug.json` and `*_debug_external_eval.json`.

Recommended workflows:

1. **Full paper runs:** confirmatory family + seeds `11 22 33`.
2. **Pilot runs:** `_pilot` family for faster end-to-end checks.
3. **Quick bug checks:** `_debug` family (very small subset + low steps).
4. **External evaluation checks:** `_external_eval` family (adds GSM8K/MATH/SVAMP entries).
5. **Ablation sweeps:** `_ablation` family + `--run-scope ablation`.
6. **Compute-controlled comparisons:** `_compute_controlled` family.

You can now select these directly:

```bash
python scripts/run_all_experiments.py \
  --config-dir experiments/configs \
  --config-family debug \
  --preset-scope all \
  --seeds 7
```

```bash
python scripts/run_all_experiments.py \
  --config-dir experiments/configs \
  --config-family compute_controlled \
  --preset-scope all \
  --seeds 11 22 33
```

### Ultra-fast debug/smoke external eval example (non-reportable)

`stage_specialized_recurrence_debug_external_eval.json` and `moe_stage_specialized_recurrence_debug_external_eval.json` are intentionally tiny and debugging-oriented:
- primary `subset_size` of `16`
- `batch_size=1`
- very low `max_steps`
- single-seed friendly
- tiny external subsets:
  - `gsm8k`: `subset_size=3`
  - `math`: `subset_size=3`
  - `svamp`: `subset_size=3`

### External dataset settings (per dataset entry)

Each object under `dataset.external_evaluations` supports:

- `name` (required): one of `gsm8k`, `math`, `svamp`
- `split` (optional, default `test`)
- `subset_size` (optional, default `0` = full split)
- `seed` (optional, default `0`)

Example:

```json
"dataset": {
  "external_evaluations": [
    {"name": "gsm8k", "split": "test", "subset_size": 3, "seed": 3},
    {"name": "math", "split": "test", "subset_size": 3, "seed": 3}
  ]
}
```

## Paper workflow (explicit)

1. **Reportable runs only:** execute study configs (`experiments/configs/*.json`) with required seeds `11,22,33`; keep pilot configs out of paper numbers.
2. **Primary headline metrics:** `final_answer_accuracy`, `final_answer_exact_match`, `normalized_numeric_answer_accuracy`.
3. **Secondary diagnostics:** `stage_2_token_accuracy`, `stage_3_token_accuracy`, loss/perplexity, and compute/fairness metrics.
4. **Artifact usage:**
   - `outputs/report_table.csv`: canonical paper-table source (run rows + aggregate rows with explicit `row_type`/`report_tier`).
   - `outputs/aggregates.json`: canonical aggregate plotting source.
   - `outputs/summary.json`: full machine-readable archive (run + aggregate + invocation context).
   - `outputs/summary.csv`: wide mixed export for quick spreadsheet inspection.

## Statistical inference workflow (confirmatory vs descriptive)

- **Primary confirmatory outcomes:** `final_answer_accuracy`, `final_answer_exact_match`, `normalized_numeric_answer_accuracy`.
- **Planned confirmatory contrasts (within each architecture):**
  - `stage_specialized_recurrence` vs `standard_lora`
  - `stage_specialized_recurrence` vs `shared_recurrence`
  - `stage_specialized_recurrence` vs `latent_refiner_only`
- **Homogeneous comparison family requirement:** confirmatory contrasts must match on `architecture_type`, `model_name`, and `dataset_name`; contrasted baselines may have different `config_name` values. Output rows retain `config_name_a` and `config_name_b` for audit traceability.
- **Repeated-run unit:** run-level metrics with pairing by `seed` when overlap exists.
- **Primary test:** `scipy.stats.wilcoxon` (two-sided, `zero_method="wilcox"`, `method="auto"`) per metric/contrast; `scipy.stats.ttest_rel` reported as sensitivity only.
- **Family-wise error control:** Holm correction across confirmatory-eligible rows only (primary outcomes × planned contrasts × architectures, excluding downgraded descriptive rows with null p-values).
- **Pairing fallback:** with `--allow-unpaired`, non-overlap yields descriptive downgraded rows (`raw_p_value=null`) excluded from Holm.
- **Interval reporting:** bootstrap percentile 95% CI for paired mean difference (`mean_difference_ci_low/high`, 5000 resamples, fixed RNG seed=0).
- **Secondary and efficiency outcomes:** analyzed in separate descriptive artifacts and not pooled into confirmatory correction by default.
- **Confirmatory purity defaults:** confirmatory analysis fails on ablation-derived rows and pilot rows unless explicitly overridden (`--allow-ablations-in-analysis`, `--allow-pilot-runs-in-analysis`).
- **External evaluation analysis mode:** external datasets remain descriptive by default and are analyzable only when explicitly selected via `--dataset-scope external` (or included in `--dataset-scope all`); confirmatory inference remains primary-dataset-only by default.
- **External row identity semantics:** external-analysis rows are flattened from `external_eval.<dataset>` using the external payload identity fields (not inherited primary-run identity). External pairing checks validate external `dataset_fingerprint` and `eval_sample_ids_hash` per seed and fail loudly on mismatches.


## Threats to validity and known limitations

- **Dataset partitioning:** current MetaMathQA protocol samples from a single upstream split (`train`) and then forms train/eval partitions locally. We now log partition strategy and signature-based overlap removals, but this is still weaker than independent upstream train/test splits.
- **Near-duplicate leakage risk:** exact problem+answer signature overlap across train/eval is filtered and counted, but semantic near-duplicates can still remain.
- **Stage extraction heuristics:** answer extraction (`####`, `\boxed{}`, `the answer is`) can miss uncommon formats; failures are reflected in preprocessing summaries and answer-eval skip counters.
- **Truncation effects:** long examples can be truncated at tokenizer max length; truncation and answer-span truncation counts are now written in preprocessing summaries and should be reviewed before inferential reporting.
- **Answer scoring scope:** symbolic parsing is attempted only when answers look expression-like and may fail for malformed or unsupported notation; numeric scoring is tolerance-based and not a full theorem prover.
- **Comparison scope:** confirmatory claims are scoped to the fixed dense-vs-MoE baseline matrix on MetaMathQA-centered staged supervision; external validity beyond this setup is not claimed.
