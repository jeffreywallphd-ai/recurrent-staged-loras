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
An explicit `final_answer_mask` is also created for answer-only tokens inside Stage 3, excluding the literal `Final Answer:` header.
Rows without stage-3 tokens are filtered.

Per-run preprocessing statistics are written to `dataset_preprocessing_summary.json`, including:
- kept examples,
- filtered examples (empty/problem-response missing, stage-3 missing),
- examples with non-empty stage-2 and stage-3.

## Train/eval protocol

- Backbone is frozen in the main study path.
- Stage-aware loss aligns recurrence steps to stage masks.
- `eval_interval_steps` is true **step-based** periodic eval (not epoch-only).
- Final eval runs at the end unless the final step already landed on an interval boundary.
- Eval token/time totals include all interval eval passes plus final eval (when present).

## Answer-span protocol and metrics semantics

- `stage_2_token_accuracy`: token accuracy on `stage2_mask` (reasoning section).
- `stage_3_token_accuracy`: token accuracy on `stage3_mask` (full Stage 3 section, including `Final Answer:` header tokens).
- `final_answer_accuracy`: normalized string match on decoded `final_answer_mask` span only. Normalization lowercases, removes `$`, unwraps simple `\\boxed{...}`, squashes whitespace, and strips trailing punctuation.
- `final_answer_exact_match`: strict raw-string equality between decoded `final_answer_mask` prediction and extracted answer text.
- `normalized_numeric_answer_accuracy`: numeric comparison using float normalization from decoded `final_answer_mask` prediction vs normalized numeric gold label.
- Reviewer-facing aliases are also written: `answer_span_normalized_accuracy`, `answer_span_exact_match`, and `answer_span_numeric_accuracy` (same values as the three answer metrics above).

Limitations:
- String normalization is intentionally conservative and does not parse full symbolic/math equivalence.
- Numeric matching uses first detectable number and cannot validate multi-value expressions.

## Canonical run-level metrics schema (`metrics.json`)

Official fields:
`run_name`, `config_name`, `baseline_name`, `dataset_name`, `dataset_train_examples`, `dataset_eval_examples`, `seed`, `architecture_type`, `model_name`, `final_train_loss`, `final_eval_loss`, `best_eval_loss`, `eval_perplexity`, `train_perplexity`, `stage_2_token_accuracy`, `stage_3_token_accuracy`, `final_answer_accuracy`, `final_answer_exact_match`, `normalized_numeric_answer_accuracy`, `answer_span_normalized_accuracy`, `answer_span_exact_match`, `answer_span_numeric_accuracy`, `answer_eval_string_count`, `answer_eval_numeric_count`, `answer_eval_skipped_no_stage3`, `answer_eval_skipped_no_answer_span`, `answer_eval_skipped_missing_answer_text`, `answer_eval_skipped_missing_numeric_target`, `wall_time_seconds_total`, `wall_time_seconds_train`, `wall_time_seconds_eval`, `tokens_seen_train`, `tokens_seen_eval`, `tokens_per_second_train`, `tokens_per_second_eval`, `steps_per_second`, `seconds_per_step`, `trainable_params`, `total_params`, `trainable_param_fraction`, `recurrence_steps`, `effective_forward_passes_per_example`, `global_steps_completed`, `epochs_completed`, `backend`, `latent_cache`.

Each run also writes `answer_eval_diagnostics.json` with answer-scoring counts and skip reasons.

## Compute fairness protocol

Comparisons report:
- trainable parameter fraction,
- recurrence steps / effective forward passes per example,
- tokens/sec and total wall-clock time.

These are used together with quality metrics to interpret performance vs compute budget.

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

For each group, mean/std are reported for:
- `final_eval_loss`
- `eval_perplexity`
- `stage_2_token_accuracy`
- `stage_3_token_accuracy`
- `final_answer_accuracy`
- `final_answer_exact_match`
- `normalized_numeric_answer_accuracy`
- `wall_time_seconds_total`
- `tokens_per_second_train`
- `trainable_param_fraction`

## Presets: study vs pilot

- **Study presets:** base config names (for reportable runs).
- **Pilot presets:** `*_pilot.json` variants for smoke checks and quick iteration.
- Default orchestration mode is reportable study runs only: `--preset-scope study`.

## Paper workflow (explicit)

1. **Reportable runs only:** execute study configs (`experiments/configs/*.json`) with required seeds `11,22,33`; keep pilot configs out of paper numbers.
2. **Primary headline metrics:** `final_answer_accuracy`, `final_answer_exact_match`, `normalized_numeric_answer_accuracy`.
3. **Secondary diagnostics:** `stage_2_token_accuracy`, `stage_3_token_accuracy`, loss/perplexity, and compute/fairness metrics.
4. **Artifact usage:**
   - `outputs/report_table.csv`: canonical paper-table source (run rows + aggregate rows with explicit `row_type`/`report_tier`).
   - `outputs/aggregates.json`: canonical aggregate plotting source.
   - `outputs/summary.json`: full machine-readable archive (run + aggregate + invocation context).
   - `outputs/summary.csv`: wide mixed export for quick spreadsheet inspection.
