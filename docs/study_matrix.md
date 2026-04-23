# Dense vs MoE Study Matrix (Review-Ready)

## Families and paired baselines

Dense family (`architecture_type=dense`):
- `base.json`
- `standard_lora.json`
- `latent_refiner_only.json`
- `shared_recurrence.json`
- `stage_specialized_recurrence.json`

MoE family (`architecture_type=moe`):
- `moe_base.json`
- `moe_standard_lora.json`
- `moe_latent_refiner_only.json`
- `moe_shared_recurrence.json`
- `moe_stage_specialized_recurrence.json`

Pilot (non-reportable) counterparts:
- `*_pilot.json` and `moe_*_pilot.json`

Additional explicit family suffixes are shipped for both dense and MoE baselines where applicable:
- `*_compute_controlled.json`
- `*_external_eval.json`
- `*_ablation.json` (for ablation-compatible baselines)
- `*_debug.json`
- `*_debug_external_eval.json` (stage-specialized paired dense+MoE tiny external smoke configs)

## Best available study workflow (recommended full robustness workflow)

Under the current design, the strongest available study is a **multi-family workflow** (not one all-in-one JSON):

1. **Confirmatory study sweep (reportable main claims):** no-suffix confirmatory family.
2. **Compute-controlled robustness sweep (supporting robustness table):** `*_compute_controlled.json`.
3. **External-eval robustness sweep (descriptive generalization table):** `*_external_eval.json`.
4. **Ablation robustness sweep (mechanistic/structure plots/tables):** `*_ablation.json` with `--run-scope ablation`.
5. **Debug sweep (non-reportable bug hunting only):** `*_debug.json` and `*_debug_external_eval.json`.

This keeps confirmatory inference clean while still exposing a reproducible high-coverage robustness path with shipped configs only.

## Required seeds

Primary multi-seed report runs use: `11`, `22`, `33`.

## Primary reported metrics

- `final_answer_accuracy`
- `final_answer_exact_match`
- `final_answer_normalized_match`
- `normalized_numeric_answer_accuracy`
- `symbolic_answer_accuracy` (secondary robustness metric; conditional on parse success)

## Secondary diagnostics

- `stage_2_token_accuracy`
- `stage_3_token_accuracy`
- `final_eval_loss`
- `eval_perplexity`

## Fairness / compute metrics

- `trainable_param_fraction`
- `recurrence_steps`
- `effective_forward_passes_per_example`
- `wall_time_seconds_total`
- `tokens_per_second_train`
- `compute_control_enabled`
- `compute_control_mode`
- `adjusted_max_steps`
- `effective_optimizer_steps`
- `tokens_per_optimizer_step`

Interpretation note: `effective_forward_passes_per_example` equalizes forward-pass budget, not full optimization dynamics. Token and wall-time compute-control modes apply different stop criteria.

## External evaluations (optional, non-confirmatory by default)

Supported config-driven external datasets:
- `gsm8k`
- `math`
- `svamp`

External scores are reported separately under `external_eval.<dataset>` and in `report_table.csv` rows with `report_tier=external_eval`.
External rows are labeled `dataset_type=external` and excluded from primary aggregates and confirmatory inference by default.
Each external payload is self-describing and carries its own dataset identity (`dataset_name`, `dataset_type`, `dataset_split`, `dataset_seed`, `dataset_subset_size`, `dataset_eval_fraction`, `dataset_fingerprint`, `train_sample_ids_hash`, `eval_sample_ids_hash`) alongside metrics.

## Ablations (optional, separated from confirmatory matrix)

Supported ablation grid controls:
- `ablations.recurrent_steps`
- `ablations.lora_rank`

Validity constraints (fail-fast):
- `ablations.recurrent_steps` requires `model.latent_refiner.enabled=true`.
- `ablations.lora_rank` requires at least one active adapter (`standard_lora.enabled` or `latent_refiner.adapter.enabled`).
- Rank ablations are applied only to active adapter paths.

Expanded run names use only requested suffix dimensions:
- recurrence-only `_r{steps}`
- rank-only `_rank{rank}`
- two-dimensional `_r{steps}_rank{rank}`

Metrics always write explicit fields:
- `ablation_recurrent_steps`
- `ablation_lora_rank`

Do not mix ablation-derived runs with confirmatory study reporting by default.

## Run scopes

`scripts/run_all_experiments.py` supports explicit isolation via `--run-scope`:
- `confirmatory` (default): excludes ablation configs.
- `ablation`: includes only ablation-derived runs.
- `all`: includes both confirmatory and ablation runs.

Artifacts store `run_scope` per run (`confirmatory` or `ablation`) so downstream filters can prevent accidental mixing.
Confirmatory analysis fails fast on ablation rows by default unless `--allow-ablations-in-analysis` is provided.

## Baseline naming rules

- `baseline_name`: full executed baseline (including ablation suffixes such as `_r3_rank8`).
- `baseline_family`: original baseline before ablation suffixing (e.g., `stage_specialized_recurrence`).

## Reportability protocol

- Pilot presets are excluded by default in `scripts/run_all_experiments.py` (`--preset-scope study`).
- Use `--preset-scope all` only for engineering checks.
- Paper/report tables should be produced from `outputs/report_table.csv` generated by study-only runs.
- Aggregate plots should be generated from `outputs/aggregates.json`; deeper diagnostics can use `outputs/summary.json` and per-run artifacts.

### Reportable vs robustness-only vs debug-only

- **Reportable confirmatory configs (main paper tables):**
  - no-suffix confirmatory configs (dense + MoE pairs).
- **Robustness/supporting configs:**
  - `*_compute_controlled.json` (compute robustness tables),
  - `*_external_eval.json` (descriptive external-eval tables by default),
  - `*_ablation.json` (ablation robustness plots/tables, non-confirmatory by default).
- **Non-reportable engineering/debug configs:**
  - `*_debug.json`,
  - `*_debug_external_eval.json`,
  - `*_pilot.json`.

### Which config should I use?

- **Full paper runs (reportable):** no-suffix confirmatory configs.
- **Pilot runs (non-reportable):** `*_pilot.json`.
- **Quick bug checks (non-reportable):** `*_debug.json`.
- **External evaluation checks (descriptive by default):** `*_external_eval.json`.
- **Ablation sweeps (non-confirmatory by default):** `*_ablation.json` with `--run-scope ablation`.
- **Compute-controlled comparisons:** `*_compute_controlled.json`.

Run helper:

```bash
python scripts/run_all_experiments.py --config-dir experiments/configs --config-family debug --preset-scope all --seeds 7
```

Recommended top-level sweep commands:

```bash
python scripts/run_all_experiments.py --config-dir experiments/configs --config-family confirmatory --preset-scope study --run-scope confirmatory --seeds 11 22 33
```

```bash
python scripts/run_all_experiments.py --config-dir experiments/configs --config-family compute_controlled --preset-scope study --run-scope confirmatory --seeds 11 22 33
```

```bash
python scripts/run_all_experiments.py --config-dir experiments/configs --config-family external_eval --preset-scope study --run-scope confirmatory --seeds 11 22 33
```

```bash
python scripts/run_all_experiments.py --config-dir experiments/configs --config-family ablation --preset-scope study --run-scope ablation --seeds 11 22 33
```

```bash
python scripts/run_all_experiments.py --config-dir experiments/configs --config-family debug --preset-scope all --run-scope all --seeds 7
```

Concrete tiny-sample external debug example:
- `stage_specialized_recurrence_debug_external_eval.json`
- `moe_stage_specialized_recurrence_debug_external_eval.json`

Both include:
- primary `subset_size=16`
- external `gsm8k/math/svamp` with `subset_size=3`, `split=test`, `seed=3`
- low-step fast smoke settings and explicit `NON-REPORTABLE` notes.

## Confirmatory contrasts and error control

Planned confirmatory contrasts are fixed in advance and evaluated separately inside each architecture (`dense`, `moe`):

- `stage_specialized_recurrence` vs `standard_lora`
- `stage_specialized_recurrence` vs `shared_recurrence`
- `stage_specialized_recurrence` vs `latent_refiner_only`

Confirmatory inference is limited to the primary outcomes (`final_answer_accuracy`, `final_answer_exact_match`, `final_answer_normalized_match`, `normalized_numeric_answer_accuracy`); symbolic accuracy is retained as secondary because applicability depends on parse success rates with Holm family-wise correction across the full confirmatory family. Secondary and efficiency outcomes remain descriptive by default.

Confirmatory comparison validity requires matched `architecture_type`, `model_name`, and `dataset_name`; compared baselines can come from different config files and are audited with `config_name_a` / `config_name_b` in analysis outputs.

External evaluation datasets are descriptive by default. They are included in analysis only in explicit external-analysis mode (`--dataset-scope external` or `--dataset-scope all`), and confirmatory inference remains primary-dataset-only unless intentionally changed.
In external-analysis mode, flattened rows explicitly overwrite dataset identity from `external_eval.<dataset>` and are tagged `dataset_scope=external`; they do not inherit primary-run dataset fingerprints or sample hashes.
Pilot rows (`*_pilot.json`) are also rejected by confirmatory analysis by default unless `--allow-pilot-runs-in-analysis` is explicitly provided.

## Dataset reproducibility identity (required)

Every run writes stable dataset identity fields for pairing auditability:
- `dataset_fingerprint`
- `train_sample_ids_hash`
- `eval_sample_ids_hash`
- `dataset_name`, `dataset_split`, `dataset_seed`, `dataset_subset_size`, `dataset_eval_fraction`

Paired confirmatory comparisons require matching `dataset_fingerprint` and `eval_sample_ids_hash` per paired seed; mismatches fail loudly as non-comparable runs.
