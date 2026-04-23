# recurrent-staged-loras

Controlled dense-vs-MoE adaptation study pipeline for staged supervision on MetaMathQA.

## Defaults

- Dense family: `Qwen/Qwen3-8B`
- MoE family: `allenai/OLMoE-1B-7B-0125-Instruct`
- Baselines: `base`, `standard_lora`, `latent_refiner_only`, `shared_recurrence`, `stage_specialized_recurrence`
- Dataset: `meta-math/MetaMathQA` with explicit 3-stage masks.

## Study outputs

Each run writes:
- `metrics.json`
- `dataset_preprocessing_summary.json`
- `answer_eval_diagnostics.json`

Multi-seed orchestration writes:
- `outputs/summary.json` (runs + aggregates)
- `outputs/summary.csv` (run and aggregate rows)
- `outputs/aggregates.json` (aggregate-only artifact)
- `outputs/report_table.csv` (canonical paper/report table source with explicit run vs aggregate row typing)
- `outputs/statistical_analysis_confirmatory.json` / `.csv` (confirmatory primary-outcome contrasts with Holm-adjusted p-values)
- `outputs/statistical_analysis_secondary.json` and `outputs/statistical_analysis_efficiency.json` (separate descriptive families)
- `outputs/statistical_analysis_report.md` (reviewer-facing inferential summary)

## Presets

- Study presets: `experiments/configs/*.json`
- Pilot/smoke presets: `experiments/configs/*_pilot.json`
- Default orchestration scope is study-only (`--preset-scope study`).

## Quick start

```bash
python -m training.train --config experiments/configs/standard_lora.json --run-name dense_lora_seed11
python scripts/run_all_experiments.py --config-dir experiments/configs --seeds 11 22 33 --preset-scope study
```

## Docs

- `docs/README.md`
- `docs/architecture.md`
- `docs/baselines.md`
- `docs/experiments.md` (staging construction + strict/normalized/numeric answer-eval protocol)
- `docs/study_matrix.md`
- `analysis/README.md`
