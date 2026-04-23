# recurrent-staged-loras

Controlled dense-vs-MoE adaptation study pipeline for staged supervision on MetaMathQA.

## Defaults

- Dense family: `Qwen/Qwen3-8B`
- MoE family: `allenai/OLMoE-1B-7B-0125-Instruct`
- Baselines: `base`, `standard_lora`, `latent_refiner_only`, `shared_recurrence`, `stage_specialized_recurrence`
- Dataset: `meta-math/MetaMathQA` with explicit 3-stage masks.

## Study outputs

Each run writes `metrics.json` and `dataset_preprocessing_summary.json`.

Multi-seed orchestration writes:
- `outputs/summary.json` (runs + aggregates)
- `outputs/summary.csv` (run and aggregate rows)
- `outputs/aggregates.json` (aggregate-only artifact)

## Presets

- Study presets: `experiments/configs/*.json`
- Pilot/smoke presets: `experiments/configs/*_pilot.json`

## Quick start

```bash
python -m training.train --config experiments/configs/standard_lora.json --run-name dense_lora_seed11
python scripts/run_all_experiments.py --configs experiments/configs/base.json experiments/configs/moe_base.json --seeds 11 22 33
```

## Docs

- `docs/README.md`
- `docs/architecture.md`
- `docs/baselines.md`
- `docs/experiments.md`
