# recurrent-staged-loras

Core scaffolding for a controlled empirical comparison of latent adaptation baselines on top of a frozen base language model.

## Why this repo exists

This repository isolates adaptation choices while holding base model family, dataset protocol, and evaluation framing fixed. The objective is reproducible baseline-to-baseline comparison, not novelty claims.

## Baselines in scope

1. Base
2. Base + standard LoRA
3. Base + latent refiner only
4. Base + latent refiner + shared recurrence adapters
5. Base + latent refiner + stage-specialized recurrence adapters

Baseline templates live in `experiments/configs/`.

## Architecture path

**frozen base LM -> latent refiner -> LM head**

The latent refiner path is configurable for:
- recurrence with no adapters (`latent_refiner_only`),
- recurrence with one shared adapter across steps (`shared_recurrence`),
- recurrence with per-step adapters (`stage_specialized_recurrence`).

## Current status

This version now provides a minimal trainable baseline core:
- baseline definitions and config semantics,
- model composition with torch modules and trainability smoke path,
- config parsing and baseline selection,
- minimal end-to-end training loop with synthetic dataset, batching, optimizer, eval, and checkpoints,
- run metadata/config/metrics artifacts and lightweight tests.

Still deferred:
- full optimizer/scheduler sophistication beyond minimal AdamW loop,
- production-grade external LoRA backend integration,
- full benchmark reporting.

## Reading order

1. `docs/README.md`
2. `docs/architecture.md`
3. `docs/baselines.md`
4. `docs/experiments.md`

## Quick start

```bash
python -m training.train --config experiments/configs/base.json --run-name smoke_base
```
