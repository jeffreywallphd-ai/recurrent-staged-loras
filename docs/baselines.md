# Baseline Definitions

All baselines use the same base instruct model family, tokenizer assumptions, data splits, and evaluation protocol unless explicitly stated otherwise.

## 1) Base instruct model (no adaptation)

- **What changes relative to base**: nothing.
- **What stays fixed**: all model parameters and inference path.
- **Hypothesis tested**: establishes the reference capability level and variance floor.

## 2) Base + standard LoRA

- **What changes relative to base**: add trainable standard LoRA modules to selected base-model target modules.
- **What stays fixed**: base backbone remains frozen; no latent recurrence block is used.
- **Hypothesis tested**: parameter-efficient adaptation in the base path can improve task performance over the non-adapted base.

## 3) Base + latent refiner only

- **What changes relative to base**: add a latent refiner with a small fixed number of recurrence steps and no refiner adapters.
- **What stays fixed**: base backbone frozen; no standard LoRA; no step-aware/refiner adapter bank.
- **Hypothesis tested**: latent-space iterative refinement alone may improve over base and helps isolate adapter contributions in later baselines.

## 4) Base + latent refiner + shared recurrence

- **What changes relative to base**: add a latent refiner with multiple recurrence steps and one shared adapter across steps.
- **What stays fixed**: base backbone frozen; recurrence depth fixed per config; adapter reused at each step.
- **Hypothesis tested**: iterative latent refinement with shared parameters can improve adaptation quality/efficiency relative to standard LoRA and base.

## 5) Base + latent refiner + stage-specialized recurrence

- **What changes relative to base**: add a latent refiner with multiple recurrence steps and distinct per-step adapters.
- **What stays fixed**: base backbone frozen; recurrence depth fixed per config; step order and interface consistent with shared recurrence.
- **Hypothesis tested**: step specialization in latent adaptation can outperform shared recurrence at comparable training budget.

## Comparison discipline

The study is designed as an empirical comparison under a fixed setup. It intentionally avoids positioning these baselines as a new fundamental architecture family.
