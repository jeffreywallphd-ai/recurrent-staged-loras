# Baseline Definitions

All baselines use the same base instruct model family, tokenizer assumptions, data splits, and evaluation protocol unless explicitly stated otherwise.

## 1) Base instruct model (no adaptation)

- **What changes relative to base**: nothing.
- **What stays fixed**: all model parameters and inference path.
- **Hypothesis tested**: establishes the reference capability level and variance floor.

## 2) Base + standard LoRA

- **What changes relative to base**: add trainable standard LoRA modules to selected base-model target modules.
- **What stays fixed**: base backbone remains frozen; no latent refiner block is used.
- **Hypothesis tested**: parameter-efficient adaptation in the base path can improve task performance over the non-adapted base.

## 3) Base + latent refiner only

- **What changes relative to base**: add latent-space recurrence with no step-aware adapters.
- **Canonical config semantics**: `latent_refiner.enabled=true`, `recurrence_mode="latent_only"`, `adapter_sharing="none"`.
- **What stays fixed**: base backbone frozen; no standard LoRA; no shared/per-step refiner adapters.
- **Hypothesis tested**: latent recurrence alone may improve over base and isolates adapter contributions in later baselines.

## 4) Base + latent refiner + shared recurrence

- **What changes relative to base**: add latent recurrence and reuse one adapter at each recurrence step.
- **Canonical config semantics**: `recurrence_mode="shared"`, `adapter_sharing="shared"`.
- **What stays fixed**: base backbone frozen; recurrence depth fixed per config.
- **Hypothesis tested**: shared adapterized recurrence can improve adaptation efficiency/quality relative to base and standard LoRA.

## 5) Base + latent refiner + stage-specialized recurrence

- **What changes relative to base**: add latent recurrence with distinct adapter parameters per step.
- **Canonical config semantics**: `recurrence_mode="stage_specialized"`, `adapter_sharing="per_step"`.
- **What stays fixed**: base backbone frozen; recurrence depth fixed per config; step order/interface consistent with shared recurrence.
- **Hypothesis tested**: step specialization in latent adaptation can outperform shared recurrence at comparable training budget.

## Comparison discipline

The study is designed as an empirical comparison under a fixed setup. It intentionally avoids positioning these baselines as a new fundamental architecture family.
