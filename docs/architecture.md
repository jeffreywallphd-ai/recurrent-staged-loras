# Architecture Plan

## High-level pipeline

The model path for latent-refiner variants is:

**frozen base LM -> latent refiner -> LM head**

More explicitly:

1. The frozen base causal LM produces final hidden states.
2. A latent refiner block optionally applies one or more refinement steps.
3. Optional step-aware adapters are applied within recurrence steps (depending on baseline).
4. The LM head maps refined hidden states to vocabulary logits.

This placement supports controlled testing of latent adaptation without changing tokenization, dataset formatting, or decoder head behavior.

## Components

- **Frozen base wrapper** (`models/frozen_base.py`)
  - Loads and runs the base model backbone.
  - Uses a tiny internal torch causal-LM-like module for `example/*` configs.
  - Uses Hugging Face causal LM for non-example names when `transformers` is available.
  - Exposes final hidden states and LM head access.
  - Keeps base parameters frozen by default.
  - Standard LoRA is implemented directly for internal target modules and as a first-pass hidden-state residual path for Hugging Face backends.

- **Latent refiner** (`models/recurrent_refiner.py`)
  - Applies optional recurrence in latent space.
  - Supports one-step behavior and multi-step recurrence.

- **Step-aware adapter bank** (`models/lora_bank.py`)
  - Routes low-rank adapters by step index when adapterized recurrence is enabled.
  - Supports either one shared adapter (`shared`) or per-step adapters (`per_step`).

- **Top-level composition** (`models/staged_model.py`)
  - Composes base wrapper + optional latent refiner + optional adapter bank.
  - Provides a single forward interface for training/evaluation code.

## Terminology and config semantics

- **standard LoRA**: adaptation in the base-model path, with latent refiner disabled.
- **latent refiner only**: latent recurrence enabled with `recurrence_mode="latent_only"` and `adapter_sharing="none"`.
- **shared recurrence**: latent recurrence enabled with `recurrence_mode="shared"` and `adapter_sharing="shared"` (one adapter reused across steps).
- **stage-specialized recurrence**: latent recurrence enabled with `recurrence_mode="stage_specialized"` and `adapter_sharing="per_step"` (distinct adapter parameters per step).

This keeps latent recurrence semantics separate from adapter-sharing semantics while still making each baseline explicit.

## Non-claims

This project is scoped as a controlled baseline comparison. It does **not** claim:

- a fundamentally novel architecture,
- a replacement for mixture-of-experts systems,
- a replacement for recursive transformers.

Any stronger claims would require separate evidence beyond the scaffold and baseline comparisons documented here.
