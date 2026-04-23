# Architecture Plan

## High-level pipeline

The planned model path for latent-refiner variants is:

**frozen base LM -> latent refiner -> LM head**

More explicitly:

1. The frozen base causal LM produces final hidden states.
2. A latent refiner block optionally applies one or more refinement steps.
3. The LM head maps refined hidden states to vocabulary logits.

This placement supports controlled testing of latent adaptation without changing tokenization, dataset formatting, or decoder head behavior.

## Components

- **Frozen base wrapper** (`models/frozen_base.py`)
  - Loads and runs the base model backbone.
  - Exposes final hidden states and LM head access.
  - Keeps base parameters frozen by default.

- **Latent refiner** (`models/recurrent_refiner.py`)
  - Applies optional recurrence in latent space.
  - Supports one-step behavior (no recurrence baseline path).

- **Step-aware adapter bank** (`models/lora_bank.py`)
  - Routes LoRA adapters by step index when adapterized recurrence is enabled.
  - Supports either shared recurrence or stage-specialized recurrence.

- **Top-level composition** (`models/staged_model.py`)
  - Composes base wrapper + optional latent refiner.
  - Provides a single forward interface for training/evaluation code.

## Terminology and modes

- **standard LoRA**: LoRA adaptation applied in the base model path (no latent refiner recurrence).
- **latent refiner only**: latent refiner recurrence enabled without refiner adapters.
- **shared recurrence**: recurrent latent refiner with one shared adapter reused across all steps.
- **stage-specialized recurrence**: recurrent latent refiner with distinct adapter parameters for each step.
- **latent refiner**: end-mounted latent-space module between base hidden states and LM head.

In this scaffold, the latent refiner is intended to support three usage patterns: without adapters (`latent refiner only`), with shared recurrence adapters, and with stage-specialized adapters.

## Non-claims

This project is scoped as a controlled baseline comparison. It does **not** claim:

- a fundamentally novel architecture,
- a replacement for mixture-of-experts systems,
- a replacement for recursive transformers.

Any stronger claims would require separate evidence beyond the scaffold and baseline comparisons documented here.
