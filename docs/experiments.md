# Experiment Plan (Initial)

## Objective

Run a controlled comparison of latent adaptation strategies under the same frozen base model setup.

## Initial experiment matrix

| Baseline | Config file | Recurrence | Adapter scheme | Base frozen |
|---|---|---|---|---|
| Base | `experiments/configs/base.json` | No | None | Yes |
| Standard LoRA | `experiments/configs/standard_lora.json` | No | standard LoRA | Yes |
| Latent refiner only | `experiments/configs/latent_refiner_only.json` | Yes (`recurrence_mode=latent_only`) | None | Yes |
| Shared recurrence | `experiments/configs/shared_recurrence.json` | Yes (`recurrence_mode=shared`) | one shared recurrence adapter (`adapter_sharing=shared`) | Yes |
| Stage-specialized recurrence | `experiments/configs/stage_specialized_recurrence.json` | Yes (`recurrence_mode=stage_specialized`) | per-step adapters (`adapter_sharing=per_step`) | Yes |

## Minimum controlled variables

Across the matrix, keep fixed where feasible:

- base model checkpoint,
- tokenizer and prompt formatting,
- dataset versions and splits,
- training budget definition (tokens/steps/epochs),
- evaluation metrics and scripts,
- random seed protocol and number of repeats.

## Minimum success criteria for scaffold stage

1. Each baseline config loads successfully.
2. Baseline selector routes each config to the intended variant.
3. Training entrypoint can build a real model path and run a tiny forward smoke pass.
4. Training entrypoint writes run metadata for reproducibility.

## Failure criteria (scaffold stage)

- Baseline names/config semantics are ambiguous or inconsistent.
- Terminology drifts between latent-refiner-only recurrence and adapterized recurrence modes.
- The architecture path is unclear about placement of the latent refiner.
- Integrating full training internals would require major reorganization.

## Out of scope for this initial version

- Full optimizer/scheduler implementation.
- Production-grade backend integration for LoRA libraries.
- Full benchmark claims beyond smoke-level wiring checks.
