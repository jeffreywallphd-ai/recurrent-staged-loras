# Baselines

Supported baselines:
- `base`
- `standard_lora` (real PEFT LoRA on target modules)
- `latent_refiner_only`
- `shared_recurrence`
- `stage_specialized_recurrence`

The same baseline family is runnable for both `architecture_type=dense` and `architecture_type=moe`.

All baselines share the same staged supervision and answer-span evaluation protocol to keep dense-vs-MoE comparisons scientifically aligned.
