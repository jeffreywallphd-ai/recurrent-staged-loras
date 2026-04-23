# Architecture

`frozen HF backbone -> optional recurrent latent refiner -> LM head`

Stage-aware recurrence receives explicit stage masks in every batch and aligns recurrence steps to stage regions during loss computation.

Answer-level evaluation is deliberately decoupled from stage-level supervision:
- `stage3_mask` spans the full Stage 3 section (`Final Answer:` header + answer text) for stage token diagnostics.
- `final_answer_mask` spans answer tokens only for final-answer string/numeric metrics.
