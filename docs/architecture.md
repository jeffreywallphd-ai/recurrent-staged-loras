# Architecture

`frozen HF backbone -> optional recurrent latent refiner -> LM head`

Stage-aware recurrence receives explicit stage masks in every batch and aligns recurrence steps to stage regions during loss computation.
