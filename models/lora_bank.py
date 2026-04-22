"""Step-aware adapter bank for shared or stage-specialized LoRA routing."""

from __future__ import annotations

from typing import Any


class StepAwareLoRABank:
    """Container that resolves adapter weights by recurrence step.

    Modes:
    - shared recurrence: one adapter reused across all recurrence steps.
    - stage-specialized recurrence: unique adapter per recurrence step.
    """

    def __init__(self, num_steps: int, shared_across_steps: bool) -> None:
        self.num_steps = num_steps
        self.shared_across_steps = shared_across_steps
        self.adapters: dict[int, Any] = {}
        # TODO: initialize adapter modules and register trainable params.

    def get_adapter_for_step(self, step_idx: int) -> Any:
        """Return the adapter module used at the requested recurrence step."""
        if self.shared_across_steps:
            step_idx = 0
        if step_idx not in self.adapters:
            raise KeyError(f"No adapter registered for step {step_idx}")
        return self.adapters[step_idx]

    def apply(self, hidden_states: Any, step_idx: int) -> Any:
        """Apply step-routed adapter transformation to hidden states.

        TODO: define adapter invocation contract with concrete tensor types.
        """
        raise NotImplementedError("TODO: implement adapter application")
