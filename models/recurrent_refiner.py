"""Recurrent latent refiner block placed between backbone and LM head."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .lora_bank import StepAwareLoRABank


@dataclass(slots=True)
class RefinerOutput:
    """Output of latent refinement."""

    refined_hidden_states: list[list[list[float]]]
    per_step_hidden_states: list[list[list[list[float]]]]


class RecurrentLatentRefiner:
    """Applies latent-space refinement with optional recurrence."""

    def __init__(
        self,
        num_steps: int,
        hidden_size: int,
        adapter_bank: StepAwareLoRABank | None = None,
    ) -> None:
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.adapter_bank = adapter_bank

        rng = random.Random(2024)
        self.w_in = [[rng.uniform(-0.05, 0.05) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.w_out = [[rng.uniform(-0.05, 0.05) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.step_scale = 0.5

    def step(self, hidden_states: list[list[list[float]]], step_idx: int) -> list[list[list[float]]]:
        """Apply one latent refinement step."""
        out: list[list[list[float]]] = []
        for seq in hidden_states:
            seq_out: list[list[float]] = []
            for token_hidden in seq:
                inter = _tanh(_matvec(token_hidden, self.w_in))
                delta = _matvec(inter, self.w_out)
                seq_out.append([h + self.step_scale * d for h, d in zip(token_hidden, delta)])
            out.append(seq_out)
        return out

    def forward(self, hidden_states: list[list[list[float]]]) -> RefinerOutput:
        """Run latent refinement for configured number of steps."""
        current = hidden_states
        history: list[list[list[list[float]]]] = []
        for step_idx in range(self.num_steps):
            current = self.step(current, step_idx=step_idx)
            if self.adapter_bank is not None:
                current = self.adapter_bank.apply(current, step_idx=step_idx)
            history.append(current)
        return RefinerOutput(refined_hidden_states=current, per_step_hidden_states=history)


def _matvec(vector: list[float], matrix: list[list[float]]) -> list[float]:
    out_dim = len(matrix[0])
    out = [0.0 for _ in range(out_dim)]
    for in_idx, in_val in enumerate(vector):
        row = matrix[in_idx]
        for out_idx in range(out_dim):
            out[out_idx] += in_val * row[out_idx]
    return out


def _tanh(vector: list[float]) -> list[float]:
    return [math.tanh(v) for v in vector]
