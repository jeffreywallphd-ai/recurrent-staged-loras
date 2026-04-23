"""Step-aware adapter bank for shared or stage-specialized LoRA routing."""

from __future__ import annotations

from dataclasses import dataclass, field
import random


@dataclass(slots=True)
class LowRankAdapter:
    """Lightweight low-rank adapter for hidden-state refinement."""

    hidden_size: int
    rank: int
    alpha: int
    seed: int
    scale: float = field(init=False)
    a: list[list[float]] = field(init=False)
    b: list[list[float]] = field(init=False)

    def __post_init__(self) -> None:
        rng = random.Random(self.seed)
        self.scale = float(self.alpha) / float(self.rank)
        self.a = [[rng.uniform(-0.04, 0.04) for _ in range(self.rank)] for _ in range(self.hidden_size)]
        self.b = [[rng.uniform(-0.04, 0.04) for _ in range(self.hidden_size)] for _ in range(self.rank)]

    def apply_to_vector(self, vector: list[float]) -> list[float]:
        low_rank = [0.0 for _ in range(self.rank)]
        for in_idx, in_val in enumerate(vector):
            a_row = self.a[in_idx]
            for r in range(self.rank):
                low_rank[r] += in_val * a_row[r]

        lifted = [0.0 for _ in range(self.hidden_size)]
        for r_idx, r_val in enumerate(low_rank):
            b_row = self.b[r_idx]
            for out_idx in range(self.hidden_size):
                lifted[out_idx] += r_val * b_row[out_idx]

        return [v + self.scale * d for v, d in zip(vector, lifted)]


class StepAwareLoRABank:
    """Container that resolves adapter weights by recurrence step."""

    def __init__(
        self,
        num_steps: int,
        hidden_size: int,
        rank: int,
        alpha: int,
        shared_across_steps: bool,
        enabled: bool = True,
    ) -> None:
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.rank = rank
        self.alpha = alpha
        self.shared_across_steps = shared_across_steps
        self.enabled = enabled
        self.adapters: dict[int, LowRankAdapter] = {}

        if not enabled:
            return

        if shared_across_steps:
            self.adapters[0] = LowRankAdapter(hidden_size=hidden_size, rank=rank, alpha=alpha, seed=17)
        else:
            for step_idx in range(num_steps):
                self.adapters[step_idx] = LowRankAdapter(
                    hidden_size=hidden_size,
                    rank=rank,
                    alpha=alpha,
                    seed=17 + step_idx,
                )

    def get_adapter_for_step(self, step_idx: int) -> LowRankAdapter:
        """Return the adapter module used at the requested recurrence step."""
        if not self.enabled:
            raise KeyError("Adapters disabled")
        lookup = 0 if self.shared_across_steps else step_idx
        if lookup not in self.adapters:
            raise KeyError(f"No adapter registered for step {lookup}")
        return self.adapters[lookup]

    def apply(self, hidden_states: list[list[list[float]]], step_idx: int) -> list[list[list[float]]]:
        """Apply step-routed adapter transformation to hidden states."""
        if not self.enabled:
            return hidden_states
        adapter = self.get_adapter_for_step(step_idx)
        out: list[list[list[float]]] = []
        for seq in hidden_states:
            out.append([adapter.apply_to_vector(token_hidden) for token_hidden in seq])
        return out
