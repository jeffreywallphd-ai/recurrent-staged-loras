from __future__ import annotations

import sys
from types import SimpleNamespace

from models.config import BaseModelConfig
from models.frozen_base import FrozenBaseCausalLM


def test_full_gpu_loading_prefers_cuda_when_available(monkeypatch) -> None:
    class _FakeModel:
        def __init__(self) -> None:
            self.moved_to: str | None = None
            self.config = SimpleNamespace(hidden_size=16, vocab_size=32)

        def to(self, device: str):
            self.moved_to = device
            return self

        def parameters(self):
            return iter([])

    class _FakeAutoModelForCausalLM:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return _FakeModel()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModelForCausalLM=_FakeAutoModelForCausalLM))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    cfg = BaseModelConfig(
        model_name="fake/model",
        model_loading_mode="full_gpu",
        device_map="auto",
    )
    model = FrozenBaseCausalLM(cfg)
    assert model.hf_model is not None
    assert model.hf_model.moved_to == "cuda"
