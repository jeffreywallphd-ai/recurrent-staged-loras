from pathlib import Path
import json

import torch

from publish.model_serialization import serialize_checkpoint_to_hf_directory


def test_serialization_converts_checkpoint_to_safetensors_and_shards(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(
        {
            "model_state_dict": {
                "layer.weight": torch.zeros(64, 64),
                "layer.bias": torch.zeros(64),
            }
        },
        run_dir / "checkpoint.pt",
    )
    (run_dir / "metadata.json").write_text(json.dumps({"run_name": "unit"}), encoding="utf-8")
    output_dir = tmp_path / "hf"

    payload = serialize_checkpoint_to_hf_directory(
        run_dir=run_dir,
        output_dir=output_dir,
        runtime_config={"variant": {"base": {"model_name": "test/tiny"}}},
        max_shard_size="1KB",
    )

    assert (output_dir / "config.json").exists()
    assert payload["sharded"] is True
    assert (output_dir / "model.safetensors.index.json").exists()
    assert any(output_dir.glob("model-*.safetensors"))
