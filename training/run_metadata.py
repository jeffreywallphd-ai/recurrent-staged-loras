"""Run metadata and logging structures for experiment reproducibility."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json


@dataclass(slots=True)
class RunMetadata:
    """Metadata recorded for each experiment run."""

    run_name: str
    baseline: str
    dataset_name: str
    model_name: str
    output_dir: str
    git_commit: str = "unknown"
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def write(self, path: str | Path | None = None) -> Path:
        """Persist run metadata to disk."""
        out_path = Path(path) if path is not None else Path(self.output_dir) / "metadata.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return out_path
