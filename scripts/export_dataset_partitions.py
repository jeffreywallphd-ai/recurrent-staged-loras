"""Export train/eval dataset partitions from a run directory to local files."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export train/eval partition artifacts from an existing run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    required = ["dataset_partitions.json", "dataset_preprocessing_summary.json", "config.json"]
    for name in required:
        src = run_dir / name
        if not src.exists():
            raise SystemExit(f"Missing required artifact: {src}")
        shutil.copy2(src, out_dir / name)


if __name__ == "__main__":
    main()
