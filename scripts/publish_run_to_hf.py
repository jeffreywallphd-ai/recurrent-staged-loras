"""Publish an existing run directory to Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from publish.huggingface_export import publish_run_directory_from_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish recurrent-staged-loras run artifacts to HF Hub")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--model-repo", default=None)
    parser.add_argument("--dataset-repo", default=None)
    parser.add_argument("--private", action="store_true", default=False)
    parser.add_argument("--commit-message", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    publish_run_directory_from_paths(
        run_dir=Path(args.run_dir),
        model_repo=args.model_repo,
        dataset_repo=args.dataset_repo,
        private=bool(args.private),
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
