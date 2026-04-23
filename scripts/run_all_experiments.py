"""Run multiple experiment configs sequentially and aggregate metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.config_loader import load_runtime_config
from training.engine import build_training_components, run_training_loop


def _collect_config_paths(configs: list[str], config_dir: str | None) -> list[Path]:
    paths = [Path(c) for c in configs]
    if config_dir:
        paths.extend(sorted(Path(config_dir).glob("*.json")))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for p in paths:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            deduped.append(p)
    if not deduped:
        raise ValueError("No config files provided. Use --configs and/or --config-dir.")
    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all experiment configs sequentially")
    parser.add_argument("--configs", nargs="*", default=[], help="Explicit config paths")
    parser.add_argument("--config-dir", default=None, help="Directory containing *.json experiment configs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_paths = _collect_config_paths(args.configs, args.config_dir)

    summary: dict[str, dict[str, float | int | str]] = {}
    for config_path in config_paths:
        runtime = load_runtime_config(config_path)
        runtime.output["dir"] = str(Path("output") / runtime.baseline)
        run_name = config_path.stem

        components = build_training_components(runtime)
        result = run_training_loop(components=components, run_name=run_name)

        metrics_path = result.output_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summary[runtime.baseline] = metrics
        print(f"[ok] baseline={runtime.baseline} run={run_name} metrics={metrics_path}")

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[ok] summary={summary_path}")


if __name__ == "__main__":
    main()
