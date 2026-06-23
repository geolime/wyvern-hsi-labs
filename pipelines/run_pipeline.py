"""Run a project's pipeline: load config, execute its stages in order.

  python pipelines/run_pipeline.py --config configs/forest_landcover.yaml
  python pipelines/run_pipeline.py --config configs/forest_landcover.yaml --from 15
  python pipelines/run_pipeline.py --config configs/water_quality.yaml --stages 04
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from wyvernhsi.config import load_config


def _load_stage_main(script_path: Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise AttributeError(f"Stage {script_path.name} has no main(config).")
    return module.main


def _select(stages: list, only: list | None, start: str | None) -> list:
    if only:
        keep = [s for s in stages if any(tok in s for tok in only)]
        if not keep:
            raise SystemExit(f"--stages matched nothing in {stages}")
        return keep
    if start:
        idx = next((i for i, s in enumerate(stages) if start in s), None)
        if idx is None:
            raise SystemExit(f"--from '{start}' matched no stage in {stages}")
        return stages[idx:]
    return stages


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a wyvern-hsi pipeline.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--stages", help="comma-separated tokens; run only matching stages")
    ap.add_argument("--from", dest="start", help="token; run from the first matching stage")
    args = ap.parse_args()

    config = load_config(args.config)
    scripts_dir = config.project_dir / "scripts"
    only = [t.strip() for t in args.stages.split(",")] if args.stages else None
    selected = _select(config.stages, only, args.start)

    print(f"Project: {config.name}")
    print(f"Stages:  {', '.join(selected)}")
    for stage in selected:
        script_path = scripts_dir / stage
        if not script_path.exists():
            raise FileNotFoundError(f"Stage script not found: {script_path}")
        print(f"\n=== {stage} ===")
        _load_stage_main(script_path)(config)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()