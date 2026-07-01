"""Run provenance: a manifest.json (what ran, on what, with which versions) and a
markdown report that embeds the run's figures so the README never drifts from outputs."""
from __future__ import annotations

import dataclasses
import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from wyvernhsi.config import Config
from wyvernhsi.paths import find_radiance_scene


def _git_sha(repo_dir: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_dir,
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _versions(packages: list[str]) -> dict:
    out = {}
    for p in packages:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = "unknown"
    return out


def build_manifest(config: Config, run_id: str) -> dict:
    """Collect provenance for a completed run from the project's data and outputs dirs."""
    data_dir = config.project_dir / "data"
    outputs = config.project_dir / "outputs"
    radiance = find_radiance_scene(data_dir)

    def listing(pattern: str) -> list:
        return sorted(p.relative_to(outputs).as_posix() for p in outputs.rglob(pattern))

    return {
        "run_id": run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": config.name,
        "git_sha": _git_sha(config.project_dir),
        "scene": {
            "radiance": radiance.name,
            "reflectance": f"{radiance.stem}_toa_reflectance.tif",
        },
        "package_versions": _versions(["numpy", "rasterio", "scikit-learn"]),
        "config": dataclasses.asdict(config),
        "outputs": {
            "figures": listing("*.png"),
            "tables": listing("*.csv"),
            "rasters": listing("*.tif"),
        },
    }


def write_manifest(manifest: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def write_report(manifest: dict, report_path: Path) -> Path:
    """Write report.md next to the figures (relative image links resolve from outputs/)."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {manifest['project']} — run report",
        "",
        f"- Run: `{manifest['run_id']}` ({manifest['generated_utc']})",
        f"- Commit: `{manifest['git_sha']}`",
        f"- Scene: `{manifest['scene']['radiance']}` → TOA reflectance",
        "",
        "## Figures",
        "",
    ]
    for fig in manifest["outputs"]["figures"]:
        lines += [f"### {fig}", f"![{fig}]({fig})", ""]
    if manifest["outputs"]["tables"]:
        lines += ["## Tables", ""]
        lines += [f"- `{tbl}`" for tbl in manifest["outputs"]["tables"]]
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path