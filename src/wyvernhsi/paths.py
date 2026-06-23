"""Filesystem layout helpers. Pure functions — no work at import; fail fast when called."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def repo_root() -> Path:
    """Repo root, from this file's location (src/wyvernhsi/paths.py)."""
    return Path(__file__).resolve().parents[2]


def project_dir_of(script_file: str) -> Path:
    """The projects/<name> dir a script under projects/<name>/scripts/ belongs to."""
    return Path(script_file).resolve().parents[1]


@dataclass(frozen=True)
class ScenePaths:
    project_dir: Path
    data_dir: Path
    outputs_dir: Path
    radiance: Path
    reflectance: Path
    mask: Path


def find_radiance_scene(data_dir: Path) -> Path:
    """The single raw Wyvern scene in data_dir (not a mask, not a derived reflectance)."""
    tiffs = sorted(data_dir.glob("*.tif*"))
    scenes = [
        p for p in tiffs
        if not p.name.endswith(("_data_mask.tiff", "_data_mask.tif"))
        and "_toa_reflectance" not in p.stem
    ]
    if not scenes:
        raise FileNotFoundError(
            f"No raw Wyvern scene in {data_dir} "
            "(expected a *.tif that is not *_data_mask and not *_toa_reflectance)."
        )
    if len(scenes) > 1:
        raise RuntimeError(
            f"Multiple raw scenes in {data_dir}; keep exactly one active scene per project."
        )
    return scenes[0]


def resolve_scene(project_dir: Path, *, require_reflectance: bool = True) -> ScenePaths:
    """
    Resolve all scene paths for a project, raising immediately if required inputs are missing.
    Conversion stages that *produce* the reflectance pass require_reflectance=False.
    """
    data_dir = project_dir / "data"
    radiance = find_radiance_scene(data_dir)  # exists (globbed)
    reflectance = data_dir / "derived" / f"{radiance.stem}_toa_reflectance.tif"
    mask = data_dir / f"{radiance.stem}_data_mask.tiff"

    if not mask.exists():
        raise FileNotFoundError(f"Data mask not found: {mask}")
    if require_reflectance and not reflectance.exists():
        raise FileNotFoundError(
            f"Derived TOA reflectance not found: {reflectance}\n"
            "Run the TOA reflectance conversion stage first."
        )

    return ScenePaths(
        project_dir=project_dir,
        data_dir=data_dir,
        outputs_dir=project_dir / "outputs",
        radiance=radiance,
        reflectance=reflectance,
        mask=mask,
    )