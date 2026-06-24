"""Typed configuration loaded from a project's YAML file."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from wyvernhsi.paths import repo_root


@dataclass(frozen=True)
class ClusteringCfg:
    k: int
    pca_components: int
    n_samples: int
    tile_size: int
    use_subset: bool
    subset_pad: int
    subset_points: list


@dataclass(frozen=True)
class FeaturesCfg:
    red_nm: float
    red_edge_nm: float
    nir_nm: float
    min_pixels: int


@dataclass(frozen=True)
class MaskingCfg:
    green_nm: float
    red_nm: float
    nir_nm: float
    rgb_nm: list
    ndwi_min: float
    ndvi_max: float
    nir_percentile: float
    red_percentile: float
    percentile_lo: float
    percentile_hi: float

@dataclass(frozen=True)
class SamCfg:
    angle_threshold_rad: float
    roi_half_size_px: int
    tile_size: int
    reference_classes: dict


@dataclass(frozen=True)
class Config:
    name: str
    project_dir: Path
    random_seed: int
    stages: list
    clustering: Optional[ClusteringCfg] = None
    features: Optional[FeaturesCfg] = None
    masking: Optional[MaskingCfg] = None
    sam: Optional[SamCfg] = None


def load_config(path) -> Config:
    """Load and validate a project config. Missing/extra keys fail fast."""
    raw = yaml.safe_load(Path(path).read_text())
    name = raw["project"]["name"]
    return Config(
        name=name,
        project_dir=repo_root() / "projects" / name,
        random_seed=raw["project"]["random_seed"],
        stages=list(raw["pipeline"]["stages"]),
        clustering=ClusteringCfg(**raw["clustering"]) if "clustering" in raw else None,
        features=FeaturesCfg(**raw["features"]) if "features" in raw else None,
        masking=MaskingCfg(**raw["masking"]) if "masking" in raw else None,
        sam=SamCfg(**raw["sam"]) if "sam" in raw else None,
    )