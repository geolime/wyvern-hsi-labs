"""Typed configuration loaded from a project's YAML file."""
from __future__ import annotations

from dataclasses import dataclass, field
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
    use_subset: bool = False
    subset_pad: int = 0
    subset_points: list = field(default_factory=list)
    sample_silhouette: int = 0
    stability_runs: int = 0


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
class ProxiesCfg:
    ndti_green_nm: float
    ndti_red_nm: float
    ndci_red_edge_nm: float
    ndci_red_nm: float
    ngb_nm: list
    percentile_lo: float
    percentile_hi: float
    hotspot_top_pct: float
    n_bins: int
    composite_w_ndti: float
    composite_w_ndci: float
    composite_hotspot_top_pct: float
    state_split_pct: float

@dataclass(frozen=True)
class SamCfg:
    angle_threshold_rad: float
    roi_half_size_px: int
    tile_size: int
    reference_classes: dict

@dataclass(frozen=True)
class SfaCfg:
    ndti_green_nm: float
    ndti_red_nm: float
    ndci_red_edge_nm: float
    ndci_red_nm: float
    nir_red_nir_nm: float
    nir_red_red_nm: float
    rgb_nm: list
    ngb_nm: list

@dataclass(frozen=True)
class ValidationCfg:
    reference_tif: str
    reference_ignore: list
    crosswalk: dict   # {class_name: [reference_code, ...]}

@dataclass(frozen=True)
class RandomForestCfg:
    max_samples_per_class: int
    n_blocks: int
    test_frac: float
    n_estimators: int
    max_depth: int


@dataclass(frozen=True)
class Config:
    name: str
    project_dir: Path
    random_seed: int
    stages: list
    clustering: Optional[ClusteringCfg] = None
    features: Optional[FeaturesCfg] = None
    masking: Optional[MaskingCfg] = None
    proxies: Optional[ProxiesCfg] = None
    sam: Optional[SamCfg] = None
    sfa: Optional[SfaCfg] = None
    validation: Optional[ValidationCfg] = None
    random_forest: Optional[RandomForestCfg] = None

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
        proxies=ProxiesCfg(**raw["proxies"]) if "proxies" in raw else None,
        sam=SamCfg(**raw["sam"]) if "sam" in raw else None,
        sfa=SfaCfg(**raw["sfa"]) if "sfa" in raw else None,
        validation=ValidationCfg(**raw["validation"]) if "validation" in raw else None,
        random_forest=RandomForestCfg(**raw["random_forest"]) if "random_forest" in raw else None,
    )