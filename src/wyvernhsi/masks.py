"""QA and water masks for Wyvern scenes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def _band_index_by_name(ds: rasterio.DatasetReader, name: str) -> int:
    if ds.descriptions is None:
        raise RuntimeError("Mask dataset has no band descriptions.")
    try:
        return ds.descriptions.index(name) + 1  # rasterio bands are 1-based
    except ValueError as e:
        raise RuntimeError(f"Band '{name}' not found. Available: {ds.descriptions}") from e


def load_valid_mask(mask_path) -> np.ndarray:
    """True where a pixel is OK: QA_CLEAR_MASK==1 and not cloud/haze/shadow."""
    with rasterio.open(mask_path) as ds:
        b_clear = _band_index_by_name(ds, "QA_CLEAR_MASK")
        b_cloud = _band_index_by_name(ds, "QA_CLOUD_MASK")
        b_haze = _band_index_by_name(ds, "QA_HAZE_MASK")
        b_shadow = _band_index_by_name(ds, "QA_CLOUD_SHADOW_MASK")
        clear = ds.read(b_clear).astype(np.uint8)
        cloud = ds.read(b_cloud).astype(np.uint8)
        haze = ds.read(b_haze).astype(np.uint8)
        shadow = ds.read(b_shadow).astype(np.uint8)
    return (clear == 1) & ~((cloud == 1) | (haze == 1) | (shadow == 1))


def load_water_mask(mask_path) -> np.ndarray:
    """Load a water-mask GeoTIFF produced by the water-mask stage. True = water."""
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Water mask not found: {mask_path}. Run the water-mask stage first.")
    with rasterio.open(mask_path) as ds:
        return ds.read(1) == 1