from __future__ import annotations
import numpy as np
import rasterio


def _band_index_by_name(ds: rasterio.DatasetReader, name: str) -> int:
    # rasterio bands are 1-based; descriptions is 0-based list
    if ds.descriptions is None:
        raise RuntimeError("Mask dataset has no band descriptions.")
    try:
        return ds.descriptions.index(name) + 1
    except ValueError as e:
        raise RuntimeError(f"Band '{name}' not found. Available: {ds.descriptions}") from e


def load_valid_mask(mask_path) -> np.ndarray:
    """
    True where pixel is OK to use.
    Rule:
      keep: QA_CLEAR_MASK == 1
      drop: cloud==1 or haze==1 or cloud_shadow==1
    """
    with rasterio.open(mask_path) as ds:
        # Ensure names exist; fails fast with helpful message
        b_clear = _band_index_by_name(ds, "QA_CLEAR_MASK")
        b_cloud = _band_index_by_name(ds, "QA_CLOUD_MASK")
        b_haze = _band_index_by_name(ds, "QA_HAZE_MASK")
        b_shadow = _band_index_by_name(ds, "QA_CLOUD_SHADOW_MASK")

        clear = ds.read(b_clear).astype(np.uint8)
        cloud = ds.read(b_cloud).astype(np.uint8)
        haze = ds.read(b_haze).astype(np.uint8)
        shadow = ds.read(b_shadow).astype(np.uint8)

    clear_ok = (clear == 1)
    ground_interference = (cloud == 1) | (haze == 1) | (shadow == 1)

    return clear_ok & (~ground_interference)
