from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from wyvernhsi.wavelengths import (
    parse_wavelengths_nm_from_descriptions,
    pick_band_index_nearest,
)


def read_band(ds: rasterio.DatasetReader, band_1based: int) -> np.ndarray:
    """Read one band as float32 with nodata set to NaN."""
    arr = ds.read(band_1based).astype(np.float32)
    if ds.nodata is not None:
        arr[arr == ds.nodata] = np.nan
    return arr


def read_cube(ds: rasterio.DatasetReader) -> np.ndarray:
    """Read all bands as an (H, W, B) float32 cube with nodata set to NaN."""
    cube = ds.read().astype(np.float32)  # (B, H, W)
    if ds.nodata is not None:
        cube[cube == ds.nodata] = np.nan
    return np.transpose(cube, (1, 2, 0))  # (H, W, B)


def wavelengths_nm(ds: rasterio.DatasetReader) -> np.ndarray:
    """Per-band wavelengths (nm) parsed from band descriptions; NaN where unknown."""
    return parse_wavelengths_nm_from_descriptions(ds.descriptions)


def band_index_for_nm(ds: rasterio.DatasetReader, target_nm: float) -> int:
    """1-based rasterio band index nearest to target_nm. Raises if no wavelengths."""
    return pick_band_index_nearest(wavelengths_nm(ds), target_nm) + 1


def read_band_nm(ds: rasterio.DatasetReader, target_nm: float) -> np.ndarray:
    """Read the band nearest to target_nm as float32 with NaN nodata."""
    return read_band(ds, band_index_for_nm(ds, target_nm))


def write_geotiff(
    ref_profile: dict,
    out_path: Path,
    array: np.ndarray,
    *,
    nodata: float | int | None,
    dtype: str,
    descriptions: list[str] | None = None,
) -> None:
    """
    Write a (H, W) or (H, W, B) array as a tiled, deflate-compressed GeoTIFF,
    reusing ref_profile for CRS/transform. predictor is chosen from dtype.
    """
    if array.ndim == 2:
        array = array[:, :, None]
    _, _, count = array.shape

    profile = ref_profile.copy()
    profile.update(
        count=count,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        predictor=2 if np.issubdtype(np.dtype(dtype), np.integer) else 3,
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        for b in range(count):
            dst.write(array[:, :, b].astype(dtype), b + 1)
            if descriptions is not None:
                dst.set_band_description(b + 1, descriptions[b])