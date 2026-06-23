"""
Create a water-only mask for water_quality workflows (visible + NIR only; no SWIR).

Method:
- Start from the QA-based clear-only mask (load_valid_mask).
- NDWI-like index from Green (549) and NIR (764); reject vegetation via NDVI (NIR, Red 660).
- NIR-darkness and Red constraints, then morphological cleanup, then keep the two
  largest connected water bodies.

Outputs (outputs/masks/):
- water_mask_raw.png, rgb_water_only_raw.png   (pre-morphology previews)
- water_mask.png, rgb_water_only.png, water_mask.tif
"""
from __future__ import annotations

import numpy as np
import rasterio
from scipy.ndimage import binary_opening, binary_closing, label

from wyvernhsi import indices, io, visualization
from wyvernhsi.masks import load_valid_mask
from wyvernhsi.paths import project_dir_of, resolve_scene

# Target wavelengths (nm) — resolved to nearest band, not hardcoded indices
NM_GREEN, NM_RED, NM_NIR = 549.0, 660.0, 764.0
NM_RGB = (660.0, 549.0, 510.0)  # R, G, B preview
P_LO, P_HI = 2.0, 98.0

# Water-mask thresholds (heuristic; tune per scene)
NDWI_MIN = 0.15  # higher => stricter water-only
NDVI_MAX = 0.10  # lower  => stricter vegetation rejection


def main() -> None:
    scene = resolve_scene(project_dir_of(__file__))
    img_path = scene.reflectance
    out_dir = scene.outputs_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_mask = load_valid_mask(scene.mask)  # True = clear & not cloud/haze/shadow

    with rasterio.open(img_path) as ds:
        profile = ds.profile
        green = io.read_band_nm(ds, NM_GREEN)
        red = io.read_band_nm(ds, NM_RED)
        nir = io.read_band_nm(ds, NM_NIR)
        rgb = np.dstack([io.read_band_nm(ds, nm) for nm in NM_RGB]).astype(np.float32)

    for arr in (green, red, nir):
        arr[~valid_mask] = np.nan

    ndwi = indices.normalized_difference(green, nir)
    ndvi = indices.normalized_difference(nir, red)

    nir_valid = nir[np.isfinite(nir)]
    if nir_valid.size == 0:
        raise RuntimeError("No valid pixels after QA mask.")
    nir_p = np.percentile(nir_valid, 35.0)
    red_p = np.percentile(red[np.isfinite(red)], 55.0)

    water_mask_raw = (
        np.isfinite(ndwi) & (ndwi >= NDWI_MIN)
        & np.isfinite(ndvi) & (ndvi <= NDVI_MAX)
        & np.isfinite(nir) & (nir <= nir_p)
        & (red <= red_p)
    )

    # Pre-morphology previews
    visualization.save_png(out_dir / "water_mask_raw.png", water_mask_raw.astype(np.uint8) * 255)
    rgb_raw = rgb.copy()
    rgb_raw[~valid_mask, :] = np.nan
    rgb_raw[~water_mask_raw, :] = np.nan
    visualization.save_png(out_dir / "rgb_water_only_raw.png", visualization.stretch_rgb(rgb_raw, P_LO, P_HI))

    # Morphological cleanup
    water_mask = binary_opening(water_mask_raw, structure=np.ones((3, 3)))  # drop thin features
    water_mask = binary_closing(water_mask, structure=np.ones((3, 3)))      # fill small holes

    # Keep the two largest connected water bodies
    lbl, n = label(water_mask)
    if n > 0:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0  # ignore background
        keep_labels = np.argsort(sizes)[-2:]
        water_mask = np.isin(lbl, keep_labels)

    # Final previews + GeoTIFF
    visualization.save_png(out_dir / "water_mask.png", water_mask.astype(np.uint8) * 255)
    rgb_final = rgb.copy()
    rgb_final[~valid_mask, :] = np.nan
    rgb_final[~water_mask, :] = np.nan
    visualization.save_png(out_dir / "rgb_water_only.png", visualization.stretch_rgb(rgb_final, P_LO, P_HI))

    io.write_geotiff(
        profile, out_dir / "water_mask.tif", water_mask.astype(np.uint8),
        nodata=0, dtype="uint8", descriptions=["WATER_MASK"],
    )

    print("Water mask stats:")
    print(f"  water pixels: {water_mask.sum():,}")
    print(f"  valid pixels: {valid_mask.sum():,}")
    print(f"  water share of valid: {water_mask.sum() / max(valid_mask.sum(), 1):.3f}")
    print("Wrote:", out_dir / "water_mask.tif", out_dir / "water_mask.png", out_dir / "rgb_water_only.png")


if __name__ == "__main__":
    main()