"""
Create a water-only mask for water_quality workflows (visible + NIR only; no SWIR).

Method:
- Start from the QA-based clear-only mask (load_valid_mask).
- NDWI-like index from Green (549) and NIR (764); reject vegetation via NDVI (NIR, Red 660).
- NIR-darkness and Red constraints, then morphological cleanup, then keep the two
  largest connected water bodies.

Water-only mask (visible + NIR; no SWIR). QA clear-mask -> NDWI/NDVI thresholds ->
NIR-darkness/Red cutoffs -> morphology -> keep the two largest water bodies.
Outputs under outputs/masks/.

Outputs (outputs/masks/):
- water_mask_raw.png, rgb_water_only_raw.png   (pre-morphology previews)
- water_mask.png, rgb_water_only.png, water_mask.tif
"""
from __future__ import annotations

import numpy as np
import rasterio
from scipy.ndimage import binary_opening, binary_closing, label
import logging

from wyvernhsi import indices, io, visualization
from wyvernhsi.config import Config, load_config
from wyvernhsi.logging_setup import configure_logging
from wyvernhsi.masks import load_valid_mask
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene

logger = logging.getLogger(__name__)

def main(config: Config) -> None:
    m = config.masking
    scene = resolve_scene(config.project_dir)
    out_dir = scene.outputs_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_mask = load_valid_mask(scene.mask)

    with rasterio.open(scene.reflectance) as ds:
        profile = ds.profile
        green = io.read_band_nm(ds, m.green_nm)
        red = io.read_band_nm(ds, m.red_nm)
        nir = io.read_band_nm(ds, m.nir_nm)
        rgb = np.dstack([io.read_band_nm(ds, nm) for nm in m.rgb_nm]).astype(np.float32)

    for arr in (green, red, nir):
        arr[~valid_mask] = np.nan

    ndwi = indices.normalized_difference(green, nir)
    ndvi = indices.normalized_difference(nir, red)

    nir_valid = nir[np.isfinite(nir)]
    if nir_valid.size == 0:
        raise RuntimeError("No valid pixels after QA mask.")
    nir_p = np.percentile(nir_valid, m.nir_percentile)
    red_p = np.percentile(red[np.isfinite(red)], m.red_percentile)

    water_mask_raw = (
        np.isfinite(ndwi) & (ndwi >= m.ndwi_min)
        & np.isfinite(ndvi) & (ndvi <= m.ndvi_max)
        & np.isfinite(nir) & (nir <= nir_p)
        & (red <= red_p)
    )

    lo, hi = m.percentile_lo, m.percentile_hi
    visualization.save_png(out_dir / "water_mask_raw.png", water_mask_raw.astype(np.uint8) * 255)
    rgb_raw = rgb.copy()
    rgb_raw[~valid_mask, :] = np.nan
    rgb_raw[~water_mask_raw, :] = np.nan
    visualization.save_png(out_dir / "rgb_water_only_raw.png", visualization.stretch_rgb(rgb_raw, lo, hi))

    water_mask = binary_opening(water_mask_raw, structure=np.ones((3, 3)))
    water_mask = binary_closing(water_mask, structure=np.ones((3, 3)))

    lbl, n = label(water_mask)
    if n > 0:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0
        keep_labels = np.argsort(sizes)[-2:]
        water_mask = np.isin(lbl, keep_labels)

    visualization.save_png(out_dir / "water_mask.png", water_mask.astype(np.uint8) * 255)
    rgb_final = rgb.copy()
    rgb_final[~valid_mask, :] = np.nan
    rgb_final[~water_mask, :] = np.nan
    visualization.save_png(out_dir / "rgb_water_only.png", visualization.stretch_rgb(rgb_final, lo, hi))

    io.write_geotiff(
        profile, out_dir / "water_mask.tif", water_mask.astype(np.uint8),
        nodata=0, dtype="uint8", descriptions=["WATER_MASK"],
    )

    logger.info(
        "Water mask: %d water px / %d valid px (share %.3f)",
        int(water_mask.sum()), int(valid_mask.sum()),
        water_mask.sum() / max(valid_mask.sum(), 1),
    )
    logger.info("Wrote: %s", out_dir / "water_mask.tif")


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))