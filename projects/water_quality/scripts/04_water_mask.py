"""
Goal:
- Create a water-only mask for water_quality workflows.
- Uses only visible + NIR bands (no SWIR available).

Method:
- Start from valid_mask (your QA-based clear-only mask).
- Compute NDWI-like index using Green (549) and NIR (764).
- Reject vegetation using NDVI using Red (660) and NIR (764).
- Optional NIR darkness constraint.

Outputs:
- outputs/masks/water_mask.png          (binary preview)
- outputs/masks/rgb_water_only.png      (RGB preview with land removed)

Notes:
- Thresholds are heuristic; tweak per scene.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import binary_opening, binary_closing, label, distance_transform_edt


from wyvernhsi.paths import ACTIVE_WYVERN_FILE, ACTIVE_WYVERN_MASK, OUTPUTS_DIR
from wyvernhsi.masks import load_valid_mask

# Bands (1-based)
B_GREEN = 5   # 549
B_RED = 12    # 660
B_NIR = 21    # 764

# For quick RGB preview (same as before)
BANDS_RGB = {"R": 12, "G": 5, "B": 2}  # 660 / 549 / 510
P_LO, P_HI = 2.0, 98.0

# --- Water mask thresholds (start here, then tune) ---
NDWI_MIN = 0.15     # higher => stricter water-only
NDVI_MAX = 0.10     # lower => stricter vegetation rejection
NIR_MAX_PCT = 60.0  # water tends to be dark in NIR; keep pixels below this percentile
#MIN_WATER_COMPONENT_PX = 5_000  # tune: 500–50_000 depending on resolution/scene
#MIN_WATER_THICKNESS_PX = 3  # try 2; if roads remain, try 3. If lakes disappear, drop to 1.




def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_band(ds: rasterio.DatasetReader, band_index_1based: int) -> np.ndarray:
    arr = ds.read(band_index_1based).astype(np.float32)
    nodata = ds.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr


def safe_normdiff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    out = np.full_like(a, np.nan, dtype=np.float32)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(denom) > 1e-10)
    out[ok] = (a[ok] - b[ok]) / denom[ok]
    return out


def percentile_stretch_rgb(rgb: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        band = rgb[:, :, i]
        valid = np.isfinite(band)
        if not np.any(valid):
            continue
        lo = np.nanpercentile(band[valid], p_lo)
        hi = np.nanpercentile(band[valid], p_hi)
        if hi <= lo:
            continue
        out[:, :, i] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    out = np.nan_to_num(out, nan=0.0)
    return (out * 255.0 + 0.5).astype(np.uint8)


def load_rgb_stack(image_path: Path, bands: dict[str, int]) -> np.ndarray:
    with rasterio.open(image_path) as ds:
        r = read_band(ds, bands["R"])
        g = read_band(ds, bands["G"])
        b = read_band(ds, bands["B"])
    return np.dstack([r, g, b]).astype(np.float32)

def save_mask_geotiff(ref_image_path: Path, out_path: Path, mask: np.ndarray) -> None:
    """
    Save a boolean mask as uint8 GeoTIFF aligned to ref image.
    mask: True = keep (water), False = not water
    """
    with rasterio.open(ref_image_path) as src:
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=0,
            compress="deflate",
            tiled=True,
        )

        out = mask.astype(np.uint8)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(out, 1)
            dst.set_band_description(1, "WATER_MASK")

def save_png(path: Path, arr: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    if arr.ndim == 2:
        plt.imshow(arr)
    else:
        plt.imshow(arr)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def main() -> None:
    img_path = Path(ACTIVE_WYVERN_FILE)
    msk_path = Path(ACTIVE_WYVERN_MASK)

    out_dir = OUTPUTS_DIR / "masks"
    ensure_dir(out_dir)

    valid_mask = load_valid_mask(msk_path)  # True = OK pixels (clear & not cloud/haze/shadow)

    with rasterio.open(img_path) as ds:
        green = read_band(ds, B_GREEN)
        red = read_band(ds, B_RED)
        nir = read_band(ds, B_NIR)

    # apply QA validity
    green[~valid_mask] = np.nan
    red[~valid_mask] = np.nan
    nir[~valid_mask] = np.nan

    ndwi = safe_normdiff(green, nir)  # (G - NIR) / (G + NIR)
    ndvi = safe_normdiff(nir, red)    # (NIR - R) / (NIR + R)

    # NIR darkness cutoff computed on valid pixels only
    nir_valid = nir[np.isfinite(nir)]
    if nir_valid.size == 0:
        raise RuntimeError("No valid pixels after QA mask.")
    nir_max = np.percentile(nir_valid, NIR_MAX_PCT)
    nir_p = np.percentile(nir_valid, 35.0)     # try 25–45
    red_valid = red[np.isfinite(red)]
    red_p = np.percentile(red_valid, 55.0)     # try 40–60


    water_mask_raw = (
        np.isfinite(ndwi) &
        (ndwi >= NDWI_MIN) &
        np.isfinite(ndvi) &
        (ndvi <= NDVI_MAX) &
        np.isfinite(nir) &
        (nir <= nir_p) &
        (red <= red_p)
    )

    # --- exports before morphology ---
    water_preview_raw = (water_mask_raw.astype(np.uint8) * 255)
    save_png(out_dir / "water_mask_raw.png", water_preview_raw)

    rgb_raw = load_rgb_stack(img_path, BANDS_RGB)
    rgb_raw[~valid_mask, :] = np.nan
    rgb_raw[~water_mask_raw, :] = np.nan
    rgb_raw_u8 = percentile_stretch_rgb(rgb_raw, P_LO, P_HI)
    save_png(out_dir / "rgb_water_only_raw.png", rgb_raw_u8)

    # -------------------------
    # Morphological cleanup
    # -------------------------

    water_mask = water_mask_raw.copy()

    # Remove thin linear features (roads, ditches)
    water_mask = binary_opening(water_mask,structure=np.ones((3, 3)))
    # 2) THIN-LINE KILLER (roads/ditches are thin)
    #dist = distance_transform_edt(water_mask)
    #water_mask = water_mask & (dist >= MIN_WATER_THICKNESS_PX)
    # Fill small holes inside water bodies
    water_mask = binary_closing(water_mask,structure=np.ones((3, 3)))

    # -------------------------
    # Keep largest connected components
    # -------------------------

    lbl, n = label(water_mask)
    if n > 0:
        sizes = np.bincount(lbl.ravel())
        sizes[0] = 0  # ignore background

        # Keep only the 2 largest water regions
        keep_labels = np.argsort(sizes)[-2:]
        #keep_labels = np.where(sizes >= MIN_WATER_COMPONENT_PX)[0]
        water_mask = np.isin(lbl, keep_labels)


    # --- exports ---
    # 1) water mask preview (white=water, black=not water)
    water_preview = (water_mask.astype(np.uint8) * 255)
    save_png(out_dir / "water_mask.png", water_preview)

    # 2) RGB water-only preview
    rgb = load_rgb_stack(img_path, BANDS_RGB)
    rgb[~valid_mask, :] = np.nan
    rgb[~water_mask, :] = np.nan
    rgb_u8 = percentile_stretch_rgb(rgb, P_LO, P_HI)
    save_png(out_dir / "rgb_water_only.png", rgb_u8)

    save_mask_geotiff(img_path, out_dir / "water_mask.tif", water_mask)
    print(f"- {out_dir / 'water_mask.tif'}")


    print("Water mask stats:")
    print(f"  water pixels: {water_mask.sum():,}")
    print(f"  valid pixels: {valid_mask.sum():,}")
    print(f"  water share of valid: {water_mask.sum() / max(valid_mask.sum(), 1):.3f}")
    print("Wrote:")
    print(f"- {out_dir / 'water_mask.png'}")
    print(f"- {out_dir / 'rgb_water_only.png'}")


if __name__ == "__main__":
    main()
