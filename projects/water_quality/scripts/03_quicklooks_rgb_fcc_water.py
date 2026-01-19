"""
Exports:
- RGB quicklook (R=660, G=549, B=510)
- Water-friendly composite (NIR/Green/Blue): R=764, G=549, B=510

Masking:
- Cloud mask only (per repo convention)
- Masked pixels -> NaN
- Percentile stretch (2–98) for visualization
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, ACTIVE_WYVERN_MASK, OUTPUTS_DIR
from wyvernhsi.masks import load_valid_mask

# Band indices (1-based), verified by band descriptions
BANDS_RGB = {"R": 12, "G": 5, "B": 2}   # 660 / 549 / 510
BANDS_NGB = {"R": 21, "G": 5, "B": 2}   # 764 / 549 / 510

P_LO, P_HI = 2.0, 98.0


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_band(ds: rasterio.DatasetReader, band_index_1based: int) -> np.ndarray:
    arr = ds.read(band_index_1based).astype(np.float32)
    nodata = ds.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr


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

    # Replace NaNs (clouds / nodata) with black
    out = np.nan_to_num(out, nan=0.0)

    return (out * 255.0 + 0.5).astype(np.uint8)


def load_rgb_stack(image_path: Path, bands: dict[str, int]) -> np.ndarray:
    with rasterio.open(image_path) as ds:
        r = read_band(ds, bands["R"])
        g = read_band(ds, bands["G"])
        b = read_band(ds, bands["B"])
    return np.dstack([r, g, b]).astype(np.float32)


def save_png(path: Path, rgb_u8: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_u8)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def main() -> None:
    previews_dir = OUTPUTS_DIR / "previews"
    ensure_dir(previews_dir)

    img_path = Path(ACTIVE_WYVERN_FILE)
    msk_path = Path(ACTIVE_WYVERN_MASK)

    if not img_path.exists():
        raise FileNotFoundError(f"Missing ACTIVE_WYVERN_FILE: {img_path}")
    if not msk_path.exists():
        raise FileNotFoundError(f"Missing ACTIVE_WYVERN_MASK: {msk_path}")

    print("Band mapping:")
    print(f"  RGB: R=Band_660 (12), G=Band_549 (5), B=Band_510 (2)")
    print(f"  NGB: R=Band_764 (21), G=Band_549 (5), B=Band_510 (2)")
    print(f"Stretch: p{P_LO}–p{P_HI}")
    print("Mask: cloud only")

    valid_mask = load_valid_mask(msk_path)  # True where OK to use
    invalid_mask = ~valid_mask              # True where we should mask out

    # RGB
    rgb = load_rgb_stack(img_path, BANDS_RGB)
    rgb[invalid_mask, :] = np.nan

    rgb_u8 = percentile_stretch_rgb(rgb, P_LO, P_HI)
    save_png(previews_dir / "rgb_quicklook.png", rgb_u8)

    # NGB
    ngb = load_rgb_stack(img_path, BANDS_NGB)
    ngb[invalid_mask, :] = np.nan

    ngb_u8 = percentile_stretch_rgb(ngb, P_LO, P_HI)
    save_png(previews_dir / "ngb_water_composite.png", ngb_u8)

    print("Wrote:")
    print(f"- {previews_dir / 'rgb_quicklook.png'}")
    print(f"- {previews_dir / 'ngb_water_composite.png'}")


if __name__ == "__main__":
    main()
