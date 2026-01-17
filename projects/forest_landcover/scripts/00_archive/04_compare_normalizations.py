from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import rasterio

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions, pick_band_index_nearest

NAN_VALUE = -9999  # Wyvern tutorial uses this; local file may already be masked


def wyvern_minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to 0–1, with nodata handling similar to Wyvern docs."""
    arr = np.where(arr == NAN_VALUE, np.nan, arr)
    mn = np.nanmin(arr)
    mx = np.nanmax(arr)
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)


def stretch01_percentile(x: np.ndarray, lo=2, hi=98) -> np.ndarray:
    a = np.nanpercentile(x, lo)
    b = np.nanpercentile(x, hi)
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - a) / (b - a)
    return np.clip(y, 0, 1).astype(np.float32)


def get_rgb_arrays(da, wl_nm: np.ndarray, targets_nm: tuple[float, float, float]) -> np.ndarray:
    """Return raw RGB stack as float32, shape (H, W, 3)."""
    idx_r = pick_band_index_nearest(wl_nm, targets_nm[0])
    idx_g = pick_band_index_nearest(wl_nm, targets_nm[1])
    idx_b = pick_band_index_nearest(wl_nm, targets_nm[2])

    r = np.asarray(da.isel(band=idx_r).data.compute(), dtype=np.float32)
    g = np.asarray(da.isel(band=idx_g).data.compute(), dtype=np.float32)
    b = np.asarray(da.isel(band=idx_b).data.compute(), dtype=np.float32)

    return np.dstack([r, g, b])


def normalize_rgb_wyvern(rgb_raw: np.ndarray) -> np.ndarray:
    # Wyvern example normalizes the 3-band stack together (global min/max)
    return wyvern_minmax_normalize(rgb_raw)


def normalize_rgb_percentile_per_channel(rgb_raw: np.ndarray, lo=2, hi=98) -> np.ndarray:
    # Percentile stretch per channel (common for composites)
    out = np.empty_like(rgb_raw, dtype=np.float32)
    for c in range(3):
        out[..., c] = stretch01_percentile(rgb_raw[..., c], lo=lo, hi=hi)
    return out


def plot_and_save(img_a, img_b, title_a: str, title_b: str, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(img_a)
    axes[0].set_title(title_a)
    axes[0].axis("off")

    axes[1].imshow(img_b)
    axes[1].set_title(title_b)
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    if not ACTIVE_WYVERN_FILE.exists():
        raise FileNotFoundError(f"Missing local file: {ACTIVE_WYVERN_FILE}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    src = str(ACTIVE_WYVERN_FILE)

    # Lazy load data
    da = rxr.open_rasterio(src, masked=True, chunks={"x": 1024, "y": 1024})

    # Get wavelengths from descriptions (reliable)
    with rasterio.open(src) as ds:
        desc = list(ds.descriptions)
    wl_nm = parse_wavelengths_nm_from_descriptions(desc)

    # Targets
    rgb_targets = (660.0, 560.0, 490.0)
    cir_targets = (800.0, 660.0, 560.0)

    # RGB comparison
    rgb_raw = get_rgb_arrays(da, wl_nm, rgb_targets)
    rgb_wyvern = normalize_rgb_wyvern(rgb_raw)
    rgb_pct = normalize_rgb_percentile_per_channel(rgb_raw, lo=2, hi=98)

    plot_and_save(
        rgb_wyvern,
        rgb_pct,
        "RGB - Wyvern min-max (global)",
        "RGB - Percentile (2–98) per-channel",
        OUTPUTS_DIR / "compare_rgb_wyvern_vs_percentile.png",
    )

    # CIR comparison
    cir_raw = get_rgb_arrays(da, wl_nm, cir_targets)  # same function: 3-channel composite
    cir_wyvern = normalize_rgb_wyvern(cir_raw)
    cir_pct = normalize_rgb_percentile_per_channel(cir_raw, lo=2, hi=98)

    plot_and_save(
        cir_wyvern,
        cir_pct,
        "CIR - Wyvern min-max (global)",
        "CIR - Percentile (2–98) per-channel",
        OUTPUTS_DIR / "compare_cir_wyvern_vs_percentile.png",
    )

    print("Wrote:")
    print(" -", OUTPUTS_DIR / "compare_rgb_wyvern_vs_percentile.png")
    print(" -", OUTPUTS_DIR / "compare_cir_wyvern_vs_percentile.png")


if __name__ == "__main__":
    main()
