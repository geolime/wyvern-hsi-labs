from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, ACTIVE_WYVERN_MASK, OUTPUTS_DIR
from wyvernhsi.masks import load_valid_mask, load_water_mask


# ---------------- Settings ----------------
# Bands (1-based)
B570 = 6
B660 = 12
B669 = 13
B711 = 17
B764 = 21

# NGB background for overlays
BANDS_NGB = {"R": 21, "G": 5, "B": 2}  # 764/549/510

P_LO, P_HI = 2.0, 98.0
EPS = 1e-6

HOTSPOT_TOP_PCT = 10.0

OUT_DIR = OUTPUTS_DIR / "water_features" / "proxies"
OUT_WHOLE = OUT_DIR / "whole_scene"
OUT_WATER = OUT_DIR / "water_only"


# ---------------- Helpers ----------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_band(ds: rasterio.DatasetReader, b1: int) -> np.ndarray:
    x = ds.read(b1).astype(np.float32)
    if ds.nodata is not None:
        x[x == ds.nodata] = np.nan
    return x


def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float32)
    denom = a + b
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(denom) > EPS)
    out[ok] = (a[ok] - b[ok]) / denom[ok]
    return out


def percentile_stretch01(x: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    v = x[np.isfinite(x)]
    if v.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = np.percentile(v, p_lo)
    hi = np.percentile(v, p_hi)
    y = (x - lo) / (hi - lo + 1e-12)
    return np.clip(y, 0, 1).astype(np.float32)


def build_composite(ds: rasterio.DatasetReader, bands: dict[str, int]) -> np.ndarray:
    r = read_band(ds, bands["R"])
    g = read_band(ds, bands["G"])
    b = read_band(ds, bands["B"])
    rgb = np.dstack([
        percentile_stretch01(r, P_LO, P_HI),
        percentile_stretch01(g, P_LO, P_HI),
        percentile_stretch01(b, P_LO, P_HI),
    ])
    return np.nan_to_num(rgb, nan=0.0)


def robust_limits(x: np.ndarray, lo: float = 2.0, hi: float = 98.0):
    v = x[np.isfinite(x)]
    if v.size == 0:
        return None, None
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def save_continuous_png(path: Path, arr2d: np.ndarray, title: str) -> None:
    vmin, vmax = robust_limits(arr2d)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(arr2d, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.title(title)

    fig = plt.gcf()
    fig.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.88, 0.12, 0.03, 0.76])
    plt.colorbar(im, cax=cax)

    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("Wrote:", path)


def bin_by_quantiles(arr2d: np.ndarray, mask: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """
    Returns int16 array with values 0..n_bins-1, NaN elsewhere.
    Bins are quantiles over arr2d[mask & finite].
    """
    out = np.full(arr2d.shape, -1, dtype=np.int16)
    v = arr2d[mask & np.isfinite(arr2d)]
    if v.size == 0:
        return out

    qs = np.quantile(v, np.linspace(0, 1, n_bins + 1))
    # ensure strictly increasing edges (rare but can happen)
    qs = np.unique(qs)
    if qs.size < 3:
        return out

    # digitize into bins
    idx = np.digitize(arr2d, qs[1:-1], right=True)  # 0..(len(qs)-2)
    ok = mask & np.isfinite(arr2d)
    out[ok] = idx[ok].astype(np.int16)
    return out


def save_binned_png(path: Path, binned: np.ndarray, title: str, n_bins: int = 5) -> None:
    plot = np.ma.masked_where(binned < 0, binned)

    cmap = plt.get_cmap("viridis", n_bins).copy()
    cmap.set_bad(alpha=0.0)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(plot, cmap=cmap, vmin=0, vmax=n_bins - 1)
    plt.axis("off")
    plt.title(title)

    # discrete colorbar
    fig = plt.gcf()
    fig.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.88, 0.12, 0.03, 0.76])
    cb = plt.colorbar(im, cax=cax, ticks=list(range(n_bins)))
    cb.ax.set_yticklabels([f"Q{i+1}" for i in range(n_bins)])

    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("Wrote:", path)


def threshold_top_percent(arr2d: np.ndarray, mask: np.ndarray, top_pct: float) -> np.ndarray:
    v = arr2d[mask & np.isfinite(arr2d)]
    if v.size == 0:
        return np.zeros(arr2d.shape, dtype=bool)
    thr = np.percentile(v, 100.0 - top_pct)
    return (mask & np.isfinite(arr2d) & (arr2d >= thr))


def save_hotspot_overlay(out_png: Path, ngb: np.ndarray, hotspot: np.ndarray, water_mask: np.ndarray, title: str) -> None:
    plt.figure(figsize=(12, 10))
    plt.imshow(ngb)

    # outline hotspots
    plt.contour(hotspot.astype(np.uint8), levels=[0.5], colors="yellow", linewidths=1.2)

    # shoreline outline
    plt.contour(water_mask.astype(np.uint8), levels=[0.5], colors="white", linewidths=0.6)


    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Wrote:", out_png)


# ---------------- Main ----------------
def main() -> None:
    ensure_dir(OUT_DIR)
    ensure_dir(OUT_WHOLE)
    ensure_dir(OUT_WATER)

    if not ACTIVE_WYVERN_FILE.exists():
        raise FileNotFoundError(f"Missing image: {ACTIVE_WYVERN_FILE}")
    if not ACTIVE_WYVERN_MASK.exists():
        raise FileNotFoundError(f"Missing QA mask: {ACTIVE_WYVERN_MASK}")

    valid = load_valid_mask(ACTIVE_WYVERN_MASK).astype(bool)
    water = load_water_mask().astype(bool)
    use_water = valid & water
    use_whole = valid

    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        r570 = read_band(ds, B570)
        r660 = read_band(ds, B660)
        r669 = read_band(ds, B669)
        r711 = read_band(ds, B711)

        ndti = nd(r660, r570)          # turbidity proxy
        ndci = nd(r711, r669)          # chlorophyll/red-edge proxy

        # backgrounds for overlays
        ngb = build_composite(ds, BANDS_NGB)

    # whole-scene (valid only)
    ndti_whole = ndti.copy()
    ndci_whole = ndci.copy()
    ndti_whole[~use_whole] = np.nan
    ndci_whole[~use_whole] = np.nan

    # water-only (valid & water)
    ndti_water = ndti.copy()
    ndci_water = ndci.copy()
    ndti_water[~use_water] = np.nan
    ndci_water[~use_water] = np.nan

    # ---- Continuous maps ----
    save_continuous_png(OUT_WHOLE / "ndti_continuous.png", ndti_whole, "NDTI (660/570) turbidity proxy (whole scene, QA-valid)")
    save_continuous_png(OUT_WATER / "ndti_continuous.png", ndti_water, "NDTI (660/570) turbidity proxy (water-only)")

    save_continuous_png(OUT_WHOLE / "ndci_continuous.png", ndci_whole, "NDCI (711/669) chlorophyll proxy (whole scene, QA-valid)")
    save_continuous_png(OUT_WATER / "ndci_continuous.png", ndci_water, "NDCI (711/669) chlorophyll proxy (water-only)")

    # ---- Binned maps (quantiles) ----
    ndti_b_whole = bin_by_quantiles(ndti, use_whole, n_bins=5)
    ndti_b_water = bin_by_quantiles(ndti, use_water, n_bins=5)
    ndci_b_whole = bin_by_quantiles(ndci, use_whole, n_bins=5)
    ndci_b_water = bin_by_quantiles(ndci, use_water, n_bins=5)

    save_binned_png(OUT_WHOLE / "ndti_binned5.png", ndti_b_whole, "NDTI binned into 5 quantiles (whole scene, QA-valid)", n_bins=5)
    save_binned_png(OUT_WATER / "ndti_binned5.png", ndti_b_water, "NDTI binned into 5 quantiles (water-only)", n_bins=5)

    save_binned_png(OUT_WHOLE / "ndci_binned5.png", ndci_b_whole, "NDCI binned into 5 quantiles (whole scene, QA-valid)", n_bins=5)
    save_binned_png(OUT_WATER / "ndci_binned5.png", ndci_b_water, "NDCI binned into 5 quantiles (water-only)", n_bins=5)

    # ---- Hotspots (top 10%) on NGB ----
    ndti_hot = threshold_top_percent(ndti, use_water, top_pct=HOTSPOT_TOP_PCT)
    ndci_hot = threshold_top_percent(ndci, use_water, top_pct=HOTSPOT_TOP_PCT)

    save_hotspot_overlay(
        OUT_DIR / "ndti_hotspots_top10_on_ngb.png",
        ngb,
        ndti_hot,
        water,
        "NDTI hotspots (top 10% on water-only) over NGB",
    )

    save_hotspot_overlay(
        OUT_DIR / "ndci_hotspots_top10_on_ngb.png",
        ngb,
        ndci_hot,
        water,
        "NDCI hotspots (top 10% on water-only) over NGB",
    )


if __name__ == "__main__":
    main()
