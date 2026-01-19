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

# NGB background
BANDS_NGB = {"R": 21, "G": 5, "B": 2}

P_LO, P_HI = 2.0, 98.0
EPS = 1e-6

HOTSPOT_TOP_PCT = 5.0
W_NDTI = 1.0
W_NDCI = 0.5

OUT_DIR = OUTPUTS_DIR / "water_features" / "proxies"


# ---------------- Helpers ----------------
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


def zscore_on_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=np.float32)
    v = x[mask & np.isfinite(x)]
    if v.size == 0:
        return out
    mu = float(np.mean(v))
    sd = float(np.std(v)) + 1e-12
    out[mask & np.isfinite(x)] = ((x[mask & np.isfinite(x)] - mu) / sd).astype(np.float32)
    return out


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


def threshold_top_percent(arr2d: np.ndarray, mask: np.ndarray, top_pct: float) -> np.ndarray:
    v = arr2d[mask & np.isfinite(arr2d)]
    if v.size == 0:
        return np.zeros(arr2d.shape, dtype=bool)
    thr = np.percentile(v, 100.0 - top_pct)
    return (mask & np.isfinite(arr2d) & (arr2d >= thr))


def save_hotspot_overlay(out_png: Path, ngb: np.ndarray, hotspot: np.ndarray, water_mask: np.ndarray, title: str) -> None:
    plt.figure(figsize=(12, 10))
    plt.imshow(ngb)

    plt.contour(hotspot.astype(np.uint8), levels=[0.5], linewidths=1.0)  # hotspot outline
    plt.contour(water_mask.astype(np.uint8), levels=[0.5], linewidths=0.6)  # shoreline

    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Wrote:", out_png)


# ---------------- Main ----------------
def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    valid = load_valid_mask(ACTIVE_WYVERN_MASK).astype(bool)
    water = load_water_mask().astype(bool)
    use = valid & water

    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        r570 = read_band(ds, B570)
        r660 = read_band(ds, B660)
        r669 = read_band(ds, B669)
        r711 = read_band(ds, B711)

        ngb = build_composite(ds, BANDS_NGB)

    ndti = nd(r660, r570)
    ndci = nd(r711, r669)

    ndti[~use] = np.nan
    ndci[~use] = np.nan

    z_ndti = zscore_on_mask(ndti, use)
    z_ndci = zscore_on_mask(ndci, use)

    risk = (W_NDTI * z_ndti + W_NDCI * z_ndci).astype(np.float32)
    risk[~use] = np.nan

    save_continuous_png(
        OUT_DIR / "risk_proxy_continuous.png",
        risk,
        f"Composite proxy = {W_NDTI}*z(NDTI) + {W_NDCI}*z(NDCI) (water-only)",
    )

    hot = threshold_top_percent(risk, use, top_pct=HOTSPOT_TOP_PCT)
    save_hotspot_overlay(
        OUT_DIR / "risk_proxy_hotspots_top5_on_ngb.png",
        ngb,
        hot,
        water,
        "Composite proxy hotspots (top 5% on water-only) over NGB",
    )


if __name__ == "__main__":
    main()
