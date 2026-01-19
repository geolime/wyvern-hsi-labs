"""
Spectral Feature Analysis (SFA) feature maps for Wyvern Dragonette hyperspectral TOA radiance.

Exports TWO versions for each feature:
  1) whole_scene: masked by QA valid mask only
  2) water_only:  masked by QA valid mask AND saved water mask

Also exports:
  - per-cluster feature stats CSV (water-only, using water_features_kmeans_K5.tif if present)
  - feature correlation heatmap (water-only)

Output folders:
  outputs/water_features/sfa/whole_scene/
  outputs/water_features/sfa/water_only/
  outputs/water_features/sfa/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, ACTIVE_WYVERN_MASK, OUTPUTS_DIR
from wyvernhsi.masks import load_valid_mask, load_water_mask


# ---------------- Settings ----------------
TILE_SIZE = 1024

# Wyvern Dragonette band indices (1-based) from your list:
#  2: 510
#  6: 570
#  8: 600
#  9: 614
# 10: 635
# 11: 649
# 12: 660
# 13: 669
# 14: 679
# 16: 699
# 17: 711
# 18: 722
# 21: 764
B510 = 2
B570 = 6
B600 = 8
B614 = 9
B635 = 10
B649 = 11
B660 = 12
B669 = 13
B679 = 14
B699 = 16
B711 = 17
B722 = 18
B764 = 21

EPS = 1e-6

# If you already have a water-features KMeans output, we can summarize features per cluster.
KMEANS_LABELS_TIF = OUTPUTS_DIR / "water_features" / "water_features_kmeans_K5.tif"


# ---------------- Helpers ----------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_band(ds: rasterio.DatasetReader, b1: int, win: Window) -> np.ndarray:
    x = ds.read(b1, window=win).astype(np.float32)
    if ds.nodata is not None:
        x[x == ds.nodata] = np.nan
    return x


def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a-b)/(a+b) with NaN handling."""
    out = np.full_like(a, np.nan, dtype=np.float32)
    denom = a + b
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(denom) > EPS)
    out[ok] = (a[ok] - b[ok]) / denom[ok]
    return out


def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a/b with NaN handling."""
    out = np.full_like(a, np.nan, dtype=np.float32)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > EPS)
    out[ok] = a[ok] / b[ok]
    return out


def robust_limits(x: np.ndarray, lo: float = 2, hi: float = 98):
    v = x[np.isfinite(x)]
    if v.size == 0:
        return None, None
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def save_map_png(path: Path, arr2d: np.ndarray, title: str) -> None:
    vmin, vmax = robust_limits(arr2d)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(arr2d, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.title(title)

    # colorbar that never gets clipped
    fig = plt.gcf()
    fig.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.88, 0.12, 0.03, 0.76])
    plt.colorbar(im, cax=cax)

    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def band_depth(center: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """
    Simple band depth proxy:
      BD = 1 - center / mean(left, right)
    """
    cont = 0.5 * (left + right)
    out = np.full_like(center, np.nan, dtype=np.float32)
    ok = np.isfinite(center) & np.isfinite(cont) & (np.abs(cont) > EPS)
    out[ok] = 1.0 - (center[ok] / cont[ok])
    return out


def compute_features_from_tile(
    r510: np.ndarray,
    r570: np.ndarray,
    r600: np.ndarray,
    r614: np.ndarray,
    r635: np.ndarray,
    r649: np.ndarray,
    r660: np.ndarray,
    r669: np.ndarray,
    r679: np.ndarray,
    r699: np.ndarray,
    r711: np.ndarray,
    r722: np.ndarray,
    r764: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Core SFA features (all 2D arrays).
    """
    feat: dict[str, np.ndarray] = {}

    # turbidity-ish
    feat["ndti_660_570"] = nd(r660, r570)          # (R-G)/(R+G)
    feat["rg_660_570"] = ratio(r660, r570)         # R/G
    feat["rbr_660_510"] = ratio(r660, r510)        # Red/Blue ratio

    # chlorophyll-ish / red-edge
    feat["ndci_711_669"] = nd(r711, r669)          # (RE-R)/(RE+R)
    feat["ndci_722_669"] = nd(r722, r669)          # alt NDCI
    feat["nir_red_764_669"] = ratio(r764, r669)    # NIR/Red ratio
    feat["re_slope_nd_711_669"] = nd(r711, r669)   # same as ndci, kept for naming clarity

    # red-edge curvature proxy around ~699
    # curvature = (722-699) - (699-679) = 722 - 2*699 + 679
    feat["re_curvature_722_699_679"] = (r722 - 2.0 * r699 + r679).astype(np.float32)

    # visible absorption-ish band depth proxy around 635 (using 600 and 649 as shoulders)
    feat["bd_635_600_649"] = band_depth(r635, r600, r649)

    # another visible dip proxy around 614 (using 600 and 635)
    feat["bd_614_600_635"] = band_depth(r614, r600, r635)

    return feat


def plot_corr_heatmap(path: Path, df: pd.DataFrame, title: str) -> None:
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(9, 8))
    im = plt.imshow(corr.values, interpolation="nearest")
    plt.xticks(range(corr.shape[1]), corr.columns, rotation=45, ha="right")
    plt.yticks(range(corr.shape[0]), corr.index)
    plt.title(title)

    fig = plt.gcf()
    fig.subplots_adjust(right=0.86)
    cax = fig.add_axes([0.88, 0.12, 0.03, 0.76])
    plt.colorbar(im, cax=cax)

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()


# ---------------- Main ----------------
def main() -> None:
    ensure_dir(OUTPUTS_DIR)

    out_root = OUTPUTS_DIR / "water_features" / "sfa"
    out_whole = out_root / "whole_scene"
    out_water = out_root / "water_only"
    ensure_dir(out_root)
    ensure_dir(out_whole)
    ensure_dir(out_water)

    if not ACTIVE_WYVERN_FILE.exists():
        raise FileNotFoundError(f"Missing image: {ACTIVE_WYVERN_FILE}")
    if not ACTIVE_WYVERN_MASK.exists():
        raise FileNotFoundError(f"Missing mask: {ACTIVE_WYVERN_MASK}")

    valid_full = load_valid_mask(ACTIVE_WYVERN_MASK).astype(bool)
    water_full = load_water_mask().astype(bool)
    use_water = valid_full & water_full
    use_whole = valid_full

    print("Masks:")
    print("  valid pixels:", int(valid_full.sum()))
    print("  water pixels:", int(water_full.sum()))
    print("  use_water (valid & water):", int(use_water.sum()))

    # Prepare in-memory arrays for outputs (float32 with NaNs)
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        H, W = ds.height, ds.width

        feature_names = [
            "ndti_660_570",
            "rg_660_570",
            "rbr_660_510",
            "ndci_711_669",
            "ndci_722_669",
            "nir_red_764_669",
            "re_curvature_722_699_679",
            "bd_635_600_649",
            "bd_614_600_635",
        ]

        whole_maps = {nm: np.full((H, W), np.nan, dtype=np.float32) for nm in feature_names}
        water_maps = {nm: np.full((H, W), np.nan, dtype=np.float32) for nm in feature_names}

        for r0 in range(0, H, TILE_SIZE):
            for c0 in range(0, W, TILE_SIZE):
                r1 = min(H, r0 + TILE_SIZE)
                c1 = min(W, c0 + TILE_SIZE)
                win = Window.from_slices((r0, r1), (c0, c1))

                # read needed bands
                r510 = read_band(ds, B510, win)
                r570 = read_band(ds, B570, win)
                r600 = read_band(ds, B600, win)
                r614 = read_band(ds, B614, win)
                r635 = read_band(ds, B635, win)
                r649 = read_band(ds, B649, win)
                r660 = read_band(ds, B660, win)
                r669 = read_band(ds, B669, win)
                r679 = read_band(ds, B679, win)
                r699 = read_band(ds, B699, win)
                r711 = read_band(ds, B711, win)
                r722 = read_band(ds, B722, win)
                r764 = read_band(ds, B764, win)

                feats = compute_features_from_tile(
                    r510, r570, r600, r614, r635, r649, r660, r669, r679, r699, r711, r722, r764
                )

                # apply masks for this tile
                vm = use_whole[r0:r1, c0:c1]
                wm = use_water[r0:r1, c0:c1]

                for nm in feature_names:
                    a = feats[nm]

                    # whole scene (valid only)
                    aa = a.copy()
                    aa[~vm] = np.nan
                    whole_maps[nm][r0:r1, c0:c1] = aa

                    # water-only (valid & water)
                    bb = a.copy()
                    bb[~wm] = np.nan
                    water_maps[nm][r0:r1, c0:c1] = bb

    # ---- Export maps ----
    print("Exporting feature maps...")
    for nm in feature_names:
        save_map_png(out_whole / f"{nm}.png", whole_maps[nm], f"{nm} (whole scene, valid only)")
        save_map_png(out_water / f"{nm}.png", water_maps[nm], f"{nm} (water only)")

    print("Wrote feature PNGs to:")
    print(" -", out_whole)
    print(" -", out_water)

    # ---- Water-only correlation + sampling table ----
    # Sample water pixels for correlation stats (keeps memory sane)
    rng = np.random.default_rng(42)
    water_idx = np.flatnonzero(np.isfinite(water_maps["ndti_660_570"]).ravel())
    take = min(120_000, water_idx.size)
    if take > 5_000:
        sel = rng.choice(water_idx, size=take, replace=False)
        df_s = pd.DataFrame({nm: water_maps[nm].ravel()[sel] for nm in feature_names})
        df_s = df_s.replace([np.inf, -np.inf], np.nan).dropna()

        corr_png = out_root / "sfa_feature_correlation_water_only.png"
        plot_corr_heatmap(corr_png, df_s, "SFA feature correlation (water-only sample)")
        print("Wrote:", corr_png)

        corr_csv = out_root / "sfa_feature_correlation_water_only.csv"
        df_s.corr(numeric_only=True).to_csv(corr_csv)
        print("Wrote:", corr_csv)
    else:
        print("Skipping correlation export (not enough water pixels).")

    # ---- Per-cluster feature stats (water-only) if labels exist ----
    if KMEANS_LABELS_TIF.exists():
        with rasterio.open(KMEANS_LABELS_TIF) as ds_lab:
            lab = ds_lab.read(1).astype(np.int16)

        m = (lab >= 0) & use_water
        if np.any(m):
            rows = []
            for k in sorted(np.unique(lab[m]).tolist()):
                mk = m & (lab == k)
                n = int(mk.sum())
                row = {"cluster": int(k), "n": n}
                for nm in feature_names:
                    v = water_maps[nm][mk]
                    row[f"{nm}_mean"] = float(np.nanmean(v))
                    row[f"{nm}_median"] = float(np.nanmedian(v))
                    row[f"{nm}_std"] = float(np.nanstd(v))
                rows.append(row)

            dfc = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
            out_stats = out_root / "sfa_feature_stats_by_cluster_water_only.csv"
            dfc.to_csv(out_stats, index=False)
            print("Wrote:", out_stats)
        else:
            print("Labels exist but no overlap with water mask; skipping cluster stats.")
    else:
        print("No KMeans labels found at:", KMEANS_LABELS_TIF)
        print("Skipping per-cluster feature stats.")


if __name__ == "__main__":
    main()
