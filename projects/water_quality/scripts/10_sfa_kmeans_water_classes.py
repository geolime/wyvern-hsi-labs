"""
10_sfa_kmeans_water_classes.py

KMeans clustering on physically interpretable SFA features (water-only):
  - NDTI (660/570): turbidity proxy
  - NDCI (711/669): chlorophyll/red-edge proxy
  - NIR/Red (764/669): red-edge strength proxy

Outputs:
  outputs/water_features/sfa_kmeans/
    - sfa_kmeans_K5.tif
    - sfa_kmeans_K5.png                 (class-only, transparent background)
    - sfa_kmeans_K5_legend.txt
    - sfa_kmeans_K5_feature_stats.csv
    - rgb_with_sfa_kmeans_K5.png         (overlay on RGB)
    - ngb_with_sfa_kmeans_K5.png         (overlay on NGB)
    - sfa_kmeans_K5_scatter_ndti_ndci.png (quick diagnostic scatter)

Notes:
- Uses QA valid mask + your saved water mask
- Uses StandardScaler for features before KMeans
- Deterministic via RANDOM_SEED
- Reorders labels by median NDTI (0 = clearest, 4 = most turbid)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib import colormaps

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, ACTIVE_WYVERN_MASK, OUTPUTS_DIR
from wyvernhsi.masks import load_valid_mask, load_water_mask


# ---------------- Settings ----------------
K = 5
TILE_SIZE = 1024
RANDOM_SEED = 42

# Bands (1-based)
B510 = 2
B549 = 5
B570 = 6
B660 = 12
B669 = 13
B711 = 17
B764 = 21

# RGB / NGB display composites
BANDS_RGB = {"R": 12, "G": 5, "B": 2}   # 660/549/510
BANDS_NGB = {"R": 21, "G": 5, "B": 2}   # 764/549/510

# Visualization
P_LO, P_HI = 2.0, 98.0
ALPHA = 0.45
SHORELINE_LINEWIDTH = 0.6

EPS = 1e-6

OUT_DIR = OUTPUTS_DIR / "water_features" / "sfa_kmeans"
OUT_PREFIX = f"sfa_kmeans_K{K}"

# Ordered-by-turbidity meanings after relabeling (0=clearest)
LABELS = {
    0: "clear / deep water",
    1: "low turbidity (transitional)",
    2: "moderate turbidity",
    3: "turbid water",
    4: "high turbidity / sediment plume",
}


# ---------------- Helpers ----------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_band(ds: rasterio.DatasetReader, b1: int, win: Window | None = None) -> np.ndarray:
    x = ds.read(b1, window=win).astype(np.float32)
    if ds.nodata is not None:
        x[x == ds.nodata] = np.nan
    return x


def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float32)
    denom = a + b
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(denom) > EPS)
    out[ok] = (a[ok] - b[ok]) / denom[ok]
    return out


def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float32)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > EPS)
    out[ok] = a[ok] / b[ok]
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


def save_overlay(bg: np.ndarray, lab: np.ndarray, water_mask: np.ndarray, out_png: Path, title: str) -> None:
    VIRIDIS = colormaps["viridis"]
    cmap = VIRIDIS.copy()
    cmap.set_bad(alpha=0.0)

    lab_plot = np.ma.masked_where(lab < 0, lab)

    def viridis_color(k: int, K: int):
        return VIRIDIS(k / (K - 1))[:3]

    plt.figure(figsize=(12, 10))
    plt.imshow(bg)
    plt.imshow(lab_plot, cmap=cmap, vmin=0, vmax=K - 1, alpha=ALPHA)

    # shoreline outline only
    plt.contour(water_mask.astype(np.uint8), levels=[0.5], linewidths=SHORELINE_LINEWIDTH)

    plt.axis("off")
    plt.title(title)

    handles, labels = [], []
    for k in range(K):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=viridis_color(k, K)))
        labels.append(f"{k}: {LABELS[k]}")
    plt.legend(handles, labels, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Wrote:", out_png)


def save_class_only(lab: np.ndarray, water_mask: np.ndarray, out_png: Path, title: str) -> None:
    VIRIDIS = colormaps["viridis"]
    cmap = VIRIDIS.copy()
    cmap.set_bad(alpha=0.0)

    lab_plot = np.ma.masked_where(lab < 0, lab)

    def viridis_color(k: int, K: int):
        return VIRIDIS(k / (K - 1))[:3]

    plt.figure(figsize=(12, 10))
    plt.imshow(lab_plot, cmap=cmap, vmin=0, vmax=K - 1, alpha=1.0)

    # shoreline outline only
    plt.contour(water_mask.astype(np.uint8), levels=[0.5], linewidths=SHORELINE_LINEWIDTH)

    plt.axis("off")
    plt.title(title)

    handles, labels = [], []
    for k in range(K):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=viridis_color(k, K)))
        labels.append(f"{k}: {LABELS[k]}")
    plt.legend(handles, labels, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, transparent=True)
    plt.close()
    print("Wrote:", out_png)


# ---------------- Main ----------------
def main() -> None:
    ensure_dir(OUT_DIR)

    if not ACTIVE_WYVERN_FILE.exists():
        raise FileNotFoundError(f"Missing image: {ACTIVE_WYVERN_FILE}")
    if not ACTIVE_WYVERN_MASK.exists():
        raise FileNotFoundError(f"Missing QA mask: {ACTIVE_WYVERN_MASK}")

    valid_full = load_valid_mask(ACTIVE_WYVERN_MASK).astype(bool)
    water_full = load_water_mask().astype(bool)
    use = valid_full & water_full

    print("Masks:")
    print("  valid:", int(valid_full.sum()))
    print("  water:", int(water_full.sum()))
    print("  use (valid & water):", int(use.sum()))

    out_tif = OUT_DIR / f"{OUT_PREFIX}.tif"

    # ---- Compute feature arrays (full scene, then mask) ----
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        H, W = ds.height, ds.width

        r570 = read_band(ds, B570)
        r660 = read_band(ds, B660)
        r669 = read_band(ds, B669)
        r711 = read_band(ds, B711)
        r764 = read_band(ds, B764)

        ndti = nd(r660, r570)           # turbidity proxy
        ndci = nd(r711, r669)           # chl / red-edge proxy
        nir_red = ratio(r764, r669)     # red-edge strength proxy

    # Apply masks
    ndti[~use] = np.nan
    ndci[~use] = np.nan
    nir_red[~use] = np.nan

    # Build X (N,3)
    ok = np.isfinite(ndti) & np.isfinite(ndci) & np.isfinite(nir_red)
    X = np.stack([ndti[ok], ndci[ok], nir_red[ok]], axis=1).astype(np.float32)

    if X.shape[0] == 0:
        raise RuntimeError("No valid water pixels for SFA KMeans.")

    # ---- Fit scaler + KMeans ----
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=K, n_init="auto", random_state=RANDOM_SEED)
    y = km.fit_predict(Xs).astype(np.int16)

    # ---- Create label raster (full scene) ----
    lab = np.full((ndti.shape[0], ndti.shape[1]), -1, dtype=np.int16)
    lab[ok] = y

    # ---- Per-cluster stats (pre-relabel) ----
    rows = []
    for k in range(K):
        mk = (y == k)
        if not np.any(mk):
            continue
        rows.append({
            "cluster": int(k),
            "n": int(np.sum(mk)),
            "ndti_median": float(np.median(X[mk, 0])),
            "ndci_median": float(np.median(X[mk, 1])),
            "nir_red_median": float(np.median(X[mk, 2])),
            "ndti_mean": float(np.mean(X[mk, 0])),
            "ndci_mean": float(np.mean(X[mk, 1])),
            "nir_red_mean": float(np.mean(X[mk, 2])),
            "ndti_std": float(np.std(X[mk, 0])),
            "ndci_std": float(np.std(X[mk, 1])),
            "nir_red_std": float(np.std(X[mk, 2])),
        })
    df = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)

    # ---- Relabel by median NDTI: 0=clearest, 4=most turbid ----
    order = df.sort_values("ndti_median")["cluster"].values  # low -> high
    label_map = {int(old): int(new) for new, old in enumerate(order)}

    print("Relabel map (old -> new):", label_map)

    lab2 = np.full_like(lab, -1, dtype=np.int16)
    for old, new in label_map.items():
        lab2[lab == old] = new
    lab = lab2

    # Remap stats table to ordered labels
    df["cluster_ordered"] = df["cluster"].map(label_map)
    df_out = df.sort_values("cluster_ordered").reset_index(drop=True)

    # ---- Write GeoTIFF ----
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        profile = ds.profile.copy()
        profile.update(
            count=1,
            dtype="int16",
            nodata=-1,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=512,
            blockysize=512,
        )

    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(lab, 1)

    with rasterio.open(out_tif, "r+") as dst:
        dst.update_tags(
            method="kmeans_on_sfa_features_water_only",
            features="NDTI(660,570),NDCI(711,669),NIR/Red(764,669)",
            kmeans_K=str(K),
            random_seed=str(RANDOM_SEED),
            relabel="ordered_by_ndti_median (0=clearest)",
        )

    print("Wrote:", out_tif)

    # ---- Write legend text ----
    legend_path = OUT_DIR / f"{OUT_PREFIX}_legend.txt"
    with open(legend_path, "w", encoding="utf-8") as f:
        f.write("SFA KMeans water classes (water-only)\n")
        f.write("Relabeled by median NDTI: 0=clearest, higher=more turbid\n\n")
        for k in range(K):
            f.write(f"{k}: {LABELS[k]}\n")
        f.write("\nCluster feature medians (ordered):\n")
        for _, row in df_out.iterrows():
            f.write(
                f"cluster {int(row['cluster_ordered'])}: "
                f"n={int(row['n'])}, "
                f"ndti_med={row['ndti_median']:.3f}, "
                f"ndci_med={row['ndci_median']:.3f}, "
                f"nir_red_med={row['nir_red_median']:.3f}\n"
            )
    print("Wrote:", legend_path)

    # ---- Chlorophyll ranking of turbidity-ordered clusters ----

    chl_rank_path = OUT_DIR / f"{OUT_PREFIX}_chlorophyll_ranking.txt"

    tmp = df_out[
        ["cluster_ordered", "n", "ndci_median", "ndti_median", "nir_red_median"]
    ].copy()

    tmp = tmp.sort_values("ndci_median", ascending=False).reset_index(drop=True)

    with open(chl_rank_path, "w", encoding="utf-8") as f:
        f.write("Clusters ranked by NDCI median (higher = more chlorophyll-like signal)\n\n")
        for _, r in tmp.iterrows():
            f.write(
                f"cluster {int(r['cluster_ordered'])}: "
                f"ndci_med={r['ndci_median']:.3f}, "
                f"ndti_med={r['ndti_median']:.3f}, "
                f"nir_red_med={r['nir_red_median']:.3f}, "
                f"n={int(r['n'])}\n"
            )

    print("Wrote:", chl_rank_path)


    # ---- Write stats CSV ----
    out_csv = OUT_DIR / f"{OUT_PREFIX}_feature_stats.csv"
    df_out.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

    # ---- Diagnostic scatter (sample) ----
    rng = np.random.default_rng(RANDOM_SEED)
    take = min(80_000, X.shape[0])
    sel = rng.choice(X.shape[0], size=take, replace=False)
    xs = X[sel, 0]
    zs = X[sel, 1]
    ys = y[sel]

    plt.figure(figsize=(7, 6))
    plt.scatter(xs, zs, s=1, c=ys, alpha=0.35)
    plt.xlabel("NDTI (660/570)")
    plt.ylabel("NDCI (711/669)")
    plt.title("SFA feature space (sample, colored by raw KMeans)")
    plt.tight_layout()
    out_scatter = OUT_DIR / f"{OUT_PREFIX}_scatter_ndti_ndci.png"
    plt.savefig(out_scatter, dpi=200)
    plt.close()
    print("Wrote:", out_scatter)

    # ---- Export maps (class-only + overlays) ----
    # Background composites for context
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        rgb = build_composite(ds, BANDS_RGB)
        ngb = build_composite(ds, BANDS_NGB)

    save_class_only(
        lab,
        water_full,
        OUT_DIR / f"{OUT_PREFIX}.png",
        f"SFA KMeans classes (K={K}, water-only)",
    )

    save_overlay(
        rgb,
        lab,
        water_full,
        OUT_DIR / f"rgb_with_{OUT_PREFIX}.png",
        f"SFA KMeans over RGB (alpha={ALPHA})",
    )

    save_overlay(
        ngb,
        lab,
        water_full,
        OUT_DIR / f"ngb_with_{OUT_PREFIX}.png",
        f"SFA KMeans over NGB (alpha={ALPHA})",
    )

# ---- Extra chlorophyll (NDCI) visuals ----
    # Save NDCI water-only map + NGB overlay heatmap

    def save_ndci_png(out_png: Path, arr2d: np.ndarray, title: str) -> None:
        v = arr2d[np.isfinite(arr2d)]
        if v.size == 0:
            return
        vmin = float(np.percentile(v, 2))
        vmax = float(np.percentile(v, 98))

        plt.figure(figsize=(10, 8))
        im = plt.imshow(arr2d, vmin=vmin, vmax=vmax)
        plt.axis("off")
        plt.title(title)

        fig = plt.gcf()
        fig.subplots_adjust(right=0.86)
        cax = fig.add_axes([0.88, 0.12, 0.03, 0.76])
        plt.colorbar(im, cax=cax)

        plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
        plt.close()
        print("Wrote:", out_png)


    def save_ngb_with_ndci(out_png: Path, ngb_bg: np.ndarray, ndci_map: np.ndarray, water_mask: np.ndarray) -> None:
        # Mask non-water to transparent
        ndci_plot = ndci_map.copy()
        ndci_plot[~water_mask] = np.nan
        ndci_plot = np.ma.masked_where(~np.isfinite(ndci_plot), ndci_plot)

        v = ndci_map[np.isfinite(ndci_map) & water_mask]
        if v.size == 0:
            return
        vmin = float(np.percentile(v, 2))
        vmax = float(np.percentile(v, 98))

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(alpha=0.0)

        plt.figure(figsize=(12, 10))
        plt.imshow(ngb_bg)
        plt.imshow(ndci_plot, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.55)
        plt.contour(water_mask.astype(np.uint8), levels=[0.5], linewidths=SHORELINE_LINEWIDTH)
        plt.axis("off")
        plt.title("NDCI (711/669) over NGB (water-only)")

        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print("Wrote:", out_png)


    # 1) NDCI map (water-only)
    ndci_water = ndci.copy()
    ndci_water[~use] = np.nan
    save_ndci_png(
        OUT_DIR / "ndci_711_669_water_only.png",
        ndci_water,
        "NDCI (711/669) chlorophyll proxy (water-only)",
    )

    # 2) NGB + NDCI overlay
    save_ngb_with_ndci(
        OUT_DIR / "ngb_with_ndci_711_669.png",
        ngb,
        ndci,
        use,
    )



if __name__ == "__main__":
    main()
