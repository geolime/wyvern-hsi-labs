from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, ACTIVE_WYVERN_MASK, OUTPUTS_DIR
from wyvernhsi.masks import load_valid_mask, load_water_mask


# ---------------- Settings ----------------
K = 5
TILE_SIZE = 512
RANDOM_SEED = 42

# Feature band mapping for your Wyvern Dragonette band list:
# Green ~ 560: use 570 (band 6)
# Red ~ 660: use 660 (band 12)
# Red ~ 665 for NDCI: use 669 (band 13)
# Red-edge ~ 708: use 711 (band 17)
B_G = 6     # 570
B_R = 12    # 660
B_R665 = 13 # 669
B_RE = 17   # 711

EPS = 1e-6

OUT_PREFIX = f"water_features_kmeans_K{K}"


# ---------------- Helpers ----------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_band(ds: rasterio.DatasetReader, b1: int, win: Window | None) -> np.ndarray:
    x = ds.read(b1, window=win).astype(np.float32)
    if ds.nodata is not None:
        x[x == ds.nodata] = np.nan
    return x


def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalized difference (a-b)/(a+b) with NaN handling."""
    denom = a + b
    out = np.full_like(a, np.nan, dtype=np.float32)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(denom) > EPS)
    out[ok] = (a[ok] - b[ok]) / denom[ok]
    return out


def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Ratio a/b with NaN handling."""
    out = np.full_like(a, np.nan, dtype=np.float32)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > EPS)
    out[ok] = a[ok] / b[ok]
    return out


def robust_limits(x: np.ndarray, lo: float = 2, hi: float = 98):
    v = x[np.isfinite(x)]
    if v.size == 0:
        return None, None
    return np.percentile(v, lo), np.percentile(v, hi)


def build_feature_stack(ds: rasterio.DatasetReader, win: Window | None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Returns:
      F: (H,W,4) feature cube
      maps: dict of raw feature maps (H,W) for export
    Features:
      0 NDTI: (R660 - G570)/(R660 + G570)   turbidity proxy
      1 NDCI: (RE711 - R669)/(RE711 + R669) chlorophyll proxy
      2 RG  : R660 / G570                  turbidity-ish
      3 RE_R: RE711 / R669                 chlorophyll-ish
    """
    g = read_band(ds, B_G, win)
    r = read_band(ds, B_R, win)
    r665 = read_band(ds, B_R665, win)
    re = read_band(ds, B_RE, win)

    ndti = nd(r, g)
    ndci = nd(re, r665)
    rg = ratio(r, g)
    re_r = ratio(re, r665)

    F = np.stack([ndti, ndci, rg, re_r], axis=2).astype(np.float32)
    maps = {"ndti": ndti, "ndci": ndci, "rg": rg, "re_r": re_r}
    return F, maps


def flatten_features(F: np.ndarray, use_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    F: (H,W,4)
    use_mask: (H,W) bool
    Returns:
      X: (N,4)
      valid2d: (H,W) bool where used
    """
    valid2d = use_mask & np.isfinite(F).all(axis=2)
    X = F[valid2d]
    return X, valid2d


def save_png(path: Path, arr: np.ndarray, title: str | None = None, vmin=None, vmax=None) -> None:
    plt.figure(figsize=(10, 8))
    im = plt.imshow(arr, vmin=vmin, vmax=vmax)
    plt.axis("off")

    if title:
        plt.title(title)

    # Put colorbar in dedicated axis so it never gets clipped
    if arr.ndim == 2:
        fig = plt.gcf()
        fig.subplots_adjust(right=0.86)
        cax = fig.add_axes([0.88, 0.12, 0.03, 0.76])
        plt.colorbar(im, cax=cax)

    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()


# ---------------- Main ----------------
def main() -> None:
    ensure_dir(OUTPUTS_DIR)
    out_dir = OUTPUTS_DIR / "water_features"
    ensure_dir(out_dir)

    valid_full = load_valid_mask(ACTIVE_WYVERN_MASK)   # True = OK pixels
    water_full = load_water_mask()                     # True = water pixels
    use_full = valid_full & water_full                 # True = pixels used in clustering

    out_tif = out_dir / f"{OUT_PREFIX}.tif"

    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        H, W = ds.height, ds.width

        # ---- Build full-scene feature stack (in memory) ----
        F, maps = build_feature_stack(ds, None)
        F[~use_full] = np.nan

        # ---- Fit KMeans on all valid water pixels ----
        X, _ = flatten_features(F, use_full)
        if X.shape[0] == 0:
            raise RuntimeError("No valid water pixels after masks.")

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        km = KMeans(n_clusters=K, n_init="auto", random_state=RANDOM_SEED)
        km.fit(Xs)

        # ---------------- Stability check (feature clustering) ----------------

        STABILITY_RUNS = 5

        labels_list = []

        for i in range(STABILITY_RUNS):
            km_tmp = KMeans(
                n_clusters=K,
                n_init="auto",
                random_state=RANDOM_SEED + i * 17,
            )
            lab_tmp = km_tmp.fit_predict(Xs)
            labels_list.append(lab_tmp)

        ari = np.zeros((STABILITY_RUNS, STABILITY_RUNS), dtype=np.float32)

        for i in range(STABILITY_RUNS):
            for j in range(STABILITY_RUNS):
                ari[i, j] = adjusted_rand_score(labels_list[i], labels_list[j])

        mean_ari = (ari.sum() - np.trace(ari)) / (ari.size - STABILITY_RUNS)

        print("Water feature clustering stability:")
        print("Mean ARI:", float(mean_ari))


        # ---- Predict full scene in tiles ----
        profile = ds.profile.copy()
        profile.update(
            count=1,
            dtype="int16",
            nodata=-1,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=TILE_SIZE,
            blockysize=TILE_SIZE,
        )

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(np.full((H, W), -1, dtype=np.int16), 1)

            for r0 in range(0, H, TILE_SIZE):
                for c0 in range(0, W, TILE_SIZE):
                    r1 = min(H, r0 + TILE_SIZE)
                    c1 = min(W, c0 + TILE_SIZE)
                    win = Window.from_slices((r0, r1), (c0, c1))

                    tile_use = use_full[r0:r1, c0:c1]
                    if not np.any(tile_use):
                        continue

                    Ft, _ = build_feature_stack(ds, win)
                    Ft[~tile_use] = np.nan

                    Xt, valid2d = flatten_features(Ft, tile_use)
                    out = np.full((r1 - r0, c1 - c0), -1, dtype=np.int16)
                    if Xt.shape[0] > 0:
                        Xt_s = scaler.transform(Xt)
                        out[valid2d] = km.predict(Xt_s).astype(np.int16)

                    dst.write(out, 1, window=win)

        with rasterio.open(out_tif, "r+") as dst:
            dst.update_tags(
                method="kmeans_on_water_features",
                features="NDTI,NDCI,R660/G570,RE711/R669",
                bands=f"G570={B_G},R660={B_R},R669={B_R665},RE711={B_RE}",
                kmeans_K=str(K),
                random_seed=str(RANDOM_SEED),
            )

        print("Wrote:", out_tif)

        # ---- Export feature maps for sanity checking (masked) ----
        ndti = maps["ndti"].copy()
        ndci = maps["ndci"].copy()
        ndti[~use_full] = np.nan
        ndci[~use_full] = np.nan

        m = np.isfinite(ndti) & np.isfinite(ndci) & use_full
        corr = np.corrcoef(ndti[m], ndci[m])[0, 1]
        print("Corr(NDTI, NDCI) on water pixels:", float(corr))

        v0, v1 = robust_limits(ndti)
        save_png(out_dir / "ndti_map.png", ndti, "NDTI (R660,G570) turbidity proxy", vmin=v0, vmax=v1)

        v0, v1 = robust_limits(ndci)
        save_png(out_dir / "ndci_map.png", ndci, "NDCI (RE711,R669) chlorophyll proxy", vmin=v0, vmax=v1)

        print("Wrote:", out_dir / "ndti_map.png")
        print("Wrote:", out_dir / "ndci_map.png")

        # ---- Load labels raster ----
        with rasterio.open(out_tif) as ds_lab:
            lab = ds_lab.read(1).astype(np.int16)

        # ---- Per-cluster feature summary (original labels) ----
        used = use_full & (lab >= 0) & np.isfinite(F).all(axis=2)
        Xall = F[used]
        yall = lab[used]

        rows = []
        names = ["ndti", "ndci", "rg", "re_r"]
        for k in range(K):
            mk = (yall == k)
            if not np.any(mk):
                continue
            vals = Xall[mk]
            row = {"cluster": k, "n": int(np.sum(mk))}
            for j, nm in enumerate(names):
                row[f"{nm}_mean"] = float(np.nanmean(vals[:, j]))
                row[f"{nm}_median"] = float(np.nanmedian(vals[:, j]))
                row[f"{nm}_std"] = float(np.nanstd(vals[:, j]))
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
        out_csv = out_dir / f"{OUT_PREFIX}_cluster_feature_stats.csv"
        df.to_csv(out_csv, index=False)
        print("Wrote:", out_csv)

        # =========================================================
        # Reorder clusters by turbidity (median NDTI)
        # New label 0 = clearest water
        # =========================================================

        order = df.sort_values("ndti_median")["cluster"].values
        label_map = {int(old): int(new) for new, old in enumerate(order)}

        print("Cluster relabel mapping (old -> new):")
        print(label_map)

        lab_reordered = np.full_like(lab, -1, dtype=np.int16)
        for old, new in label_map.items():
            lab_reordered[lab == old] = new
        lab = lab_reordered

        # ---- Write reordered labels back to GeoTIFF ----
        with rasterio.open(out_tif, "r+") as ds_lab:
            ds_lab.write(lab, 1)

        # ---- Legend text file (ordered labels) ----
        legend_path = out_dir / f"{OUT_PREFIX}_legend.txt"
        df2 = df.copy()
        df2["cluster_ordered"] = df2["cluster"].map(label_map)
        df2 = df2.sort_values("cluster_ordered").reset_index(drop=True)

        with open(legend_path, "w", encoding="utf-8") as f:
            f.write("Cluster legend (ordered by turbidity, based on median NDTI)\n")
            f.write("0 = clearest water\n")
            f.write("Higher = more turbid / optically active\n\n")
            for _, row in df2.iterrows():
                f.write(
                    f"Cluster {int(row['cluster_ordered'])}: "
                    f"NDTI_med={row['ndti_median']:.3f}, "
                    f"NDCI_med={row['ndci_median']:.3f}, "
                    f"n={int(row['n'])}\n"
                )

        print("Wrote:", legend_path)

        # ---- Export reordered cluster PNG (legend box with meanings) ----
        from matplotlib import colormaps

        VIRIDIS = colormaps["viridis"]

        def viridis_color(k: int, K: int):
            return VIRIDIS(k / (K - 1))[:3]

        LABELS = {
            0: "clear / deep water",
            1: "low turbidity (transitional)",
            2: "moderate turbidity",
            3: "algae-enhanced / optically active",
            4: "high turbidity / sediment plume",
        }

        out_png = out_dir / f"{OUT_PREFIX}.png"

        # ---------------- Mask nodata as transparent ----------------
        lab_plot = np.ma.masked_where(lab < 0, lab)

        cmap = VIRIDIS.copy()
        cmap.set_bad(alpha=0.0)   # fully transparent nodata

        # ---------------- Plot ----------------
        plt.figure(figsize=(12, 10))
        plt.imshow(lab_plot, vmin=0, vmax=K - 1, cmap=cmap)
        plt.axis("off")
        plt.title(f"KMeans water classes (ordered by turbidity) — K={K}")

        # ---------------- Legend ----------------
        handles = []
        labels = []

        for k in range(K):
            handles.append(plt.Rectangle((0, 0), 1, 1, color=viridis_color(k, K)))
            labels.append(f"{k}: {LABELS[k]}")

        plt.legend(handles, labels, loc="lower right", framealpha=0.9)

        plt.tight_layout()
        plt.savefig(out_png, dpi=200, transparent=True)
        plt.close()

        print("Wrote:", out_png)



        # ---- Bar plots (original labels) ----
        plt.figure(figsize=(8, 4))
        plt.bar(df["cluster"].astype(int).astype(str), df["ndti_median"])
        plt.xlabel("Cluster (original)")
        plt.ylabel("Median NDTI")
        plt.title("Median turbidity proxy (NDTI) per cluster")
        plt.tight_layout()
        plt.savefig(out_dir / f"{OUT_PREFIX}_ndti_median.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.bar(df["cluster"].astype(int).astype(str), df["ndci_median"])
        plt.xlabel("Cluster (original)")
        plt.ylabel("Median NDCI")
        plt.title("Median chlorophyll proxy (NDCI) per cluster")
        plt.tight_layout()
        plt.savefig(out_dir / f"{OUT_PREFIX}_ndci_median.png", dpi=200)
        plt.close()

        print("Wrote:", out_dir / f"{OUT_PREFIX}_ndti_median.png")
        print("Wrote:", out_dir / f"{OUT_PREFIX}_ndci_median.png")


if __name__ == "__main__":
    main()
