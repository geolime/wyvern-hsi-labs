from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score

from scipy.ndimage import label

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, ACTIVE_WYVERN_MASK, OUTPUTS_DIR
from wyvernhsi.masks import load_valid_mask, load_water_mask
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions


# ---------------- Settings ----------------
K = 5
PCA_N = 8

SAMPLE_FIT = 200_000        # PCA+KMeans fit
SAMPLE_SIL = 80_000         # silhouette sample size
STABILITY_RUNS = 6          # ARI runs

TILE_SIZE = 512
RANDOM_SEED = 42

OUT_PREFIX = f"water_kmeans_K{K}_PCA{PCA_N}"


# ---------------- Helpers ----------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def read_cube(ds: rasterio.DatasetReader, win: Window | None) -> np.ndarray:
    """Return cube as float32 with shape (Y, X, B)."""
    arr = ds.read(window=win).astype(np.float32)  # (B,Y,X)
    if ds.nodata is not None:
        arr[arr == ds.nodata] = np.nan
    cube = np.transpose(arr, (1, 2, 0))          # (Y,X,B)
    return cube


def flatten_valid(cube_yxb: np.ndarray, use_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    cube_yxb: (H,W,B)
    use_mask: (H,W) bool where we want to keep pixels (valid & water)
    Returns:
      X: (N,B)
      valid2d: (H,W) bool for selected finite pixels
    """
    valid2d = use_mask & np.isfinite(cube_yxb).all(axis=2)
    X = cube_yxb[valid2d]
    return X, valid2d


def percentile_stretch01(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    a = np.nanpercentile(x, lo)
    b = np.nanpercentile(x, hi)
    y = (x - a) / (b - a + 1e-12)
    return np.clip(y, 0, 1)


def predict_tile(tile_yxb: np.ndarray, tile_use_mask: np.ndarray, pca: PCA, kmeans: KMeans) -> np.ndarray:
    """
    Predict cluster IDs for one tile.
    Returns int16 (H,W) with -1 for non-use pixels.
    """
    out = np.full(tile_yxb.shape[:2], -1, dtype=np.int16)

    X, valid2d = flatten_valid(tile_yxb, tile_use_mask)
    if X.shape[0] == 0:
        return out

    Xn = l2_normalize_rows(X)
    Z = pca.transform(Xn)
    lab = kmeans.predict(Z).astype(np.int16)
    out[valid2d] = lab
    return out


def compute_cluster_means(ds: rasterio.DatasetReader, labels_path: Path, wl_nm: np.ndarray, use_mask_full: np.ndarray) -> None:
    """
    Second pass over tiles: mean spectrum per cluster (on L2-normalized spectra).
    """
    sums = np.zeros((K, ds.count), dtype=np.float64)
    counts = np.zeros((K,), dtype=np.int64)

    with rasterio.open(labels_path) as lab_ds:
        H, W = ds.height, ds.width

        for r0 in range(0, H, TILE_SIZE):
            for c0 in range(0, W, TILE_SIZE):
                r1 = min(H, r0 + TILE_SIZE)
                c1 = min(W, c0 + TILE_SIZE)
                w = Window.from_slices((r0, r1), (c0, c1))

                cube = read_cube(ds, w)  # (y,x,b)
                labs = lab_ds.read(1, window=w).astype(np.int16)

                tile_use = use_mask_full[r0:r1, c0:c1]
                valid = tile_use & (labs >= 0) & np.isfinite(cube).all(axis=2)
                if not np.any(valid):
                    continue

                X = cube[valid]
                y = labs[valid]

                X = l2_normalize_rows(X)

                for k in range(K):
                    m = (y == k)
                    if np.any(m):
                        sums[k] += np.sum(X[m], axis=0)
                        counts[k] += int(np.sum(m))

    means = np.full((K, ds.count), np.nan, dtype=np.float32)
    for k in range(K):
        if counts[k] > 0:
            means[k] = (sums[k] / counts[k]).astype(np.float32)

    plt.figure(figsize=(12, 7))
    for k in range(K):
        if np.isfinite(means[k]).any():
            plt.plot(wl_nm, means[k], label=f"cluster {k} (n={counts[k]})")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("L2-normalized radiance")
    plt.title(f"Cluster mean spectra (water-only) — K={K}, PCA={PCA_N}")
    plt.legend()
    plt.tight_layout()
    out_png = OUTPUTS_DIR / f"{OUT_PREFIX}_cluster_mean_spectra.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Wrote:", out_png)


# ---------------- Main ----------------
def main() -> None:
    ensure_dir(OUTPUTS_DIR)

    if not ACTIVE_WYVERN_FILE.exists():
        raise FileNotFoundError(f"Missing local file: {ACTIVE_WYVERN_FILE}")
    if not ACTIVE_WYVERN_MASK.exists():
        raise FileNotFoundError(f"Missing local file: {ACTIVE_WYVERN_MASK}")

    # Masks (full scene)
    valid_mask = load_valid_mask(ACTIVE_WYVERN_MASK)   # True where OK
    water_mask = load_water_mask()                     # True where water
    use_mask_full = valid_mask & water_mask            # True where we will operate

    out_tif = OUTPUTS_DIR / f"{OUT_PREFIX}.tif"

    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        wl_nm = parse_wavelengths_nm_from_descriptions(list(ds.descriptions))

        # -------- Fit PCA + KMeans on sampled water pixels --------
        cube_full = read_cube(ds, None)  # (H,W,B) (ok for your sizes; if huge, we can tile-sample instead)
        cube_full[~use_mask_full] = np.nan

        X, _ = flatten_valid(cube_full, use_mask_full)
        if X.shape[0] == 0:
            raise RuntimeError("No valid water pixels after masks. Check water_mask + QA mask overlap.")

        rng = np.random.default_rng(RANDOM_SEED)
        take = min(SAMPLE_FIT, X.shape[0])
        sel = rng.choice(X.shape[0], size=take, replace=False)
        Xs = X[sel].astype(np.float32)
        Xs = l2_normalize_rows(Xs)

        pca = PCA(n_components=min(PCA_N, Xs.shape[1]), random_state=RANDOM_SEED)
        Zs = pca.fit_transform(Xs)

        km = KMeans(n_clusters=K, n_init="auto", random_state=RANDOM_SEED)
        y_ref = km.fit_predict(Zs)

        print(f"Fit: samples={take:,}, bands={Xs.shape[1]}, PCA={pca.n_components_}, K={K}")

        # -------- PCA diagnostics --------
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)

        plt.figure(figsize=(7, 4))
        plt.plot(np.arange(1, len(evr) + 1), evr, marker="o")
        plt.xlabel("PC")
        plt.ylabel("Explained variance ratio")
        plt.title("PCA explained variance (water-only)")
        plt.tight_layout()
        out_scree = OUTPUTS_DIR / f"{OUT_PREFIX}_pca_scree.png"
        plt.savefig(out_scree, dpi=200)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(np.arange(1, len(cum) + 1), cum, marker="o")
        plt.xlabel("PC")
        plt.ylabel("Cumulative explained variance")
        plt.title("PCA cumulative explained variance (water-only)")
        plt.tight_layout()
        out_cum = OUTPUTS_DIR / f"{OUT_PREFIX}_pca_cumulative.png"
        plt.savefig(out_cum, dpi=200)
        plt.close()

        # -------- Silhouette (sampled) --------
        sil_take = min(SAMPLE_SIL, Zs.shape[0])
        sil_sel = rng.choice(Zs.shape[0], size=sil_take, replace=False)
        sil = silhouette_score(Zs[sil_sel], y_ref[sil_sel], metric="euclidean")
        print("Silhouette (sampled):", float(sil))

        # -------- ARI stability (on same Zs sample) --------
        seeds = [RANDOM_SEED + i * 17 for i in range(STABILITY_RUNS)]
        ys = []
        for s in seeds:
            km_s = KMeans(n_clusters=K, n_init="auto", random_state=s)
            ys.append(km_s.fit_predict(Zs))

        ari = np.zeros((STABILITY_RUNS, STABILITY_RUNS), dtype=np.float32)
        for i in range(STABILITY_RUNS):
            for j in range(STABILITY_RUNS):
                ari[i, j] = adjusted_rand_score(ys[i], ys[j])

        plt.figure(figsize=(6, 5))
        plt.imshow(ari, interpolation="nearest")
        plt.xticks(range(STABILITY_RUNS), [str(s) for s in seeds], rotation=45, ha="right")
        plt.yticks(range(STABILITY_RUNS), [str(s) for s in seeds])
        plt.title("KMeans stability (ARI) — water-only")
        plt.colorbar()
        plt.tight_layout()
        out_ari_png = OUTPUTS_DIR / f"{OUT_PREFIX}_stability_ari.png"
        plt.savefig(out_ari_png, dpi=200)
        plt.close()

        # -------- Predict full scene in tiles (water-only) --------
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
            dst.write(np.full((ds.height, ds.width), -1, dtype=np.int16), 1)

            H, W = ds.height, ds.width
            for r0 in range(0, H, TILE_SIZE):
                for c0 in range(0, W, TILE_SIZE):
                    r1 = min(H, r0 + TILE_SIZE)
                    c1 = min(W, c0 + TILE_SIZE)
                    w = Window.from_slices((r0, r1), (c0, c1))

                    tile = read_cube(ds, w)
                    tile_use = use_mask_full[r0:r1, c0:c1]

                    labs = predict_tile(tile, tile_use, pca, km)
                    dst.write(labs, 1, window=w)

        with rasterio.open(out_tif, "r+") as dst:
            dst.update_tags(
                kmeans_K=str(K),
                pca_components=str(pca.n_components_),
                samples=str(take),
                mask="valid&water",
                random_seed=str(RANDOM_SEED),
                silhouette=str(float(sil)),
            )

        print("Wrote:", out_tif)

        # -------- PNG cluster preview --------
        with rasterio.open(out_tif) as ds_lab:
            lab = ds_lab.read(1).astype(np.int16)

        plt.figure(figsize=(12, 10))
        plt.imshow(lab, vmin=0, vmax=K - 1)
        plt.axis("off")
        plt.title(f"KMeans clusters (water-only) — K={K}")
        plt.tight_layout()
        out_png = OUTPUTS_DIR / f"{OUT_PREFIX}.png"
        plt.savefig(out_png, dpi=200)
        plt.close()
        print("Wrote:", out_png)

        # =========================================================
        # --- Simple physical-ish diagnostics (water-only) ---
        # =========================================================

        with rasterio.open(ACTIVE_WYVERN_FILE) as ds2:
            cube = ds2.read().astype(np.float32)   # (B,H,W)
            cube = np.transpose(cube, (1, 2, 0))  # (H,W,B)

        use = use_mask_full & (lab >= 0) & np.isfinite(cube).all(axis=2)
        X = cube[use]          # (N,B)
        y = lab[use]           # (N,)

        # Brightness proxy: mean radiance across bands
        bright = np.mean(X, axis=1)

        # Visible slope proxy (660 - 510)
        b510 = 2 - 1
        b660 = 12 - 1
        slope = (X[:, b660] - X[:, b510]) / (660 - 510)

        # Plot distributions per cluster
        plt.figure(figsize=(10, 4))
        plt.boxplot(
            [bright[y == k] for k in range(K)],
            tick_labels=[str(k) for k in range(K)],
            showfliers=False,
        )

        plt.xlabel("Cluster")
        plt.ylabel("Mean radiance (all bands)")
        plt.title("Cluster brightness proxy (water-only)")
        plt.tight_layout()
        out_b = OUTPUTS_DIR / f"{OUT_PREFIX}_cluster_brightness.png"
        plt.savefig(out_b, dpi=200)
        plt.close()

        plt.figure(figsize=(10, 4))
        plt.boxplot(
            [slope[y == k] for k in range(K)],
            tick_labels=[str(k) for k in range(K)],
            showfliers=False,
        )

        plt.xlabel("Cluster")
        plt.ylabel("Slope (Band660 - Band510) / 150nm")
        plt.title("Cluster red–blue slope proxy (water-only)")
        plt.tight_layout()
        out_s = OUTPUTS_DIR / f"{OUT_PREFIX}_cluster_slope.png"
        plt.savefig(out_s, dpi=200)
        plt.close()

        print("Wrote:", out_b)
        print("Wrote:", out_s)

        # -------- PCA PC1/2/3 composite (water-only) --------
        # Transform all water pixels (full scene) and write PC1/2/3 into image for viz
        H, W, B = cube_full.shape
        pcs = np.full((H, W, 3), np.nan, dtype=np.float32)

        flat = cube_full.reshape(-1, B)
        use_flat = np.isfinite(flat).all(axis=1)  # already NaN outside water

        Xv = flat[use_flat].astype(np.float32)
        Xv = l2_normalize_rows(Xv)
        Zv = pca.transform(Xv)[:, :3].astype(np.float32)

        pcs_flat = pcs.reshape(-1, 3)
        pcs_flat[use_flat] = Zv

        pc1 = percentile_stretch01(pcs[:, :, 0], 2, 98)
        pc2 = percentile_stretch01(pcs[:, :, 1], 2, 98)
        pc3 = percentile_stretch01(pcs[:, :, 2], 2, 98)
        pca_rgb = np.dstack([pc1, pc2, pc3])

        plt.figure(figsize=(14, 10))
        plt.imshow(pca_rgb)
        plt.axis("off")
        plt.title("PCA composite (PC1, PC2, PC3) — water-only")
        plt.tight_layout()
        out_pca_rgb = OUTPUTS_DIR / f"{OUT_PREFIX}_pca_pc123.png"
        plt.savefig(out_pca_rgb, dpi=200)
        plt.close()
        print("Wrote:", out_pca_rgb)

        # -------- Mean spectra per cluster --------
        compute_cluster_means(ds, out_tif, wl_nm, use_mask_full)

        # -------- Summary text (easy README paste) --------
        out_txt = OUTPUTS_DIR / f"{OUT_PREFIX}_summary.txt"
        ari_mean_offdiag = float((ari.sum() - np.trace(ari)) / (ari.size - STABILITY_RUNS))
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(f"K: {K}\n")
            f.write(f"PCA_N: {pca.n_components_}\n")
            f.write(f"Samples_fit: {take}\n")
            f.write(f"Silhouette_sampled: {float(sil)}\n")
            f.write(f"Explained_variance_ratio: {evr.tolist()}\n")
            f.write(f"Cumulative_explained_variance: {cum.tolist()}\n")
            f.write(f"ARI_runs: {STABILITY_RUNS}\n")
            f.write(f"ARI_mean_offdiag: {ari_mean_offdiag}\n")
        print("Wrote:", out_txt)


if __name__ == "__main__":
    main()
