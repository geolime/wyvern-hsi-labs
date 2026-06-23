"""
Unsupervised land-cover grouping: PCA + KMeans over TOA reflectance.

Fits PCA->KMeans on a sample of valid pixels, predicts the (full or subset) scene
tile-by-tile, writes a class GeoTIFF + preview PNG, and plots per-cluster mean spectra.

Input:  ACTIVE_WYVERN_FILE (TOA reflectance), ACTIVE_WYVERN_MASK (QA)
Output: kmeans_clusters_K{K}.tif, kmeans_clusters_K{K}.png, kmeans_cluster_spectra_K{K}.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window

from wyvernhsi import clustering, io
from wyvernhsi.masks import load_valid_mask
from wyvernhsi.paths import ACTIVE_WYVERN_FILE, ACTIVE_WYVERN_MASK, OUTPUTS_DIR
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions

# ---------- Settings ----------
K = 5
N_SAMPLES = 50000
PCA_COMPONENTS = 8
TILE_SIZE = 512
RANDOM_STATE = 42

USE_SUBSET = True
SUBSET_PAD = 1200
MANUAL_SUBSET_BOUNDS = None  # (row0, row1, col0, col1) or None

POINTS_FOR_SUBSET = [
    (896, 2163), (313, 3065), (1043, 3477), (3745, 2629), (3309, 3839),
    (3856, 3127), (3598, 3219), (239, 3999), (3966, 3200), (4378, 3489),
    (3690, 2789), (4421, 2838), (1350, 640), (3101, 1229), (2536, 1352),
]


def _clip(r0, r1, c0, c1, height, width):
    return max(0, r0), min(height, r1), max(0, c0), min(width, c1)


def subset_window(ds) -> Window | None:
    if not USE_SUBSET:
        return None
    if MANUAL_SUBSET_BOUNDS is not None:
        r0, r1, c0, c1 = _clip(*MANUAL_SUBSET_BOUNDS, ds.height, ds.width)
        return Window.from_slices((r0, r1), (c0, c1))
    rows = [r for r, _ in POINTS_FOR_SUBSET]
    cols = [c for _, c in POINTS_FOR_SUBSET]
    r0, r1, c0, c1 = _clip(
        min(rows) - SUBSET_PAD, max(rows) + SUBSET_PAD,
        min(cols) - SUBSET_PAD, max(cols) + SUBSET_PAD,
        ds.height, ds.width,
    )
    return Window.from_slices((r0, r1), (c0, c1))


def _subset_mask(valid_full: np.ndarray, win: Window | None) -> np.ndarray:
    if win is None:
        return valid_full
    r0, c0 = int(win.row_off), int(win.col_off)
    return valid_full[r0:r0 + int(win.height), c0:c0 + int(win.width)]


def plot_cluster_spectra(means, counts, wl_nm, out_png) -> None:
    plt.figure(figsize=(12, 7))
    for k in range(means.shape[0]):
        if np.isfinite(means[k]).any():
            plt.plot(wl_nm, means[k], label=f"cluster {k} (n={counts[k]})")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("L2-normalized TOA reflectance")  # corrected: data is reflectance, not radiance
    plt.title(f"KMeans cluster mean spectra (K={means.shape[0]}, PCA={PCA_COMPONENTS})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Wrote:", out_png)


def main() -> None:
    if not ACTIVE_WYVERN_FILE.exists():
        raise FileNotFoundError(f"Missing local file: {ACTIVE_WYVERN_FILE}")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_tif = OUTPUTS_DIR / f"kmeans_clusters_K{K}.tif"

    valid_full = load_valid_mask(ACTIVE_WYVERN_MASK)

    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        wl_nm = parse_wavelengths_nm_from_descriptions(list(ds.descriptions))
        win = subset_window(ds)
        print("Mode:", "FULL SCENE" if win is None else
              f"SUBSET rows[{int(win.row_off)}:{int(win.row_off + win.height)}] "
              f"cols[{int(win.col_off)}:{int(win.col_off + win.width)}]")

        # Fit on sampled valid pixels
        cube = io.read_cube(ds, window=win)
        cube[~_subset_mask(valid_full, win)] = np.nan
        X, _ = clustering.flatten_valid(cube)
        pca, kmeans = clustering.fit_pca_kmeans(
            X, k=K, pca_components=PCA_COMPONENTS, n_samples=N_SAMPLES, random_state=RANDOM_STATE,
        )
        print("Fitted PCA + KMeans.")

        # Tiled prediction
        profile = ds.profile.copy()
        profile.update(
            count=1, dtype="int16", nodata=-1, compress="deflate",
            predictor=2, tiled=True, blockxsize=TILE_SIZE, blockysize=TILE_SIZE,
        )
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(np.full((ds.height, ds.width), -1, dtype=np.int16), 1)
            for w in io.iter_windows(ds, TILE_SIZE, window=win):
                tile = io.read_cube(ds, window=w)
                vmask = valid_full[
                    int(w.row_off):int(w.row_off + w.height),
                    int(w.col_off):int(w.col_off + w.width),
                ]
                tile[~vmask] = np.nan
                dst.write(clustering.predict_tile(tile, pca, kmeans), 1, window=w)

    with rasterio.open(out_tif, "r+") as dst:
        dst.update_tags(
            kmeans_K=str(K), pca_components=str(PCA_COMPONENTS),
            samples=str(min(N_SAMPLES, X.shape[0])),
            mode="subset" if win is not None else "full",
        )
    print("Wrote:", out_tif)

    # Preview PNG
    with rasterio.open(out_tif) as ds_lab:
        lab = ds_lab.read(1)
    plt.figure(figsize=(12, 10))
    plt.imshow(lab, vmin=0, vmax=K - 1)
    plt.axis("off")
    plt.title(f"KMeans clusters (K={K})")
    plt.tight_layout()
    out_png = OUTPUTS_DIR / f"kmeans_clusters_K{K}.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Wrote:", out_png)

    # Cluster mean spectra (second streaming pass)
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds, rasterio.open(out_tif) as lab_ds:
        tiles = (
            (io.read_cube(ds, window=w), lab_ds.read(1, window=w).astype(np.int16))
            for w in io.iter_windows(ds, TILE_SIZE, window=win)
        )
        means, counts = clustering.cluster_mean_spectra(tiles, k=K, n_bands=ds.count, normalize=True)
    plot_cluster_spectra(means, counts, wl_nm, OUTPUTS_DIR / f"kmeans_cluster_spectra_K{K}.png")


if __name__ == "__main__":
    main()