from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, ACTIVE_WYVERN_MASK, OUTPUTS_DIR
from wyvernhsi.masks import load_valid_mask
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions




# ---------- Settings ----------
K = 5                       # number of clusters
N_SAMPLES = 50000           # random pixels for fitting k-means
PCA_COMPONENTS = 8          # reduce spectral dims before k-means (helps stability)
TILE_SIZE = 512             # tiling for prediction
RANDOM_STATE = 42

# Start fast on a subset. Set USE_SUBSET=False for full scene once you're happy.
USE_SUBSET = True
SUBSET_PAD = 1200           # pixels around the clicked ROI extents (used only if USE_SUBSET=True)

# If you don't want any ROI dependence, you can instead hardcode a window here.
# Leave None to auto-build subset window from POINTS.
MANUAL_SUBSET_BOUNDS = None  # (row0, row1, col0, col1) or None

# Optional: re-use your earlier points to define a sensible subset quickly
POINTS_FOR_SUBSET = [
    (896, 2163),
    (313, 3065),
    (1043, 3477),
    (3745, 2629),
    (3309, 3839),
    (3856, 3127),
    (3598, 3219),
    (239, 3999),
    (3966, 3200),
    (4378, 3489),
    (3690, 2789),
    (4421, 2838),
    (1350, 640),
    (3101, 1229),
    (2536, 1352),
]


def clip_bounds(r0, r1, c0, c1, height, width):
    return max(0, r0), min(height, r1), max(0, c0), min(width, c1)


def subset_window(ds) -> Window | None:
    if not USE_SUBSET:
        return None

    if MANUAL_SUBSET_BOUNDS is not None:
        r0, r1, c0, c1 = MANUAL_SUBSET_BOUNDS
        r0, r1, c0, c1 = clip_bounds(r0, r1, c0, c1, ds.height, ds.width)
        return Window.from_slices((r0, r1), (c0, c1))

    rows = [r for r, _ in POINTS_FOR_SUBSET]
    cols = [c for _, c in POINTS_FOR_SUBSET]
    r0 = min(rows) - SUBSET_PAD
    r1 = max(rows) + SUBSET_PAD
    c0 = min(cols) - SUBSET_PAD
    c1 = max(cols) + SUBSET_PAD
    r0, r1, c0, c1 = clip_bounds(r0, r1, c0, c1, ds.height, ds.width)
    return Window.from_slices((r0, r1), (c0, c1))


def read_cube(ds, win: Window | None) -> np.ndarray:
    """Return cube as float32 with shape (Y, X, B)."""
    arr = ds.read(window=win).astype(np.float32)  # (B,Y,X)
    if ds.nodata is not None:
        arr[arr == ds.nodata] = np.nan
    cube = np.transpose(arr, (1, 2, 0))          # (Y,X,B)
    return cube


def flatten_valid(cube_yxb: np.ndarray):
    """Return (X, valid_mask) where X is (N,B) for pixels with all-finite spectra."""
    valid = np.isfinite(cube_yxb).all(axis=2)
    X = cube_yxb[valid]
    return X, valid


def fit_models(X: np.ndarray):
    """
    Fit PCA -> KMeans on sampled data.
    Returns (pca, kmeans).
    """
    if X.shape[0] == 0:
        raise ValueError("No valid pixels found to sample from.")

    n = min(N_SAMPLES, X.shape[0])
    idx = np.random.default_rng(RANDOM_STATE).choice(X.shape[0], size=n, replace=False)
    Xs = X[idx]

    pca = PCA(n_components=min(PCA_COMPONENTS, Xs.shape[1]), random_state=RANDOM_STATE)
    Zs = pca.fit_transform(Xs)

    kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init="auto")
    kmeans.fit(Zs)

    return pca, kmeans


def predict_tile(tile_yxb: np.ndarray, pca: PCA, kmeans: KMeans) -> np.ndarray:
    """
    Predict cluster IDs for one tile.
    Returns int16 array (Y,X) with -1 for invalid pixels.
    """
    X, valid = flatten_valid(tile_yxb)
    out = np.full(tile_yxb.shape[:2], -1, dtype=np.int16)
    if X.shape[0] == 0:
        return out

    Z = pca.transform(X)
    lab = kmeans.predict(Z).astype(np.int16)
    out[valid] = lab
    return out


def compute_cluster_means(ds, win: Window | None, labels_path: Path, wl_nm: np.ndarray):
    """
    Compute mean spectrum per cluster using a second pass over tiles.
    Reads labels from disk to avoid holding everything in memory.
    """
    with rasterio.open(labels_path) as lab_ds:
        # Determine processing bounds
        if win is None:
            row0, col0 = 0, 0
            height, width = ds.height, ds.width
        else:
            row0, col0 = int(win.row_off), int(win.col_off)
            height, width = int(win.height), int(win.width)

        sums = np.zeros((K, ds.count), dtype=np.float64)
        counts = np.zeros((K,), dtype=np.int64)

        for r in range(0, height, TILE_SIZE):
            for c in range(0, width, TILE_SIZE):
                rr0 = row0 + r
                cc0 = col0 + c
                rr1 = min(row0 + height, rr0 + TILE_SIZE)
                cc1 = min(col0 + width, cc0 + TILE_SIZE)

                w = Window.from_slices((rr0, rr1), (cc0, cc1))

                cube = read_cube(ds, w)  # (y,x,b)
                labs = lab_ds.read(1, window=w).astype(np.int16)

                valid = (labs >= 0) & np.isfinite(cube).all(axis=2)
                if not np.any(valid):
                    continue

                X = cube[valid]          # (N,B)
                y = labs[valid]          # (N,)

                for k in range(K):
                    m = (y == k)
                    if np.any(m):
                        sums[k] += np.sum(X[m], axis=0)
                        counts[k] += int(np.sum(m))

        means = np.full((K, ds.count), np.nan, dtype=np.float32)
        for k in range(K):
            if counts[k] > 0:
                means[k] = (sums[k] / counts[k]).astype(np.float32)

    # Plot
    plt.figure(figsize=(12, 7))
    for k in range(K):
        if np.isfinite(means[k]).any():
            plt.plot(wl_nm, means[k], label=f"cluster {k} (n={counts[k]})")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Radiance (raw units)")
    plt.title(f"Unsupervised KMeans cluster mean spectra (K={K}, PCA={PCA_COMPONENTS})")
    plt.legend()
    plt.tight_layout()
    out_png = OUTPUTS_DIR / f"kmeans_cluster_spectra_K{K}.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Wrote:", out_png)


def main():
    if not ACTIVE_WYVERN_FILE.exists():
        raise FileNotFoundError(f"Missing local file: {ACTIVE_WYVERN_FILE}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    out_tif = OUTPUTS_DIR / f"kmeans_clusters_K{K}.tif"

    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        wl_nm = parse_wavelengths_nm_from_descriptions(list(ds.descriptions))

        win = subset_window(ds)
        if win is None:
            print("Mode: FULL SCENE")
        else:
            print(f"Mode: SUBSET window rows[{int(win.row_off)}:{int(win.row_off+win.height)}] "
                  f"cols[{int(win.col_off)}:{int(win.col_off+win.width)}]")
            
        valid_mask_full = load_valid_mask(ACTIVE_WYVERN_MASK)

        # Read cube for fitting (subset or full)
        cube = read_cube(ds, win)

        
        # Apply QA mask to fitting cube
        if win is None:
            valid_mask = valid_mask_full
        else:
            r0, c0 = int(win.row_off), int(win.col_off)
            r1, c1 = r0 + int(win.height), c0 + int(win.width)
            valid_mask = valid_mask_full[r0:r1, c0:c1]
        cube[~valid_mask] = np.nan

        X, _ = flatten_valid(cube)

        np.random.seed(RANDOM_STATE)
        pca, kmeans = fit_models(X)
        print("Fitted PCA + KMeans.")

        # Output GeoTIFF profile
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
            if win is None:
                row0, col0 = 0, 0
                height, width = ds.height, ds.width
            else:
                row0, col0 = int(win.row_off), int(win.col_off)
                height, width = int(win.height), int(win.width)

            # Initialize whole output as nodata (-1)
            # (Not strictly necessary but makes intent clear; tiles write over it.)
            dst.write(np.full((ds.height, ds.width), -1, dtype=np.int16), 1)

            for r in range(0, height, TILE_SIZE):
                for c in range(0, width, TILE_SIZE):
                    rr0 = row0 + r
                    cc0 = col0 + c
                    rr1 = min(row0 + height, rr0 + TILE_SIZE)
                    cc1 = min(col0 + width, cc0 + TILE_SIZE)

                    w = Window.from_slices((rr0, rr1), (cc0, cc1))
                    tile = read_cube(ds, w)

                    # Apply QA mask to this tile
                    valid_tile = valid_mask_full[
                        int(w.row_off):int(w.row_off + w.height),
                        int(w.col_off):int(w.col_off + w.width)
                    ]
                    tile[~valid_tile] = np.nan

                    labs = predict_tile(tile, pca, kmeans)
                    dst.write(labs, 1, window=w)


        # Store metadata
        with rasterio.open(out_tif, "r+") as dst:
            dst.update_tags(
                kmeans_K=str(K),
                pca_components=str(PCA_COMPONENTS),
                samples=str(min(N_SAMPLES, X.shape[0])),
                mode="subset" if win is not None else "full",
            )

        print("Wrote:", out_tif)

        # -------- PNG preview export --------
        with rasterio.open(out_tif) as ds_lab:
            lab = ds_lab.read(1)

        plt.figure(figsize=(12, 10))
        plt.imshow(lab, vmin=0, vmax=K-1)
        plt.axis("off")
        plt.title(f"KMeans clusters (K={K})")
        plt.tight_layout()
        out_png = OUTPUTS_DIR / f"kmeans_clusters_K{K}.png"
        plt.savefig(out_png, dpi=200)
        plt.close()

        print("Wrote:", out_png)
        

        # Compute and plot mean spectra per cluster
        compute_cluster_means(ds, win, out_tif, wl_nm)


if __name__ == "__main__":
    main()
