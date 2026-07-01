"""
Unsupervised land-cover grouping: PCA + KMeans over TOA reflectance.

Fits PCA->KMeans on a sample of valid pixels, predicts the (full or subset) scene
tile-by-tile, writes a class GeoTIFF + preview PNG, and plots per-cluster mean spectra.

Unsupervised land-cover grouping: PCA + KMeans over TOA reflectance.
Fits on sampled valid pixels, predicts tile-by-tile, writes a class GeoTIFF + preview,
and plots per-cluster mean spectra.

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
import logging

from wyvernhsi import clustering, io
from wyvernhsi.config import Config, load_config
from wyvernhsi.logging_setup import configure_logging
from wyvernhsi.masks import load_valid_mask
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions

logger = logging.getLogger(__name__)

def _clip(r0, r1, c0, c1, height, width):
    return max(0, r0), min(height, r1), max(0, c0), min(width, c1)


def subset_window(ds, cl) -> Window | None:
    if not cl.use_subset:
        return None
    rows = [r for r, _ in cl.subset_points]
    cols = [c for _, c in cl.subset_points]
    r0, r1, c0, c1 = _clip(
        min(rows) - cl.subset_pad, max(rows) + cl.subset_pad,
        min(cols) - cl.subset_pad, max(cols) + cl.subset_pad,
        ds.height, ds.width,
    )
    return Window.from_slices((r0, r1), (c0, c1))


def _subset_mask(valid_full, win):
    if win is None:
        return valid_full
    r0, c0 = int(win.row_off), int(win.col_off)
    return valid_full[r0:r0 + int(win.height), c0:c0 + int(win.width)]


def plot_cluster_spectra(means, counts, wl_nm, pca_components, out_png):
    plt.figure(figsize=(12, 7))
    for k in range(means.shape[0]):
        if np.isfinite(means[k]).any():
            plt.plot(wl_nm, means[k], label=f"cluster {k} (n={counts[k]})")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("L2-normalized TOA reflectance")
    plt.title(f"KMeans cluster mean spectra (K={means.shape[0]}, PCA={pca_components})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    logger.info("Wrote: %s", out_png)


def main(config: Config) -> None:
    cl = config.clustering
    seed = config.random_seed
    scene = resolve_scene(config.project_dir)
    scene.outputs_dir.mkdir(parents=True, exist_ok=True)
    out_tif = scene.outputs_dir / f"kmeans_clusters_K{cl.k}.tif"

    valid_full = load_valid_mask(scene.mask)

    with rasterio.open(scene.reflectance) as ds:
        wl_nm = parse_wavelengths_nm_from_descriptions(list(ds.descriptions))
        win = subset_window(ds, cl)
        logger.info(
            "Mode: %s",
            "FULL SCENE" if win is None else
            f"SUBSET rows[{int(win.row_off)}:{int(win.row_off + win.height)}] "
            f"cols[{int(win.col_off)}:{int(win.col_off + win.width)}]",
    )

        cube = io.read_cube(ds, window=win)
        cube[~_subset_mask(valid_full, win)] = np.nan
        X, _ = clustering.flatten_valid(cube)
        fit = clustering.fit_pca_kmeans(
            X, k=cl.k, pca_components=cl.pca_components, n_samples=cl.n_samples, random_state=seed,
        )
        pca, kmeans = fit.pca, fit.kmeans
        logger.info("Fitted PCA + KMeans.")

        profile = ds.profile.copy()
        profile.update(
            count=1, dtype="int16", nodata=-1, compress="deflate",
            predictor=2, tiled=True, blockxsize=cl.tile_size, blockysize=cl.tile_size,
        )
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(np.full((ds.height, ds.width), -1, dtype=np.int16), 1)
            for w in io.iter_windows(ds, cl.tile_size, window=win):
                tile = io.read_cube(ds, window=w)
                vmask = valid_full[
                    int(w.row_off):int(w.row_off + w.height),
                    int(w.col_off):int(w.col_off + w.width),
                ]
                tile[~vmask] = np.nan
                dst.write(clustering.predict_tile(tile, pca, kmeans), 1, window=w)

    with rasterio.open(out_tif, "r+") as dst:
        dst.update_tags(
            kmeans_K=str(cl.k), pca_components=str(cl.pca_components),
            samples=str(min(cl.n_samples, X.shape[0])),
            mode="subset" if win is not None else "full",
        )
    logger.info("Wrote: %s", out_tif)

    with rasterio.open(out_tif) as ds_lab:
        lab = ds_lab.read(1)
    plt.figure(figsize=(12, 10))
    plt.imshow(lab, vmin=0, vmax=cl.k - 1)
    plt.axis("off")
    plt.title(f"KMeans clusters (K={cl.k})")
    plt.tight_layout()
    out_png = scene.outputs_dir / f"kmeans_clusters_K{cl.k}.png"
    plt.savefig(out_png, dpi=200)
    plt.close()
    logger.info("Wrote: %s", out_png)

    with rasterio.open(scene.reflectance) as ds, rasterio.open(out_tif) as lab_ds:
        tiles = (
            (io.read_cube(ds, window=w), lab_ds.read(1, window=w).astype(np.int16))
            for w in io.iter_windows(ds, cl.tile_size, window=win)
        )
        means, counts = clustering.cluster_mean_spectra(tiles, k=cl.k, n_bands=ds.count, normalize=True)
    plot_cluster_spectra(means, counts, wl_nm, cl.pca_components,
                         scene.outputs_dir / f"kmeans_cluster_spectra_K{cl.k}.png")


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))