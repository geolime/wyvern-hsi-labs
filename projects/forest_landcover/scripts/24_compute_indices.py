"""
Spectral-index separability of KMeans clusters and SAM classes.

Computes NDVI and red-edge slope, then summarises each class's index distribution
to a CSV and a boxplot figure for the README.

Spectral-index separability of KMeans clusters and SAM classes (NDVI, red-edge slope).

Inputs:  reflectance, kmeans_clusters_K5.tif, sam_fullscene_class.tif
Outputs: spectral_index_stats.csv, spectral_indices_kmeans.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from wyvernhsi import indices, io
from wyvernhsi.config import Config, load_config
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene

SAM_NAMES = {0: "trees", 1: "vegetation", 2: "soil"}


def _stats(source, name, mask, ndvi, re_slope) -> dict:
    return {
        "source": source,
        "class": name,
        "ndvi_mean": float(np.nanmean(ndvi[mask])),
        "ndvi_std": float(np.nanstd(ndvi[mask])),
        "re_slope_mean": float(np.nanmean(re_slope[mask])),
        "re_slope_std": float(np.nanstd(re_slope[mask])),
        "count": int(mask.sum()),
    }


def main(config: Config) -> None:
    f = config.features
    k_clusters = config.clustering.k
    scene = resolve_scene(config.project_dir)
    outputs = scene.outputs_dir
    outputs.mkdir(parents=True, exist_ok=True)
    kmeans_tif = outputs / f"kmeans_clusters_K{k_clusters}.tif"
    sam_tif = outputs / "sam_fullscene_class.tif"

    with rasterio.open(scene.reflectance) as ds:
        red = io.read_band_nm(ds, f.red_nm)
        red_edge = io.read_band_nm(ds, f.red_edge_nm)
        nir = io.read_band_nm(ds, f.nir_nm)

    ndvi = indices.ndvi(nir, red)
    re_slope = indices.red_edge_slope(red_edge, red, f.red_edge_nm, f.red_nm)

    with rasterio.open(kmeans_tif) as ds:
        km = ds.read(1)
    with rasterio.open(sam_tif) as ds:
        sam = ds.read(1)

    results = []
    for kk in range(k_clusters):
        msk = km == kk
        if msk.sum() >= f.min_pixels:
            results.append(_stats("kmeans", f"cluster_{kk}", msk, ndvi, re_slope))
    for kk, name in SAM_NAMES.items():
        msk = sam == kk
        if msk.sum() >= f.min_pixels:
            results.append(_stats("sam", name, msk, ndvi, re_slope))

    out_csv = outputs / "spectral_index_stats.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, values, title in (
        (axes[0], ndvi, "NDVI by KMeans cluster"),
        (axes[1], re_slope, "Red-edge slope by KMeans cluster"),
    ):
        data, labels = [], []
        for kk in range(k_clusters):
            msk = km == kk
            if msk.sum() > f.min_pixels:
                vals = values[msk]
                data.append(vals[np.isfinite(vals)])
                labels.append(f"K{kk}")
        ax.boxplot(data, showfliers=False)
        ax.set_title(title)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)

    fig.tight_layout()
    out_png = outputs / "spectral_indices_kmeans.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print("Wrote:", out_png)


if __name__ == "__main__":
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))