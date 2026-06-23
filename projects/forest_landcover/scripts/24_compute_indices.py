"""
Spectral-index separability of KMeans clusters and SAM classes.

Computes NDVI and red-edge slope, then summarises each class's index distribution
to a CSV and a boxplot figure for the README.

Inputs:  ACTIVE_WYVERN_FILE (TOA reflectance), kmeans_clusters_K5.tif, sam_fullscene_class.tif
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
from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR

KMEANS_TIF = OUTPUTS_DIR / "kmeans_clusters_K5.tif"
SAM_TIF = OUTPUTS_DIR / "sam_fullscene_class.tif"

NM_RED, NM_RED_EDGE, NM_NIR = 660.0, 720.0, 800.0
N_CLUSTERS = 5
MIN_PIXELS = 1000
SAM_NAMES = {0: "trees", 1: "vegetation", 2: "soil"}


def _stats(source: str, name: str, mask: np.ndarray, ndvi: np.ndarray, re_slope: np.ndarray) -> dict:
    return {
        "source": source,
        "class": name,
        "ndvi_mean": float(np.nanmean(ndvi[mask])),
        "ndvi_std": float(np.nanstd(ndvi[mask])),
        "re_slope_mean": float(np.nanmean(re_slope[mask])),
        "re_slope_std": float(np.nanstd(re_slope[mask])),
        "count": int(mask.sum()),
    }


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        red = io.read_band_nm(ds, NM_RED)
        red_edge = io.read_band_nm(ds, NM_RED_EDGE)
        nir = io.read_band_nm(ds, NM_NIR)

    ndvi = indices.ndvi(nir, red)
    re_slope = indices.red_edge_slope(red_edge, red, NM_RED_EDGE, NM_RED)

    with rasterio.open(KMEANS_TIF) as ds:
        km = ds.read(1)
    with rasterio.open(SAM_TIF) as ds:
        sam = ds.read(1)

    results = []
    for k in range(N_CLUSTERS):
        m = km == k
        if m.sum() >= MIN_PIXELS:
            results.append(_stats("kmeans", f"cluster_{k}", m, ndvi, re_slope))
    for k, name in SAM_NAMES.items():
        m = sam == k
        if m.sum() >= MIN_PIXELS:
            results.append(_stats("sam", name, m, ndvi, re_slope))

    out_csv = OUTPUTS_DIR / "spectral_index_stats.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, values, title in (
        (axes[0], ndvi, "NDVI by KMeans cluster"),
        (axes[1], re_slope, "Red-edge slope by KMeans cluster"),
    ):
        data, labels = [], []
        for k in range(N_CLUSTERS):
            m = km == k
            if m.sum() > MIN_PIXELS:
                vals = values[m]
                data.append(vals[np.isfinite(vals)])
                labels.append(f"K{k}")
        ax.boxplot(data, showfliers=False)
        ax.set_title(title)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)

    fig.tight_layout()
    out_png = OUTPUTS_DIR / "spectral_indices_kmeans.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print("Wrote:", out_png)


if __name__ == "__main__":
    main()