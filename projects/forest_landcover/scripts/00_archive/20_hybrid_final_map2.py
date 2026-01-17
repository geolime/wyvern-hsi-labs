from __future__ import annotations
from pathlib import Path
import numpy as np
import rasterio

from wyvernhsi.paths import OUTPUTS_DIR


SAM_BOOT = OUTPUTS_DIR / "sam_boot_fullscene_class.tif"
KMEANS_SEM = OUTPUTS_DIR / "kmeans_semantic_K5.tif"
OUT_TIF = OUTPUTS_DIR / "final_hybrid_3class.tif"

# Map KMeans semantic IDs -> 3 classes
# 0=trees, 1=vegetation, 2=soil, -1=nodata
KMEANS_TO_3 = {
    1: 0,  # dense_forest -> trees
    0: 1,  # crop_transition -> vegetation
    2: 1,  # vegetation_other -> vegetation
    3: 1,  # crop_field -> vegetation
    4: 2,  # soil_or_dry_veg -> soil
}


def ensure_same_grid(a: rasterio.DatasetReader, b: rasterio.DatasetReader):
    if (a.width, a.height) != (b.width, b.height):
        raise ValueError("Shape mismatch between rasters.")
    if a.transform != b.transform:
        raise ValueError("Transform mismatch between rasters.")
    if a.crs != b.crs:
        raise ValueError("CRS mismatch between rasters.")


def main():
    with rasterio.open(SAM_BOOT) as ds_sam, rasterio.open(KMEANS_SEM) as ds_km:
        ensure_same_grid(ds_sam, ds_km)
        sam = ds_sam.read(1).astype(np.int16)
        km = ds_km.read(1).astype(np.int16)
        profile = ds_sam.profile.copy()

    # Remap kmeans to 3 classes
    km3 = np.full_like(km, -1, dtype=np.int16)
    for k, v in KMEANS_TO_3.items():
        km3[km == k] = np.int16(v)
    km3[km == -1] = -1

    # Hybrid: prefer SAM where classified, else use KMeans
    out = sam.copy()
    fill_mask = (out == -1) & (km3 != -1)
    out[fill_mask] = km3[fill_mask]

    # Stats
    total = out.size
    u, c = np.unique(out, return_counts=True)
    print("Hybrid counts:", list(zip(u.tolist(), c.tolist())))
    print("Hybrid unclassified fraction:", float(c[u == -1][0] / total) if np.any(u == -1) else 0.0)

    # Write GeoTIFF
    profile.update(count=1, dtype="int16", nodata=-1, compress="deflate", predictor=2)
    with rasterio.open(OUT_TIF, "w", **profile) as dst:
        dst.write(out, 1)
        dst.update_tags(
            classes="0:trees;1:vegetation;2:soil;-1:nodata",
            sam_source="sam_boot_fullscene_class.tif",
            kmeans_source="kmeans_semantic_K5.tif",
            rule="use_SAM_if_classified_else_KMeans",
        )

    print("Wrote:", OUT_TIF)


if __name__ == "__main__":
    main()
