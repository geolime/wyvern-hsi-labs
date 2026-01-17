from __future__ import annotations
import numpy as np
import rasterio
from rasterio.windows import Window
from skimage.morphology import footprint_rectangle
from skimage.filters.rank import modal

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, ACTIVE_WYVERN_MASK, OUTPUTS_DIR
from wyvernhsi.masks import load_valid_mask
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions


KMEANS_TIF = OUTPUTS_DIR / "kmeans_clusters_K5.tif"

# Map KMeans semantic IDs -> SAM classes (0 trees, 1 veg, 2 soil)
KMEANS_TO_SAM = {
    2: 0,  # dense_forest -> trees
    0: 1,  # field A -> veg
    1: 1,  # field B -> veg
    4: 1,  # mixed transition -> veg
    3: 2,  # bare soil -> soil
}

# How many "core" pixels to sample per SAM class
N_PER_CLASS = 20000
RANDOM_STATE = 42

# Neighborhood size for purity (3 or 5). Start with 3.
PURITY_KERNEL = 3


def core_mask(label_img: np.ndarray, target: int, kernel: int) -> np.ndarray:
    """
    A pixel is 'core' if the modal label in its neighborhood equals target.
    This is a quick purity test.
    """
    # rank filters need non-negative => shift by +1 so nodata -1 -> 0
    shifted = (label_img + 1).astype(np.uint16)
    fp = footprint_rectangle((kernel, kernel))
    mode = modal(shifted, fp) - 1  # shift back
    return (label_img == target) & (mode == target)


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_npz = OUTPUTS_DIR / "bootstrap_endmembers_from_kmeans.npz"

    rng = np.random.default_rng(RANDOM_STATE)

    with rasterio.open(KMEANS_TIF) as ds_km:
        km = ds_km.read(1).astype(np.int16)
    
    # Load QA valid mask (full scene)
    valid_mask_full = load_valid_mask(ACTIVE_WYVERN_MASK)

    # Build SAM-class label image from KMeans labels
    sam_like = np.full_like(km, -1, dtype=np.int16)
    for k, s in KMEANS_TO_SAM.items():
        sam_like[km == k] = np.int16(s)
    sam_like[km == -1] = -1

    # Core masks for each SAM class
    core = {}
    for s in [0, 1, 2]:
        m = core_mask(sam_like, s, PURITY_KERNEL) & valid_mask_full
        core[s] = m
        print(f"SAM class {s}: core pixels = {int(m.sum())}")


    # Open hyperspectral cube
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        wl_nm = parse_wavelengths_nm_from_descriptions(list(ds.descriptions))
        bands = ds.count

        endmembers = {}
        counts = {}

        for s in [0, 1, 2]:
            idx = np.argwhere(core[s])
            if idx.shape[0] == 0:
                raise RuntimeError(f"No core pixels found for class {s}. Try PURITY_KERNEL=3 or mapping check.")

            n = min(N_PER_CLASS, idx.shape[0])
            pick = idx[rng.choice(idx.shape[0], size=n, replace=False)]
            rows = pick[:, 0]
            cols = pick[:, 1]

            # Read spectra in chunks to avoid tiny random reads being too slow
            # We'll do a simple approach: read bands once (can be big but OK for 31 bands),
            # then index. If memory is an issue, we can tile it.
            arr = ds.read().astype(np.float32)  # (B, Y, X)
            if ds.nodata is not None:
                arr[arr == ds.nodata] = np.nan
            cube = np.transpose(arr, (1, 2, 0))  # (Y, X, B)

            X = cube[rows, cols, :]  # (n, B)
            X = X[np.isfinite(X).all(axis=1)]
            if X.shape[0] == 0:
                raise RuntimeError(f"All sampled spectra were invalid for class {s}.")

            endmembers[s] = np.nanmean(X, axis=0).astype(np.float32)
            counts[s] = int(X.shape[0])
            print(f"SAM class {s}: used {counts[s]} spectra for mean endmember.")

    # Save endmembers
    np.savez(
        out_npz,
        wavelengths_nm=wl_nm.astype(np.float32),
        endmember_trees=endmembers[0],
        endmember_veg=endmembers[1],
        endmember_soil=endmembers[2],
        n_trees=np.int32(counts[0]),
        n_veg=np.int32(counts[1]),
        n_soil=np.int32(counts[2]),
        purity_kernel=np.int32(PURITY_KERNEL),
        n_per_class=np.int32(N_PER_CLASS),
    )

    print("Wrote:", out_npz)


if __name__ == "__main__":
    main()
