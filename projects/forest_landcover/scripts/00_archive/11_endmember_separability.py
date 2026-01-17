import numpy as np
import rasterio
from rasterio.windows import Window

from wyvernhsi.paths import ACTIVE_WYVERN_FILE


POINTS = {
    "dense_trees": [
        (896, 2163),
        (313, 3065),
        (1043, 3477),
        (3745, 2629),
        (3309, 3839),
    ],
    "bright_veg": [
        (3856, 3127),
        (3598, 3219),
        (239, 3999),
        (3966, 3200),
        (4378, 3489),
    ],
    "low_veg_soil": [
        (3690, 2789),
        (4421, 2838),
        (1350, 640),
        (3101, 1229),
        (2536, 1352),
    ],
    "young_sparse_canopy": [
        (2511, 4171),
        (2486, 3894),
        (3641, 4465),
        (3506, 2463),
        (2143, 3569),
    ],
}

HALF_SIZE_ROI = 12  # 25x25


def clip_bounds(r0, r1, c0, c1, height, width):
    return max(0, r0), min(height, r1), max(0, c0), min(width, c1)


def roi_window(ds, row, col, half):
    r0, r1, c0, c1 = clip_bounds(row-half, row+half+1, col-half, col+half+1, ds.height, ds.width)
    return Window.from_slices((r0, r1), (c0, c1))


def mean_spectrum(ds, row, col, half):
    arr = ds.read(window=roi_window(ds, row, col, half)).astype(np.float32)  # (bands, r, c)
    if ds.nodata is not None:
        arr[arr == ds.nodata] = np.nan
    flat = arr.reshape(arr.shape[0], -1)
    return np.nanmean(flat, axis=1)


def build_library(ds):
    names, specs = [], []
    for cls, pts in POINTS.items():
        ss = [mean_spectrum(ds, r, c, HALF_SIZE_ROI) for r, c in pts]
        names.append(cls)
        specs.append(np.nanmean(np.stack(ss, axis=0), axis=0))
    return names, np.stack(specs, axis=0).astype(np.float32)


def sam_angle(a, b, eps=1e-12):
    a = a / (np.linalg.norm(a) + eps)
    b = b / (np.linalg.norm(b) + eps)
    cos = np.clip(np.dot(a, b), -1.0, 1.0)
    return float(np.arccos(cos))


def main():
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        names, E = build_library(ds)

    print("Pairwise SAM angles between class spectra (radians):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ang = sam_angle(E[i], E[j])
            print(f"  {names[i]} vs {names[j]}: {ang:.4f} rad")


if __name__ == "__main__":
    main()
