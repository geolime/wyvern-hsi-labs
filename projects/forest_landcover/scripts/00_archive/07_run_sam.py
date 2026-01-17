from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

from pysptools.classification import SAM

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR

if not hasattr(np, "float"):
    np.float = float

# --- Same points as before (row, col) ---
POINTS = {
    "trees": [
        (3620, 2640),
        (2688, 2394),
        (2776, 2051),
        (4206, 3912),
    ],
    "vegetated_non_tree": [
        (3879, 3121),
        (3608, 3240),
        (4394, 3898),
        (4507, 4061),
    ],
    "low_veg_soil": [
        (3697, 2826),
        (4631, 2845),
        (2536, 3832),
        (4634, 2635),
    ],
}

# ROI size around points to compute class spectra (same as before)
HALF_SIZE_ROI = 12  # 25x25

# Subset window padding around all points (in pixels)
SUBSET_PAD = 300

# SAM threshold (smaller = stricter). 0.10 is a reasonable start.
SAM_THRESHOLD = 0.10


def clip_bounds(r0, r1, c0, c1, height, width):
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(height, r1)
    c1 = min(width, c1)
    return r0, r1, c0, c1


def roi_window_around_point(ds, row, col, half):
    r0, r1 = row - half, row + half + 1
    c0, c1 = col - half, col + half + 1
    r0, r1, c0, c1 = clip_bounds(r0, r1, c0, c1, ds.height, ds.width)
    return Window.from_slices((r0, r1), (c0, c1))


def mean_spectrum_from_roi(ds, row, col, half):
    win = roi_window_around_point(ds, row, col, half)
    arr = ds.read(window=win).astype(np.float32)  # (bands, rows, cols)
    if ds.nodata is not None:
        arr[arr == ds.nodata] = np.nan
    flat = arr.reshape(arr.shape[0], -1)          # (bands, pixels)
    return np.nanmean(flat, axis=1)               # (bands,)


def compute_class_library(ds):
    """Return (class_names, E) where E is (n_classes, n_bands)."""
    class_names = []
    class_specs = []

    for cls, pts in POINTS.items():
        specs = []
        for (r, c) in pts:
            specs.append(mean_spectrum_from_roi(ds, r, c, HALF_SIZE_ROI))
        cls_mean = np.nanmean(np.stack(specs, axis=0), axis=0)
        class_names.append(cls)
        class_specs.append(cls_mean)

    E = np.stack(class_specs, axis=0).astype(np.float32)
    return class_names, E


def subset_window_covering_points(ds, pad):
    rows = [r for pts in POINTS.values() for (r, _) in pts]
    cols = [c for pts in POINTS.values() for (_, c) in pts]
    r0 = min(rows) - pad
    r1 = max(rows) + pad
    c0 = min(cols) - pad
    c1 = max(cols) + pad
    r0, r1, c0, c1 = clip_bounds(r0, r1, c0, c1, ds.height, ds.width)
    return Window.from_slices((r0, r1), (c0, c1)), (r0, r1, c0, c1)


def main():
    if not ACTIVE_WYVERN_FILE.exists():
        raise FileNotFoundError(f"Missing local file: {ACTIVE_WYVERN_FILE}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    out_png = OUTPUTS_DIR / "sam_classmap.png"
    out_npy = OUTPUTS_DIR / "sam_classmap.npy"

    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        print(f"Using: {ACTIVE_WYVERN_FILE.name}")
        print(f"Image size: {ds.height} x {ds.width}, bands={ds.count}")

        # 1) Build library from ROIs
        class_names, E = compute_class_library(ds)
        print("Classes:", class_names)

        # 2) Read a subset window for classification (keeps it fast)
        win, bounds = subset_window_covering_points(ds, SUBSET_PAD)
        r0, r1, c0, c1 = bounds
        print(f"Classifying subset window rows[{r0}:{r1}] cols[{c0}:{c1}]")

        arr = ds.read(window=win).astype(np.float32)  # (bands, rows, cols)
        if ds.nodata is not None:
            arr[arr == ds.nodata] = np.nan

    # Convert to (rows, cols, bands) for pysptools
    cube = np.transpose(arr, (1, 2, 0))

    # 3) Run SAM
    sam = SAM()
    cls_map = sam.classify(cube, E, threshold=SAM_THRESHOLD)

    # Save raw results (so you can reuse without recomputing)
    np.save(out_npy, cls_map)

    # 4) Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(cls_map)
    plt.title(f"SAM class map (threshold={SAM_THRESHOLD})")
    plt.axis("off")

    # Add a simple legend text block (index -> class)
    legend_lines = [f"{i}: {name}" for i, name in enumerate(class_names)]
    legend_text = "\n".join(legend_lines)
    plt.gcf().text(0.02, 0.02, legend_text, fontsize=10, family="monospace")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print("Wrote:")
    print(" -", out_png)
    print(" -", out_npy)


if __name__ == "__main__":
    main()
