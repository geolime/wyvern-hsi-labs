from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions


# --- Your clicked points (row, col) ---
# Assumption: first 4 = trees, next 4 = vegetated_non_tree, rest = low_veg_soil
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

# ROI size: 25x25 pixels (half_size=12 => 2*12+1 = 25)
HALF_SIZE = 12


def clip_window(row0, col0, height, width, half):
    r0 = max(0, row0 - half)
    r1 = min(height, row0 + half + 1)
    c0 = max(0, col0 - half)
    c1 = min(width, col0 + half + 1)
    return r0, r1, c0, c1


def read_roi_all_bands(ds, row, col, half):
    r0, r1, c0, c1 = clip_window(row, col, ds.height, ds.width, half)
    win = Window.from_slices((r0, r1), (c0, c1))
    arr = ds.read(window=win).astype(np.float32)  # (bands, rows, cols)

    # Convert nodata to NaN if present
    nodata = ds.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan

    return arr, (r0, r1, c0, c1)


def summarize_spectrum(arr_brc):
    # arr: (bands, rows, cols)
    flat = arr_brc.reshape(arr_brc.shape[0], -1)  # (bands, pixels)
    mean = np.nanmean(flat, axis=1)
    median = np.nanmedian(flat, axis=1)
    std = np.nanstd(flat, axis=1)
    return mean, median, std


def main():
    if not ACTIVE_WYVERN_FILE.exists():
        raise FileNotFoundError(f"Missing local file: {ACTIVE_WYVERN_FILE}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUTPUTS_DIR / "roi_spectra.png"
    out_csv = OUTPUTS_DIR / "roi_spectra_summary.csv"

    src = str(ACTIVE_WYVERN_FILE)
    print(f"Using: {ACTIVE_WYVERN_FILE}")

    with rasterio.open(src) as ds:
        wl_nm = parse_wavelengths_nm_from_descriptions(list(ds.descriptions))

        # store per-point spectra + per-class aggregate
        per_point = []
        per_class_means = {}

        for cls, pts in POINTS.items():
            cls_means = []
            for i, (row, col) in enumerate(pts, start=1):
                roi, bounds = read_roi_all_bands(ds, row, col, HALF_SIZE)
                mean, median, std = summarize_spectrum(roi)
                cls_means.append(mean)

                per_point.append({
                    "class": cls,
                    "point_id": i,
                    "row": row,
                    "col": col,
                    "r0": bounds[0],
                    "r1": bounds[1],
                    "c0": bounds[2],
                    "c1": bounds[3],
                    "mean": mean,
                    "median": median,
                    "std": std,
                })

            per_class_means[cls] = np.nanmean(np.stack(cls_means, axis=0), axis=0)

    # --- Plot per-class mean spectra ---
    plt.figure(figsize=(12, 7))
    for cls, spec in per_class_means.items():
        plt.plot(wl_nm, spec, label=cls)

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Radiance (raw units)")
    plt.title(f"ROI mean spectra (ROI {2*HALF_SIZE+1}x{2*HALF_SIZE+1})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    # --- Write a compact CSV (per-point summary) ---
    # CSV: one row per point per band is too big; instead write per-point mean at a few key wavelengths + metadata.
    key_wls = [490, 560, 660, 800]
    key_idx = [int(np.nanargmin(np.abs(wl_nm - w))) for w in key_wls]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "class", "point_id", "row", "col", "r0", "r1", "c0", "c1",
            *[f"mean_{key_wls[i]}nm" for i in range(len(key_wls))],
        ])
        for rec in per_point:
            w.writerow([
                rec["class"], rec["point_id"], rec["row"], rec["col"],
                rec["r0"], rec["r1"], rec["c0"], rec["c1"],
                *[float(rec["mean"][j]) for j in key_idx],
            ])

    print("Wrote:")
    print(" -", out_png)
    print(" -", out_csv)


if __name__ == "__main__":
    main()
