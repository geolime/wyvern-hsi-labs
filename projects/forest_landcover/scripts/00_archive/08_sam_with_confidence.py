from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR

# ---- Your classes / points (same as before) ----
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
}


HALF_SIZE_ROI = 12          # ROI size around clicks
SUBSET_PAD = 800            # make it larger so you see more context
ANGLE_THRESHOLD_RAD = 0.10
MARGIN_THRESHOLD_RAD = 0.005  # very gentle, avoids over-unclassifying


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
    class_names = []
    E = []
    for cls, pts in POINTS.items():
        specs = [mean_spectrum(ds, r, c, HALF_SIZE_ROI) for r, c in pts]
        class_names.append(cls)
        E.append(np.nanmean(np.stack(specs, axis=0), axis=0))
    return class_names, np.stack(E, axis=0).astype(np.float32)  # (K, B)


def subset_window_covering_points(ds, pad):
    rows = [r for pts in POINTS.values() for (r, _) in pts]
    cols = [c for pts in POINTS.values() for (_, c) in pts]
    r0, r1, c0, c1 = clip_bounds(min(rows)-pad, max(rows)+pad, min(cols)-pad, max(cols)+pad, ds.height, ds.width)
    return Window.from_slices((r0, r1), (c0, c1)), (r0, r1, c0, c1)


def sam_angles(cube_yxb: np.ndarray, E_kb: np.ndarray) -> np.ndarray:
    """
    Compute SAM angles for each pixel to each endmember.
    cube: (Y, X, B)
    E:    (K, B)
    returns angles: (Y, X, K) in radians
    """
    # normalize vectors
    eps = 1e-12
    cube = cube_yxb.astype(np.float32)

    # mask invalid pixels (any nan across bands)
    valid = np.isfinite(cube).all(axis=2)

    # norms
    #cube_norm = np.linalg.norm(cube, axis=2) + eps          # (Y, X)
    #E_norm = np.linalg.norm(E_kb, axis=1) + eps             # (K,)

    # dot products: (Y,X,K)
    #dots = np.tensordot(cube, E_kb, axes=([2], [1]))        # (Y, X, K)

    # cos(theta) = dot/(||a||*||b||)
    #denom = cube_norm[..., None] * E_norm[None, None, :]
    #cosang = np.clip(dots / denom, -1.0, 1.0)

    #ang = np.arccos(cosang).astype(np.float32)
    # Normalize cube and endmembers to unit length (shape-only comparison)
    cube_unit = cube / (np.linalg.norm(cube, axis=2, keepdims=True) + eps)        # (Y,X,B)
    E_unit = E_kb / (np.linalg.norm(E_kb, axis=1, keepdims=True) + eps)          # (K,B)

    dots = np.tensordot(cube_unit, E_unit, axes=([2], [1]))                      # (Y,X,K)
    cosang = np.clip(dots, -1.0, 1.0)
    ang = np.arccos(cosang).astype(np.float32)

    ang[~valid] = np.nan
    return ang


def main():
    if not ACTIVE_WYVERN_FILE.exists():
        raise FileNotFoundError(f"Missing local file: {ACTIVE_WYVERN_FILE}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        class_names, E = build_library(ds)
        print("Library finite fraction:", float(np.isfinite(E).mean()))
        print("Library bandwise NaNs:", int(np.isnan(E).sum()))


        win, bounds = subset_window_covering_points(ds, SUBSET_PAD)
        r0, r1, c0, c1 = bounds
        print(f"Subset: rows[{r0}:{r1}] cols[{c0}:{c1}]")

        arr = ds.read(window=win).astype(np.float32)  # (B, Y, X)
        if ds.nodata is not None:
            arr[arr == ds.nodata] = np.nan

    cube = np.transpose(arr, (1, 2, 0))  # (Y, X, B)
    print("Cube finite fraction:", float(np.isfinite(cube).mean()))
    print("Pixels with any NaN across bands:", int((~np.isfinite(cube).all(axis=2)).sum()))


    ang = sam_angles(cube, E)            # (Y, X, K)
    # Sort angles so we can compare best vs second-best
    ang_sorted = np.sort(ang, axis=2)  # (Y, X, K)

    best_angle = ang_sorted[..., 0].astype(np.float32)
    second_angle = ang_sorted[..., 1].astype(np.float32)
    margin = (second_angle - best_angle).astype(np.float32)

    # Compute best class indices separately (argmin across classes)
    # Only safe on pixels with at least one finite angle.
    valid = np.isfinite(best_angle)
    best_class = np.full(best_angle.shape, -1, dtype=np.int16)
    best_class[valid] = np.argmin(ang[valid], axis=1).astype(np.int16)
    # Mask low-confidence pixels
    masked = best_class.copy()
    masked[~valid] = -1
    masked[best_angle > ANGLE_THRESHOLD_RAD] = -1
    masked[margin < MARGIN_THRESHOLD_RAD] = -1
    print("Angle threshold:", ANGLE_THRESHOLD_RAD, "Margin threshold:", MARGIN_THRESHOLD_RAD)
    print("Classified fraction:", float((masked != -1).mean()))
    np.save(OUTPUTS_DIR / "sam_margin.npy", margin)


    # Plot masked classes
    plt.figure(figsize=(10, 8))
    plt.imshow(masked)
    plt.title(f"SAM masked class map (angle <= {ANGLE_THRESHOLD_RAD} rad)")
    plt.axis("off")
    legend = "\n".join([f"{i}: {n}" for i, n in enumerate(class_names)] + ["-1: unclassified"])
    plt.gcf().text(0.02, 0.02, legend, fontsize=10, family="monospace")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "sam_masked_class.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Plot angle map (lower = better)
    plt.figure(figsize=(10, 8))
    plt.imshow(best_angle)
    plt.title("SAM best angle (radians) — lower is better")
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "sam_best_angle.png", dpi=200, bbox_inches="tight")
    plt.close()

    print("Wrote:")
    print(" - outputs/sam_masked_class.png")
    print(" - outputs/sam_best_angle.png")


if __name__ == "__main__":
    main()
