from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR


# --- 3-class points (no young_sparse_canopy) ---
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

HALF_SIZE_ROI = 12          # 25x25 ROI around clicks
TILE_SIZE = 512             # adjust 256/512/1024
ANGLE_THRESHOLD_RAD = 0.10  # stricter = fewer mislabels, more unclassified


def clip_bounds(r0, r1, c0, c1, height, width):
    return max(0, r0), min(height, r1), max(0, c0), min(width, c1)


def roi_window(ds, row, col, half):
    r0, r1, c0, c1 = clip_bounds(row - half, row + half + 1, col - half, col + half + 1, ds.height, ds.width)
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
    E = np.stack(E, axis=0).astype(np.float32)  # (K, B)
    return class_names, E


def sam_best_angle_and_class(tile_yxb: np.ndarray, E_kb: np.ndarray):
    """
    tile: (Y, X, B) float32, may contain NaNs
    E: (K, B) float32
    returns:
      best_class int16 (Y,X) in [0..K-1] or -1 for invalid
      best_angle float32 (Y,X) radians (NaN for invalid)
    """
    eps = 1e-12
    cube = tile_yxb.astype(np.float32)

    valid = np.isfinite(cube).all(axis=2)
    best_class = np.full((cube.shape[0], cube.shape[1]), -1, dtype=np.int16)
    best_angle = np.full((cube.shape[0], cube.shape[1]), np.nan, dtype=np.float32)

    if not np.any(valid):
        return best_class, best_angle

    # unit vectors (shape-only)
    cube_unit = cube[valid] / (np.linalg.norm(cube[valid], axis=1, keepdims=True) + eps)  # (N,B)
    E_unit = E_kb / (np.linalg.norm(E_kb, axis=1, keepdims=True) + eps)                   # (K,B)

    dots = cube_unit @ E_unit.T                                                            # (N,K)
    dots = np.clip(dots, -1.0, 1.0)
    ang = np.arccos(dots).astype(np.float32)                                               # (N,K)

    bc = np.argmin(ang, axis=1).astype(np.int16)
    ba = np.min(ang, axis=1).astype(np.float32)

    # apply threshold
    bc_thr = bc.copy()
    bc_thr[ba > ANGLE_THRESHOLD_RAD] = -1

    best_class[valid] = bc_thr
    best_angle[valid] = ba
    return best_class, best_angle


def main():
    if not ACTIVE_WYVERN_FILE.exists():
        raise FileNotFoundError(f"Missing local file: {ACTIVE_WYVERN_FILE}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    out_cls = OUTPUTS_DIR / "sam_fullscene_class.tif"
    out_ang = OUTPUTS_DIR / "sam_fullscene_best_angle.tif"

    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        class_names, E = build_library(ds)
        print("Classes:", class_names)
        print("Writing:")
        print(" -", out_cls)
        print(" -", out_ang)

        profile_cls = ds.profile.copy()
        profile_cls.update(
            count=1,
            dtype="int16",
            nodata=-1,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=TILE_SIZE,
            blockysize=TILE_SIZE,
        )

        profile_ang = ds.profile.copy()
        profile_ang.update(
            count=1,
            dtype="float32",
            nodata=np.nan,
            compress="deflate",
            predictor=2,
            tiled=True,
            blockxsize=TILE_SIZE,
            blockysize=TILE_SIZE,
        )

        with rasterio.open(out_cls, "w", **profile_cls) as dst_cls, rasterio.open(out_ang, "w", **profile_ang) as dst_ang:
            height, width = ds.height, ds.width

            for row0 in range(0, height, TILE_SIZE):
                for col0 in range(0, width, TILE_SIZE):
                    row1 = min(height, row0 + TILE_SIZE)
                    col1 = min(width, col0 + TILE_SIZE)
                    win = Window.from_slices((row0, row1), (col0, col1))

                    arr = ds.read(window=win).astype(np.float32)  # (B, y, x)
                    if ds.nodata is not None:
                        arr[arr == ds.nodata] = np.nan

                    tile = np.transpose(arr, (1, 2, 0))            # (y, x, B)
                    cls_tile, ang_tile = sam_best_angle_and_class(tile, E)

                    dst_cls.write(cls_tile, 1, window=win)
                    dst_ang.write(ang_tile.astype(np.float32), 1, window=win)

            # Store legend in tags for convenience
            dst_cls.update_tags(
                sam_classes=";".join([f"{i}:{n}" for i, n in enumerate(class_names)]) + ";-1:unclassified",
                angle_threshold_rad=str(ANGLE_THRESHOLD_RAD),
            )

    print("Done.")


if __name__ == "__main__":
    main()
