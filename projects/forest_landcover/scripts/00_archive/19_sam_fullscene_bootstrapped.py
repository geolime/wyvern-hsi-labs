from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR


BOOT_NPZ = OUTPUTS_DIR / "bootstrap_endmembers_from_kmeans.npz"

TILE_SIZE = 512
ANGLE_THRESHOLD_RAD = 0.10  # start here; try 0.08 (stricter) or 0.12 (looser)

# Class order in output GeoTIFF:
# 0 = trees, 1 = vegetation, 2 = soil, -1 = unclassified/nodata
CLASS_NAMES = ["trees", "vegetation", "soil"]


def sam_best_angle_and_class(tile_yxb: np.ndarray, E_kb: np.ndarray):
    """
    tile: (Y, X, B) float32, may contain NaNs
    E: (K, B) float32
    returns:
      best_class int16 (Y,X) in [0..K-1] or -1 for invalid/low-confidence
      best_angle float32 (Y,X) radians (NaN for invalid)
    """
    eps = 1e-12
    cube = tile_yxb.astype(np.float32)

    valid = np.isfinite(cube).all(axis=2)
    best_class = np.full((cube.shape[0], cube.shape[1]), -1, dtype=np.int16)
    best_angle = np.full((cube.shape[0], cube.shape[1]), np.nan, dtype=np.float32)

    if not np.any(valid):
        return best_class, best_angle

    X = cube[valid]  # (N,B)

    # unit vectors (shape-only)
    X_unit = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)             # (N,B)
    E_unit = E_kb / (np.linalg.norm(E_kb, axis=1, keepdims=True) + eps)       # (K,B)

    dots = X_unit @ E_unit.T
    dots = np.clip(dots, -1.0, 1.0)
    ang = np.arccos(dots).astype(np.float32)                                  # (N,K)

    bc = np.argmin(ang, axis=1).astype(np.int16)
    ba = np.min(ang, axis=1).astype(np.float32)

    bc[ba > ANGLE_THRESHOLD_RAD] = -1

    best_class[valid] = bc
    best_angle[valid] = ba
    return best_class, best_angle


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if not ACTIVE_WYVERN_FILE.exists():
        raise FileNotFoundError(f"Missing local file: {ACTIVE_WYVERN_FILE}")
    if not BOOT_NPZ.exists():
        raise FileNotFoundError(f"Missing bootstrapped endmembers: {BOOT_NPZ}")

    out_cls = OUTPUTS_DIR / "sam_boot_fullscene_class.tif"
    out_ang = OUTPUTS_DIR / "sam_boot_fullscene_best_angle.tif"

    data = np.load(BOOT_NPZ)
    E = np.stack(
        [data["endmember_trees"], data["endmember_veg"], data["endmember_soil"]],
        axis=0
    ).astype(np.float32)  # (3,B)

    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
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

                    arr = ds.read(window=win).astype(np.float32)  # (B,y,x)
                    if ds.nodata is not None:
                        arr[arr == ds.nodata] = np.nan

                    tile = np.transpose(arr, (1, 2, 0))  # (y,x,B)
                    cls_tile, ang_tile = sam_best_angle_and_class(tile, E)

                    dst_cls.write(cls_tile, 1, window=win)
                    dst_ang.write(ang_tile.astype(np.float32), 1, window=win)

            dst_cls.update_tags(
                sam_classes=";".join([f"{i}:{n}" for i, n in enumerate(CLASS_NAMES)]) + ";-1:unclassified",
                angle_threshold_rad=str(ANGLE_THRESHOLD_RAD),
                endmembers_source="kmeans_bootstrap",
            )

    print("Done.")


if __name__ == "__main__":
    main()
