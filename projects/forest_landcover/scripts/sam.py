"""
Reference-spectrum mapping (Spectral Angle Mapper) over the full scene.

Builds endmembers from a few hand-picked reference ROIs per cover type (from config),
assigns each valid pixel to the nearest endmember by spectral angle (subject to a
threshold), and writes a class GeoTIFF + a best-angle GeoTIFF.

NOTE: this is reference-spectrum mapping, NOT a validated classification — there is no
accuracy assessment. See the project README.
"""
from __future__ import annotations

import logging

import numpy as np
import rasterio
from rasterio.windows import Window

from wyvernhsi import classification, io
from wyvernhsi.config import Config, load_config
from wyvernhsi.logging_setup import configure_logging
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene

logger = logging.getLogger(__name__)


def _build_endmembers(ds, sam):
    """Mean reference spectrum per class, averaged over its ROIs. Returns (names, E (K, B))."""
    names, endmembers = [], []
    half = sam.roi_half_size_px
    for name, points in sam.reference_classes.items():
        specs = []
        for row, col in points:
            win = Window.from_slices(
                (max(0, row - half), min(ds.height, row + half + 1)),
                (max(0, col - half), min(ds.width, col + half + 1)),
            )
            cube = io.read_cube(ds, window=win)  # (y, x, B), nodata -> NaN
            specs.append(np.nanmean(cube.reshape(-1, cube.shape[2]), axis=0))
        names.append(name)
        endmembers.append(np.nanmean(np.stack(specs), axis=0))
    return names, np.stack(endmembers).astype(np.float32)


def main(config: Config) -> None:
    sam = config.sam
    scene = resolve_scene(config.project_dir)
    outputs = scene.outputs_dir
    outputs.mkdir(parents=True, exist_ok=True)
    out_cls = outputs / "sam_fullscene_class.tif"
    out_ang = outputs / "sam_fullscene_best_angle.tif"

    with rasterio.open(scene.reflectance) as ds:
        names, endmembers = _build_endmembers(ds, sam)
        logger.info("SAM endmembers: %s", names)
        profile = ds.profile
        cls = np.full((ds.height, ds.width), -1, dtype=np.int16)
        ang = np.full((ds.height, ds.width), np.nan, dtype=np.float32)
        for w in io.iter_windows(ds, sam.tile_size):
            tile = io.read_cube(ds, window=w)
            c, a = classification.spectral_angle_classify(tile, endmembers, sam.angle_threshold_rad)
            r0, c0 = int(w.row_off), int(w.col_off)
            cls[r0:r0 + int(w.height), c0:c0 + int(w.width)] = c
            ang[r0:r0 + int(w.height), c0:c0 + int(w.width)] = a

    io.write_geotiff(profile, out_cls, cls, nodata=-1, dtype="int16", descriptions=["SAM_CLASS"])
    io.write_geotiff(profile, out_ang, ang, nodata=np.nan, dtype="float32", descriptions=["SAM_ANGLE_RAD"])

    legend = ";".join(f"{i}:{n}" for i, n in enumerate(names)) + ";-1:unclassified"
    with rasterio.open(out_cls, "r+") as dst:
        dst.update_tags(sam_classes=legend, angle_threshold_rad=str(sam.angle_threshold_rad))

    logger.info("Wrote: %s", out_cls)
    logger.info("Wrote: %s", out_ang)


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))