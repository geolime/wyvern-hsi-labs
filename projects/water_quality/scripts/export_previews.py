"""
Water previews: cloud-masked RGB + NGB quicklooks, and water-class overlays on RGB/NGB
backgrounds plus a transparent class-only map (classes from stage 06).

Class labels are relative turbidity tiers (ordered by median NDTI), not validated types.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from wyvernhsi import io, visualization
from wyvernhsi.config import Config, load_config
from wyvernhsi.logging_setup import configure_logging
from wyvernhsi.masks import load_valid_mask, load_water_mask
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene

logger = logging.getLogger(__name__)

P_LO, P_HI = 2.0, 98.0
ALPHA = 0.45
TURBIDITY_LABELS = {
    0: "clearest (lowest NDTI)",
    1: "low turbidity",
    2: "moderate turbidity",
    3: "turbid",
    4: "most turbid (highest NDTI)",
}


def _save_quicklook(path, u8):
    plt.figure(figsize=(8, 8)); plt.imshow(u8); plt.axis("off")
    plt.tight_layout(pad=0); plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0); plt.close()
    logger.info("Wrote: %s", path)


def _draw_classes(path, lab, water, k, title, *, bg=None, alpha=1.0, transparent=False):
    plt.figure(figsize=(12, 10))
    if bg is not None:
        plt.imshow(bg.astype(np.float32) / 255.0 * 0.5)
    plt.imshow(np.ma.masked_where(lab < 0, lab), cmap=visualization.masked_cmap(),
               vmin=0, vmax=k - 1, alpha=alpha)
    plt.contour(water.astype(np.uint8), levels=[0.5], linewidths=0.5)
    plt.axis("off"); plt.title(title)
    visualization.add_class_legend([f"{i}: {TURBIDITY_LABELS.get(i, '')}" for i in range(k)])
    plt.tight_layout(); plt.savefig(path, dpi=200, transparent=transparent); plt.close()
    logger.info("Wrote: %s", path)


def main(config: Config) -> None:
    s = config.sfa
    k = config.clustering.k
    scene = resolve_scene(config.project_dir)
    out = scene.outputs_dir
    previews = out / "previews"
    water_dir = out / "water_features"
    previews.mkdir(parents=True, exist_ok=True)
    water_dir.mkdir(parents=True, exist_ok=True)

    valid = load_valid_mask(scene.mask)
    water = load_water_mask(out / "masks" / "water_mask.tif")

    with rasterio.open(scene.reflectance) as ds:
        rgb_raw = io.read_composite(ds, s.rgb_nm)
        ngb_raw = io.read_composite(ds, s.ngb_nm)

    # Cloud-masked quicklooks
    for raw, name in ((rgb_raw, "rgb_quicklook.png"), (ngb_raw, "ngb_water_composite.png")):
        masked = raw.copy()
        masked[~valid] = np.nan
        _save_quicklook(previews / name, visualization.stretch_rgb(masked, P_LO, P_HI))

    # Class overlays (need stage 06 labels)
    labels_tif = water_dir / f"water_features_kmeans_K{k}.tif"
    if not labels_tif.exists():
        logger.warning("Missing %s; skipping class overlays (run stage 06).", labels_tif)
        return
    with rasterio.open(labels_tif) as dl:
        lab = dl.read(1).astype(np.int16)

    rgb_bg = visualization.stretch_rgb(rgb_raw, P_LO, P_HI)
    ngb_bg = visualization.stretch_rgb(ngb_raw, P_LO, P_HI)
    _draw_classes(water_dir / "rgb_with_water_classes.png", lab, water, k,
                  f"Water classes over RGB (alpha={ALPHA})", bg=rgb_bg, alpha=ALPHA)
    _draw_classes(water_dir / "ngb_with_water_classes.png", lab, water, k,
                  f"Water classes over NGB (alpha={ALPHA})", bg=ngb_bg, alpha=ALPHA)
    _draw_classes(water_dir / "water_classes_only.png", lab, water, k,
                  "Water classes (transparent background)", transparent=True)


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))