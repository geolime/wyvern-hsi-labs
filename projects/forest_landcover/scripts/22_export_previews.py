"""
Forest classification previews: class-only map + CIR overlay, for both the KMeans clusters
and the SAM reference-class map.

KMeans IDs are arbitrary (unsupervised) so its legend shows IDs; SAM class names come from
config.sam.reference_classes (ROI-defined).
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib import colormaps
from matplotlib.colors import BoundaryNorm, ListedColormap

from wyvernhsi import io, visualization
from wyvernhsi.config import Config, load_config
from wyvernhsi.logging_setup import configure_logging
from wyvernhsi.masks import load_valid_mask
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene

logger = logging.getLogger(__name__)

CIR_NM = (800.0, 660.0, 560.0)
GAMMA = 0.85
NODATA_COLOR = (0.92, 0.92, 0.92)
SAM_PALETTE = [(0.10, 0.45, 0.10), (0.30, 0.75, 0.30), (0.80, 0.70, 0.50)]


def _cir(ds):
    return np.dstack([visualization.percentile_stretch(io.read_band_nm(ds, nm), 1, 99) ** GAMMA
                      for nm in CIR_NM])


def _boundaries(lab):
    b = np.zeros(lab.shape, dtype=bool)
    dv = lab[:-1, :] != lab[1:, :]
    dh = lab[:, :-1] != lab[:, 1:]
    b[:-1, :] |= dv; b[1:, :] |= dv
    b[:, :-1] |= dh; b[:, 1:] |= dh
    return b


def _preview(lab, cir, *, class_colors, class_names, nodata_color, draw_boundaries,
             title, class_only_png, overlay_png):
    nclass = len(class_names)
    idx = np.where(lab == -1, 0, lab + 1)
    cmap = ListedColormap([nodata_color, *class_colors])
    norm = BoundaryNorm([-0.5] + [i + 0.5 for i in range(nclass + 1)], cmap.N)

    plt.figure(figsize=(14, 10), facecolor="white")
    plt.imshow(idx, cmap=cmap, norm=norm, interpolation="nearest",
               alpha=np.where(idx == 0, 0.0, 1.0).astype(np.float32))
    if draw_boundaries:
        outline = np.zeros((*lab.shape, 4), dtype=np.float32)
        outline[_boundaries(lab)] = (1, 1, 1, 0.8)
        plt.imshow(outline)
    plt.axis("off"); plt.title(title)
    visualization.add_class_legend(["nodata", *class_names], colors=[nodata_color, *class_colors])
    plt.tight_layout(); plt.savefig(class_only_png, dpi=200); plt.close()

    overlay = np.zeros((*lab.shape, 4), dtype=np.float32)
    overlay[lab == -1] = (*nodata_color, 0.20)
    for i in range(nclass):
        overlay[lab == i] = (*class_colors[i], 0.55)
    plt.figure(figsize=(14, 10)); plt.imshow(cir); plt.imshow(overlay)
    plt.axis("off"); plt.title(f"{title} — overlay (CIR)")
    plt.tight_layout(); plt.savefig(overlay_png, dpi=200); plt.close()
    logger.info("Wrote: %s, %s", class_only_png, overlay_png)


def main(config: Config) -> None:
    k = config.clustering.k
    scene = resolve_scene(config.project_dir)
    out = scene.outputs_dir
    out.mkdir(parents=True, exist_ok=True)

    with rasterio.open(scene.reflectance) as ds:
        cir = _cir(ds)  # read once, reused for both overlays

    with rasterio.open(out / f"kmeans_clusters_K{k}.tif") as ds:
        km = ds.read(1).astype(np.int16)
    _preview(km, cir,
             class_colors=visualization.class_colors(k),
             class_names=[f"cluster {c}" for c in range(k)],
             nodata_color=colormaps["viridis"](0.0)[:3], draw_boundaries=True,
             title=f"KMeans clusters (K={k})",
             class_only_png=out / f"kmeans_K{k}_class_only.png",
             overlay_png=out / f"kmeans_K{k}_overlay_cir.png")

    names = list(config.sam.reference_classes.keys())
    with rasterio.open(out / "sam_fullscene_class.tif") as ds:
        sam = ds.read(1).astype(np.int16)
    sam[~load_valid_mask(scene.mask)] = -1
    _preview(sam, cir,
             class_colors=SAM_PALETTE[:len(names)],
             class_names=names, nodata_color=NODATA_COLOR, draw_boundaries=False,
             title=f"SAM classification ({len(names)} classes)",
             class_only_png=out / "sam_class_only.png",
             overlay_png=out / "sam_overlay.png")


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))