"""
Composite optical proxy map: a weighted z-score blend of NDTI (turbidity) and NDCI
(chlorophyll) over water, as a continuous map and a top-percentile hotspot overlay on NGB.

NOTE: a heuristic optical composite on TOA reflectance with arbitrary weights — NOT a
validated environmental risk index. Renamed from "risk_proxy" to avoid that overclaim.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from wyvernhsi import indices, io, visualization
from wyvernhsi.config import Config, load_config
from wyvernhsi.logging_setup import configure_logging
from wyvernhsi.masks import load_valid_mask, load_water_mask
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene

logger = logging.getLogger(__name__)


def _zscore_on_mask(x, mask):
    out = np.full(x.shape, np.nan, dtype=np.float32)
    sel = mask & np.isfinite(x)
    v = x[sel]
    if v.size == 0:
        return out
    out[sel] = ((x[sel] - float(v.mean())) / (float(v.std()) + 1e-12)).astype(np.float32)
    return out


def _save_continuous(path, arr, title):
    v = arr[np.isfinite(arr)]
    vmin, vmax = (float(np.percentile(v, 2)), float(np.percentile(v, 98))) if v.size else (None, None)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(arr, vmin=vmin, vmax=vmax); plt.axis("off"); plt.title(title)
    fig = plt.gcf(); fig.subplots_adjust(right=0.86)
    plt.colorbar(im, cax=fig.add_axes([0.88, 0.12, 0.03, 0.76]))
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05); plt.close()
    logger.info("Wrote: %s", path)


def _save_hotspots(path, ngb, hotspot, water, title):
    plt.figure(figsize=(12, 10))
    plt.imshow(ngb)
    plt.contour(hotspot.astype(np.uint8), levels=[0.5], colors="yellow", linewidths=1.0)
    plt.contour(water.astype(np.uint8), levels=[0.5], colors="white", linewidths=0.6)
    plt.axis("off"); plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()
    logger.info("Wrote: %s", path)


def main(config: Config) -> None:
    p = config.proxies
    scene = resolve_scene(config.project_dir)
    out_dir = scene.outputs_dir / "water_features" / "proxies"
    out_dir.mkdir(parents=True, exist_ok=True)

    valid = load_valid_mask(scene.mask)
    water = load_water_mask(scene.outputs_dir / "masks" / "water_mask.tif")
    use = valid & water

    with rasterio.open(scene.reflectance) as ds:
        ndti = indices.ndti(io.read_band_nm(ds, p.ndti_red_nm), io.read_band_nm(ds, p.ndti_green_nm))
        ndci = indices.ndci(io.read_band_nm(ds, p.ndci_red_edge_nm), io.read_band_nm(ds, p.ndci_red_nm))
        ngb = visualization.stretch_rgb(io.read_composite(ds, p.ngb_nm), p.percentile_lo, p.percentile_hi)

    ndti[~use] = np.nan
    ndci[~use] = np.nan
    composite = (p.composite_w_ndti * _zscore_on_mask(ndti, use)
                 + p.composite_w_ndci * _zscore_on_mask(ndci, use)).astype(np.float32)
    composite[~use] = np.nan

    _save_continuous(out_dir / "optical_proxy_composite_continuous.png", composite,
                     f"Composite optical proxy = {p.composite_w_ndti}*z(NDTI) + "
                     f"{p.composite_w_ndci}*z(NDCI) (water-only)")

    top = int(p.composite_hotspot_top_pct)
    v = composite[use & np.isfinite(composite)]
    thr = np.percentile(v, 100.0 - p.composite_hotspot_top_pct) if v.size else np.inf
    hot = use & np.isfinite(composite) & (composite >= thr)
    _save_hotspots(out_dir / f"optical_proxy_composite_hotspots_top{top}_on_ngb.png", ngb, hot, water,
                   f"Composite optical-proxy hotspots (top {top}% water-only) over NGB")


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))