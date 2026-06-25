"""
Optical water-quality proxy maps: NDTI (turbidity) and NDCI (chlorophyll) over a Wyvern
scene — continuous, 5-quantile binned, and top-percentile hotspot overlays on an NGB
composite — for the whole valid scene and water-only.

NOTE: NDTI/NDCI are relative optical proxies on TOA reflectance, NOT calibrated turbidity
or chlorophyll concentrations (no atmospheric correction, no in-situ validation).
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


def _robust_limits(x, lo=2.0, hi=98.0):
    v = x[np.isfinite(x)]
    if v.size == 0:
        return None, None
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def _save_continuous(path, arr, title):
    vmin, vmax = _robust_limits(arr)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(arr, vmin=vmin, vmax=vmax)
    plt.axis("off"); plt.title(title)
    fig = plt.gcf(); fig.subplots_adjust(right=0.86)
    plt.colorbar(im, cax=fig.add_axes([0.88, 0.12, 0.03, 0.76]))
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05); plt.close()
    logger.info("Wrote: %s", path)


def _bin_by_quantiles(arr, mask, n_bins):
    out = np.full(arr.shape, -1, dtype=np.int16)
    v = arr[mask & np.isfinite(arr)]
    if v.size == 0:
        return out
    qs = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    if qs.size < 3:
        return out
    ok = mask & np.isfinite(arr)
    out[ok] = np.digitize(arr, qs[1:-1], right=True)[ok].astype(np.int16)
    return out


def _save_binned(path, binned, title, n_bins):
    cmap = plt.get_cmap("viridis", n_bins).copy()
    cmap.set_bad(alpha=0.0)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(np.ma.masked_where(binned < 0, binned), cmap=cmap, vmin=0, vmax=n_bins - 1)
    plt.axis("off"); plt.title(title)
    fig = plt.gcf(); fig.subplots_adjust(right=0.86)
    cb = plt.colorbar(im, cax=fig.add_axes([0.88, 0.12, 0.03, 0.76]), ticks=list(range(n_bins)))
    cb.ax.set_yticklabels([f"Q{i + 1}" for i in range(n_bins)])
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05); plt.close()
    logger.info("Wrote: %s", path)


def _top_percent(arr, mask, top_pct):
    v = arr[mask & np.isfinite(arr)]
    if v.size == 0:
        return np.zeros(arr.shape, dtype=bool)
    thr = np.percentile(v, 100.0 - top_pct)
    return mask & np.isfinite(arr) & (arr >= thr)


def _save_hotspots(path, ngb, hotspot, water, title):
    plt.figure(figsize=(12, 10))
    plt.imshow(ngb)
    plt.contour(hotspot.astype(np.uint8), levels=[0.5], colors="yellow", linewidths=1.2)
    plt.contour(water.astype(np.uint8), levels=[0.5], colors="white", linewidths=0.6)
    plt.axis("off"); plt.title(title); plt.tight_layout()
    plt.savefig(path, dpi=200); plt.close()
    logger.info("Wrote: %s", path)


def main(config: Config) -> None:
    p = config.proxies
    scene = resolve_scene(config.project_dir)
    out_dir = scene.outputs_dir / "water_features" / "proxies"
    out_whole, out_water = out_dir / "whole_scene", out_dir / "water_only"
    for d in (out_whole, out_water):
        d.mkdir(parents=True, exist_ok=True)

    valid = load_valid_mask(scene.mask)
    water = load_water_mask(scene.outputs_dir / "masks" / "water_mask.tif")
    use_water, use_whole = valid & water, valid

    with rasterio.open(scene.reflectance) as ds:
        ndti = indices.ndti(io.read_band_nm(ds, p.ndti_red_nm), io.read_band_nm(ds, p.ndti_green_nm))
        ndci = indices.ndci(io.read_band_nm(ds, p.ndci_red_edge_nm), io.read_band_nm(ds, p.ndci_red_nm))
        ngb = visualization.stretch_rgb(io.read_composite(ds, p.ngb_nm), p.percentile_lo, p.percentile_hi)

    top = int(p.hotspot_top_pct)
    for name, arr, title in (("ndti", ndti, "NDTI (turbidity proxy)"),
                             ("ndci", ndci, "NDCI (chlorophyll proxy)")):
        _save_continuous(out_whole / f"{name}_continuous.png",
                         np.where(use_whole, arr, np.nan), f"{title} — whole scene, QA-valid")
        _save_continuous(out_water / f"{name}_continuous.png",
                         np.where(use_water, arr, np.nan), f"{title} — water-only")
        _save_binned(out_whole / f"{name}_binned5.png",
                     _bin_by_quantiles(arr, use_whole, p.n_bins), f"{title} — 5 quantiles, whole scene", p.n_bins)
        _save_binned(out_water / f"{name}_binned5.png",
                     _bin_by_quantiles(arr, use_water, p.n_bins), f"{title} — 5 quantiles, water-only", p.n_bins)
        _save_hotspots(out_dir / f"{name}_hotspots_top{top}_on_ngb.png",
                       ngb, _top_percent(arr, use_water, p.hotspot_top_pct), water,
                       f"{title} hotspots (top {top}% water-only) over NGB")


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))