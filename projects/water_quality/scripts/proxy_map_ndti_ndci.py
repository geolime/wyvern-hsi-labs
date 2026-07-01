"""
Optical water-quality proxy maps over water (TOA reflectance, Wyvern scene):

  - NDTI (turbidity) and NDCI (chlorophyll) as continuous water-only heatmaps.
  - NDTI and NDCI percentile tier maps, each binned into terciles of its OWN water-pixel
    distribution, labelled low / moderate / high RELATIVE TO THIS SCENE.
  - An NDTI-NDCI density scatter reporting the Pearson correlation: the honest joint view
    of how the two proxies co-vary across water. They measure physically distinct things
    (sediment vs algae), so they are mapped independently, not blended into one score here.

NOTE: NDTI/NDCI are relative optical proxies on TOA reflectance, NOT calibrated turbidity
or chlorophyll concentrations (no atmospheric correction, no in-situ validation). Tiers are
within-scene relative orderings, not absolute levels and not comparable across dates.
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

TIER_LABELS_3 = ["low", "moderate", "high"]


def _tier_labels(n_bins: int) -> list[str]:
    base = TIER_LABELS_3 if n_bins == 3 else [f"tier {i + 1}" for i in range(n_bins)]
    return [f"{b} (relative to scene)" for b in base]


def _bin_by_quantiles(arr, mask, n_bins):
    """Assign each masked, finite pixel to one of n_bins equal-count percentile bins."""
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


def _scatter(out_png, x, y, mask, xlabel, ylabel, title):
    """Hexbin density of two indices over water, with the Pearson correlation in the title."""
    sel = mask & np.isfinite(x) & np.isfinite(y)
    xv, yv = x[sel], y[sel]
    if xv.size == 0:
        logger.warning("No finite water pixels for scatter; skipping %s", out_png)
        return
    r = float(np.corrcoef(xv, yv)[0, 1])
    logger.info("Corr(%s, %s) over water: r = %.4f (n = %d)", xlabel, ylabel, r, xv.size)
    plt.figure(figsize=(7, 6))
    plt.hexbin(xv, yv, gridsize=80, bins="log", mincnt=1)
    plt.colorbar(label="log10(pixel count)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{title}\nPearson r = {r:.3f} (n = {xv.size:,} water pixels)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    logger.info("Wrote: %s", out_png)


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

    ndti_w = np.where(use, ndti, np.nan)
    ndci_w = np.where(use, ndci, np.nan)

    # Continuous water-only heatmaps
    visualization.save_heatmap(out_dir / "ndti_continuous.png", ndti_w,
                               "NDTI turbidity proxy (water-only, relative)")
    visualization.save_heatmap(out_dir / "ndci_continuous.png", ndci_w,
                               "NDCI chlorophyll proxy (water-only, relative)")

    # Independent tier maps: each index binned on its OWN water-pixel distribution
    labels = _tier_labels(p.n_bins)
    visualization.save_binned(out_dir / "ndti_tiers.png",
                              _bin_by_quantiles(ndti, use, p.n_bins), p.n_bins,
                              f"NDTI turbidity tiers (water-only, {p.n_bins} percentile bins)", labels)
    visualization.save_binned(out_dir / "ndci_tiers.png",
                              _bin_by_quantiles(ndci, use, p.n_bins), p.n_bins,
                              f"NDCI chlorophyll tiers (water-only, {p.n_bins} percentile bins)", labels)

    # Joint view: how the two distinct proxies co-vary across water
    _scatter(out_dir / "ndti_ndci_scatter.png", ndti, ndci, use,
             "NDTI (turbidity proxy)", "NDCI (chlorophyll proxy)",
             "NDTI vs NDCI across water")


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))