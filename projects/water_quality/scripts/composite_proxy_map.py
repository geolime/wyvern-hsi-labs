"""
Illustrative optical composite: an equal-weight z-score blend of NDTI (turbidity) and NDCI
(chlorophyll) over water, as a single continuous water-only map.

This is a DEMOTED, secondary product with NO physical basis. NDTI and NDCI measure
physically distinct things (sediment vs algae), and averaging their z-scores does not
estimate any real quantity. The primary, honest products are the independent NDTI and NDCI
tier maps and the NDTI-NDCI scatter (proxy_map_ndti_ndci.py). Equal weights are used
deliberately, to signal that the weighting encodes no claim.

NOTE: TOA reflectance, no atmospheric correction. A relative optical blend, NOT a validated
environmental or "risk" index.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
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

    ndti[~use] = np.nan
    ndci[~use] = np.nan
    composite = (p.composite_w_ndti * _zscore_on_mask(ndti, use)
                 + p.composite_w_ndci * _zscore_on_mask(ndci, use)).astype(np.float32)
    composite[~use] = np.nan

    visualization.save_heatmap(
        out_dir / "optical_proxy_composite_continuous.png", composite,
        f"Illustrative NDTI+NDCI blend "
        f"({p.composite_w_ndti:g}*z(NDTI) + {p.composite_w_ndci:g}*z(NDCI), water-only)\n"
        f"secondary, no physical basis: see the NDTI and NDCI tier maps")


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))