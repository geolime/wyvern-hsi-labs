"""
Two combined views of NDTI (turbidity) and NDCI (chlorophyll) over water, TOA reflectance.

1. Optical-state map (primary combined product): each water pixel is labelled by whether its
   NDTI and NDCI sit above or below their own within-scene split (default: the median),
   giving four interpretable states, clearest / sediment-dominated / algae-dominated / both
   elevated. This names WHICH optical signal is elevated where, which a single blended score
   hides. It is NOT a "water quality" score: it carries no good/bad judgement and no units.
   The state boundaries are relative to this scene's own distribution.

2. Illustrative blend (secondary): an equal-weight z-score average of the two indices as one
   continuous map. NDTI and NDCI measure physically distinct things, so this blend has NO
   physical basis; equal weights are used deliberately, to signal the weighting encodes no
   claim. Kept only as an exploratory view, secondary to the state map and the index maps.

NOTE: TOA reflectance, no atmospheric correction. Relative optical proxies, not concentrations
and not validated water-quality classes.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from wyvernhsi import indices, io, visualization
from wyvernhsi.config import Config, load_config
from wyvernhsi.logging_setup import configure_logging
from wyvernhsi.masks import load_valid_mask, load_water_mask
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene

logger = logging.getLogger(__name__)

# Encoding: state = ndti_high + 2 * ndci_high  (0..3). Order matches STATE_COLORS.
STATE_LABELS = [
    "clearest (low turbidity, low chlorophyll signal)",
    "sediment-dominated (high turbidity, low chlorophyll signal)",
    "algae-dominated (low turbidity, high chlorophyll signal)",
    "both elevated (high turbidity, high chlorophyll signal)",
]
STATE_COLORS = ["#2c7fb8", "#a6611a", "#31a354", "#762a83"]


def _zscore_on_mask(x, mask):
    out = np.full(x.shape, np.nan, dtype=np.float32)
    sel = mask & np.isfinite(x)
    v = x[sel]
    if v.size == 0:
        return out
    out[sel] = ((x[sel] - float(v.mean())) / (float(v.std()) + 1e-12)).astype(np.float32)
    return out


def _split_high(x, mask, pct):
    """Boolean 'high' mask for x, thresholded at the given within-water percentile."""
    thr = float(np.quantile(x[mask], pct / 100.0))
    return x >= thr, thr


def _save_state_map(out_png, state, fractions, title):
    disp = np.ma.masked_less(state, 0)
    plt.figure(figsize=(12, 9))
    plt.imshow(disp, cmap=ListedColormap(STATE_COLORS), vmin=0, vmax=3, interpolation="nearest")
    plt.axis("off")
    plt.title(title)
    handles = [
        Patch(facecolor=STATE_COLORS[i], label=f"{STATE_LABELS[i]}: {fractions[i] * 100:.1f}%")
        for i in range(4)
    ]
    plt.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=1, frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
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

    ok = use & np.isfinite(ndti) & np.isfinite(ndci)
    n = int(ok.sum())
    if n == 0:
        logger.warning("No finite water pixels; skipping composite stage.")
        return

    # 1. Optical-state map (NDTI x NDCI split at a within-scene percentile)
    ndti_high, ndti_thr = _split_high(ndti, ok, p.state_split_pct)
    ndci_high, ndci_thr = _split_high(ndci, ok, p.state_split_pct)
    state = np.full(ndti.shape, -1, dtype=np.int8)
    state[ok] = ndti_high[ok].astype(np.int8) + 2 * ndci_high[ok].astype(np.int8)
    fractions = [float(np.count_nonzero(state == i)) / n for i in range(4)]
    logger.info("Optical-state split at p%g: NDTI thr = %.4f, NDCI thr = %.4f (n = %d water px)",
                p.state_split_pct, ndti_thr, ndci_thr, n)
    for i in range(4):
        logger.info("  state %d %s: %.1f%%", i, STATE_LABELS[i], fractions[i] * 100.0)
    _save_state_map(
        out_dir / "optical_state_2x2.png", state, fractions,
        f"Optical water state (NDTI x NDCI, split at within-scene p{p.state_split_pct:g}), water-only")

    # 2. Demoted illustrative blend (equal-weight z-score average)
    ndti[~use] = np.nan
    ndci[~use] = np.nan
    composite = (p.composite_w_ndti * _zscore_on_mask(ndti, use)
                 + p.composite_w_ndci * _zscore_on_mask(ndci, use)).astype(np.float32)
    composite[~use] = np.nan
    visualization.save_heatmap(
        out_dir / "optical_proxy_composite_continuous.png", composite,
        f"Illustrative NDTI+NDCI blend "
        f"({p.composite_w_ndti:g}*z(NDTI) + {p.composite_w_ndci:g}*z(NDCI), water-only)\n"
        f"secondary, no physical basis: see the optical-state map and the NDTI/NDCI tier maps")


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))