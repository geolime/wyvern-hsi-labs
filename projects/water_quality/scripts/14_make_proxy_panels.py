"""
Multi-panel proxy figures: NGB + binned NDTI/NDCI + composite optical proxy + SFA classes.

Builds value-range-binned NDTI/NDCI maps, a labeled composite optical-proxy map, and two
3-panel figures (NGB | binned NDTI | binned NDCI; and NGB | SFA classes | composite proxy).

NOTE: NDTI/NDCI and the weighted composite are optical proxies on TOA reflectance, NOT
validated concentrations or an environmental "risk" index.
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

TURBIDITY_LABELS = {
    0: "clearest (lowest NDTI)",
    1: "low turbidity",
    2: "moderate turbidity",
    3: "turbid",
    4: "most turbid (highest NDTI)",
}


def _zscore(x, mask):
    out = np.full(x.shape, np.nan, dtype=np.float32)
    sel = mask & np.isfinite(x)
    v = x[sel]
    if v.size == 0:
        return out
    out[sel] = ((x[sel] - float(v.mean())) / (float(v.std()) + 1e-12)).astype(np.float32)
    return out


def _quantile_edges(x, mask, n_bins):
    v = x[mask & np.isfinite(x)]
    if v.size == 0:
        return np.array([], dtype=np.float32)
    return np.maximum.accumulate(np.quantile(v, np.linspace(0, 1, n_bins + 1))).astype(np.float32)


def _bin_with_edges(x, mask, edges):
    out = np.full(x.shape, -1, dtype=np.int16)
    if edges.size < 2:
        return out
    ok = mask & np.isfinite(x)
    out[ok] = np.digitize(x, edges[1:-1], right=True)[ok].astype(np.int16)
    return out


def _fmt_edges(edges):
    return [f"[{edges[i]:.3f}, {edges[i + 1]:.3f}]" for i in range(len(edges) - 1)]


def _robust_limits(x, mask, lo=2.0, hi=98.0):
    v = x[mask & np.isfinite(x)]
    if v.size == 0:
        return None, None
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def _bin_to_rgba(binned, n_bins):
    cmap = plt.get_cmap("viridis", n_bins).copy(); cmap.set_bad(alpha=0.0)
    return cmap(np.ma.masked_where(binned < 0, binned) / max(1, n_bins - 1))


def _panel_three(out_png, panels):
    plt.figure(figsize=(18, 6))
    for i, (img, t) in enumerate(panels, 1):
        ax = plt.subplot(1, 3, i); ax.imshow(img); ax.set_title(t); ax.axis("off")
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    logger.info("Wrote: %s", out_png)


def main(config: Config) -> None:
    p = config.proxies
    k = config.clustering.k
    scene = resolve_scene(config.project_dir)
    out_dir = scene.outputs_dir / "water_features" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    valid = load_valid_mask(scene.mask)
    water = load_water_mask(scene.outputs_dir / "masks" / "water_mask.tif")
    use = valid & water

    with rasterio.open(scene.reflectance) as ds:
        ngb = visualization.stretch_rgb(io.read_composite(ds, p.ngb_nm), p.percentile_lo, p.percentile_hi)
        ndti = indices.ndti(io.read_band_nm(ds, p.ndti_red_nm), io.read_band_nm(ds, p.ndti_green_nm))
        ndci = indices.ndci(io.read_band_nm(ds, p.ndci_red_edge_nm), io.read_band_nm(ds, p.ndci_red_nm))

    ndti[~use] = np.nan
    ndci[~use] = np.nan
    composite = (p.composite_w_ndti * _zscore(ndti, use)
                 + p.composite_w_ndci * _zscore(ndci, use)).astype(np.float32)
    composite[~use] = np.nan

    ndti_edges = _quantile_edges(ndti, use, p.n_bins)
    ndci_edges = _quantile_edges(ndci, use, p.n_bins)
    ndti_bin = _bin_with_edges(ndti, use, ndti_edges)
    ndci_bin = _bin_with_edges(ndci, use, ndci_edges)

    visualization.save_binned(out_dir / "ndti_binned5_value_ranges.png", ndti_bin, len(ndti_edges) - 1,
                              "NDTI binned (water-only) — bins are value ranges", _fmt_edges(ndti_edges))
    visualization.save_binned(out_dir / "ndci_binned5_value_ranges.png", ndci_bin, len(ndci_edges) - 1,
                              "NDCI binned (water-only) — bins are value ranges", _fmt_edges(ndci_edges))
    visualization.save_heatmap(out_dir / "optical_proxy_composite_labeled.png", composite,
                               "Composite optical proxy (water-only)",
                               cbar_label=f"{p.composite_w_ndti}*z(NDTI) + {p.composite_w_ndci}*z(NDCI)")

    _panel_three(out_dir / "panel_ngb_ndti_ndci_binned.png", [
        (ngb, "NGB (764/549/510)"),
        (_bin_to_rgba(ndti_bin, p.n_bins), "NDTI binned (water-only)"),
        (_bin_to_rgba(ndci_bin, p.n_bins), "NDCI binned (water-only)"),
    ])

    sfa_tif = scene.outputs_dir / "water_features" / "sfa_kmeans" / f"sfa_kmeans_K{k}.tif"
    if not sfa_tif.exists():
        logger.warning("SFA class raster not found (%s); skipping composite panel.", sfa_tif)
        return

    with rasterio.open(sfa_tif) as ds_lab:
        lab = ds_lab.read(1).astype(np.int16)

    plt.figure(figsize=(18, 6))
    ax1 = plt.subplot(1, 3, 1); ax1.imshow(ngb); ax1.set_title("NGB (764/549/510)"); ax1.axis("off")
    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(np.ma.masked_where(lab < 0, lab), cmap=visualization.masked_cmap(), vmin=0, vmax=k - 1)
    ax2.set_title(f"SFA KMeans classes (K={k})"); ax2.axis("off")
    visualization.add_class_legend([f"{i}: {TURBIDITY_LABELS.get(i, '')}" for i in range(k)], ax=ax2)
    ax3 = plt.subplot(1, 3, 3)
    vmin, vmax = _robust_limits(composite, use)
    im = ax3.imshow(np.ma.masked_where(~np.isfinite(composite), composite),
                    cmap=visualization.masked_cmap(), vmin=vmin, vmax=vmax)
    ax3.set_title(f"Composite proxy: {p.composite_w_ndti}*z(NDTI) + {p.composite_w_ndci}*z(NDCI)")
    ax3.axis("off")
    fig = plt.gcf()
    cb = plt.colorbar(im, cax=fig.add_axes([0.92, 0.17, 0.012, 0.66])); cb.set_label("Relative (water-only)")
    plt.tight_layout(rect=[0, 0, 0.91, 1])
    out_png = out_dir / "panel_ngb_classes_composite.png"
    plt.savefig(out_png, dpi=200); plt.close()
    logger.info("Wrote: %s", out_png)


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))