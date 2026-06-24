"""
Unsupervised water-type grouping on interpretable SFA features (water-only):
  NDTI (turbidity proxy), NDCI (chlorophyll proxy), NIR/Red (red-edge strength).

StandardScaler + KMeans, clusters relabeled by median NDTI (0 = clearest, K-1 = most turbid).
Outputs class GeoTIFF/PNG, legend, NDCI ranking, feature stats CSV, scatter, and RGB/NGB
overlays under outputs/water_features/sfa_kmeans/.

NOTE: an unsupervised grouping of optically similar water on TOA-reflectance proxies — NOT a
validated water-quality classification, and NDCI is a proxy, not a chlorophyll concentration.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib import colormaps

from wyvernhsi import clustering, indices, io, visualization
from wyvernhsi.config import Config, load_config
from wyvernhsi.logging_setup import configure_logging
from wyvernhsi.masks import load_valid_mask, load_water_mask
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene

logger = logging.getLogger(__name__)

P_LO, P_HI = 2.0, 98.0
ALPHA = 0.45
SHORELINE_LW = 0.6
SCATTER_SAMPLE = 80_000

# Relabeled by median NDTI: relative optical turbidity tiers (not validated concentrations)
TURBIDITY_LABELS = {
    0: "clearest water (lowest NDTI)",
    1: "low turbidity",
    2: "moderate turbidity",
    3: "turbid water",
    4: "most turbid (highest NDTI)",
}


def _feature_row(c, mk, X):
    return {
        "cluster": c, "n": int(mk.sum()),
        "ndti_median": float(np.median(X[mk, 0])), "ndci_median": float(np.median(X[mk, 1])),
        "nir_red_median": float(np.median(X[mk, 2])),
        "ndti_mean": float(X[mk, 0].mean()), "ndci_mean": float(X[mk, 1].mean()),
        "nir_red_mean": float(X[mk, 2].mean()),
        "ndti_std": float(X[mk, 0].std()), "ndci_std": float(X[mk, 1].std()),
        "nir_red_std": float(X[mk, 2].std()),
    }


def _save_class_map(lab, water, out_png, title, k, *, bg=None, alpha=1.0, transparent=False):
    viridis = colormaps["viridis"]
    cmap = viridis.copy(); cmap.set_bad(alpha=0.0)
    plt.figure(figsize=(12, 10))
    if bg is not None:
        plt.imshow(bg)
    plt.imshow(np.ma.masked_where(lab < 0, lab), cmap=cmap, vmin=0, vmax=k - 1, alpha=alpha)
    plt.contour(water.astype(np.uint8), levels=[0.5], linewidths=SHORELINE_LW)
    plt.axis("off"); plt.title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, color=viridis(i / (k - 1))[:3]) for i in range(k)]
    plt.legend(handles, [f"{i}: {TURBIDITY_LABELS[i]}" for i in range(k)],
               loc="lower right", framealpha=0.9)
    plt.tight_layout(); plt.savefig(out_png, dpi=200, transparent=transparent); plt.close()
    logger.info("Wrote: %s", out_png)


def _save_ndci_continuous(out_png, arr, title):
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return
    vmin, vmax = float(np.percentile(v, 2)), float(np.percentile(v, 98))
    plt.figure(figsize=(10, 8))
    im = plt.imshow(arr, vmin=vmin, vmax=vmax); plt.axis("off"); plt.title(title)
    fig = plt.gcf(); fig.subplots_adjust(right=0.86)
    plt.colorbar(im, cax=fig.add_axes([0.88, 0.12, 0.03, 0.76]))
    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05); plt.close()
    logger.info("Wrote: %s", out_png)


def _save_ngb_ndci(out_png, ngb, ndci, water):
    v = ndci[np.isfinite(ndci) & water]
    if v.size == 0:
        return
    vmin, vmax = float(np.percentile(v, 2)), float(np.percentile(v, 98))
    cmap = plt.get_cmap("viridis").copy(); cmap.set_bad(alpha=0.0)
    plot = np.ma.masked_where(~(np.isfinite(ndci) & water), ndci)
    plt.figure(figsize=(12, 10))
    plt.imshow(ngb)
    plt.imshow(plot, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.55)
    plt.contour(water.astype(np.uint8), levels=[0.5], linewidths=SHORELINE_LW)
    plt.axis("off"); plt.title("NDCI over NGB (water-only)"); plt.tight_layout()
    plt.savefig(out_png, dpi=200); plt.close()
    logger.info("Wrote: %s", out_png)


def main(config: Config) -> None:
    s = config.sfa
    k = config.clustering.k
    seed = config.random_seed
    scene = resolve_scene(config.project_dir)
    out_dir = scene.outputs_dir / "water_features" / "sfa_kmeans"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"sfa_kmeans_K{k}"

    valid = load_valid_mask(scene.mask)
    water = load_water_mask(scene.outputs_dir / "masks" / "water_mask.tif")
    use = valid & water
    logger.info("Masks: valid=%d water=%d use=%d", int(valid.sum()), int(water.sum()), int(use.sum()))

    with rasterio.open(scene.reflectance) as ds:
        profile = ds.profile
        ndti = indices.ndti(io.read_band_nm(ds, s.ndti_red_nm), io.read_band_nm(ds, s.ndti_green_nm))
        ndci = indices.ndci(io.read_band_nm(ds, s.ndci_red_edge_nm), io.read_band_nm(ds, s.ndci_red_nm))
        nirred = indices.ratio(io.read_band_nm(ds, s.nir_red_nir_nm), io.read_band_nm(ds, s.nir_red_red_nm))
        rgb = visualization.stretch_rgb(io.read_composite(ds, s.rgb_nm), P_LO, P_HI)
        ngb = visualization.stretch_rgb(io.read_composite(ds, s.ngb_nm), P_LO, P_HI)

    for arr in (ndti, ndci, nirred):
        arr[~use] = np.nan

    ok = np.isfinite(ndti) & np.isfinite(ndci) & np.isfinite(nirred)
    X = np.stack([ndti[ok], ndci[ok], nirred[ok]], axis=1).astype(np.float32)
    if X.shape[0] == 0:
        raise RuntimeError("No valid water pixels for SFA KMeans.")

    y = clustering.fit_standardized_kmeans(X, k=k, random_state=seed).labels

    lab = np.full(ndti.shape, -1, dtype=np.int16)
    lab[ok] = y

    df = pd.DataFrame([_feature_row(c, y == c, X) for c in range(k) if np.any(y == c)])
    df = df.sort_values("cluster").reset_index(drop=True)

    # Relabel by median NDTI (0 = clearest)
    order = df.sort_values("ndti_median")["cluster"].values
    label_map = {int(old): new for new, old in enumerate(order)}
    logger.info("Relabel (old->new): %s", label_map)
    lab2 = np.full_like(lab, -1)
    for old, new in label_map.items():
        lab2[lab == old] = new
    lab = lab2
    df["cluster_ordered"] = df["cluster"].map(label_map)
    df_out = df.sort_values("cluster_ordered").reset_index(drop=True)

    out_tif = out_dir / f"{prefix}.tif"
    io.write_geotiff(profile, out_tif, lab, nodata=-1, dtype="int16", descriptions=["SFA_WATER_CLASS"])
    with rasterio.open(out_tif, "r+") as dst:
        dst.update_tags(method="kmeans_on_sfa_features_water_only",
                        features="NDTI,NDCI,NIR/Red", kmeans_K=str(k),
                        random_seed=str(seed), relabel="ordered_by_ndti_median (0=clearest)")
    logger.info("Wrote: %s", out_tif)

    legend_lines = ["SFA KMeans water classes (water-only)",
                    "Relabeled by median NDTI: 0=clearest, higher=more turbid", ""]
    legend_lines += [f"{i}: {TURBIDITY_LABELS[i]}" for i in range(k)]
    legend_lines += ["", "Cluster feature medians (ordered):"]
    for _, r in df_out.iterrows():
        legend_lines.append(f"cluster {int(r['cluster_ordered'])}: n={int(r['n'])}, "
                            f"ndti_med={r['ndti_median']:.3f}, ndci_med={r['ndci_median']:.3f}, "
                            f"nir_red_med={r['nir_red_median']:.3f}")
    (out_dir / f"{prefix}_legend.txt").write_text("\n".join(legend_lines), encoding="utf-8")

    rank_lines = ["Clusters ranked by median NDCI (higher = stronger chlorophyll-like optical "
                  "signal; proxy, not concentration)", ""]
    for _, r in df_out.sort_values("ndci_median", ascending=False).iterrows():
        rank_lines.append(f"cluster {int(r['cluster_ordered'])}: ndci_med={r['ndci_median']:.3f}, "
                          f"ndti_med={r['ndti_median']:.3f}, nir_red_med={r['nir_red_median']:.3f}, "
                          f"n={int(r['n'])}")
    (out_dir / f"{prefix}_ndci_ranking.txt").write_text("\n".join(rank_lines), encoding="utf-8")

    df_out.to_csv(out_dir / f"{prefix}_feature_stats.csv", index=False)
    logger.info("Wrote legend, NDCI ranking, feature stats")

    rng = np.random.default_rng(seed)
    sel = rng.choice(X.shape[0], min(SCATTER_SAMPLE, X.shape[0]), replace=False)
    plt.figure(figsize=(7, 6))
    plt.scatter(X[sel, 0], X[sel, 1], s=1, c=y[sel], alpha=0.35)
    plt.xlabel("NDTI"); plt.ylabel("NDCI")
    plt.title("SFA feature space (sample, colored by raw KMeans)")
    plt.tight_layout(); plt.savefig(out_dir / f"{prefix}_scatter_ndti_ndci.png", dpi=200); plt.close()

    _save_class_map(lab, water, out_dir / f"{prefix}.png",
                    f"SFA KMeans classes (K={k}, water-only)", k, transparent=True)
    _save_class_map(lab, water, out_dir / f"rgb_with_{prefix}.png",
                    f"SFA KMeans over RGB (alpha={ALPHA})", k, bg=rgb, alpha=ALPHA)
    _save_class_map(lab, water, out_dir / f"ngb_with_{prefix}.png",
                    f"SFA KMeans over NGB (alpha={ALPHA})", k, bg=ngb, alpha=ALPHA)

    _save_ndci_continuous(out_dir / "ndci_711_669_water_only.png", ndci,
                          "NDCI (711/669) chlorophyll proxy (water-only)")
    _save_ngb_ndci(out_dir / "ngb_with_ndci_711_669.png", ngb, ndci, use)


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))