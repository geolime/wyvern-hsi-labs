"""
Unsupervised water-type grouping on 4 engineered water features (water-only):
  NDTI (turbidity), NDCI (chlorophyll), R660/G570, RE711/R669.

StandardScaler + KMeans, relabeled by median NDTI (0 = clearest). Reports KMeans ARI
stability, writes a class GeoTIFF/PNG, NDTI/NDCI proxy maps, per-cluster feature stats,
and median bar charts under outputs/water_features/.

NOTE: unsupervised grouping on TOA-reflectance optical proxies — NOT a validated
water-quality classification; NDTI/NDCI are proxies, not concentrations.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from wyvernhsi import clustering, indices, io, visualization
from wyvernhsi.config import Config, load_config
from wyvernhsi.logging_setup import configure_logging
from wyvernhsi.masks import load_valid_mask, load_water_mask
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene

logger = logging.getLogger(__name__)

FEATURES = ["ndti", "ndci", "rg", "re_r"]
TURBIDITY_LABELS = {
    0: "clearest water (lowest NDTI)",
    1: "low turbidity",
    2: "moderate turbidity",
    3: "turbid water",
    4: "most turbid (highest NDTI)",
}

def _feature_row(c, mk, X):
    row = {"cluster": c, "n": int(mk.sum())}
    for j, nm in enumerate(FEATURES):
        row[f"{nm}_mean"] = float(np.mean(X[mk, j]))
        row[f"{nm}_median"] = float(np.median(X[mk, j]))
        row[f"{nm}_std"] = float(np.std(X[mk, j]))
    return row


def _save_class_png(lab, out_png, title, k):
    plt.figure(figsize=(12, 10))
    plt.imshow(np.ma.masked_where(lab < 0, lab), vmin=0, vmax=k - 1, cmap=visualization.masked_cmap())
    plt.axis("off")
    plt.title(title)
    visualization.add_class_legend([f"{i}: {TURBIDITY_LABELS[i]}" for i in range(k)])
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, transparent=True)
    plt.close()
    logger.info("Wrote: %s", out_png)


def main(config: Config) -> None:
    s = config.sfa
    cl = config.clustering
    k, seed = cl.k, config.random_seed
    scene = resolve_scene(config.project_dir)
    out_dir = scene.outputs_dir / "water_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"water_features_kmeans_K{k}"

    valid = load_valid_mask(scene.mask)
    water = load_water_mask(scene.outputs_dir / "masks" / "water_mask.tif")
    use = valid & water

    with rasterio.open(scene.reflectance) as ds:
        profile = ds.profile
        g = io.read_band_nm(ds, s.ndti_green_nm)
        r = io.read_band_nm(ds, s.ndti_red_nm)
        r665 = io.read_band_nm(ds, s.ndci_red_nm)
        re = io.read_band_nm(ds, s.ndci_red_edge_nm)

    ndti = indices.ndti(r, g)
    ndci = indices.ndci(re, r665)
    F = np.stack([ndti, ndci, indices.ratio(r, g), indices.ratio(re, r665)], axis=2).astype(np.float32)
    F[~use] = np.nan

    X, valid2d = clustering.flatten_valid(F)
    if X.shape[0] == 0:
        raise RuntimeError("No valid water pixels after masks.")

    fit = clustering.fit_standardized_kmeans(X, k=k, random_state=seed)
    lab = np.full(F.shape[:2], -1, dtype=np.int16)
    lab[valid2d] = fit.labels

    seeds = [seed + i * 17 for i in range(cl.stability_runs)]
    ari = clustering.ari_stability(fit.scaled, k=k, seeds=seeds)
    mean_ari = float((ari.sum() - np.trace(ari)) / (ari.size - len(seeds)))
    logger.info("Water-feature clustering stability: mean ARI = %.4f", mean_ari)

    df = pd.DataFrame([_feature_row(c, fit.labels == c, X) for c in range(k) if np.any(fit.labels == c)])
    df = df.sort_values("cluster").reset_index(drop=True)

    order = df.sort_values("ndti_median")["cluster"].values
    label_map = {int(old): new for new, old in enumerate(order)}
    logger.info("Relabel (old->new): %s", label_map)
    lab2 = np.full_like(lab, -1)
    for old, new in label_map.items():
        lab2[lab == old] = new
    lab = lab2

    out_tif = out_dir / f"{prefix}.tif"
    io.write_geotiff(profile, out_tif, lab, nodata=-1, dtype="int16", descriptions=["WATER_FEATURE_CLASS"])
    with rasterio.open(out_tif, "r+") as dst:
        dst.update_tags(method="kmeans_on_water_features", features="NDTI,NDCI,R/G,RE/R",
                        kmeans_K=str(k), random_seed=str(seed), mean_ari=str(mean_ari),
                        relabel="ordered_by_ndti_median (0=clearest)")
    logger.info("Wrote: %s", out_tif)

    df["cluster_ordered"] = df["cluster"].map(label_map)
    df_out = df.sort_values("cluster_ordered").reset_index(drop=True)
    df_out.to_csv(out_dir / f"{prefix}_cluster_feature_stats.csv", index=False)

    legend = ["Cluster legend (ordered by turbidity, median NDTI; 0 = clearest)", ""]
    legend += [f"{i}: {TURBIDITY_LABELS[i]}" for i in range(k)]
    legend += ["", "Cluster feature medians (ordered):"]
    for _, row in df_out.iterrows():
        legend.append(f"cluster {int(row['cluster_ordered'])}: n={int(row['n'])}, "
                      f"ndti_med={row['ndti_median']:.3f}, ndci_med={row['ndci_median']:.3f}")
    (out_dir / f"{prefix}_legend.txt").write_text("\n".join(legend), encoding="utf-8")

    ndti_w, ndci_w = np.where(use, ndti, np.nan), np.where(use, ndci, np.nan)
    m = np.isfinite(ndti_w) & np.isfinite(ndci_w)
    if m.any():
        logger.info("Corr(NDTI, NDCI) on water pixels: %.4f",
                    float(np.corrcoef(ndti_w[m], ndci_w[m])[0, 1]))
    visualization.save_heatmap(out_dir / "ndti_map.png", ndti_w, "NDTI (turbidity proxy)")
    visualization.save_heatmap(out_dir / "ndci_map.png", ndci_w, "NDCI (chlorophyll proxy)")

    _save_class_png(lab, out_dir / f"{prefix}.png",
                    f"KMeans water classes (ordered by turbidity) — K={k}", k)

    for feat, title in (("ndti_median", "Median NDTI (turbidity proxy)"),
                        ("ndci_median", "Median NDCI (chlorophyll proxy)")):
        plt.figure(figsize=(8, 4))
        plt.bar(df_out["cluster_ordered"].astype(int).astype(str), df_out[feat])
        plt.xlabel("Cluster (ordered)")
        plt.ylabel(title.split("(")[0].strip())
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_{feat.replace('_median', '')}_median.png", dpi=200)
        plt.close()
    logger.info("Wrote feature maps, class map, bar charts")


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))