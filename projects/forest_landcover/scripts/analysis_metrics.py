"""
Forest clustering analysis metrics: PCA scree/cumulative, sampled silhouette, KMeans ARI
stability, cluster proportions, a PC1/2/3 composite, and NDVI / red-edge-slope separability
(Welch t-test + Cohen's d) between cluster pairs.

NOTE: separability quantifies how spectrally distinct the clusters are — it is NOT external
validation against ground truth. Cluster IDs are arbitrary; pairs are reported by ID with
each cluster's median NDVI for interpretation, not hardcoded land-cover labels.
"""
from __future__ import annotations

import logging
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import ttest_ind
from sklearn.metrics import silhouette_score

from wyvernhsi import clustering, indices, io, visualization
from wyvernhsi.config import Config, load_config
from wyvernhsi.logging_setup import configure_logging
from wyvernhsi.masks import load_valid_mask
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene

logger = logging.getLogger(__name__)

SEP_SAMPLE = 60_000  # pixels per cluster for separability tests


def _welch_t_and_d(a, b):
    t, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
    sa, sb = np.nanstd(a), np.nanstd(b)
    d = (np.nanmean(a) - np.nanmean(b)) / (np.sqrt((sa * sa + sb * sb) / 2.0) + 1e-12)
    return float(t), float(p), float(d)


def main(config: Config) -> None:
    cl = config.clustering
    f = config.features
    seed = config.random_seed
    scene = resolve_scene(config.project_dir)
    out = scene.outputs_dir
    out.mkdir(parents=True, exist_ok=True)

    valid = load_valid_mask(scene.mask)
    with rasterio.open(scene.reflectance) as ds:
        cube = io.read_cube(ds)
    cube[~valid] = np.nan

    X, _ = clustering.flatten_valid(cube)
    logger.info("Valid spectra: %s", X.shape)

    fit = clustering.fit_pca_kmeans(X, k=cl.k, pca_components=cl.pca_components,
                                    n_samples=cl.n_samples, random_state=seed)
    pca, Zs, y_ref = fit.pca, fit.sample_z, fit.sample_labels

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    xs = np.arange(1, len(evr) + 1)
    for ys, ylabel, title, name in (
        (evr, "Explained variance ratio", "PCA explained variance (scree)", "pca_scree.png"),
        (cum, "Cumulative explained variance", "PCA cumulative explained variance", "pca_cumulative.png"),
    ):
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, marker="o")
        plt.xlabel("PC")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out / name, dpi=200)
        plt.close()

    rng = np.random.default_rng(seed)
    sil_sel = rng.choice(Zs.shape[0], min(cl.sample_silhouette, Zs.shape[0]), replace=False)
    sil = float(silhouette_score(Zs[sil_sel], y_ref[sil_sel]))
    logger.info("Silhouette (sampled): %.4f", sil)

    with rasterio.open(out / f"kmeans_clusters_K{cl.k}.tif") as ds_lab:
        lab = ds_lab.read(1).astype(np.int16)
    u, c = np.unique(lab, return_counts=True)
    df_props = pd.DataFrame([{"label": int(uu), "count": int(cc), "fraction": float(cc / lab.size)}
                             for uu, cc in zip(u.tolist(), c.tolist())]).sort_values("label")
    df_props.to_csv(out / f"kmeans_cluster_proportions_K{cl.k}.csv", index=False)
    dfp = df_props[df_props["label"] >= 0]
    plt.figure(figsize=(7, 4))
    plt.bar(dfp["label"].astype(str), dfp["fraction"])
    plt.xlabel("Cluster")
    plt.ylabel("Fraction of scene")
    plt.title("KMeans cluster proportions")
    plt.tight_layout()
    plt.savefig(out / f"kmeans_cluster_proportions_K{cl.k}.png", dpi=200)
    plt.close()

    seeds = [seed + i * 17 for i in range(cl.stability_runs)]
    ari = clustering.ari_stability(Zs, k=cl.k, seeds=seeds)
    pd.DataFrame(ari, index=[f"seed_{s}" for s in seeds], columns=[f"seed_{s}" for s in seeds]) \
        .to_csv(out / f"kmeans_stability_ari_K{cl.k}.csv")
    plt.figure(figsize=(6, 5))
    plt.imshow(ari, interpolation="nearest")
    plt.xticks(range(len(seeds)), [str(s) for s in seeds], rotation=45, ha="right")
    plt.yticks(range(len(seeds)), [str(s) for s in seeds])
    plt.title("KMeans stability (ARI)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out / f"kmeans_stability_ari_K{cl.k}.png", dpi=200)
    plt.close()

    # PCA PC1/2/3 composite (whole valid scene; cube already in memory)
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)
    vflat = np.isfinite(flat).all(axis=1)
    Zv = pca.transform(clustering.l2_normalize_rows(flat[vflat]))[:, :3].astype(np.float32)
    pcs = np.full((H * W, 3), np.nan, dtype=np.float32)
    pcs[vflat] = Zv
    pcs = pcs.reshape(H, W, 3)
    pca_rgb = np.dstack([visualization.percentile_stretch(pcs[:, :, i]) for i in range(3)])
    plt.figure(figsize=(14, 10))
    plt.imshow(pca_rgb)
    plt.axis("off")
    plt.title("PCA composite (PC1, PC2, PC3)")
    plt.tight_layout()
    plt.savefig(out / "pca_rgb_pc123.png", dpi=200)
    plt.close()

    # NDVI + red-edge slope separability between all cluster pairs (no hardcoded IDs)
    with rasterio.open(scene.reflectance) as ds:
        red = io.read_band_nm(ds, f.red_nm)
        red_edge = io.read_band_nm(ds, f.red_edge_nm)
        nir = io.read_band_nm(ds, f.nir_nm)
    for arr in (red, red_edge, nir):
        arr[~valid] = np.nan
    ndvi = indices.ndvi(nir, red)
    re_slope = indices.red_edge_slope(red_edge, red, f.red_edge_nm, f.red_nm)

    ndvi_flat, rs_flat, lab_flat = ndvi.ravel(), re_slope.ravel(), lab.ravel()
    clusters = sorted(int(k) for k in np.unique(lab) if k >= 0)
    median_ndvi = {k: float(np.nanmedian(ndvi_flat[lab_flat == k])) for k in clusters}

    rows = []
    for a, b in combinations(clusters, 2):
        ia, ib = np.flatnonzero(lab_flat == a), np.flatnonzero(lab_flat == b)
        if ia.size == 0 or ib.size == 0:
            continue
        sa = rng.choice(ia, min(SEP_SAMPLE, ia.size), replace=False)
        sb = rng.choice(ib, min(SEP_SAMPLE, ib.size), replace=False)
        t_nd, p_nd, d_nd = _welch_t_and_d(ndvi_flat[sa], ndvi_flat[sb])
        t_rs, p_rs, d_rs = _welch_t_and_d(rs_flat[sa], rs_flat[sb])
        rows.append({"cluster_a": a, "cluster_b": b,
                     "ndvi_median_a": median_ndvi[a], "ndvi_median_b": median_ndvi[b],
                     "ndvi_t": t_nd, "ndvi_p": p_nd, "ndvi_cohens_d": d_nd,
                     "re_slope_t": t_rs, "re_slope_p": p_rs, "re_slope_cohens_d": d_rs})
    pd.DataFrame(rows).to_csv(out / f"index_separability_tests_K{cl.k}.csv", index=False)

    ari_off = float((ari.sum() - np.trace(ari)) / (ari.size - len(seeds)))
    (out / "analysis_summary.txt").write_text(
        f"PCA components: {pca.n_components_}\n"
        f"Explained variance ratio: {evr.tolist()}\n"
        f"Cumulative explained variance: {cum.tolist()}\n"
        f"Silhouette score (sampled): {sil}\n"
        f"ARI stability runs: {len(seeds)}\n"
        f"ARI mean off-diagonal: {ari_off}\n"
        f"Cluster median NDVI: {median_ndvi}\n",
        encoding="utf-8",
    )
    logger.info("Wrote analysis metrics (scree, silhouette, ARI, proportions, PC123, separability, summary)")


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))