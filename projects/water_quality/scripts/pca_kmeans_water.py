"""
Unsupervised water-type grouping: PCA + KMeans over water-only TOA reflectance.

Fits PCA->KMeans on sampled water pixels, predicts the water-only scene, and produces
diagnostics: PCA scree/cumulative, sampled silhouette, KMeans ARI stability, a class map,
PC1/2/3 composite, per-cluster brightness/slope proxies, mean spectra, and a summary.

NOTE: an unsupervised grouping of optically similar water, NOT a validated classification.
Products are TOA reflectance (no atmospheric correction).
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.metrics import silhouette_score

from wyvernhsi import clustering, io, visualization
from wyvernhsi.config import Config, load_config
from wyvernhsi.logging_setup import configure_logging
from wyvernhsi.masks import load_valid_mask, load_water_mask
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions, pick_band_index_nearest

logger = logging.getLogger(__name__)


def _save_line(xs, ys, ylabel, title, out_png):
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("PC"); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def _save_boxplot(values_per_cluster, ylabel, title, out_png):
    plt.figure(figsize=(10, 4))
    plt.boxplot(values_per_cluster,
                tick_labels=[str(k) for k in range(len(values_per_cluster))], showfliers=False)
    plt.xlabel("Cluster"); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()


def main(config: Config) -> None:
    cl = config.clustering
    seed = config.random_seed
    scene = resolve_scene(config.project_dir)
    out = scene.outputs_dir
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"water_kmeans_K{cl.k}_PCA{cl.pca_components}"

    valid_mask = load_valid_mask(scene.mask)
    water_mask = load_water_mask(out / "masks" / "water_mask.tif")
    use_mask = valid_mask & water_mask

    with rasterio.open(scene.reflectance) as ds:
        wl_nm = parse_wavelengths_nm_from_descriptions(list(ds.descriptions))
        profile = ds.profile
        cube = io.read_cube(ds)  # whole scene, read ONCE and reused throughout

    cube[~use_mask] = np.nan  # water-only
    X, _ = clustering.flatten_valid(cube)
    if X.shape[0] == 0:
        raise RuntimeError("No valid water pixels after masks. Check water-mask / QA overlap.")

    fit = clustering.fit_pca_kmeans(X, k=cl.k, pca_components=cl.pca_components,
                                    n_samples=cl.n_samples, random_state=seed)
    pca, Zs, y_ref = fit.pca, fit.sample_z, fit.sample_labels
    logger.info("Fit: samples=%d, bands=%d, PCA=%d, K=%d",
                Zs.shape[0], cube.shape[2], pca.n_components_, cl.k)

    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    pcs_x = np.arange(1, len(evr) + 1)
    _save_line(pcs_x, evr, "Explained variance ratio",
               "PCA explained variance (water-only)", out / f"{prefix}_pca_scree.png")
    _save_line(pcs_x, cum, "Cumulative explained variance",
               "PCA cumulative explained variance (water-only)", out / f"{prefix}_pca_cumulative.png")

    rng = np.random.default_rng(seed)
    sil_sel = rng.choice(Zs.shape[0], min(cl.sample_silhouette, Zs.shape[0]), replace=False)
    sil = float(silhouette_score(Zs[sil_sel], y_ref[sil_sel]))
    logger.info("Silhouette (sampled): %.4f", sil)

    seeds = [seed + i * 17 for i in range(cl.stability_runs)]
    ari = clustering.ari_stability(Zs, k=cl.k, seeds=seeds)
    plt.figure(figsize=(6, 5))
    plt.imshow(ari, interpolation="nearest")
    plt.xticks(range(len(seeds)), [str(s) for s in seeds], rotation=45, ha="right")
    plt.yticks(range(len(seeds)), [str(s) for s in seeds])
    plt.title("KMeans stability (ARI) — water-only"); plt.colorbar(); plt.tight_layout()
    plt.savefig(out / f"{prefix}_stability_ari.png", dpi=200); plt.close()

    # Whole water-only prediction (per-pixel, so identical to tiling)
    lab = clustering.predict_tile(cube, pca, fit.kmeans)
    out_tif = out / f"{prefix}.tif"
    io.write_geotiff(profile, out_tif, lab, nodata=-1, dtype="int16", descriptions=["WATER_CLUSTER"])
    with rasterio.open(out_tif, "r+") as dst:
        dst.update_tags(kmeans_K=str(cl.k), pca_components=str(pca.n_components_),
                        samples=str(Zs.shape[0]), mask="valid&water",
                        random_seed=str(seed), silhouette=str(sil))
    logger.info("Wrote: %s", out_tif)

    plt.figure(figsize=(12, 10))
    plt.imshow(lab, vmin=0, vmax=cl.k - 1); plt.axis("off")
    plt.title(f"KMeans clusters (water-only) — K={cl.k}"); plt.tight_layout()
    plt.savefig(out / f"{prefix}.png", dpi=200); plt.close()

    # Per-cluster proxies (bands resolved by wavelength, not hardcoded indices)
    keep = (lab >= 0) & np.isfinite(cube).all(axis=2)
    Xk, yk = cube[keep], lab[keep]
    bright = np.mean(Xk, axis=1)
    b_lo = pick_band_index_nearest(wl_nm, 510.0)
    b_hi = pick_band_index_nearest(wl_nm, 660.0)
    slope = (Xk[:, b_hi] - Xk[:, b_lo]) / (660.0 - 510.0)
    _save_boxplot([bright[yk == k] for k in range(cl.k)], "Mean TOA reflectance (all bands)",
                  "Cluster brightness proxy (water-only)", out / f"{prefix}_cluster_brightness.png")
    _save_boxplot([slope[yk == k] for k in range(cl.k)], "Slope (R660 - R510) / 150 nm",
                  "Cluster red-blue slope proxy (water-only)", out / f"{prefix}_cluster_slope.png")

    # PC1/2/3 composite (water-only)
    H, W, B = cube.shape
    flat = cube.reshape(-1, B)
    use_flat = np.isfinite(flat).all(axis=1)
    Zv = pca.transform(clustering.l2_normalize_rows(flat[use_flat]))[:, :3].astype(np.float32)
    pcs = np.full((H * W, 3), np.nan, dtype=np.float32)
    pcs[use_flat] = Zv
    pcs = pcs.reshape(H, W, 3)
    rgb = np.dstack([visualization.percentile_stretch(pcs[:, :, i]) for i in range(3)])
    plt.figure(figsize=(14, 10)); plt.imshow(rgb); plt.axis("off")
    plt.title("PCA composite (PC1, PC2, PC3) — water-only"); plt.tight_layout()
    plt.savefig(out / f"{prefix}_pca_pc123.png", dpi=200); plt.close()

    # Mean spectra per cluster (in-memory; cube already loaded)
    means, counts = clustering.cluster_mean_spectra([(cube, lab)], k=cl.k, n_bands=B, normalize=True)
    plt.figure(figsize=(12, 7))
    for k in range(cl.k):
        if np.isfinite(means[k]).any():
            plt.plot(wl_nm, means[k], label=f"cluster {k} (n={counts[k]})")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("L2-normalized TOA reflectance")  # corrected from "radiance"
    plt.title(f"Cluster mean spectra (water-only) — K={cl.k}, PCA={cl.pca_components}")
    plt.legend(); plt.tight_layout()
    plt.savefig(out / f"{prefix}_cluster_mean_spectra.png", dpi=200); plt.close()

    ari_offdiag = float((ari.sum() - np.trace(ari)) / (ari.size - len(seeds)))
    (out / f"{prefix}_summary.txt").write_text(
        f"K: {cl.k}\nPCA_N: {pca.n_components_}\nSamples_fit: {Zs.shape[0]}\n"
        f"Silhouette_sampled: {sil}\nExplained_variance_ratio: {evr.tolist()}\n"
        f"Cumulative_explained_variance: {cum.tolist()}\n"
        f"ARI_runs: {len(seeds)}\nARI_mean_offdiag: {ari_offdiag}\n",
        encoding="utf-8",
    )
    logger.info("Wrote: %s", out / f"{prefix}_summary.txt")


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))