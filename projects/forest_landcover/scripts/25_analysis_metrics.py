import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

from scipy.stats import ttest_ind

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR, ACTIVE_WYVERN_MASK
from wyvernhsi.masks import load_valid_mask
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions, pick_band_index_nearest


# -------- Settings --------
K = 5
PCA_N = 8

# Sampling sizes (keep these reasonable for speed)
SAMPLE_FIT = 250_000       # pixels used to fit PCA + KMeans for analysis
SAMPLE_SIL = 80_000        # pixels used for silhouette score
STABILITY_RUNS = 6         # number of kmeans runs for ARI stability

RANDOM_SEED = 42


def stretch01(x, lo=2, hi=98):
    a = np.nanpercentile(x, lo)
    b = np.nanpercentile(x, hi)
    y = (x - a) / (b - a + 1e-12)
    return np.clip(y, 0, 1)


def l2_normalize_rows(X, eps=1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + eps)


def flatten_valid(cube, valid_mask_2d):
    """
    cube: (H,W,B), valid_mask_2d: (H,W) bool
    Returns X: (N,B), idx: (N,) flattened indices
    """
    m = valid_mask_2d & np.all(np.isfinite(cube), axis=2)
    X = cube[m].astype(np.float32)
    idx = np.flatnonzero(m.ravel())
    return X, idx


def read_cube_full(ds):
    arr = ds.read().astype(np.float32)              # (B,H,W)
    arr = np.transpose(arr, (1, 2, 0))              # (H,W,B)
    return arr


def welch_t_and_d(a, b):
    # Welch t-test + Cohen's d (using pooled-ish denom for effect size)
    t, p = ttest_ind(a, b, equal_var=False, nan_policy="omit")
    ma, mb = np.nanmean(a), np.nanmean(b)
    sa, sb = np.nanstd(a), np.nanstd(b)
    d = (ma - mb) / (np.sqrt((sa * sa + sb * sb) / 2.0) + 1e-12)
    return float(t), float(p), float(d)


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load hyperspectral cube ---
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        wl = parse_wavelengths_nm_from_descriptions(list(ds.descriptions))
        cube = read_cube_full(ds)  # (H,W,B)

    # --- QA mask: keep valid pixels (cloud-only / clear-only depending on your masks.py) ---
    valid = load_valid_mask(ACTIVE_WYVERN_MASK)
    cube[~valid] = np.nan

    # --- Flatten valid pixels ---
    X, idx = flatten_valid(cube, valid)
    print("Valid spectra:", X.shape)

    # --- Sample for analysis PCA/KMeans ---
    rng = np.random.default_rng(RANDOM_SEED)
    n = X.shape[0]
    take = min(SAMPLE_FIT, n)
    sel = rng.choice(n, size=take, replace=False)
    Xs = X[sel]
    Xs = l2_normalize_rows(Xs)

    # --- Fit PCA ---
    pca = PCA(n_components=PCA_N, random_state=RANDOM_SEED)
    Zs = pca.fit_transform(Xs)

    # Scree plot
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, PCA_N + 1), evr, marker="o")
    plt.xlabel("PC")
    plt.ylabel("Explained variance ratio")
    plt.title("PCA explained variance (scree)")
    plt.tight_layout()
    out_scree = OUTPUTS_DIR / "pca_scree.png"
    plt.savefig(out_scree, dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, PCA_N + 1), cum, marker="o")
    plt.xlabel("PC")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA cumulative explained variance")
    plt.tight_layout()
    out_cum = OUTPUTS_DIR / "pca_cumulative.png"
    plt.savefig(out_cum, dpi=200)
    plt.close()

    # --- Silhouette score (sampled) ---
    # Fit one reference KMeans model (analysis-only, not replacing your pipeline outputs)
    km_ref = KMeans(n_clusters=K, n_init="auto", random_state=RANDOM_SEED)
    y_ref = km_ref.fit_predict(Zs)

    sil_take = min(SAMPLE_SIL, Zs.shape[0])
    sil_sel = rng.choice(Zs.shape[0], size=sil_take, replace=False)
    sil = silhouette_score(Zs[sil_sel], y_ref[sil_sel], metric="euclidean")
    print("Silhouette (sampled):", sil)

    # --- Cluster proportions from your existing output raster (preferred) ---
    # This reflects the actual spatial result you show in README.
    with rasterio.open(OUTPUTS_DIR / f"kmeans_clusters_K{K}.tif") as ds_lab:
        lab = ds_lab.read(1).astype(np.int16)

    u, c = np.unique(lab, return_counts=True)
    total = lab.size
    props = []
    for uu, cc in zip(u.tolist(), c.tolist()):
        props.append({"label": int(uu), "count": int(cc), "fraction": float(cc / total)})

    df_props = pd.DataFrame(props).sort_values("label")
    out_props = OUTPUTS_DIR / f"kmeans_cluster_proportions_K{K}.csv"
    df_props.to_csv(out_props, index=False)

    # Bar chart
    df_plot = df_props[df_props["label"] >= 0].copy()
    plt.figure(figsize=(7, 4))
    plt.bar(df_plot["label"].astype(str), df_plot["fraction"])
    plt.xlabel("Cluster")
    plt.ylabel("Fraction of scene")
    plt.title("KMeans cluster proportions")
    plt.tight_layout()
    out_prop_png = OUTPUTS_DIR / f"kmeans_cluster_proportions_K{K}.png"
    plt.savefig(out_prop_png, dpi=200)
    plt.close()

    # --- KMeans stability across seeds (ARI on same sampled PCA features) ---
    # Use Zs (same data) and compare labelings across different random seeds.
    ys = []
    seeds = [RANDOM_SEED + i * 17 for i in range(STABILITY_RUNS)]
    for s in seeds:
        km = KMeans(n_clusters=K, n_init="auto", random_state=s)
        ys.append(km.fit_predict(Zs))

    ari = np.zeros((STABILITY_RUNS, STABILITY_RUNS), dtype=np.float32)
    for i in range(STABILITY_RUNS):
        for j in range(STABILITY_RUNS):
            ari[i, j] = adjusted_rand_score(ys[i], ys[j])

    df_ari = pd.DataFrame(ari, index=[f"seed_{s}" for s in seeds], columns=[f"seed_{s}" for s in seeds])
    out_ari_csv = OUTPUTS_DIR / f"kmeans_stability_ari_K{K}.csv"
    df_ari.to_csv(out_ari_csv)

    plt.figure(figsize=(6, 5))
    plt.imshow(ari, interpolation="nearest")
    plt.xticks(range(STABILITY_RUNS), [str(s) for s in seeds], rotation=45, ha="right")
    plt.yticks(range(STABILITY_RUNS), [str(s) for s in seeds])
    plt.title("KMeans stability (ARI)")
    plt.colorbar()
    plt.tight_layout()
    out_ari_png = OUTPUTS_DIR / f"kmeans_stability_ari_K{K}.png"
    plt.savefig(out_ari_png, dpi=200)
    plt.close()

    # --- PCA RGB composite (PC1/PC2/PC3) for the full scene ---
    # Fit PCA on sampled normalized spectra above, then transform full scene in chunks.
    H, W, B = cube.shape
    pcs = np.full((H, W, 3), np.nan, dtype=np.float32)

    flat = cube.reshape(-1, B)
    valid_flat = np.all(np.isfinite(flat), axis=1)

    Xv = flat[valid_flat].astype(np.float32)
    Xv = l2_normalize_rows(Xv)

    Zv = pca.transform(Xv)[:, :3].astype(np.float32)

    pcs_flat = pcs.reshape(-1, 3)
    pcs_flat[valid_flat] = Zv

    # Stretch each PC independently and stack to RGB
    pc1 = stretch01(pcs[:, :, 0], lo=2, hi=98)
    pc2 = stretch01(pcs[:, :, 1], lo=2, hi=98)
    pc3 = stretch01(pcs[:, :, 2], lo=2, hi=98)
    pca_rgb = np.dstack([pc1, pc2, pc3])

    plt.figure(figsize=(14, 10))
    plt.imshow(pca_rgb)
    plt.axis("off")
    plt.title("PCA composite (PC1, PC2, PC3)")
    plt.tight_layout()
    out_pca_rgb = OUTPUTS_DIR / "pca_rgb_pc123.png"
    plt.savefig(out_pca_rgb, dpi=200)
    plt.close()

    # --- Index separability tests for key clusters (NDVI + red-edge slope) ---
    # Use the same indices you computed earlier (computed here to keep script standalone)
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        b_red = pick_band_index_nearest(wl, 660) + 1
        b_re  = pick_band_index_nearest(wl, 720) + 1
        b_nir = pick_band_index_nearest(wl, 800) + 1
        red = ds.read(b_red).astype(np.float32)
        re  = ds.read(b_re).astype(np.float32)
        nir = ds.read(b_nir).astype(np.float32)

    red[~valid] = np.nan
    re[~valid] = np.nan
    nir[~valid] = np.nan

    ndvi = (nir - red) / (nir + red + 1e-6)
    re_slope = (re - red) / (720 - 660)

    # Compare a few pairs that usually matter most
    pairs = [
        (1, 4, "forest_vs_soil"),       # adjust if your forest cluster differs
        (3, 4, "highveg_vs_soil"),
        (1, 3, "forest_vs_highveg"),
    ]

    rows = []
    for a, b, name in pairs:
        ma = (lab == a)
        mb = (lab == b)
        # sample to keep it light
        na = np.flatnonzero(ma.ravel())
        nb = np.flatnonzero(mb.ravel())
        if len(na) == 0 or len(nb) == 0:
            continue
        sa = rng.choice(na, size=min(60_000, len(na)), replace=False)
        sb = rng.choice(nb, size=min(60_000, len(nb)), replace=False)

        nd_a = ndvi.ravel()[sa]
        nd_b = ndvi.ravel()[sb]
        rs_a = re_slope.ravel()[sa]
        rs_b = re_slope.ravel()[sb]

        t_nd, p_nd, d_nd = welch_t_and_d(nd_a, nd_b)
        t_rs, p_rs, d_rs = welch_t_and_d(rs_a, rs_b)

        rows.append({
            "pair": name,
            "cluster_a": a,
            "cluster_b": b,
            "ndvi_t": t_nd,
            "ndvi_p": p_nd,
            "ndvi_cohens_d": d_nd,
            "re_slope_t": t_rs,
            "re_slope_p": p_rs,
            "re_slope_cohens_d": d_rs,
        })

    df_sep = pd.DataFrame(rows)
    out_sep = OUTPUTS_DIR / f"index_separability_tests_K{K}.csv"
    df_sep.to_csv(out_sep, index=False)

    # --- Summary text file (easy to paste into README) ---
    out_txt = OUTPUTS_DIR / "analysis_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"PCA components: {PCA_N}\n")
        f.write(f"Explained variance ratio: {evr.tolist()}\n")
        f.write(f"Cumulative explained variance: {cum.tolist()}\n")
        f.write(f"Silhouette score (sampled): {sil}\n")
        f.write(f"ARI stability runs: {STABILITY_RUNS}\n")
        f.write(f"ARI mean off-diagonal: {float((ari.sum() - np.trace(ari)) / (ari.size - STABILITY_RUNS))}\n")

    print("Wrote:")
    print(" -", out_scree)
    print(" -", out_cum)
    print(" -", out_pca_rgb)
    print(" -", out_props)
    print(" -", out_prop_png)
    print(" -", out_ari_csv)
    print(" -", out_ari_png)
    print(" -", out_sep)
    print(" -", out_txt)


if __name__ == "__main__":
    main()
