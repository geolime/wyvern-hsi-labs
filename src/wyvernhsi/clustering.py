"""PCA + KMeans over hyperspectral pixels (pure numpy/sklearn; no raster I/O)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score


@dataclass
class PcaKmeansFit:
    pca: PCA
    kmeans: KMeans
    sample_z: np.ndarray       # transformed fit sample (n, pca_components)
    sample_labels: np.ndarray  # KMeans labels of the fit sample (n,)


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation; makes clustering shape-driven, not brightness-driven."""
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norm + eps)


def flatten_valid(cube_yxb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (X (N, B) of all-finite pixels, valid (Y, X) boolean mask locating them)."""
    valid = np.isfinite(cube_yxb).all(axis=2)
    return cube_yxb[valid], valid


def fit_pca_kmeans(
    X: np.ndarray, *, k: int, pca_components: int, n_samples: int, random_state: int
) -> PcaKmeansFit:
    """Fit PCA then KMeans on an L2-normalised random sample of X (rows = pixels)."""
    if X.shape[0] == 0:
        raise ValueError("No valid pixels to fit on.")
    n = min(n_samples, X.shape[0])
    idx = np.random.default_rng(random_state).choice(X.shape[0], size=n, replace=False)
    Xs = l2_normalize_rows(X[idx])

    pca = PCA(n_components=min(pca_components, Xs.shape[1]), random_state=random_state)
    Zs = pca.fit_transform(Xs)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(Zs).astype(np.int32)
    return PcaKmeansFit(pca=pca, kmeans=kmeans, sample_z=Zs, sample_labels=labels)


def predict_tile(tile_yxb: np.ndarray, pca: PCA, kmeans: KMeans) -> np.ndarray:
    """Cluster IDs for one (Y, X, B) tile as int16, with -1 for invalid pixels."""
    X, valid = flatten_valid(tile_yxb)
    out = np.full(tile_yxb.shape[:2], -1, dtype=np.int16)
    if X.shape[0] == 0:
        return out
    Z = pca.transform(l2_normalize_rows(X))
    out[valid] = kmeans.predict(Z).astype(np.int16)
    return out


def ari_stability(Z: np.ndarray, *, k: int, seeds: list) -> np.ndarray:
    """Pairwise adjusted Rand index across KMeans runs with different seeds on the same Z."""
    runs = [KMeans(n_clusters=k, n_init="auto", random_state=s).fit_predict(Z) for s in seeds]
    n = len(seeds)
    ari = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            ari[i, j] = ari[j, i] = adjusted_rand_score(runs[i], runs[j])
    return ari


def cluster_mean_spectra(
    tiles: Iterable[tuple[np.ndarray, np.ndarray]], *, k: int, n_bands: int, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mean spectrum per cluster, accumulated over (cube_yxb, labels_yx) tile pairs.
    Returns (means [k, n_bands] float32 with NaN for empty clusters, counts [k] int64).
    """
    sums = np.zeros((k, n_bands), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    for cube, labels in tiles:
        valid = (labels >= 0) & np.isfinite(cube).all(axis=2)
        if not np.any(valid):
            continue
        X = cube[valid]
        y = labels[valid]
        if normalize:
            X = l2_normalize_rows(X)
        for kk in range(k):
            m = y == kk
            if np.any(m):
                sums[kk] += X[m].sum(axis=0)
                counts[kk] += int(np.sum(m))

    means = np.full((k, n_bands), np.nan, dtype=np.float32)
    nz = counts > 0
    means[nz] = (sums[nz] / counts[nz, None]).astype(np.float32)
    return means, counts