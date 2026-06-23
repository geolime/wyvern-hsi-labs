import numpy as np
import pytest

from wyvernhsi import clustering


def test_l2_normalize_rows():
    out = clustering.l2_normalize_rows(np.array([[3.0, 4.0]]))
    np.testing.assert_allclose(out, [[0.6, 0.8]], atol=1e-6)


def test_flatten_valid_drops_nan_pixels():
    cube = np.ones((2, 2, 3), dtype=np.float32)
    cube[0, 0, :] = np.nan
    X, valid = clustering.flatten_valid(cube)
    assert X.shape == (3, 3) and valid.sum() == 3


def test_fit_is_deterministic_and_predicts():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.1, (200, 4)), rng.normal(5, 0.1, (200, 4))]).astype(np.float32)
    p1, k1 = clustering.fit_pca_kmeans(X, k=2, pca_components=2, n_samples=400, random_state=42)
    p2, k2 = clustering.fit_pca_kmeans(X, k=2, pca_components=2, n_samples=400, random_state=42)
    np.testing.assert_allclose(k1.cluster_centers_, k2.cluster_centers_)

    tile = X[:4].reshape(2, 2, 4).copy()
    tile[0, 0, :] = np.nan
    labels = clustering.predict_tile(tile, p1, k1)
    assert labels.dtype == np.int16 and labels[0, 0] == -1
    assert set(np.unique(labels)).issubset({-1, 0, 1})


def test_fit_empty_raises():
    with pytest.raises(ValueError):
        clustering.fit_pca_kmeans(np.empty((0, 4)), k=2, pca_components=2, n_samples=10, random_state=0)


def test_cluster_mean_spectra():
    cube = np.array([[[1, 2, 3], [1, 2, 3]], [[4, 5, 6], [0, 0, 0]]], dtype=np.float32)
    labels = np.array([[0, 0], [1, -1]], dtype=np.int16)
    means, counts = clustering.cluster_mean_spectra([(cube, labels)], k=2, n_bands=3, normalize=False)
    np.testing.assert_array_equal(counts, [2, 1])
    np.testing.assert_allclose(means[0], [1, 2, 3])
    np.testing.assert_allclose(means[1], [4, 5, 6])