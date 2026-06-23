import numpy as np

from wyvernhsi import indices


def test_normalized_difference_basic_and_zero_denominator():
    a = np.array([0.4, 0.0], dtype=np.float32)
    b = np.array([0.2, 0.0], dtype=np.float32)
    out = indices.normalized_difference(a, b)
    assert np.isclose(out[0], 0.2 / 0.6, atol=1e-6)
    assert np.isnan(out[1])  # 0/0 -> NaN, never inf


def test_normalized_difference_propagates_nan():
    out = indices.normalized_difference(np.array([np.nan]), np.array([0.1]))
    assert np.isnan(out[0])


def test_ndvi_is_normalized_difference():
    nir = np.array([0.5], dtype=np.float32)
    red = np.array([0.1], dtype=np.float32)
    np.testing.assert_array_equal(indices.ndvi(nir, red), indices.normalized_difference(nir, red))


def test_red_edge_slope():
    out = indices.red_edge_slope(np.array([0.5]), np.array([0.3]), 720.0, 660.0)
    assert np.isclose(out[0], 0.2 / 60.0, atol=1e-6)
    assert np.isnan(indices.red_edge_slope(np.array([np.nan]), np.array([0.3]), 720.0, 660.0)[0])