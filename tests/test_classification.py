import numpy as np

from wyvernhsi import classification


def test_sam_nearest_and_threshold():
    E = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32)
    tile = np.array([
        [[1, 0, 0], [0, 0, 1]],
        [[1, 1, 0], [np.nan, np.nan, np.nan]],
    ], dtype=np.float32)
    cls, ang = classification.spectral_angle_classify(tile, E, angle_threshold_rad=0.1)
    np.testing.assert_array_equal(cls, [[0, 1], [-1, -1]])  # 45deg pixel over threshold; NaN -> -1
    assert np.isclose(ang[0, 0], 0.0, atol=1e-6)
    assert np.isnan(ang[1, 1])
    assert cls.dtype == np.int16


def test_sam_relaxed_threshold_classifies_45deg():
    E = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32)
    tile = np.array([[[1, 1, 0]]], dtype=np.float32)  # 45deg from E[0]
    cls, ang = classification.spectral_angle_classify(tile, E, angle_threshold_rad=1.0)
    assert cls[0, 0] == 0
    assert np.isclose(ang[0, 0], np.pi / 4, atol=1e-5)