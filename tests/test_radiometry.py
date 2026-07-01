from datetime import datetime

import numpy as np
import pytest

from wyvernhsi import radiometry


def test_sun_earth_distance_known_doy():
    # DOY 4 -> cos(0) -> 1 - 0.01672
    d = radiometry.sun_earth_distance_au(datetime(2025, 1, 4))
    assert abs(d - 0.98328) < 1e-5


def test_toa_reflectance_hand_computed():
    radiance = np.array([[[10.0]]], dtype=np.float32)  # (1 band, 1, 1)
    esun = np.array([100.0], dtype=np.float32)
    out = radiometry.toa_radiance_to_reflectance(radiance, esun, sun_elevation_deg=90.0,
                                                 earth_sun_distance_au=1.0)
    assert np.isclose(out[0, 0, 0], 10.0 * np.pi / 100.0, atol=1e-5)


def test_toa_reflectance_preconditions():
    esun = np.array([100.0], dtype=np.float32)
    with pytest.raises(ValueError):  # not 3D
        radiometry.toa_radiance_to_reflectance(np.zeros((1, 1)), esun, 90.0, 1.0)
    with pytest.raises(ValueError):  # band count mismatch
        radiometry.toa_radiance_to_reflectance(np.zeros((1, 1, 1)), np.array([1.0, 2.0]), 90.0, 1.0)
    with pytest.raises(ValueError):  # sun at horizon
        radiometry.toa_radiance_to_reflectance(np.ones((1, 1, 1)), esun, 0.0, 1.0)


def test_replace_nodata_with_nan():
    arr = np.array([1.0, 2.0, 9999.0], dtype=np.float32)
    out = radiometry.replace_nodata_with_nan(arr, 9999.0)
    assert np.isnan(out[2]) and out[0] == 1.0
    np.testing.assert_array_equal(radiometry.replace_nodata_with_nan(arr, None), arr)