import numpy as np

from wyvernhsi import wavelengths


def test_parse_formats():
    descs = ["Band_444", "550 nm", "Band 670nm", None, "no number here"]
    out = wavelengths.parse_wavelengths_nm_from_descriptions(descs)
    np.testing.assert_array_equal(out, np.array([444.0, 550.0, 670.0, np.nan, np.nan]))


def test_pick_nearest():
    wl = np.array([444.0, 550.0, 670.0])
    assert wavelengths.pick_band_index_nearest(wl, 560) == 1
    assert wavelengths.pick_band_index_nearest(wl, 700) == 2


def test_pick_nearest_skips_nan():
    wl = np.array([444.0, np.nan, 670.0])
    assert wavelengths.pick_band_index_nearest(wl, 600) == 2  # 670 is nearest finite


def test_pick_nearest_all_nan_raises():
    import pytest
    with pytest.raises(ValueError):
        wavelengths.pick_band_index_nearest(np.array([np.nan, np.nan]), 500)