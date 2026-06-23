import numpy as np
import pytest

from wyvernhsi import masks


def test_load_valid_mask(tmp_path, write_raster):
    stack = np.stack([
        np.array([[1, 1, 1]], dtype=np.uint8),  # QA_CLEAR_MASK
        np.array([[0, 1, 0]], dtype=np.uint8),  # QA_CLOUD_MASK
        np.array([[0, 0, 0]], dtype=np.uint8),  # QA_HAZE_MASK
        np.array([[0, 0, 0]], dtype=np.uint8),  # QA_CLOUD_SHADOW_MASK
    ])
    p = write_raster(tmp_path / "m.tif", stack,
                     descriptions=["QA_CLEAR_MASK", "QA_CLOUD_MASK", "QA_HAZE_MASK", "QA_CLOUD_SHADOW_MASK"])
    np.testing.assert_array_equal(masks.load_valid_mask(p), [[True, False, True]])


def test_load_water_mask(tmp_path, write_raster):
    p = write_raster(tmp_path / "w.tif", np.array([[0, 1, 1]], dtype=np.uint8))
    np.testing.assert_array_equal(masks.load_water_mask(p), [[False, True, True]])


def test_load_water_mask_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        masks.load_water_mask(tmp_path / "nope.tif")