import numpy as np

from wyvernhsi import visualization


def test_class_colors_count():
    cols = visualization.class_colors(5)
    assert len(cols) == 5 and all(len(c) == 3 for c in cols)


def test_masked_cmap_transparent_bad():
    assert visualization.masked_cmap().get_bad()[3] == 0.0


def test_percentile_stretch_range_and_nan():
    x = np.array([[0.0, 1.0, 2.0, np.nan]], dtype=np.float32)
    out = visualization.percentile_stretch(x, 0, 100)
    assert out.min() == 0.0 and out.max() == 1.0 and out[0, 3] == 0.0


def test_save_binned_writes_file(tmp_path):
    binned = np.array([[0, 1, -1], [2, 0, 1]], dtype=np.int16)
    out = tmp_path / "b.png"
    visualization.save_binned(out, binned, 3, "t", ["Q1", "Q2", "Q3"])
    assert out.exists() and out.stat().st_size > 0