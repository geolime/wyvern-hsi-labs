import numpy as np
import rasterio

from wyvernhsi import io


def test_read_band_nodata_to_nan(tmp_path, write_raster):
    p = write_raster(tmp_path / "r.tif",
                     np.array([[1.0, 2.0], [3.0, 9999.0]], dtype=np.float32), nodata=9999.0)
    with rasterio.open(p) as ds:
        band = io.read_band(ds, 1)
    assert np.isnan(band[1, 1]) and band[0, 0] == 1.0


def test_read_band_nm_resolves_by_wavelength(tmp_path, write_raster):
    cube = np.stack([np.full((2, 2), v, dtype=np.float32) for v in (1, 2, 3)])
    p = write_raster(tmp_path / "c.tif", cube, descriptions=["Band_500", "Band_600", "Band_700"])
    with rasterio.open(p) as ds:
        assert io.band_index_for_nm(ds, 590) == 2  # 1-based
        assert np.all(io.read_band_nm(ds, 690) == 3)


def test_iter_windows_cover_scene(tmp_path, write_raster):
    p = write_raster(tmp_path / "x.tif", np.zeros((10, 10), dtype=np.float32))
    with rasterio.open(p) as ds:
        wins = list(io.iter_windows(ds, 4))
    assert len(wins) == 9
    covered = np.zeros((10, 10), bool)
    for w in wins:
        covered[int(w.row_off):int(w.row_off + w.height),
                int(w.col_off):int(w.col_off + w.width)] = True
    assert covered.all()


def test_write_geotiff_roundtrip(tmp_path, write_raster):
    src = write_raster(tmp_path / "src.tif", np.zeros((4, 4), dtype=np.float32))
    with rasterio.open(src) as ds:
        profile = ds.profile
    arr = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = tmp_path / "out.tif"
    io.write_geotiff(profile, out, arr, nodata=np.nan, dtype="float32", descriptions=["VAL"])
    with rasterio.open(out) as ds:
        assert ds.descriptions[0] == "VAL"
        assert ds.crs == profile["crs"]
        np.testing.assert_array_equal(ds.read(1), arr)