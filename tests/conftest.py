import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin


@pytest.fixture
def write_raster():
    def _write(path, data, *, descriptions=None, nodata=None):
        if data.ndim == 2:
            data = data[None]
        count, h, w = data.shape
        profile = {
            "driver": "GTiff", "height": h, "width": w, "count": count,
            "dtype": str(data.dtype), "crs": CRS.from_epsg(4326),
            "transform": from_origin(0, h, 1, 1), "nodata": nodata,
        }
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)
            if descriptions:
                for i, d in enumerate(descriptions):
                    dst.set_band_description(i + 1, d)
        return path
    return _write