import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
import rasterio

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions, pick_band_index_nearest


def stretch01(x: np.ndarray, lo=2, hi=98) -> np.ndarray:
    a = np.nanpercentile(x, lo)
    b = np.nanpercentile(x, hi)
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - a) / (b - a)
    return np.clip(y, 0, 1).astype(np.float32)


def make_cir_image(da, wl_nm: np.ndarray) -> np.ndarray:
    # CIR = (NIR, Red, Green)
    idx_nir = pick_band_index_nearest(wl_nm, 800.0)
    idx_red = pick_band_index_nearest(wl_nm, 660.0)
    idx_grn = pick_band_index_nearest(wl_nm, 560.0)

    nir = np.asarray(da.isel(band=idx_nir).data.compute(), dtype=np.float32)
    red = np.asarray(da.isel(band=idx_red).data.compute(), dtype=np.float32)
    grn = np.asarray(da.isel(band=idx_grn).data.compute(), dtype=np.float32)

    return np.dstack([stretch01(nir), stretch01(red), stretch01(grn)])


def main():
    if not ACTIVE_WYVERN_FILE.exists():
        raise FileNotFoundError(f"Missing local file: {ACTIVE_WYVERN_FILE}")

    src = str(ACTIVE_WYVERN_FILE)

    # Load lazily (dask) so it’s not too heavy
    da = rxr.open_rasterio(src, masked=True, chunks={"x": 1024, "y": 1024})

    # Get wavelengths (nm) from rasterio descriptions
    with rasterio.open(src) as ds:
        desc = list(ds.descriptions)
        height, width = ds.height, ds.width

    wl_nm = parse_wavelengths_nm_from_descriptions(desc)

    cir = make_cir_image(da, wl_nm)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(cir)
    ax.set_title("CIR ROI picker — click to print (row, col). Close window when done.")
    ax.axis("off")

    print("\nClick points in the image window. Output will appear here.")
    print("Format: row col")
    print("-" * 40)

    def onclick(event):
        if event.inaxes != ax:
            return

        # event.xdata/ydata are in image pixel coordinates (col=x, row=y)
        col = int(round(event.xdata))
        row = int(round(event.ydata))

        # bounds check
        if row < 0 or row >= height or col < 0 or col >= width:
            return

        print(f"{row} {col}")

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)


if __name__ == "__main__":
    main()
