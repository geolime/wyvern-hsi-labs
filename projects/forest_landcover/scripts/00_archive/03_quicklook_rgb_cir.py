from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def make_rgb(da, wl_nm: np.ndarray, targets_nm: tuple[float, float, float]) -> np.ndarray:
    idx_r = pick_band_index_nearest(wl_nm, targets_nm[0])
    idx_g = pick_band_index_nearest(wl_nm, targets_nm[1])
    idx_b = pick_band_index_nearest(wl_nm, targets_nm[2])

    r = np.asarray(da.isel(band=idx_r).data.compute())
    g = np.asarray(da.isel(band=idx_g).data.compute())
    b = np.asarray(da.isel(band=idx_b).data.compute())

    return np.dstack([stretch01(r), stretch01(g), stretch01(b)])


def save_png(img: np.ndarray, out_path: Path, title: str):
    plt.figure()
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    if not ACTIVE_WYVERN_FILE.exists():
        raise FileNotFoundError(
            f"Expected local file at {ACTIVE_WYVERN_FILE}. "
            f"Put the downloaded Wyvern GeoTIFF in your data/ folder (or update src/wyvernhsi/paths.py)."
        )

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    src = str(ACTIVE_WYVERN_FILE)
    print(f"Using local file: {ACTIVE_WYVERN_FILE}")

    # Data (lazy)
    da = rxr.open_rasterio(
        src,
        masked=True,
        chunks={"x": 1024, "y": 1024},
    )

    # Metadata (reliable)
    with rasterio.open(src) as ds:
        desc = list(ds.descriptions)

    wl_nm = parse_wavelengths_nm_from_descriptions(desc)

    rgb = make_rgb(da, wl_nm, (660.0, 560.0, 490.0))
    save_png(rgb, OUTPUTS_DIR / "rgb.png", "RGB (660, 560, 490 nm)")

    cir = make_rgb(da, wl_nm, (800.0, 660.0, 560.0))
    save_png(cir, OUTPUTS_DIR / "cir.png", "CIR  (800, 660, 560 nm)")

    print("Wrote:")
    print(" -", OUTPUTS_DIR / "rgb.png")
    print(" -", OUTPUTS_DIR / "cir.png")


if __name__ == "__main__":
    main()
