import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions, pick_band_index_nearest


FINAL_TIF = OUTPUTS_DIR / "final_hybrid_3class.tif"

# Class colors (match earlier)
COLORS = {
    -1: (0.15, 0.15, 0.15),
     0: (0.00, 0.55, 0.00),   # trees
     1: (0.40, 0.80, 0.20),   # vegetation
     2: (0.80, 0.70, 0.45),   # soil
}

LABELS = {
    -1: "nodata",
     0: "trees",
     1: "vegetation",
     2: "soil"
}


def stretch01(x, lo=2, hi=98):
    a = np.nanpercentile(x, lo)
    b = np.nanpercentile(x, hi)
    y = (x - a) / (b - a)
    return np.clip(y, 0, 1)


def load_rgb(ds):
    wl = parse_wavelengths_nm_from_descriptions(list(ds.descriptions))
    r = ds.read(pick_band_index_nearest(wl, 660) + 1)
    g = ds.read(pick_band_index_nearest(wl, 560) + 1)
    b = ds.read(pick_band_index_nearest(wl, 490) + 1)
    return np.dstack([stretch01(r), stretch01(g), stretch01(b)])


def load_cir(ds):
    wl = parse_wavelengths_nm_from_descriptions(list(ds.descriptions))
    nir = ds.read(pick_band_index_nearest(wl, 800) + 1)
    red = ds.read(pick_band_index_nearest(wl, 660) + 1)
    grn = ds.read(pick_band_index_nearest(wl, 560) + 1)
    return np.dstack([stretch01(nir), stretch01(red), stretch01(grn)])


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    out_class = OUTPUTS_DIR / "final_class_only.png"
    out_rgb = OUTPUTS_DIR / "final_overlay_rgb.png"
    out_cir = OUTPUTS_DIR / "final_overlay_cir.png"

    with rasterio.open(FINAL_TIF) as ds:
        cls = ds.read(1).astype(np.int16)

    # ----- Class-only -----
    idx = np.where(cls == -1, 0, cls + 1)

    cmap = ListedColormap([
        COLORS[-1],
        COLORS[0],
        COLORS[1],
        COLORS[2],
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    plt.figure(figsize=(14, 10))
    plt.imshow(idx, cmap=cmap, norm=norm, interpolation="nearest")
    plt.axis("off")
    plt.title("Final hybrid land cover map")

    handles = []
    labels = []
    for k in [-1, 0, 1, 2]:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=COLORS[k]))
        labels.append(LABELS[k])

    plt.legend(handles, labels, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_class, dpi=200)
    plt.close()

    # ----- Overlays -----
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds_img:
        rgb = load_rgb(ds_img)
        cir = load_cir(ds_img)

    overlay = np.zeros((cls.shape[0], cls.shape[1], 4), dtype=np.float32)
    for k, col in COLORS.items():
        overlay[cls == k] = (col[0], col[1], col[2], 0.45)

    # RGB overlay
    plt.figure(figsize=(14, 10))
    plt.imshow(rgb)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Final hybrid classification (RGB)")
    plt.tight_layout()
    plt.savefig(out_rgb, dpi=200)
    plt.close()

    # CIR overlay
    plt.figure(figsize=(14, 10))
    plt.imshow(cir)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Final hybrid classification (CIR)")
    plt.tight_layout()
    plt.savefig(out_cir, dpi=200)
    plt.close()

    print("Wrote:")
    print(" -", out_class)
    print(" -", out_rgb)
    print(" -", out_cir)


if __name__ == "__main__":
    main()
