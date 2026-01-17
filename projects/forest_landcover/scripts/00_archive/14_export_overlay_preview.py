import numpy as np
import matplotlib.pyplot as plt
import rasterio

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions, pick_band_index_nearest


def stretch01(x, lo=2, hi=98):
    a = np.nanpercentile(x, lo)
    b = np.nanpercentile(x, hi)
    y = (x - a) / (b - a)
    return np.clip(y, 0, 1)


def load_cir(ds):
    wl = parse_wavelengths_nm_from_descriptions(list(ds.descriptions))

    idx_nir = pick_band_index_nearest(wl, 800)
    idx_red = pick_band_index_nearest(wl, 660)
    idx_grn = pick_band_index_nearest(wl, 560)

    nir = ds.read(idx_nir + 1).astype(np.float32)
    red = ds.read(idx_red + 1).astype(np.float32)
    grn = ds.read(idx_grn + 1).astype(np.float32)

    if ds.nodata is not None:
        nir[nir == ds.nodata] = np.nan
        red[red == ds.nodata] = np.nan
        grn[grn == ds.nodata] = np.nan

    rgb = np.dstack([
        stretch01(nir),
        stretch01(red),
        stretch01(grn),
    ])

    return rgb


def main():
    cls_path = OUTPUTS_DIR / "sam_fullscene_class.tif"
    out_png = OUTPUTS_DIR / "sam_overlay_preview.png"

    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        cir = load_cir(ds)

    with rasterio.open(cls_path) as ds_cls:
        cls = ds_cls.read(1)

    # Create transparent overlay
    overlay = np.zeros((cls.shape[0], cls.shape[1], 4), dtype=np.float32)

    # Shade unclassified so it's visually distinct
    uncl = (cls == -1)
    overlay[uncl] = (0.15, 0.15, 0.15, 0.5)  # dark gray mask

    # Color table (RGBA)
    colors = {
        0: (0.0, 0.6, 0.0, 0.45),   # dense trees
        1: (0.2, 0.9, 0.2, 0.45),   # sparse vegetation
        2: (0.8, 0.7, 0.4, 0.45),   # soil
    }


    for k, rgba in colors.items():
        mask = cls == k
        overlay[mask] = rgba

    plt.figure(figsize=(14, 10))
    plt.imshow(cir)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Wyvern Hyperspectral SAM Classification Overlay")

    legend_labels = [
        "Dense trees",
        "Bright vegetation",
        "Low vegetation / soil"
    ]

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[i][:3], alpha=colors[i][3])
        for i in range(3)
    ]
    handles.insert(0, plt.Rectangle((0, 0), 1, 1, color=(0.2,0.2,0.2), alpha=0.25))
    legend_labels.insert(0, "Unclassified / nodata")


    plt.legend(handles, legend_labels, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    

    print("Wrote:", out_png)

    plt.figure(figsize=(14, 10))
    plt.imshow(cls)
    plt.title("SAM classification (raw labels)")
    plt.axis("off")
    plt.savefig(OUTPUTS_DIR / "sam_class_only.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    main()
