import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, ACTIVE_WYVERN_MASK, OUTPUTS_DIR
from wyvernhsi.masks import load_valid_mask
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions, pick_band_index_nearest


SAM_TIF = OUTPUTS_DIR / "sam_fullscene_class.tif"

COLORS = {
    -1: (0.92, 0.92, 0.92),   # nodata light gray
     0: (0.10, 0.45, 0.10),   # dark green (trees)
     1: (0.30, 0.75, 0.30),   # bright green (veg)
     2: (0.80, 0.70, 0.50),   # tan soil
}
LABELS = {
    -1: "nodata",
     0: "trees",
     1: "vegetation",
     2: "soil",
}

def stretch01(x, lo=2, hi=98):
    a = np.nanpercentile(x, lo)
    b = np.nanpercentile(x, hi)
    y = (x - a) / (b - a)
    return np.clip(y, 0, 1)

def load_cir(ds):
    wl = parse_wavelengths_nm_from_descriptions(list(ds.descriptions))
    nir = ds.read(pick_band_index_nearest(wl, 800) + 1)
    red = ds.read(pick_band_index_nearest(wl, 660) + 1)
    grn = ds.read(pick_band_index_nearest(wl, 560) + 1)
    gamma = 0.85
    return np.dstack([
        stretch01(nir) ** gamma,
        stretch01(red) ** gamma,
        stretch01(grn) ** gamma,
    ])

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    out_class = OUTPUTS_DIR / "sam_class_only.png"
    out_overlay = OUTPUTS_DIR / "sam_overlay.png"


    with rasterio.open(SAM_TIF) as ds:
        cls = ds.read(1).astype(np.int16)

    # Mask clouds (and/or clear mask rule depending on your masks.py)
    valid = load_valid_mask(ACTIVE_WYVERN_MASK)
    cls[~valid] = -1

    # class-only
    idx = np.where(cls == -1, 0, cls + 1)  # -1->0, 0..2->1..3
    cmap = ListedColormap([COLORS[-1], COLORS[0], COLORS[1], COLORS[2]])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    plt.figure(figsize=(14, 10))
    plt.imshow(idx, cmap=cmap, norm=norm, interpolation="nearest")
    plt.axis("off")
    plt.title("SAM classification - 3 classes")
    handles = []
    for k in sorted(LABELS.keys()):
        handles.append(mpatches.Patch(color=COLORS[k], label=LABELS[k])
        )
    plt.legend(handles=handles,loc="lower right",frameon=True,framealpha=0.9,fontsize=9)
    plt.tight_layout()
    plt.savefig(out_class, dpi=200)
    plt.close()

    # overlay on CIR
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds_img:
        cir = load_cir(ds_img)

    overlay = np.zeros((cls.shape[0], cls.shape[1], 4), dtype=np.float32)
    overlay[cls == -1] = (*COLORS[-1], 0.20)
    for k in [0, 1, 2]:
        overlay[cls == k] = (*COLORS[k], 0.55)


    plt.figure(figsize=(14, 10))
    plt.imshow(cir)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("SAM overlay (CIR)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in [-1, 0, 1, 2]]
    labels = [LABELS[k] for k in [-1, 0, 1, 2]]
    plt.legend(handles, labels, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_overlay, dpi=200)
    plt.close()

    print("Wrote:")
    print(" -", out_class)
    print(" -", out_overlay)

if __name__ == "__main__":
    main()
