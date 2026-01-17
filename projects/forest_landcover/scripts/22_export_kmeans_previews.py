import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import colormaps

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions, pick_band_index_nearest


KMEANS_TIF = OUTPUTS_DIR / "kmeans_clusters_K5.tif"
K = 5

# simple distinct colors for clusters 0..4 + nodata
# Higher-contrast palette (easy on eyes, distinct)
#COLORS = {
#    -1: (0.10, 0.10, 0.10),  # nodata
#     0: (0.55, 0.35, 0.70),  # purple
#     1: (0.00, 0.45, 0.00),  # forest green
#     2: (0.20, 0.65, 0.85),  # cyan-blue
#     3: (0.95, 0.55, 0.15),  # orange
#     4: (0.55, 0.45, 0.25),  # brown/soil
#}
#COLORS = {
#    -1: (0.05, 0.05, 0.05),  # deep black
#     0: (0.90, 0.40, 0.90),  # neon purple (transition crops)
#     1: (0.00, 0.85, 0.25),  # bright emerald (forest)
#     2: (0.20, 0.85, 1.00),  # cyan (other veg)
#     3: (1.00, 0.60, 0.10),  # neon orange (fields)
#     4: (0.90, 0.80, 0.20),  # yellow-gold (soil/dry veg)
#}

VIRIDIS = colormaps["viridis"]

def viridis_color(k: int, K: int = 5):
    return VIRIDIS(k / (K - 1))[:3]  # RGB only

NODATA_COLOR = VIRIDIS(0.0)[:3]  # dark purple

# Use your current interpretation (update if you change)
LABELS = {
    0: "crop transition",
    1: "forest",
    2: "vegetation (other)",
    3: "crop field",
    4: "soil / dry veg",
}


def stretch01(x, lo=1, hi=99):
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
    out_class = OUTPUTS_DIR / "kmeans_K5_class_only.png"
    out_overlay = OUTPUTS_DIR / "kmeans_K5_overlay_cir.png"

    with rasterio.open(KMEANS_TIF) as ds:
        lab = ds.read(1).astype(np.int16)

    # class-only
    idx = np.where(lab == -1, 0, lab + 1)  # -1->0, 0..4->1..5
    # Build a viridis-based colormap: index 0 is nodata, 1..K map to clusters 0..K-1
    colors = [NODATA_COLOR] + [viridis_color(k, K) for k in range(K)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5] + [i + 0.5 for i in range(0, K + 1)], cmap.N)


    plt.figure(figsize=(14, 10), facecolor="white")
    alpha = np.ones_like(idx, dtype=np.float32)
    alpha[idx == 0] = 0.0  # nodata transparent
    plt.imshow(idx, cmap=cmap, norm=norm, interpolation="nearest", alpha=alpha)

    # Draw boundaries (makes borders obvious)
    try:
        from skimage.segmentation import find_boundaries
        b = find_boundaries(lab, mode="outer")

        # Draw ONLY the boundary pixels (transparent elsewhere)
        outline = np.zeros((lab.shape[0], lab.shape[1], 4), dtype=np.float32)
        outline[b] = (1.0, 1.0, 1.0, 0.8)  # white boundary, alpha only on edges
        plt.imshow(outline)
    except Exception:
        pass

    plt.axis("off")
    plt.title("KMeans clusters (K=5)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=NODATA_COLOR)]
    labels = ["nodata"]
    for k in range(K):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=viridis_color(k, K)))
        labels.append(f"{k}: {LABELS[k]}")
    plt.legend(handles, labels, loc="lower right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out_class, dpi=200)
    plt.close()


    # overlay on CIR
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds_img:
        cir = load_cir(ds_img)

    overlay = np.zeros((lab.shape[0], lab.shape[1], 4), dtype=np.float32)

    # nodata tint
    overlay[lab == -1] = (*NODATA_COLOR, 0.20)

    # clusters
    for k in range(K):
        col = viridis_color(k, K)
        overlay[lab == k] = (*col, 0.55)


    plt.figure(figsize=(14, 10))
    plt.imshow(cir)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("KMeans clusters overlay (CIR)")
    plt.tight_layout()
    plt.savefig(out_overlay, dpi=200)
    plt.close()

    print("Wrote:")
    print(" -", out_class)
    print(" -", out_overlay)

if __name__ == "__main__":
    main()
