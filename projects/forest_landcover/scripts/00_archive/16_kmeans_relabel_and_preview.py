import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions, pick_band_index_nearest


# --- Inputs ---
CLUSTERS_TIF = OUTPUTS_DIR / "kmeans_clusters_K5.tif"

# --- Your interpretation ---
# cluster_id -> (new_class_id, label)
# We'll keep 0..4 as class ids (same as cluster ids) but attach labels.
LABELS = {
    0: "field_type_A",
    1: "field_type_B",
    2: "dense_forest",
    3: "bare_soil",
    4: "mixed_veg_transition",
}

# RGBA colors for class-only + overlay
# (Choose anything you like; these are readable defaults)
COLORS = {
    -1: (0.15, 0.15, 0.15, 1.0),  # nodata/unclassified
    0:  (0.90, 0.75, 0.20, 1.0),  # field A
    1:  (0.60, 0.85, 0.20, 1.0),  # field B
    2:  (0.00, 0.50, 0.00, 1.0),  # dense forest
    3:  (0.80, 0.70, 0.45, 1.0),  # bare soil
    4:  (0.20, 0.70, 0.70, 1.0),  # mixed transition
}


def stretch01(x, lo=2, hi=98):
    a = np.nanpercentile(x, lo)
    b = np.nanpercentile(x, hi)
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - a) / (b - a)
    return np.clip(y, 0, 1).astype(np.float32)


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

    return np.dstack([stretch01(nir), stretch01(red), stretch01(grn)])


def build_overlay(cls):
    overlay = np.zeros((cls.shape[0], cls.shape[1], 4), dtype=np.float32)
    for k, rgba in COLORS.items():
        if k == -1:
            continue
        overlay[cls == k] = (rgba[0], rgba[1], rgba[2], 0.45)  # overlay alpha
    # nodata shaded lightly
    overlay[cls == -1] = (0.15, 0.15, 0.15, 0.40)
    return overlay


def save_class_only(cls, out_png):
    # Map values {-1,0..4} -> indices {0..5}
    cls_idx = np.where(cls == -1, 0, cls + 1).astype(np.uint8)

    cmap = ListedColormap([
        COLORS[-1],  # index 0
        COLORS[0],
        COLORS[1],
        COLORS[2],
        COLORS[3],
        COLORS[4],
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)

    plt.figure(figsize=(14, 10))
    plt.imshow(cls_idx, cmap=cmap, norm=norm, interpolation="nearest")
    plt.axis("off")
    plt.title("KMeans semantic map (classes only)")

    handles = []
    labels = []
    handles.append(plt.Rectangle((0, 0), 1, 1, color=COLORS[-1][:3]))
    labels.append("nodata")

    for k in range(5):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=COLORS[k][:3]))
        labels.append(f"{k}: {LABELS[k]}")

    plt.legend(handles, labels, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    out_tif = OUTPUTS_DIR / "kmeans_semantic_K5.tif"
    out_class_png = OUTPUTS_DIR / "kmeans_semantic_class_only.png"
    out_overlay_png = OUTPUTS_DIR / "kmeans_semantic_overlay.png"

    # Read clusters
    with rasterio.open(CLUSTERS_TIF) as ds:
        clusters = ds.read(1).astype(np.int16)
        profile = ds.profile.copy()

    # Here we keep IDs as-is (0..4), but you could remap if desired.
    cls = clusters.copy()

    # Write semantic GeoTIFF
    profile.update(count=1, dtype="int16", nodata=-1, compress="deflate", predictor=2)
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(cls, 1)
        dst.update_tags(semantic_labels=";".join([f"{k}:{LABELS[k]}" for k in range(5)]) + ";-1:nodata")

    # Preview A: class-only
    save_class_only(cls, out_class_png)

    # Preview B: CIR overlay
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds_img:
        cir = load_cir(ds_img)

    overlay = build_overlay(cls)

    plt.figure(figsize=(14, 10))
    plt.imshow(cir)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("KMeans semantic overlay (CIR background)")

    handles = []
    labels = []
    handles.append(plt.Rectangle((0, 0), 1, 1, color=COLORS[-1][:3], alpha=0.40))
    labels.append("nodata")

    for k in range(5):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=COLORS[k][:3], alpha=0.45))
        labels.append(f"{k}: {LABELS[k]}")

    plt.legend(handles, labels, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_overlay_png, dpi=200)
    plt.close()

    print("Wrote:")
    print(" -", out_tif)
    print(" -", out_class_png)
    print(" -", out_overlay_png)


if __name__ == "__main__":
    main()
