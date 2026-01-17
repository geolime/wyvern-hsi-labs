import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import ListedColormap, BoundaryNorm

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions, pick_band_index_nearest


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

    # rasterio bands are 1-based
    nir = ds.read(idx_nir + 1).astype(np.float32)
    red = ds.read(idx_red + 1).astype(np.float32)
    grn = ds.read(idx_grn + 1).astype(np.float32)

    if ds.nodata is not None:
        nir[nir == ds.nodata] = np.nan
        red[red == ds.nodata] = np.nan
        grn[grn == ds.nodata] = np.nan

    rgb = np.dstack([stretch01(nir), stretch01(red), stretch01(grn)])
    return rgb


def mute_background(rgb, strength=0.6):
    """
    strength 0..1: higher = more muted/desaturated background.
    """
    gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    gray3 = np.dstack([gray, gray, gray])
    return (1 - strength) * rgb + strength * gray3


def build_overlay_rgba(cls):
    """
    Build an RGBA overlay where cls is {-1,0,1,2}.
    -1 is shaded gray so it's obvious.
    """
    overlay = np.zeros((cls.shape[0], cls.shape[1], 4), dtype=np.float32)

    # RGBA colors for overlay (alpha included)
    colors = {
        0: (0.0, 0.6, 0.0, 0.45),   # dense trees
        1: (0.2, 0.9, 0.2, 0.45),   # bright vegetation
        2: (0.8, 0.7, 0.4, 0.45),   # low veg / soil
    }

    # Shade unclassified/nodata so it doesn't look like "just background"
    uncl = (cls == -1)
    overlay[uncl] = (0.15, 0.15, 0.15, 0.50)

    for k, rgba in colors.items():
        mask = (cls == k)
        overlay[mask] = rgba

    return overlay, colors


def save_class_only(cls, out_png):
    """
    Preview A: classification-only image (no CIR). Legend matches exactly.
    """
    # Map values {-1,0,1,2} -> indices {0,1,2,3} for colormap
    cls_idx = np.where(cls == -1, 0, cls + 1).astype(np.uint8)

    cmap = ListedColormap([
        (0.15, 0.15, 0.15, 1.0),  # 0: unclassified
        (0.0,  0.6,  0.0,  1.0),  # 1: dense trees
        (0.2,  0.9,  0.2,  1.0),  # 2: bright vegetation
        (0.8,  0.7,  0.4,  1.0),  # 3: low veg / soil
    ])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    plt.figure(figsize=(14, 10))
    plt.imshow(cls_idx, cmap=cmap, norm=norm, interpolation="nearest")
    plt.axis("off")
    plt.title("SAM classification (classes only)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=(0.15, 0.15, 0.15)),
        plt.Rectangle((0, 0), 1, 1, color=(0.0, 0.6, 0.0)),
        plt.Rectangle((0, 0), 1, 1, color=(0.2, 0.9, 0.2)),
        plt.Rectangle((0, 0), 1, 1, color=(0.8, 0.7, 0.4)),
    ]
    labels = ["Unclassified / nodata", "Dense trees", "Heavy vegetation", "Sparse vegetation / soil"]
    plt.legend(handles, labels, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    cls_path = OUTPUTS_DIR / "sam_fullscene_class.tif"
    out_overlay = OUTPUTS_DIR / "sam_overlay_preview2.png"
    out_class_only = OUTPUTS_DIR / "sam_class_only2.png"
    out_overlay_unmuted = OUTPUTS_DIR / "sam_overlay_preview_unmuted.png"


    # Load CIR background
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        cir = load_cir(ds)

    # Load class raster
    with rasterio.open(cls_path) as ds_cls:
        cls = ds_cls.read(1).astype(np.int16)

    # Build overlay
    overlay, colors = build_overlay_rgba(cls)

    # -------- Preview A: class-only (no background) --------
    save_class_only(cls, out_class_only)

    # -------- Preview B: muted CIR + overlay --------
    cir_muted = mute_background(cir, strength=0.6)

    plt.figure(figsize=(14, 10))
    plt.imshow(cir_muted)
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Wyvern Hyperspectral SAM Classification Overlay (muted CIR)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=(0.15, 0.15, 0.15), alpha=0.50),
        plt.Rectangle((0, 0), 1, 1, color=colors[0][:3], alpha=colors[0][3]),
        plt.Rectangle((0, 0), 1, 1, color=colors[1][:3], alpha=colors[1][3]),
        plt.Rectangle((0, 0), 1, 1, color=colors[2][:3], alpha=colors[2][3]),
    ]
    labels = ["Unclassified / nodata", "Dense trees", "Heavy vegetation", "Sparse vegetation / soil"]
    plt.legend(handles, labels, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_overlay, dpi=200)
    plt.close()

    # -------- Preview C: unmuted CIR + overlay --------
    plt.figure(figsize=(14, 10))
    plt.imshow(cir)          # <--- unmuted CIR
    plt.imshow(overlay)
    plt.axis("off")
    plt.title("Wyvern Hyperspectral SAM Classification Overlay (CIR)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=(0.15, 0.15, 0.15), alpha=0.50),
        plt.Rectangle((0, 0), 1, 1, color=colors[0][:3], alpha=colors[0][3]),
        plt.Rectangle((0, 0), 1, 1, color=colors[1][:3], alpha=colors[1][3]),
        plt.Rectangle((0, 0), 1, 1, color=colors[2][:3], alpha=colors[2][3]),
    ]
    labels = ["Unclassified / nodata", "Dense trees", "Heavy vegetation", "Sparse vegetation / soil"]
    plt.legend(handles, labels, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_overlay_unmuted, dpi=200)
    plt.close()

    print("Wrote:")
    print(" -", out_class_only)
    print(" -", out_overlay)
    print(" -", out_overlay_unmuted)



if __name__ == "__main__":
    main()
