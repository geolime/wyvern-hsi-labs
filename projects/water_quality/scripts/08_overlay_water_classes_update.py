from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from matplotlib import colormaps

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR
from wyvernhsi.masks import load_water_mask


# -------- Settings --------
# RGB (true color): R=660 (12), G=549 (5), B=510 (2)
BANDS_RGB = {"R": 12, "G": 5, "B": 2}

# NGB (water-friendly false color): R=764 (21), G=549 (5), B=510 (2)
BANDS_NGB = {"R": 21, "G": 5, "B": 2}

P_LO, P_HI = 2.0, 98.0      # stretch for visualization
ALPHA = 0.45                # overlay transparency

# Styling
DRAW_SHORELINE = True
DRAW_BOUNDARIES = False
BOUNDARY_LINEWIDTH = 0.35
SHORELINE_LINEWIDTH = 0.2

K = 5
OUT_PREFIX = "water_features_kmeans_K5"   # must match your clustering output name

# Ordered-by-turbidity meanings (after relabeling)
LABELS = {
    0: "clear / deep water",
    1: "low turbidity (transitional)",
    2: "moderate turbidity",
    3: "algae-enhanced / optically active",
    4: "high turbidity / sediment plume",
}


# -------- Helpers --------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_band(ds: rasterio.DatasetReader, b1: int) -> np.ndarray:
    x = ds.read(b1).astype(np.float32)
    if ds.nodata is not None:
        x[x == ds.nodata] = np.nan
    return x


def percentile_stretch01(x: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    v = x[np.isfinite(x)]
    if v.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = np.percentile(v, p_lo)
    hi = np.percentile(v, p_hi)
    y = (x - lo) / (hi - lo + 1e-12)
    return np.clip(y, 0, 1).astype(np.float32)


def build_composite(ds, bands: dict[str, int]) -> np.ndarray:
    r = read_band(ds, bands["R"])
    g = read_band(ds, bands["G"])
    b = read_band(ds, bands["B"])

    rgb = np.dstack([
        percentile_stretch01(r, P_LO, P_HI),
        percentile_stretch01(g, P_LO, P_HI),
        percentile_stretch01(b, P_LO, P_HI),
    ])

    # nodata -> black; land preserved
    return np.nan_to_num(rgb, nan=0.0)


def compute_boundaries(lab: np.ndarray) -> np.ndarray:
    """
    Return a boolean boundary mask where neighboring pixels have different labels.
    Nodata (-1) does not create boundaries.
    """
    m = lab >= 0

    # Compare with 4-neighborhood shifts
    diff = np.zeros_like(m, dtype=bool)

    # up/down
    diff |= m & np.roll(m, 1, axis=0) & (lab != np.roll(lab, 1, axis=0))
    diff |= m & np.roll(m, -1, axis=0) & (lab != np.roll(lab, -1, axis=0))
    # left/right
    diff |= m & np.roll(m, 1, axis=1) & (lab != np.roll(lab, 1, axis=1))
    diff |= m & np.roll(m, -1, axis=1) & (lab != np.roll(lab, -1, axis=1))

    # Zero-out wraparound artifacts at edges
    diff[0, :] = False
    diff[-1, :] = False
    diff[:, 0] = False
    diff[:, -1] = False

    return diff


def add_legend(K: int):
    VIRIDIS = colormaps["viridis"]

    def viridis_color(k: int, K: int):
        return VIRIDIS(k / (K - 1))[:3]

    handles, labels = [], []
    for k in range(K):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=viridis_color(k, K)))
        labels.append(f"{k}: {LABELS[k]}")
    plt.legend(handles, labels, loc="lower right", framealpha=0.9)


def main() -> None:
    ensure_dir(OUTPUTS_DIR)
    out_dir = OUTPUTS_DIR / "water_features"
    ensure_dir(out_dir)

    labels_tif = out_dir / f"{OUT_PREFIX}.tif"
    if not labels_tif.exists():
        raise FileNotFoundError(f"Missing labels GeoTIFF: {labels_tif}")

    # ---- Load water mask (for shoreline outline) ----
    water_mask = load_water_mask().astype(bool)

    # ---- Load background composites ----
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        rgb = build_composite(ds, BANDS_RGB)
        ngb = build_composite(ds, BANDS_NGB)

    # ---- Load labels ----
    with rasterio.open(labels_tif) as ds_lab:
        lab = ds_lab.read(1).astype(np.int16)

    # Transparent nodata in overlay
    lab_plot = np.ma.masked_where(lab < 0, lab)

    # Colormap with transparent nodata
    VIRIDIS = colormaps["viridis"]
    cmap = VIRIDIS.copy()
    cmap.set_bad(alpha=0.0)

    # Precompute boundaries for drawing
    boundaries = compute_boundaries(lab) if DRAW_BOUNDARIES else None

    # ---- Plot helper ----
    def plot_overlay(bg: np.ndarray, title: str, out_path: Path):
        plt.figure(figsize=(12, 10))
        plt.imshow(bg * 0.5)

        # Cluster overlay (nodata is transparent)
        plt.imshow(lab_plot, cmap=cmap, vmin=0, vmax=K - 1, alpha=ALPHA)

        # Cluster boundaries (thin)
        if DRAW_BOUNDARIES and boundaries is not None:
            # contour expects numeric; draw boundary line where True
            plt.contour(
                boundaries.astype(np.uint8),
                levels=[0.5],
                linewidths=BOUNDARY_LINEWIDTH,
            )

        # Shoreline outline from water mask
        if DRAW_SHORELINE:
            plt.contour(
                water_mask.astype(np.uint8),
                levels=[0.5],
                linewidths=SHORELINE_LINEWIDTH,
            )

        plt.axis("off")
        plt.title(title)
        add_legend(K)

        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print("Wrote:", out_path)

    # ---- Clean class-only map (transparent background) ----
    def plot_class_only(title: str, out_path: Path):
        plt.figure(figsize=(12, 10))

        # Plot classes only; make background transparent
        # Use masked array so nodata becomes transparent
        plt.imshow(lab_plot, cmap=cmap, vmin=0, vmax=K - 1, alpha=1.0)

        if DRAW_BOUNDARIES and boundaries is not None:
            plt.contour(
                boundaries.astype(np.uint8),
                levels=[0.5],
                linewidths=BOUNDARY_LINEWIDTH,
            )

        if DRAW_SHORELINE:
            plt.contour(
                water_mask.astype(np.uint8),
                levels=[0.5],
                linewidths=SHORELINE_LINEWIDTH,
            )

        plt.axis("off")
        plt.title(title)
        add_legend(K)

        plt.tight_layout()
        # IMPORTANT: transparent=True so land/background is transparent
        plt.savefig(out_path, dpi=200, transparent=True)
        plt.close()
        print("Wrote:", out_path)

    # ---- Export overlays ----
    plot_overlay(
        rgb,
        f"Water classes over RGB (alpha={ALPHA})",
        out_dir / "rgb_with_water_classes.png",
    )

    plot_overlay(
        ngb,
        f"Water classes over NGB composite (alpha={ALPHA})",
        out_dir / "ngb_with_water_classes.png",
    )

    plot_class_only(
        "Water classes (transparent background)",
        out_dir / "water_classes_only.png",
    )


if __name__ == "__main__":
    main()
