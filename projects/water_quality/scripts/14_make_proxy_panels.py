from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import rasterio

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, ACTIVE_WYVERN_MASK, OUTPUTS_DIR
from wyvernhsi.masks import load_valid_mask, load_water_mask

# ---------------- Settings ----------------
# Bands (1-based)
B510 = 2
B549 = 5
B570 = 6
B660 = 12
B669 = 13
B711 = 17
B764 = 21

BANDS_NGB = {"R": 21, "G": 5, "B": 2}  # 764/549/510

P_LO, P_HI = 2.0, 98.0
EPS = 1e-6

N_BINS = 5

# Script 10 output (if present)
SFA_KMEANS_TIF = OUTPUTS_DIR / "water_features" / "sfa_kmeans" / "sfa_kmeans_K5.tif"

# Class labels (should match your script 10)
CLASS_LABELS = {
    0: "clear / deep water",
    1: "low turbidity (transitional)",
    2: "moderate turbidity",
    3: "turbid water",
    4: "high turbidity / sediment plume",
}

OUT_DIR = OUTPUTS_DIR / "water_features" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Helpers ----------------
def read_band(ds: rasterio.DatasetReader, b1: int) -> np.ndarray:
    x = ds.read(b1).astype(np.float32)
    if ds.nodata is not None:
        x[x == ds.nodata] = np.nan
    return x


def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float32)
    denom = a + b
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(denom) > EPS)
    out[ok] = (a[ok] - b[ok]) / denom[ok]
    return out


def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float32)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > EPS)
    out[ok] = a[ok] / b[ok]
    return out


def percentile_stretch01(x: np.ndarray, p_lo: float, p_hi: float) -> np.ndarray:
    v = x[np.isfinite(x)]
    if v.size == 0:
        return np.zeros_like(x, dtype=np.float32)
    lo = np.percentile(v, p_lo)
    hi = np.percentile(v, p_hi)
    y = (x - lo) / (hi - lo + 1e-12)
    return np.clip(y, 0, 1).astype(np.float32)


def build_composite(ds: rasterio.DatasetReader, bands: dict[str, int]) -> np.ndarray:
    r = read_band(ds, bands["R"])
    g = read_band(ds, bands["G"])
    b = read_band(ds, bands["B"])
    rgb = np.dstack([
        percentile_stretch01(r, P_LO, P_HI),
        percentile_stretch01(g, P_LO, P_HI),
        percentile_stretch01(b, P_LO, P_HI),
    ])
    return np.nan_to_num(rgb, nan=0.0)


def quantile_edges(x: np.ndarray, mask: np.ndarray, n_bins: int) -> np.ndarray:
    v = x[mask & np.isfinite(x)]
    if v.size == 0:
        return np.array([], dtype=np.float32)
    edges = np.quantile(v, np.linspace(0, 1, n_bins + 1))
    # enforce monotonicity
    edges = np.maximum.accumulate(edges)
    return edges.astype(np.float32)


def bin_with_edges(x: np.ndarray, mask: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    edges length = n_bins+1
    returns int16 bin indices 0..n_bins-1; -1 elsewhere
    """
    out = np.full(x.shape, -1, dtype=np.int16)
    if edges.size < 2:
        return out
    # bins determined by edges[1:-1]
    idx = np.digitize(x, edges[1:-1], right=True)
    ok = mask & np.isfinite(x)
    out[ok] = idx[ok].astype(np.int16)
    return out


def fmt_edges(edges: np.ndarray) -> list[str]:
    # labels like: [a, b]
    labels = []
    for i in range(len(edges) - 1):
        labels.append(f"[{edges[i]:.3f}, {edges[i+1]:.3f}]")
    return labels


def robust_limits(x: np.ndarray, mask: np.ndarray, lo: float = 2.0, hi: float = 98.0):
    v = x[mask & np.isfinite(x)]
    if v.size == 0:
        return None, None
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def save_binned_with_range_legend(path: Path, binned: np.ndarray, edges: np.ndarray, title: str) -> None:
    n_bins = len(edges) - 1
    plot = np.ma.masked_where(binned < 0, binned)

    cmap = plt.get_cmap("viridis", n_bins).copy()
    cmap.set_bad(alpha=0.0)

    plt.figure(figsize=(10, 8))
    im = plt.imshow(plot, cmap=cmap, vmin=0, vmax=n_bins - 1)
    plt.axis("off")
    plt.title(title)

    fig = plt.gcf()
    fig.subplots_adjust(right=0.83)
    cax = fig.add_axes([0.85, 0.12, 0.03, 0.76])

    ticks = list(range(n_bins))
    cb = plt.colorbar(im, cax=cax, ticks=ticks)
    cb.ax.set_yticklabels(fmt_edges(edges))

    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("Wrote:", path)


def save_continuous_with_labeled_cbar(path: Path, arr: np.ndarray, mask: np.ndarray, title: str, cbar_label: str) -> None:
    vmin, vmax = robust_limits(arr, mask)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(arr, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.title(title)

    fig = plt.gcf()
    fig.subplots_adjust(right=0.83)
    cax = fig.add_axes([0.85, 0.12, 0.03, 0.76])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label(cbar_label)

    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("Wrote:", path)


def panel_three(out_png: Path, ngb: np.ndarray, mid_img, mid_title: str, right_img, right_title: str) -> None:
    plt.figure(figsize=(18, 6))

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(ngb)
    ax1.set_title("NGB (764/549/510)")
    ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(mid_img)
    ax2.set_title(mid_title)
    ax2.axis("off")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(right_img)
    ax3.set_title(right_title)
    ax3.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Wrote:", out_png)


def make_class_rgba(lab: np.ndarray, K: int = 5) -> np.ndarray:
    """
    Convert label raster into RGBA image with transparent nodata.
    """
    VIR = plt.get_cmap("viridis")
    rgba = np.zeros((lab.shape[0], lab.shape[1], 4), dtype=np.float32)

    nodata = lab < 0
    rgba[nodata, 3] = 0.0

    ok = ~nodata
    if np.any(ok):
        vals = lab[ok].astype(np.float32) / (K - 1)
        rgba[ok, :3] = VIR(vals)[:, :3]
        rgba[ok, 3] = 1.0
    return rgba


def add_class_legend(ax, K: int = 5) -> None:
    VIR = plt.get_cmap("viridis")
    handles, labels = [], []
    for k in range(K):
        handles.append(plt.Rectangle((0, 0), 1, 1, color=VIR(k / (K - 1))[:3]))
        labels.append(f"{k}: {CLASS_LABELS.get(k, '')}")
    ax.legend(handles, labels, loc="lower right", framealpha=0.9)


# ---------------- Main ----------------
def main() -> None:
    if not ACTIVE_WYVERN_FILE.exists():
        raise FileNotFoundError(f"Missing image: {ACTIVE_WYVERN_FILE}")
    if not ACTIVE_WYVERN_MASK.exists():
        raise FileNotFoundError(f"Missing QA mask: {ACTIVE_WYVERN_MASK}")

    valid = load_valid_mask(ACTIVE_WYVERN_MASK).astype(bool)
    water = load_water_mask().astype(bool)
    use_water = valid & water

    # Read image + compute features
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        ngb = build_composite(ds, BANDS_NGB)

        r570 = read_band(ds, B570)
        r660 = read_band(ds, B660)
        r669 = read_band(ds, B669)
        r711 = read_band(ds, B711)
        r764 = read_band(ds, B764)

    ndti = nd(r660, r570)
    ndci = nd(r711, r669)
    nir_red = ratio(r764, r669)

    # Mask to water for display
    ndti_w = ndti.copy(); ndti_w[~use_water] = np.nan
    ndci_w = ndci.copy(); ndci_w[~use_water] = np.nan

    # Composite proxy (continuous)
    # zscore on water-only
    def zscore(x: np.ndarray) -> np.ndarray:
        v = x[use_water & np.isfinite(x)]
        out = np.full_like(x, np.nan, dtype=np.float32)
        if v.size == 0:
            return out
        mu = float(np.mean(v))
        sd = float(np.std(v)) + 1e-12
        ok = use_water & np.isfinite(x)
        out[ok] = (x[ok] - mu) / sd
        return out

    risk = 1.0 * zscore(ndti) + 0.5 * zscore(ndci)
    risk[~use_water] = np.nan

    # Binning with value-range legends
    ndti_edges = quantile_edges(ndti, use_water, N_BINS)
    ndci_edges = quantile_edges(ndci, use_water, N_BINS)

    ndti_bin = bin_with_edges(ndti, use_water, ndti_edges)
    ndci_bin = bin_with_edges(ndci, use_water, ndci_edges)

    # Save improved standalone binned PNGs (with value ranges)
    save_binned_with_range_legend(
        OUT_DIR / "ndti_binned5_value_ranges.png",
        ndti_bin,
        ndti_edges,
        "NDTI (660/570) binned (water-only) — bins are value ranges",
    )
    save_binned_with_range_legend(
        OUT_DIR / "ndci_binned5_value_ranges.png",
        ndci_bin,
        ndci_edges,
        "NDCI (711/669) binned (water-only) — bins are value ranges",
    )

    # Also save continuous risk with labeled colorbar
    save_continuous_with_labeled_cbar(
        OUT_DIR / "risk_proxy_continuous_labeled.png",
        risk,
        use_water,
        "Composite proxy (water-only)",
        "z(NDTI) + 0.5*z(NDCI)",
    )

    # Build binned RGB images for the 3-panel figure
    # Convert bins to RGBA with transparency outside water
    def bin_to_rgba(b: np.ndarray, edges: np.ndarray) -> np.ndarray:
        n_bins = len(edges) - 1
        plot = np.ma.masked_where(b < 0, b)
        cmap = plt.get_cmap("viridis", n_bins).copy()
        cmap.set_bad(alpha=0.0)
        rgba = cmap(plot / max(1, n_bins - 1))
        return rgba

    ndti_rgba = bin_to_rgba(ndti_bin, ndti_edges)
    ndci_rgba = bin_to_rgba(ndci_bin, ndci_edges)

    # Panel A: NGB + binned NDTI + binned NDCI
    panel_three(
        OUT_DIR / "panel_ngb_ndti_ndci_binned.png",
        ngb,
        ndti_rgba,
        "NDTI binned (water-only)",
        ndci_rgba,
        "NDCI binned (water-only)",
    )

    # Panel B: NGB + SFA KMeans classes + risk proxy
    if SFA_KMEANS_TIF.exists():
        with rasterio.open(SFA_KMEANS_TIF) as ds_lab:
            lab = ds_lab.read(1).astype(np.int16)

        class_rgba = make_class_rgba(lab, K=5)

        # Risk shown as continuous viridis with transparency outside water
        risk_plot = np.ma.masked_where(~np.isfinite(risk), risk)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(alpha=0.0)

        plt.figure(figsize=(18, 6))

        ax1 = plt.subplot(1, 3, 1)
        ax1.imshow(ngb)
        ax1.set_title("NGB (764/549/510)")
        ax1.axis("off")

        ax2 = plt.subplot(1, 3, 2)
        ax2.imshow(class_rgba)
        ax2.set_title("SFA KMeans classes (K=5)")
        ax2.axis("off")
        add_class_legend(ax2, K=5)

        ax3 = plt.subplot(1, 3, 3)
        vmin, vmax = robust_limits(risk, use_water)
        im = ax3.imshow(risk_plot, cmap=cmap, vmin=vmin, vmax=vmax)
        ax3.set_title("Composite proxy: z(NDTI) + 0.5*z(NDCI)")
        ax3.axis("off")

        # colorbar attached to panel B
        fig = plt.gcf()
        cax = fig.add_axes([0.92, 0.17, 0.012, 0.66])
        cb = plt.colorbar(im, cax=cax)
        cb.set_label("Relative (water-only)")

        plt.tight_layout(rect=[0, 0, 0.91, 1])
        out_png = OUT_DIR / "panel_ngb_classes_risk.png"
        plt.savefig(out_png, dpi=200)
        plt.close()
        print("Wrote:", out_png)
    else:
        print("NOTE: SFA KMeans label raster not found, skipping panel_ngb_classes_risk.png")
        print("Expected:", SFA_KMEANS_TIF)


if __name__ == "__main__":
    main()
