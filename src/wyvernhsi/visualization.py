from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless/CI-safe; these scripts only save files
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np

def masked_cmap(name="viridis"):
    """Colormap copy with transparent 'bad' (masked/NaN) cells, for masked-array imshow."""
    cmap = colormaps[name].copy()
    cmap.set_bad(alpha=0.0)
    return cmap


def class_colors(k, name="viridis"):
    """k evenly-spaced RGB colors matching imshow(vmin=0, vmax=k-1) — for class legends."""
    cmap = colormaps[name]
    return [cmap(i / (k - 1))[:3] for i in range(k)]


def add_class_legend(labels, *, colors=None, loc="lower right", ax=None):
    """Swatch legend on the current axes (or `ax`). colors defaults to viridis class colors."""
    if colors is None:
        colors = class_colors(len(labels))
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
    (ax if ax is not None else plt).legend(handles, labels, loc=loc, framealpha=0.9)


def save_binned(path, binned, n_bins, title, tick_labels):
    """Discrete binned raster (0..n_bins-1; -1 = nodata, transparent) with a labeled colorbar."""
    cmap = colormaps["viridis"].resampled(n_bins).copy()
    cmap.set_bad(alpha=0.0)
    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(np.ma.masked_where(binned < 0, binned), cmap=cmap, vmin=0, vmax=n_bins - 1)
    plt.axis("off")
    plt.title(title)
    fig.subplots_adjust(right=0.84)
    cb = plt.colorbar(im, cax=fig.add_axes([0.86, 0.12, 0.03, 0.76]), ticks=list(range(n_bins)))
    cb.ax.set_yticklabels(tick_labels)
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def percentile_stretch(band: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    """Stretch a single 2D band to [0, 1] using percentile clipping. NaN -> 0."""
    valid = np.isfinite(band)
    if not np.any(valid):
        return np.zeros(band.shape, dtype=np.float32)
    lo, hi = np.nanpercentile(band[valid], [p_lo, p_hi])
    if hi <= lo:
        return np.zeros(band.shape, dtype=np.float32)
    out = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def stretch_rgb(rgb: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    """Per-channel percentile stretch of an (H, W, 3) float array to uint8."""
    out = np.zeros(rgb.shape, dtype=np.float32)
    for i in range(rgb.shape[2]):
        out[:, :, i] = percentile_stretch(rgb[:, :, i], p_lo, p_hi)
    return (out * 255.0 + 0.5).astype(np.uint8)

def save_heatmap(path, arr, title, *, p_lo=2.0, p_hi=98.0, cbar_label=None):
    """Continuous 2D array as a colorbar heatmap; vmin/vmax at the given finite-value percentiles."""
    v = arr[np.isfinite(arr)]
    vmin, vmax = (float(np.percentile(v, p_lo)), float(np.percentile(v, p_hi))) if v.size else (None, None)
    fig = plt.figure(figsize=(10, 8))
    im = plt.imshow(arr, vmin=vmin, vmax=vmax)
    plt.axis("off")
    plt.title(title)
    fig.subplots_adjust(right=0.86)
    cb = plt.colorbar(im, cax=fig.add_axes([0.88, 0.12, 0.03, 0.76]))
    if cbar_label:
        cb.set_label(cbar_label)
    plt.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def save_contour_overlay(path, background, contours, title):
    """Background image with contour outlines. contours: list of (mask, color, linewidth)."""
    plt.figure(figsize=(12, 10))
    plt.imshow(background)
    for mask, color, lw in contours:
        plt.contour(mask.astype(np.uint8), levels=[0.5], colors=color, linewidths=lw)
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_png(path: Path, arr: np.ndarray, *, dpi: int = 200) -> None:
    """Save a 2D or (H, W, 3) array as a borderless PNG."""
    plt.figure(figsize=(8, 8))
    plt.imshow(arr)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()