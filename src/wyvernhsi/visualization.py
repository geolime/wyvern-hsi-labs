from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless/CI-safe; these scripts only save files
import matplotlib.pyplot as plt
import numpy as np


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


def save_png(path: Path, arr: np.ndarray, *, dpi: int = 200) -> None:
    """Save a 2D or (H, W, 3) array as a borderless PNG."""
    plt.figure(figsize=(8, 8))
    plt.imshow(arr)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()