"""Spectral indices. All return float32 with NaN where inputs are invalid."""
from __future__ import annotations

import numpy as np


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b); NaN where inputs are NaN or the denominator is ~0."""
    denom = a + b
    out = np.full(a.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(denom) > 1e-10)
    out[ok] = (a[ok] - b[ok]) / denom[ok]
    return out


def ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index."""
    return normalized_difference(nir, red)


def ndti(red: np.ndarray, green: np.ndarray) -> np.ndarray:
    """Normalized Difference Turbidity Index."""
    return normalized_difference(red, green)


def ndci(red_edge: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Normalized Difference Chlorophyll Index (optical proxy)."""
    return normalized_difference(red_edge, red)


def red_edge_slope(
    red_edge: np.ndarray, red: np.ndarray, red_edge_nm: float, red_nm: float
) -> np.ndarray:
    """Reflectance gradient per nm across the red edge: (R_re - R_red) / (nm_re - nm_red)."""
    out = np.full(red_edge.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(red_edge) & np.isfinite(red)
    out[ok] = (red_edge[ok] - red[ok]) / (red_edge_nm - red_nm)
    return out

def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Safe band ratio a / b; NaN where inputs are NaN or b ~ 0."""
    out = np.full(a.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 1e-10)
    out[ok] = a[ok] / b[ok]
    return out