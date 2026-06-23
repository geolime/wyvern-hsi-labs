from __future__ import annotations

import numpy as np


def normalized_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a - b) / (a + b); NaN where inputs are NaN or the denominator is ~0."""
    denom = a + b
    out = np.full(a.shape, np.nan, dtype=np.float32)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(denom) > 1e-10)
    out[ok] = (a[ok] - b[ok]) / denom[ok]
    return out