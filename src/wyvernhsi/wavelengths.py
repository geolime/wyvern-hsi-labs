from __future__ import annotations

import re
from typing import Iterable, List, Optional

import numpy as np

# Supports:
#   "Band_444"
#   "Band 444"
#   "Band_444nm"
#   "Band 444 nm"
#   "... (444 nm) ..."
_NM_RE = re.compile(r"(\d+(\.\d+)?)\s*nm", re.IGNORECASE)
_BANDNUM_RE = re.compile(r"band[_\s-]*(\d+(\.\d+)?)", re.IGNORECASE)


def parse_wavelengths_nm_from_descriptions(descriptions: Iterable[Optional[str]]) -> np.ndarray:
    """
    Parse wavelengths (nm) from band description strings.
    Returns np.nan where parsing fails.
    """
    out: List[float] = []
    for d in descriptions:
        if not d:
            out.append(float("nan"))
            continue

        # 1) Explicit "... nm"
        m = _NM_RE.search(d)
        if m:
            out.append(float(m.group(1)))
            continue

        # 2) Wyvern-style "Band_444"
        m = _BANDNUM_RE.search(d)
        if m:
            out.append(float(m.group(1)))
            continue

        out.append(float("nan"))

    return np.asarray(out, dtype=float)


def pick_band_index_nearest(wavelengths_nm: np.ndarray, target_nm: float) -> int:
    if wavelengths_nm.ndim != 1:
        raise ValueError("wavelengths_nm must be 1D")

    good = np.isfinite(wavelengths_nm)
    if not np.any(good):
        raise ValueError("No finite wavelengths found.")

    idx_good = np.argmin(np.abs(wavelengths_nm[good] - float(target_nm)))
    return int(np.flatnonzero(good)[idx_good])

