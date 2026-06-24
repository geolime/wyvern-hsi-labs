"""Spectral Angle Mapper (SAM) classification — pure numpy."""
from __future__ import annotations

import numpy as np


def _unit_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)


def spectral_angle_classify(
    tile_yxb: np.ndarray, endmembers_kb: np.ndarray, angle_threshold_rad: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign each valid pixel to the nearest endmember by spectral angle.

    Returns (best_class int16 (Y, X), best_angle float32 (Y, X) radians). Pixels with any
    non-finite band, or whose smallest angle exceeds angle_threshold_rad, are -1 / NaN.
    """
    cube = tile_yxb.astype(np.float32)
    valid = np.isfinite(cube).all(axis=2)
    best_class = np.full(cube.shape[:2], -1, dtype=np.int16)
    best_angle = np.full(cube.shape[:2], np.nan, dtype=np.float32)
    if not np.any(valid):
        return best_class, best_angle

    cube_unit = _unit_rows(cube[valid])
    e_unit = _unit_rows(endmembers_kb.astype(np.float32))
    ang = np.arccos(np.clip(cube_unit @ e_unit.T, -1.0, 1.0)).astype(np.float32)

    bc = np.argmin(ang, axis=1).astype(np.int16)
    ba = np.min(ang, axis=1).astype(np.float32)
    bc[ba > angle_threshold_rad] = -1

    best_class[valid] = bc
    best_angle[valid] = ba
    return best_class, best_angle