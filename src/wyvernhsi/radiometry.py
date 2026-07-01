"""Radiance -> TOA reflectance math (pure numpy; STAC parsing lives in stac.py)."""
from __future__ import annotations

from datetime import datetime

import numpy as np


def replace_nodata_with_nan(arr: np.ndarray, nodata) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    if nodata is not None:
        out[out == nodata] = np.nan
    return out


def sun_earth_distance_au(dt_utc: datetime) -> float:
    """Approx Earth-Sun distance (AU): 1 - 0.01672*cos(deg2rad(0.9856*(DOY-4)))."""
    doy = int(dt_utc.strftime("%j"))
    return float(1.0 - (0.01672 * np.cos(np.deg2rad(0.9856 * (doy - 4)))))


def toa_radiance_to_reflectance(
    radiance: np.ndarray,
    solar_illumination: np.ndarray,
    sun_elevation_deg: float,
    earth_sun_distance_au: float,
) -> np.ndarray:
    """rho = (L * pi * d^2) / (Esun * sin(sun_elev)); radiance shaped (bands, rows, cols)."""
    if radiance.ndim != 3:
        raise ValueError("Expected radiance array with shape (bands, rows, cols).")
    if solar_illumination.shape[0] != radiance.shape[0]:
        raise ValueError("solar_illumination length must match number of image bands.")

    sin_sun = float(np.sin(np.deg2rad(sun_elevation_deg)))
    if sin_sun <= 0:
        raise ValueError("Sun elevation must be greater than 0 degrees.")

    d2 = earth_sun_distance_au ** 2
    out = np.empty_like(radiance, dtype=np.float32)
    for i in range(radiance.shape[0]):
        out[i] = (radiance[i] * np.pi * d2) / (float(solar_illumination[i]) * sin_sun)
    return out