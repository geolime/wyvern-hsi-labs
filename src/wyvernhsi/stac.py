"""STAC item parsing for Wyvern Dragonette scenes (pystac is isolated to this module)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pystac import Item
from pystac.extensions.eo import EOExtension


@dataclass(frozen=True)
class WyvernRadiometryMeta:
    datetime_utc: datetime
    sun_elevation_deg: float
    solar_illumination: np.ndarray


def _get_datetime_utc(item: Item) -> datetime:
    if item.datetime is not None:
        return item.datetime.astimezone(timezone.utc)
    raw = item.properties["datetime"]
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)


def _get_bands_from_item(item: Item, image_name: str):
    preferred_keys = ["Cloud optimized GeoTiff", "Cloud Optimized GeoTIFF", "cog", "image"]
    for key in preferred_keys:
        if key in item.assets:
            bands = EOExtension.ext(item.assets[key]).bands
            if bands:
                return bands
    for asset in item.assets.values():
        if Path(asset.href).name == image_name:
            bands = EOExtension.ext(asset).bands
            if bands:
                return bands
    for asset in item.assets.values():
        bands = EOExtension.ext(asset).bands
        if bands:
            return bands
    raise RuntimeError("Could not find EO band metadata in STAC item assets.")


def load_wyvern_radiometry_meta(
    stac_item_json: Path, image_name: str, n_bands: int
) -> WyvernRadiometryMeta:
    item = Item.from_file(str(stac_item_json))
    dt = _get_datetime_utc(item)
    sun_elevation = float(item.properties["view:sun_elevation"])
    bands = _get_bands_from_item(item, image_name=image_name)
    esun = np.array([float(b.solar_illumination) for b in bands], dtype=np.float32)
    if esun.size != n_bands:
        raise RuntimeError(f"STAC has {esun.size} bands, image has {n_bands} bands.")
    return WyvernRadiometryMeta(
        datetime_utc=dt, sun_elevation_deg=sun_elevation, solar_illumination=esun
    )