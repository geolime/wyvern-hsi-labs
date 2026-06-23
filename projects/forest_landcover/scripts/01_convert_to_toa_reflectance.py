from __future__ import annotations
from pathlib import Path

import rasterio

from wyvernhsi.config import Config, load_config
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene
from wyvernhsi.radiometry import (
    replace_nodata_with_nan,
    sun_earth_distance_au,
    toa_radiance_to_reflectance,
)
from wyvernhsi.stac import load_wyvern_radiometry_meta


def find_stac_item_json(image_path: Path) -> Path:
    """
    Search near the image for a STAC item JSON.
    Tries all .json files in the same folder and subfolders.
    """
    candidates = sorted(image_path.parent.rglob("*.json"))

    if not candidates:
        raise FileNotFoundError(f"No STAC JSON found under: {image_path.parent}")

    last_error = None
    for p in candidates:
        try:
            # Test whether this JSON has usable Wyvern radiometry metadata
            with rasterio.open(image_path) as ds:
                load_wyvern_radiometry_meta(
                    stac_item_json=p,
                    image_name=image_path.name,
                    n_bands=ds.count,
                )
            return p
        except Exception as e:
            last_error = e

    raise RuntimeError(
        f"Could not find usable STAC item JSON under {image_path.parent}. "
        f"Last error: {last_error}"
    )


def main(config: Config) -> None:
    scene = resolve_scene(config.project_dir, require_reflectance=False)
    image_path = scene.radiance

    stac_json = find_stac_item_json(image_path)

    out_dir = image_path.parent / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_tif = out_dir / f"{image_path.stem}_toa_reflectance.tif"

    print("Input radiance:", image_path)
    print("STAC item:", stac_json)
    print("Output reflectance:", out_tif)

    with rasterio.open(image_path) as src:
        radiance = replace_nodata_with_nan(src.read(), src.nodata)

        meta = load_wyvern_radiometry_meta(
            stac_item_json=stac_json,
            image_name=image_path.name,
            n_bands=src.count,
        )

        d = sun_earth_distance_au(meta.datetime_utc)

        reflectance = toa_radiance_to_reflectance(
            radiance=radiance,
            solar_illumination=meta.solar_illumination,
            sun_elevation_deg=meta.sun_elevation_deg,
            earth_sun_distance_au=d,
        )

        profile = src.profile.copy()
        profile.update(
            dtype="float32",
            nodata=float("nan"),
            compress="deflate",
            predictor=3,
            tiled=True,
        )

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(reflectance)

            for i, desc in enumerate(src.descriptions, start=1):
                if desc:
                    dst.set_band_description(i, desc)

            dst.update_tags(
                source_product="Wyvern Dragonette L1B TOA radiance",
                derived_product="TOA reflectance",
                sun_elevation_deg=str(meta.sun_elevation_deg),
                earth_sun_distance_au=str(d),
                stac_item=str(stac_json.name),
            )

    print("Done.")


if __name__ == "__main__":
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))
