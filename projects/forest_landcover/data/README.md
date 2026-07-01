# Data — forest_landcover

Raw Wyvern Dragonette scenes are **not** stored in git (too large). Download the
scene and place it here so the pipeline can find it:

projects/forest_landcover/data/

<scene_id>/                   wyvern_dragonette-004_20250927T145218_edb4d3af

<scene_id>.tif                ![# L1B TOA radiance raster](https://wyvern-data.com/wyvern_dragonette-004_20250927T145218_edb4d3af/wyvern_dragonette-004_20250927T145218_edb4d3af.tiff)

<scene_id>.json               [# STAC item (kept in git)](https://wyvern-odp.com/wyvern_dragonette-004_20250927T145218_edb4d3af/wyvern_dragonette-004_20250927T145218_edb4d3af.json)


- Source: <https://wyvern-data.com/wyvern_dragonette-004_20250927T145218_edb4d3af/edb4d3af-3a7c-4eab-80e7-8acecb57cb79.zip>
- Scene ID used in the README results: `wyvern_dragonette-004_20250927T145218_edb4d3af`
- Bands/units: 31-band VNIR, TOA radiance (converted to TOA reflectance by stage 01).

After placing the files, verify with: `python pipelines/run_pipeline.py --config configs/forest_landcover.yaml --stages 00`S


## Validation reference (optional): MapBiomas Bolivia

The validation and Random Forest stages compare the outputs against an independent
land-cover map. This reference is **optional**: if it is absent, both stages skip
cleanly and the rest of the pipeline runs normally.

**Source:** MapBiomas Bolivia, Collection 3 (2024 land cover & use layer, Landsat-derived, 30 m).
Download the national GeoTIFF from the official portal:
https://bolivia.mapbiomas.org/en/descargas/

**Placement:** save the downloaded file to this exact path and name (it must match
`reference_tif` in `configs/forest_landcover.yaml`):

    projects/forest_landcover/data/reference/mapbiomas-bolivia-collection-30-2024.tif

The file is the full-country raster (~2.6 billion pixels); the pipeline windows it to
the scene extent automatically, so no manual clipping is needed. It is not stored in
git (large, and licensed separately).

**License & citation:** MapBiomas data retains its own terms. Review the terms of use
before reuse, and cite per the project's required format:

> MapBiomas – Collection [3] of the [Annual] Series of [Land Cover and Use] Maps of
> Bolivia, accessed on [DD Month YYYY] through the link: [[https://bolivia.mapbiomas.org/en/descargas/](https://bolivia.mapbiomas.org/en/descargas/)]

- Terms of use: https://bolivia.mapbiomas.org/en/terminos-de-uso/
- Legend codes: https://bolivia.mapbiomas.org/en/codigos-de-la-leyenda/