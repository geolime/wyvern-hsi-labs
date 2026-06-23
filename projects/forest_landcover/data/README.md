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