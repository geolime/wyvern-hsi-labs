# Data — water_quality

Raw Wyvern Dragonette scenes are **not** stored in git (too large). Download the
scene and place it here so the pipeline can find it:

projects/water_quality/data/

<scene_id>/                   wyvern_dragonette-001_20250724T071146_fbaa00bd

<scene_id>.tif                ![# L1B TOA radiance raster](https://wyvern-data.com/wyvern_dragonette-001_20250724T071146_fbaa00bd/wyvern_dragonette-001_20250724T071146_fbaa00bd.tiff)

<scene_id>.json               [# STAC item (kept in git)](https://wyvern-odp.com/wyvern_dragonette-001_20250724T071146_fbaa00bd/wyvern_dragonette-001_20250724T071146_fbaa00bd.json)


- Source: [# STAC ZIP](https://wyvern-data.com/wyvern_dragonette-001_20250724T071146_fbaa00bd/fbaa00bd-4b3d-460a-9124-9bfcb3d719a9.zip>)
- Scene ID used in the README results: `wyvern_dragonette-004_20250927T145218_edb4d3af`
- Bands/units: 31-band VNIR, TOA radiance (converted to TOA reflectance by stage 01).

After placing the files, verify with: `python pipelines/run_pipeline.py --config configs/water_quality.yaml --stages 00`S