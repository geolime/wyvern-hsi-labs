# wyvern-hsi-labs

![ci](https://github.com/geolime/wyvern-hsi-labs/actions/workflows/ci.yml/badge.svg)

Hyperspectral remote-sensing pipelines for [Wyvern Dragonette](https://wyvern.space/) open data: a small, tested Python library (`wyvernhsi`) plus two config-driven analysis projects built on it.

The library converts L1B top-of-atmosphere (TOA) radiance to TOA reflectance from STAC metadata, then provides reusable raster I/O, QA/water masking, spectral indices, and PCA + KMeans clustering. Each project is a thin orchestration layer driven by a YAML config and a single pipeline runner.

> **TOA, not surface.** No atmospheric correction is applied. All reflectance products and figures are top-of-atmosphere; spectral indices below are optical proxies, not validated geophysical quantities.

## Projects

- **[forest_landcover](projects/forest_landcover/)** — unsupervised land-cover grouping (PCA + KMeans) and reference-spectrum mapping (SAM) over Santa Cruz de la Sierra, Bolivia.
- **[water_quality](projects/water_quality/)** — water masking and optical water-quality *proxies* (NDTI turbidity, NDCI chlorophyll proxy) over Bitter Lake, Egypt.

## Results

| forest_landcover | water_quality |
|---|---|
| ![KMeans clusters](docs/figures/forest/kmeans_clusters_K5.png) | ![Water mask](docs/figures/water/water_mask.png) |

## Reproduce

```bash
pip install -e .
```

Raw scenes are not stored in git (too large) — see each project's `data/README.md` for how to fetch and place a scene. Then run a project's pipeline:

```bash
python pipelines/run_pipeline.py --config configs/forest_landcover.yaml
python pipelines/run_pipeline.py --config configs/water_quality.yaml
```

The runner loads the config, resolves the scene, runs the stages listed under `pipeline.stages`, and writes GeoTIFFs, figures, a `manifest.json` (config + git SHA + package versions), and a `report.md` under the project's `outputs/`. All tunables (K, PCA components, band wavelengths, thresholds, ROIs) live in the YAML, not the code.

## Library layout

```
src/wyvernhsi/
  config.py        typed YAML config (fails fast on missing keys)
  paths.py         pure scene/path resolution (no import-time work)
  io.py            raster read/write + window tiling
  radiometry.py    radiance -> TOA reflectance (pure numpy)
  stac.py          STAC metadata extraction (pystac)
  masks.py         QA clear-mask + water-mask loaders
  wavelengths.py   band-description parsing + nearest-wavelength selection
  indices.py       NDVI, NDTI, NDCI, red-edge slope
  clustering.py    PCA + KMeans fit / tiled predict / cluster-mean spectra
  visualization.py percentile stretch + figure helpers
  reporting.py     run manifest + auto report
  logging_setup.py logging configuration
```

## Development

```bash
pip install -e ".[dev]"
ruff check src tests pipelines
pytest -q
```

CI (GitHub Actions) installs the package and runs lint + tests on every push.

## Status

The core spine runs end to end via the runner. Several analysis stages (forest SAM, water proxy/metric maps) are being ported from standalone scripts into runner stages; see each project README for which stages are wired today.
