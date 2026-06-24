# Hyperspectral Land Cover Mapping — Wyvern Dragonette

Unsupervised land-cover grouping and reference-spectrum mapping over a 31-band VNIR hyperspectral scene from the Wyvern Dragonette constellation, processed from L1B TOA radiance to **top-of-atmosphere (TOA) reflectance**.

Santa Cruz de la Sierra, Bolivia
2025-09-27

## Method

```text
L1B TOA radiance
   │  convert  ── STAC solar illumination, sun elevation, Earth–Sun distance
   ▼
TOA reflectance ──► QA clear-mask (cloud / haze / shadow removed)
   │
   ├─► unsupervised grouping   PCA(8) + KMeans(K=5), L2-normalised, tiled prediction
   ├─► reference-spectrum map   SAM vs hand-picked reference ROIs (3 cover types)
   └─► index separability       NDVI, red-edge slope per group
```

Three things worth stating plainly about the method:

- **KMeans is unsupervised.** The output is spectral *groups*, not labelled classes. Cluster IDs are arbitrary and not stable across runs, seeds, or scenes.
- **SAM is reference-spectrum mapping, not classification with accuracy.** Endmembers come from a handful of manually chosen ROIs per cover type. There is **no held-out validation, no confusion matrix, and no accuracy figure** — agreement between methods indicates spectral separability, not ground-truth correctness.
- **The indices measure separability, not validation.** NDVI and red-edge slope are computed from the same reflectance the clustering used, so showing that groups differ in those indices demonstrates the groups are spectrally distinct — it does not externally validate them.

## Results

The clusters separate into broad cover types distinguishable by greenness and red-edge response — closed-canopy vegetation, herbaceous/agricultural vegetation, and bare/soil — with the per-cluster NDVI and red-edge-slope distributions quantifying the separation.

![KMeans clusters](../../docs/figures/forest/kmeans_clusters_K5.png)

![Cluster mean spectra](../../docs/figures/forest/kmeans_cluster_spectra_K5.png)

Per-cluster index statistics are written to `outputs/spectral_index_stats.csv`.

## Reproduce

```bash
pip install -e .
# place the scene per data/README.md, then:
python pipelines/run_pipeline.py --config configs/forest_landcover.yaml
```

Stages run, in order, are listed under `pipeline.stages` in `configs/forest_landcover.yaml`. Parameters — `k`, `pca_components`, band wavelengths, subset ROIs — live there, not in the scripts.

**Current wiring:** the convert and KMeans stages run via the runner today. The SAM stage (`12_sam_fullscene_geotiff.py`) is being ported into the runner; the index-separability stage reads the SAM raster, so until SAM is wired you need an existing `outputs/sam_fullscene_class.tif` for the full config to complete.

## Outputs

```
outputs/
  kmeans_clusters_K5.tif / .png        cluster map + preview
  kmeans_cluster_spectra_K5.png        per-cluster mean spectra (L2-normalised TOA reflectance)
  sam_fullscene_class.tif              reference-spectrum map (SAM)
  spectral_index_stats.csv             NDVI / red-edge slope per group
  spectral_indices_kmeans.png          index distributions by cluster
  manifest.json / report.md            run provenance + auto report
```
