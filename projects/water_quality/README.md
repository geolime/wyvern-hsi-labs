# Hyperspectral Water-Quality Proxies — Wyvern Dragonette

Water masking and optical water-quality **proxies** over a 31-band VNIR hyperspectral scene from the [Wyvern](https://wyvern.space/) Dragonette constellation, processed from L1B TOA radiance to **top-of-atmosphere (TOA) reflectance**.

**Study area:** Bitter Lake, Egypt. **Sensor:** Wyvern Dragonette, 31 VNIR bands (~510–900 nm). **Processing:** TOA reflectance, no atmospheric correction.

> **Proxies, not concentrations.** NDTI and NDCI track relative optical signals associated with turbidity and chlorophyll. They are **not** calibrated concentrations: there is no atmospheric correction (products are TOA, not surface reflectance) and no in-situ validation. Every map below is a relative optical proxy.

## The scene

**NGB composite (NIR-Green-Blue), QA-masked.** Clouds are removed (rendered black); the inland water body of Bitter Lake is the analysis target. The NIR-forward band choice darkens clear water and brightens vegetation/turbid water, making the shoreline and in-water structure legible.

![NGB composite](../../docs/figures/water/ngb_water_composite.png)

## Method

```text
L1B TOA radiance
   │  convert  ── STAC solar illumination, sun elevation, Earth–Sun distance
   ▼
TOA reflectance ──► QA clear-mask
   │
   ├─► water mask        NDWI/NDVI thresholds + NIR-darkness, morphology,
   │                     keep the two largest connected water bodies
   ├─► water grouping    PCA + KMeans on water spectra; KMeans on engineered
   │                     features (NDTI, NDCI, R/G, RE/R); SFA water types
   └─► optical proxies   NDTI (turbidity), NDCI (chlorophyll), composite z-blend
```

The water-type maps are **unsupervised groupings** of in-water spectra, relabeled by median NDTI (0 = clearest, 4 = most turbid). They group optically similar water; they are not validated classifications.

## Results (reference run)

### Spectral dimensionality

**PCA composite (PC1, PC2, PC3 → R, G, B), water-only.** Water-pixel reflectance is strongly low-dimensional — PC1 explains **74.2%**, two PCs **84.9%**, eight PCs **95.9%** of variance. Colour gradients across the lake trace smooth optical variation (clear → turbid), not sharp boundaries.

![PCA composite](../../docs/figures/water/water_kmeans_K5_PCA8_pca_pc123.png)

### KMeans water grouping

**Full-spectrum PCA+KMeans (K=5), water-only.** This grouping is **very stable** — 6-seed mean ARI **0.82** (sampled silhouette 0.33), notably more reproducible than the forest scene. The spatial pattern is a coherent gradient rather than a patchwork, consistent with the smooth PCA variation.

![KMeans water classes](../../docs/figures/water/water_kmeans_K5_PCA8.png)

**Cluster mean spectra.** Each curve is a water cluster's mean L2-normalised TOA reflectance. The spread is concentrated in the visible/red-edge region where turbidity and chlorophyll modulate water-leaving signal — clearer clusters are darker and flatter, more turbid clusters lift across green–red. This is the optical basis for the turbidity ordering below.

![Cluster mean spectra](../../docs/figures/water/water_kmeans_K5_PCA8_cluster_mean_spectra.png)

### Turbidity tiers (SFA grouping)

A separate KMeans on interpretable features (NDTI / NDCI / NIR-Red), relabeled by median NDTI, splits the water into five tiers spanning roughly NDTI −0.21 (clearest) to −0.07 (most turbid):

| Tier | Class | Median NDTI | Median NDCI | Water fraction |
|---|---|---|---|---|
| clearest | 0 | −0.209 | −0.153 | 23.6% |
| low turbidity | 1 | −0.203 | −0.196 | 17.0% |
| moderate | 2 | −0.178 | −0.110 | 20.3% |
| turbid | 3 | −0.100 | −0.113 | 27.1% |
| most turbid | 4 | −0.065 | −0.069 | 12.0% |

The independent 4-feature KMeans (`water_features_kmeans`) recovers the same monotonic NDTI ordering (median −0.215 → −0.056) — a useful internal consistency check between two different feature sets.

## Reproduce

```bash
pip install -e .
# place the scene per data/README.md, then:
python pipelines/run_pipeline.py --config configs/water_quality.yaml
```

Stages and thresholds (NDWI/NDVI cutoffs, band wavelengths, K, composite weights, percentiles) live in `configs/water_quality.yaml`. Each run writes GeoTIFFs, proxy maps, CSVs, a `manifest.json`, and a `report.md` under `outputs/`.

## Limitations & next steps

Everything here is TOA and uncalibrated: NDTI/NDCI are optical proxies, not turbidity or chlorophyll concentrations, and the tier labels are relative orderings, not water-quality classes. Validation would need atmospheric correction to surface reflectance plus in-situ samples. The composite "optical proxy" map uses arbitrary NDTI/NDCI weights and should be read as an exploratory blend, not an index.

## Outputs

```
outputs/masks/        water_mask.tif/.png, rgb_water_only.png
outputs/previews/     rgb_quicklook.png, ngb_water_composite.png
outputs/
  water_kmeans_K5_PCA8.{tif,png}                    full-spectrum PCA+KMeans
  water_kmeans_K5_PCA8_pca_pc123.png                PCA composite
  water_kmeans_K5_PCA8_cluster_mean_spectra.png     cluster spectra
  water_kmeans_K5_PCA8_summary.txt                  PCA variance, silhouette, ARI
outputs/water_features/
  water_features_kmeans_K5.{tif,png} + cluster_feature_stats.csv
  ndti_map.png / ndci_map.png
  sfa_kmeans/sfa_kmeans_K5.{tif,png} + feature_stats.csv + class_summary.{csv,png}
  sfa_kmeans/sfa_kmeans_K5_ndci_ranking.txt
  proxies/{whole_scene,water_only}/ndti_*.png, ndci_*.png
  proxies/optical_proxy_composite_*.png
  figures/panel_ngb_*.png
  rgb_with_water_classes.png / ngb_with_water_classes.png / water_classes_only.png
outputs/  manifest.json / report.md
```
