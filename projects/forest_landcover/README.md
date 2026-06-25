# Hyperspectral Land Cover Mapping — Wyvern Dragonette

Unsupervised land-cover grouping and reference-spectrum mapping over a 31-band VNIR hyperspectral scene from the [Wyvern](https://wyvern.space/) Dragonette constellation, processed from L1B TOA radiance to **top-of-atmosphere (TOA) reflectance**.

**Study area:** Santa Cruz de la Sierra region, Bolivia. **Sensor:** Wyvern Dragonette, 31 VNIR bands (~510–900 nm). **Processing:** TOA reflectance, no atmospheric correction.

## The scene

**Colour-infrared quicklook (full scene).** Vegetation reflects strongly in the near-infrared and renders red; bare soil and built surfaces stay muted blue-grey. The agricultural mosaic around Santa Cruz — fields, pasture, and remnant forest — is the structure the clustering recovers.

![CIR quicklook](../../docs/figures/forest/cir_quicklook.png)

## Method

```text
L1B TOA radiance
   │  convert  ── STAC solar illumination, sun elevation, Earth–Sun distance
   ▼
TOA reflectance ──► QA clear-mask (cloud / haze / shadow removed)
   │
   ├─► unsupervised grouping   PCA(8) + KMeans(K=5), L2-normalised, tiled prediction
   ├─► reference-spectrum map   SAM vs hand-picked reference ROIs (3 cover types)
   └─► index separability       NDVI, red-edge slope per group (effect sizes)
```

Three caveats stated up front:

- **KMeans is unsupervised** — the output is spectral *groups*, not labelled classes, and cluster IDs are arbitrary and unstable across runs.
- **SAM is reference-spectrum mapping, not validated classification** — endmembers come from a handful of manually chosen ROIs; there is **no accuracy assessment**.
- **Index separability measures distinctness, not correctness** — it shows the groups differ spectrally, it does not validate them against ground truth.

## Results (reference run)

### Spectral dimensionality

**PCA composite (PC1, PC2, PC3 → R, G, B).** The 31 bands collapse onto very few axes: PC1 alone explains **77.5%** of variance, the first two PCs **96.1%**, and eight PCs **99.0%**. Distinct colours here mark spectrally distinct surfaces — the visual basis for the clustering.

![PCA composite](../../docs/figures/forest/pca_rgb_pc123.png)

### KMeans clusters

**K=5 cluster map.** Sampled silhouette **0.39**, 6-seed mean ARI **0.65** — moderate stability: the broad partition is reproducible, the exact boundaries less so. Ordered by median NDVI:

| Cluster | Median NDVI | Scene fraction | Reading |
|---|---|---|---|
| 0 | 0.19 | 30.6% | least-vegetated / bright surfaces |
| 2 | 0.26 | 6.2% | sparse vegetation |
| 4 | 0.27 | 16.6% | sparse–moderate vegetation |
| 1 | 0.35 | 7.6% | moderate vegetation |
| 3 | 0.40 | 3.9% | densest / closed-canopy vegetation |

(35.2% of the scene is QA-masked nodata.)

![KMeans clusters](../../docs/figures/forest/kmeans_K5_class_only.png)

**Cluster mean spectra.** Each curve is a cluster's mean L2-normalised TOA reflectance. They fan out exactly where land cover should differ — clusters separate by the height of the near-infrared plateau and the steepness of the red-edge rise (~700–740 nm), which is what drives the NDVI gradient in the table above. The two lowest-NDVI clusters (0, 2) sit close together; the densest-vegetation cluster (3) has the highest NIR shoulder.

![Cluster mean spectra](../../docs/figures/forest/kmeans_cluster_spectra_K5.png)

### Separability

With 1–9 million pixels per cluster, every pairwise t-test returns p ≈ 0, so the informative quantity is the **effect size** (`index_separability_tests_K5.csv`). By Cohen's d on NDVI, the extremes are strongly separable — cluster 0 vs 1 (d ≈ −6.2) and 0 vs 3 (d ≈ −4.9) — while clusters 2 and 4 are nearly indistinguishable in NDVI (d ≈ −0.40), consistent with their almost-equal medians. Red-edge slope tells the same story, with the densest-vegetation cluster the most distinct.

### SAM cross-check

**Spectral Angle Mapper vs 3 reference ROIs.** Mapping each pixel to the nearest of three reference spectra assigns most of the scene to the soil endmember (~13.2M px) versus trees (~6.9M) and vegetation (~0.42M), with median NDVI of 0.21 / 0.31 / 0.47 respectively. The trees → vegetation → soil NDVI ordering agrees with the KMeans greenness gradient, but the heavy soil assignment reflects the coarseness of a 3-ROI reference set — SAM here is a sanity check on the unsupervised grouping, not a land-cover product.

![SAM classes](../../docs/figures/forest/sam_class_only.png)

## Reproduce

```bash
pip install -e .
# place the scene per data/README.md, then:
python pipelines/run_pipeline.py --config configs/forest_landcover.yaml
```

Stages run, in order, are listed under `pipeline.stages` in `configs/forest_landcover.yaml`; all parameters (K, PCA components, band wavelengths, reference ROIs) live there. Each run writes GeoTIFFs, figures, CSVs, a `manifest.json`, and a `report.md` under `outputs/`.

## Limitations & next steps

The clustering is unsupervised and uncalibrated — without field data the cover-type readings are interpretive, anchored only to NDVI/red-edge behaviour. SAM's 3-ROI reference set is too coarse to separate soil sub-types. A natural next step is matching pixel spectra against a curated reference spectral library (USGS/ECOSTRESS-style), which would require resampling library spectra onto the Dragonette's bands — a sibling to the existing SAM stage.

## Outputs

```
outputs/
  previews/rgb_quicklook.png / cir_quicklook.png   full-scene quicklooks
  kmeans_clusters_K5.tif / .png            cluster map + preview
  kmeans_cluster_spectra_K5.png            per-cluster mean spectra
  kmeans_K5_class_only.png / _overlay_cir.png   previews
  sam_fullscene_class.tif                  reference-spectrum map (SAM)
  sam_class_only.png / sam_overlay.png     SAM previews
  spectral_index_stats.csv                 NDVI / red-edge slope per group
  index_separability_tests_K5.csv          all-pairs Welch t + Cohen's d
  kmeans_cluster_proportions_K5.{csv,png}  scene fractions
  kmeans_stability_ari_K5.{csv,png}        6-seed ARI stability
  pca_scree.png / pca_cumulative.png / pca_rgb_pc123.png
  analysis_summary.txt / manifest.json / report.md
```
