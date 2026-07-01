# Hyperspectral Water-Quality Proxies — Wyvern Dragonette

Water masking and optical water-quality **proxies** over a 31-band VNIR hyperspectral scene from the [Wyvern](https://wyvern.space/) Dragonette constellation, processed from L1B TOA radiance to **top-of-atmosphere (TOA) reflectance**.

**Study area:** Bitter Lake, Egypt. **Sensor:** Wyvern Dragonette, 31 VNIR bands (~510–900 nm). **Processing:** TOA reflectance, no atmospheric correction.

> **Proxies, not concentrations.** NDTI and NDCI track relative optical signals associated with turbidity and chlorophyll. They are **not** calibrated concentrations: there is no atmospheric correction (products are TOA, not surface reflectance) and no in-situ validation. Every map below is a relative optical proxy, meaningful only in relation to other water pixels in the same scene.

## What this project maps

The pipeline does two separate things over the lake, and it is worth keeping them apart. First, it groups water pixels by the overall shape of their reflectance spectra, an unsupervised view of how many optically distinct water types the scene contains. Second, it computes two published water indices, NDTI for turbidity and NDCI for chlorophyll, and maps each one on its own. The first answers how many kinds of water are present; the second answers where each optical signal runs high or low. Neither produces a calibrated water-quality number, and this project is careful not to imply one.

## The scene

**NGB composite (NIR-Green-Blue), QA-masked.** Clouds are removed (rendered black); the inland water body of Bitter Lake is the analysis target. The NIR-forward band choice darkens clear water and brightens vegetation and turbid water, making the shoreline and in-water structure legible.

![NGB composite](../../docs/figures/water/ngb_water_composite.png)

## Method

```text
L1B TOA radiance
   │  convert  ── STAC solar illumination, sun elevation, Earth–Sun distance
   ▼
TOA reflectance ──► QA clear-mask
   │
   ├─► water mask       NDWI/NDVI thresholds + NIR-darkness, morphology,
   │                    keep the two largest connected water bodies
   ├─► water grouping   PCA(8) + KMeans(K=5) on water spectra,
   │                    unsupervised optical water types (cluster IDs arbitrary)
   └─► optical proxies  NDTI (turbidity) and NDCI (chlorophyll) as independent
                        tercile tier maps, plus an NDTI–NDCI scatter and an
                        illustrative equal-weight composite
```

Stated up front: the KMeans grouping is unsupervised, so its cluster IDs are arbitrary and carry no turbidity ranking. NDTI and NDCI are relative optical proxies on uncalibrated TOA reflectance, and the tier maps are within-scene percentile orderings, not concentrations and not comparable across dates.

## Results (reference run)

### Spectral dimensionality

**PCA composite (PC1, PC2, PC3 → R, G, B), water-only.** Water-pixel reflectance is strongly low-dimensional: PC1 explains **74.2%**, two PCs **84.9%**, and eight PCs **95.9%** of variance. Colour gradients across the lake trace smooth optical variation from clear to turbid, not sharp boundaries.

![PCA composite](../../docs/figures/water/water_kmeans_K5_PCA8_pca_pc123.png)

### KMeans water grouping

**Full-spectrum PCA+KMeans (K=5), water-only.** This grouping is very stable, with a 6-seed mean ARI of **0.82** (sampled silhouette 0.33), notably more reproducible than the forest scene. The cluster IDs are arbitrary unsupervised labels, not a turbidity ranking. What the map shows is a coherent spatial gradient rather than a patchwork, consistent with the smooth PCA variation.

![KMeans water classes](../../docs/figures/water/water_kmeans_K5_PCA8.png)

**Cluster mean spectra.** Each curve is a water cluster's mean L2-normalised TOA reflectance. The spread concentrates in the visible and red-edge region where turbidity and chlorophyll modulate the water-leaving signal: darker, flatter curves correspond to clearer water, and curves that lift across green to red to more turbid water. This optical spread is what the NDTI and NDCI indices below quantify directly.

![Cluster mean spectra](../../docs/figures/water/water_kmeans_K5_PCA8_cluster_mean_spectra.png)

### Optical proxy maps (NDTI, NDCI)

NDTI (turbidity) and NDCI (chlorophyll) are mapped independently, each binned into three relative tiers (terciles of that index's own water-pixel distribution) labelled low, moderate, and high relative to this scene. They are mapped separately on purpose: they track physically distinct signals, sediment load versus algal pigment, so collapsing them into one number would hide which signal drives a given location.

**NDTI turbidity tiers, water-only.** The turbidity proxy split into three within-scene terciles, each holding roughly a third of water pixels. Tier boundaries are percentiles of this scene's own NDTI distribution, so the labels are relative orderings, not absolute turbidity levels.

![NDTI turbidity tiers](../../docs/figures/water/ndti_tiers.png)

**NDCI chlorophyll tiers, water-only.** The chlorophyll proxy split the same way, into three within-scene terciles of the NDCI distribution. As with NDTI, low, moderate, and high are relative to this scene and carry no concentration units.

![NDCI chlorophyll tiers](../../docs/figures/water/ndci_tiers.png)

**How the two proxies co-vary.** Across 4.6 million water pixels the two indices correlate at Pearson **r = 0.50** (r² ≈ 0.24), a moderate positive relationship: they share about a quarter of their variance and roughly three quarters is independent. That independence is the quantitative reason for keeping them as two maps rather than one blended score. Read the correlation cautiously, though. On TOA reflectance over dark water, NDTI and NDCI both sit in the red and red-edge region and both ride the same atmospheric path, so part of the shared variance is very likely a common confounder (atmosphere, sun glint, overall brightness) rather than genuine sediment-and-algae co-occurrence. Disentangling that confound is what atmospheric correction would buy, and is the natural next step.

![NDTI vs NDCI scatter](../../docs/figures/water/ndti_ndci_scatter.png)

## Reproduce

```bash
pip install -e .
# place the scene per data/README.md, then:
python pipelines/run_pipeline.py --config configs/water_quality.yaml
```

Stages and thresholds (NDWI/NDVI cutoffs, band wavelengths, K, percentile bin count, composite weights) live in `configs/water_quality.yaml`. The proxy tiers use three percentile bins, and the composite uses equal NDTI/NDCI weights by design. Each run writes GeoTIFFs, proxy maps, CSVs, a `manifest.json`, and a `report.md` under `outputs/`.

## Limitations & next steps

Everything here is TOA and uncalibrated. NDTI and NDCI are optical proxies, not turbidity or chlorophyll concentrations, and the tier labels are within-scene relative orderings, not water-quality classes. The composite map is an illustrative equal-weight blend of the two z-scored indices with no physical basis; equal weights are used deliberately, to signal that the weighting encodes no claim. It is a secondary product, and the honest results are the two independent index maps and their scatter.

Two steps would move these from relative proxies toward estimates. Atmospheric correction to surface reflectance is the high-value one for water specifically, because water is dark and the water-leaving signal is a small fraction of TOA, so the atmosphere dominates and is the likely source of the shared NDTI/NDCI variance seen above. On top of that, a multi-date time series over Bitter Lake would let the indices track change, which is a defensible use of relative proxies, but only once the radiometry is consistent enough to compare dates. Full validation would additionally need in-situ samples, which were not available for this scene.

## Outputs

```
outputs/masks/        water_mask.tif/.png, rgb_water_only.png
outputs/previews/     rgb_quicklook.png, ngb_water_composite.png
outputs/
  water_kmeans_K5_PCA8.{tif,png}                    full-spectrum PCA+KMeans
  water_kmeans_K5_PCA8_pca_pc123.png                PCA composite
  water_kmeans_K5_PCA8_cluster_mean_spectra.png     cluster spectra
  water_kmeans_K5_PCA8_summary.txt                  PCA variance, silhouette, ARI
outputs/water_features/proxies/
  ndti_continuous.png / ndci_continuous.png         continuous water-only proxy heatmaps
  ndti_tiers.png / ndci_tiers.png                   three-tier (tercile) proxy maps
  ndti_ndci_scatter.png                             NDTI vs NDCI density scatter (+ correlation)
  optical_proxy_composite_continuous.png            illustrative equal-weight blend
outputs/  manifest.json / report.md
```

## Data sources & attribution

**Imagery — Wyvern Dragonette** open hyperspectral data, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

> © 2025 Wyvern Incorporated. All Rights Reserved.

No external validation reference is used for this project (no in-situ water-quality measurements or independent map were available for Bitter Lake); the water-quality indices are uncalibrated optical proxies, as noted above.