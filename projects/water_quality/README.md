# Hyperspectral Water-Quality Proxies — Wyvern Dragonette

Water masking and optical water-quality **proxies** over a VNIR hyperspectral scene from the Wyvern Dragonette constellation, processed from L1B TOA radiance to **top-of-atmosphere (TOA) reflectance**.

Bitter Lake, Egypt
2025-07-24

> **Proxies, not concentrations.** These indices track relative optical signals associated with turbidity and chlorophyll. They are **not** calibrated concentrations: there is no atmospheric correction (products are TOA, not surface reflectance) and no in-situ measurements for validation. Treat every map below as a relative optical proxy.

## Method

```text
L1B TOA radiance
   │  convert  ── STAC solar illumination, sun elevation, Earth–Sun distance
   ▼
TOA reflectance ──► QA clear-mask
   │
   ├─► water mask        NDWI/NDVI thresholds + NIR-darkness, morphology,
   │                     keep the two largest connected water bodies
   └─► optical proxies   NDTI (turbidity proxy), NDCI (chlorophyll proxy),
                         unsupervised water-type grouping (PCA + KMeans)
```

The water-type map is an **unsupervised grouping** of in-water spectra (PCA + KMeans), ordered by median NDCI — it groups optically similar water, it is not a validated classification.

## Results

![Water mask](../../docs/figures/water/water_mask.png)

![Water pixels, true colour](../../docs/figures/water/rgb_water_only.png)

NDTI and NDCI proxy maps and the water-type grouping are written under `outputs/` when their stages run.

## Reproduce

```bash
pip install -e .
# place the scene per data/README.md, then:
python pipelines/run_pipeline.py --config configs/water_quality.yaml
```

Stages and thresholds (NDWI/NDVI cutoffs, band wavelengths, percentile stretch) live in `configs/water_quality.yaml`.

**Current wiring:** the convert and water-mask stages run via the runner today. The proxy-map and water-type stages are being ported from standalone scripts into runner stages.

## Outputs

```
outputs/masks/
  water_mask.tif / .png                water-only mask + preview
  rgb_water_only.png                   true-colour, water pixels only
outputs/
  ndti_turbidity_proxy.* / ndci_*      optical proxy maps  (naming standardised on
                                       optical-proxy terminology; older runs used
                                       "risk"/"chlorophyll" filenames)
  manifest.json / report.md            run provenance + auto report
```
