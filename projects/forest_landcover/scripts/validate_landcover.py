"""
Validate the forest outputs against MapBiomas Bolivia 2024 (independent regional reference).

KMeans: cross-tabulate clusters vs the crosswalked reference, label each cluster by majority
overlap, then score. SAM: crosswalk its 3 classes directly to the target scheme, then score.

NOTE: agreement with MapBiomas (itself a ~model with its own error), not absolute accuracy.
"""
from __future__ import annotations

import json
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import rasterio

from wyvernhsi import io, validation
from wyvernhsi.config import Config, load_config
from wyvernhsi.logging_setup import configure_logging
from wyvernhsi.masks import load_valid_mask
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene

logger = logging.getLogger(__name__)

# SAM class index (from config.sam.reference_classes order) -> target scheme name
SAM_TO_TARGET = {0: "trees", 1: "crops", 2: "bare"}  # dense_trees, bright_veg, low_veg_soil


def _log_metrics(tag, m):
    logger.info("%s: kappa=%.3f | overall=%.3f (inflated by imbalance) | n=%d",
                tag, m["kappa"], m["overall_agreement"], m["n_pixels"])
    for name, pc in m["per_class"].items():
        if pc["reference_support"] == 0:
            continue
        pr = f"{pc['precision']:.3f}" if pc["precision"] is not None else "n/a"
        rc = f"{pc['recall']:.3f}" if pc["recall"] is not None else "n/a"
        logger.info("  %-6s support=%9d  precision=%s  recall=%s", name,
                    pc["reference_support"], pr, rc)
        

# Palette high->low (Oxford blue = high agreement ... Giants orange = low)
_CM_PALETTE_HI_LO = ["#01204E", "#028391", "#F6DCAC", "#FAA968", "#F85525"]
_CM_CMAP = LinearSegmentedColormap.from_list("retro", list(reversed(_CM_PALETTE_HI_LO)))


def _save_confusion(cm_df, title, out_png):
    cm = cm_df.values.astype(float)
    total = cm.sum()
    norm = cm / cm.max() if cm.max() else cm   # colour by relative magnitude
    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(norm, cmap=_CM_CMAP, vmin=0, vmax=1)
    plt.xticks(range(len(cm_df.columns)), [c.replace("ref_", "") for c in cm_df.columns])
    plt.yticks(range(len(cm_df.index)), [c.replace("pred_", "") for c in cm_df.index])
    plt.xlabel("Reference")
    plt.ylabel("Predicted")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            n = int(cm[i, j])
            pct = (n / total * 100) if total else 0.0
            # white text on the dark (high) end, dark text on the light/orange end
            color = "white" if norm[i, j] > 0.55 else "#01204E"
            plt.text(j, i, f"{n:,}\n{pct:.1f}%", ha="center", va="center",
                     fontsize=9, color=color)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    logger.info("Wrote: %s", out_png)


def main(config: Config) -> None:
    v = config.validation
    if v is None or not (config.project_dir / v.reference_tif).exists():
        logger.warning("No validation reference (%s); skipping validation stage.",
                       v.reference_tif if v else "validation config absent")
        return
    classes = list(v.crosswalk.keys())
    scene = resolve_scene(config.project_dir)
    out = scene.outputs_dir / "validation"
    out.mkdir(parents=True, exist_ok=True)

    valid = load_valid_mask(scene.mask)
    with rasterio.open(scene.reflectance) as ds:
        profile = ds.profile

    ref_codes = io.align_reference(config.project_dir / v.reference_tif, profile)
    ref_codes[np.isin(ref_codes, v.reference_ignore)] = 0
    ref_class = validation.crosswalk(ref_codes, v.crosswalk, classes)
    ref_class[~valid] = -1
    present = {classes[i]: int((ref_class == i).sum()) for i in range(len(classes))}
    logger.info("Reference class pixels in scene: %s", present)

    # --- KMeans: cross-tab -> majority labels -> score ---
    k = config.clustering.k
    with rasterio.open(scene.outputs_dir / f"kmeans_clusters_K{k}.tif") as ds:
        km = ds.read(1).astype(np.int16)
    tab = validation.crosstab(km, ref_class, k, classes)
    tab.to_csv(out / "kmeans_vs_reference_crosstab.csv")
    labels = validation.majority_labels(tab)
    logger.info("Cluster -> majority reference class: %s", labels)

    km_pred = np.full(km.shape, -1, dtype=np.int16)
    for cl, name in labels.items():
        km_pred[km == cl] = classes.index(name)
    cm_km, m_km = validation.score(km_pred, ref_class, classes)
    cm_km.to_csv(out / "kmeans_confusion_matrix.csv")
    _save_confusion(cm_km, "KMeans (cluster-labeled) vs MapBiomas", out / "kmeans_confusion_matrix.png")

    # --- SAM: direct crosswalk -> score ---
    sam_path = scene.outputs_dir / "sam_fullscene_class.tif"
    m_sam = None
    if sam_path.exists():
        with rasterio.open(sam_path) as ds:
            sam = ds.read(1).astype(np.int16)
        sam_pred = np.full(sam.shape, -1, dtype=np.int16)
        for sam_idx, name in SAM_TO_TARGET.items():
            sam_pred[sam == sam_idx] = classes.index(name)
        cm_sam, m_sam = validation.score(sam_pred, ref_class, classes)
        cm_sam.to_csv(out / "sam_confusion_matrix.csv")
        _save_confusion(cm_sam, "SAM vs MapBiomas", out / "sam_confusion_matrix.png")
    else:
        logger.warning("No SAM raster; skipping SAM validation.")

    summary = {"reference": "MapBiomas Bolivia 2024", "classes": classes,
               "reference_pixels_in_scene": present,
               "cluster_majority_labels": labels,
               "kmeans": m_km, "sam": m_sam}
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log_metrics("KMeans", m_km)
    if m_sam:
        _log_metrics("SAM", m_sam)


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))