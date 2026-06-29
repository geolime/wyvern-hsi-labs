"""
Supervised land-cover classification (Random Forest), trained on MapBiomas-sampled labels
with a SPATIALLY-BLOCKED train/test split (so adjacent-pixel leakage can't inflate scores).

Features per pixel: TOA reflectance (all bands) + NDVI + red-edge slope. Labels: crosswalked
MapBiomas (trees/crops/bare). This measures how well the spectra REPRODUCE MapBiomas labels
on held-out spatial blocks — concordance with a reference model, not ground truth.
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
from sklearn.ensemble import RandomForestClassifier

from wyvernhsi import indices, io, validation, visualization
from wyvernhsi.config import Config, load_config
from wyvernhsi.logging_setup import configure_logging
from wyvernhsi.masks import load_valid_mask
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions

logger = logging.getLogger(__name__)

_CM_CMAP = LinearSegmentedColormap.from_list(
    "retro", list(reversed(["#01204E", "#028391", "#F6DCAC", "#FAA968", "#F85525"])))


def _save_confusion(cm_df, title, out_png):
    cm = cm_df.values.astype(float)
    total = cm.sum()
    norm = cm / cm.max() if cm.max() else cm
    plt.figure(figsize=(7.5, 6.5))
    plt.imshow(norm, cmap=_CM_CMAP, vmin=0, vmax=1)
    plt.xticks(range(len(cm_df.columns)), [c.replace("ref_", "") for c in cm_df.columns])
    plt.yticks(range(len(cm_df.index)), [c.replace("pred_", "") for c in cm_df.index])
    plt.xlabel("Reference"); plt.ylabel("Predicted")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            n = int(cm[i, j]); pct = (n / total * 100) if total else 0.0
            plt.text(j, i, f"{n:,}\n{pct:.1f}%", ha="center", va="center",
                     fontsize=9, color="white" if norm[i, j] > 0.55 else "#01204E")
    plt.title(title); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()
    logger.info("Wrote: %s", out_png)


def _save_class_map(lab, classes, out_png, title):
    k = len(classes)
    plt.figure(figsize=(12, 10))
    plt.imshow(np.ma.masked_where(lab < 0, lab), cmap=visualization.masked_cmap(), vmin=0, vmax=k - 1)
    plt.axis("off"); plt.title(title)
    visualization.add_class_legend(classes, colors=visualization.class_colors(k))
    plt.figtext(0.5, 0.02, "white / transparent = nodata, cloud & QA-masked areas",
                ha="center", fontsize=8, color="0.4")
    plt.tight_layout(); plt.savefig(out_png, dpi=200, transparent=True); plt.close()
    logger.info("Wrote: %s", out_png)


def _balanced_sample(label_yx, mask, classes, max_per_class, seed):
    """Pixel indices sampled up to max_per_class per class, within mask. Returns flat indices."""
    rng = np.random.default_rng(seed)
    flat_lab = label_yx.ravel()
    flat_ok = mask.ravel()
    picks = []
    for ci in range(len(classes)):
        idx = np.flatnonzero(flat_ok & (flat_lab == ci))
        if idx.size:
            picks.append(rng.choice(idx, size=min(max_per_class, idx.size), replace=False))
    return np.concatenate(picks) if picks else np.array([], dtype=int)


def main(config: Config) -> None:
    rf_cfg = config.random_forest
    v = config.validation
    if v is None or rf_cfg is None or not (config.project_dir / v.reference_tif).exists():
        logger.warning("Missing validation reference or RF config; skipping RF stage.")
        return

    classes = list(v.crosswalk.keys())
    f = config.features
    seed = config.random_seed
    scene = resolve_scene(config.project_dir)
    out = scene.outputs_dir / "rf"
    out.mkdir(parents=True, exist_ok=True)

    valid = load_valid_mask(scene.mask)
    with rasterio.open(scene.reflectance) as ds:
        profile = ds.profile
        cube = io.read_cube(ds)                       # (H, W, B)
        scene_nm = np.array(parse_wavelengths_nm_from_descriptions(list(ds.descriptions)), dtype=np.float32)
        red = io.read_band_nm(ds, f.red_nm)
        red_edge = io.read_band_nm(ds, f.red_edge_nm)
        nir = io.read_band_nm(ds, f.nir_nm)

    ndvi = indices.ndvi(nir, red)
    re_slope = indices.red_edge_slope(red_edge, red, f.red_edge_nm, f.red_nm)
    feats = np.dstack([cube, ndvi, re_slope]).astype(np.float32)   # (H, W, B+2)
    feat_names = [f"b{int(nm)}nm" for nm in scene_nm] + ["ndvi", "re_slope"]

    ref_codes = io.align_reference(config.project_dir / v.reference_tif, profile)
    ref_codes[np.isin(ref_codes, v.reference_ignore)] = 0
    label = validation.crosswalk(ref_codes, v.crosswalk, classes)   # -1 where unmapped

    usable = valid & (label >= 0) & np.isfinite(feats).all(axis=2)
    train_mask, test_mask = validation.block_split(
        usable, n_blocks=rf_cfg.n_blocks, test_frac=rf_cfg.test_frac, seed=seed)
    logger.info("Usable=%d  train=%d  test=%d (spatial blocks)",
                int(usable.sum()), int(train_mask.sum()), int(test_mask.sum()))

    F = feats.reshape(-1, feats.shape[2])
    L = label.ravel()
    train_idx = _balanced_sample(label, train_mask, classes, rf_cfg.max_samples_per_class, seed)
    logger.info("Training samples (balanced): %d", train_idx.size)

    clf = RandomForestClassifier(n_estimators=rf_cfg.n_estimators, max_depth=rf_cfg.max_depth,
                                 random_state=seed, n_jobs=-1, class_weight="balanced")
    clf.fit(F[train_idx], L[train_idx])

    # Evaluate on held-out spatial blocks (all test pixels, not a sample)
    test_idx = np.flatnonzero(test_mask.ravel())
    pred_test = clf.predict(F[test_idx])
    pred_map = np.full(label.shape, -1, dtype=np.int16)
    pred_map.ravel()[test_idx] = pred_test
    cm, metrics = validation.score(pred_map, label, classes)
    cm.to_csv(out / "rf_confusion_matrix.csv")

    # Full-scene classification map (predict everywhere valid; scoring above used test blocks only)
    full_ok = valid & np.isfinite(feats).all(axis=2)
    full_idx = np.flatnonzero(full_ok.ravel())
    rf_map = np.full(label.shape, -1, dtype=np.int16)
    rf_map.ravel()[full_idx] = clf.predict(F[full_idx]).astype(np.int16)

    io.write_geotiff(profile, out / "rf_classification.tif", rf_map, nodata=-1, dtype="int16",
                     descriptions=["RF_CLASS"])
    with rasterio.open(out / "rf_classification.tif", "r+") as dst:
        dst.update_tags(classes=";".join(classes), method="random_forest",
                        eval="held-out spatial blocks",
                        note="labels from MapBiomas; concordance not ground truth")
    _save_class_map(rf_map, classes, out / "rf_classification.png", "Random Forest land cover")
    _save_confusion(cm, "Random Forest vs MapBiomas (held-out blocks)", out / "rf_confusion_matrix.png")

    importance = sorted(zip(feat_names, clf.feature_importances_.tolist()),
                        key=lambda kv: kv[1], reverse=True)
    (out / "rf_feature_importance.csv").write_text(
        "feature,importance\n" + "\n".join(f"{n},{v:.6f}" for n, v in importance), encoding="utf-8")

    summary = {"reference": "MapBiomas Bolivia (held-out spatial blocks)", "classes": classes,
               "n_train": int(train_idx.size), "n_test": int(test_idx.size),
               "metrics": metrics, "top_features": importance[:10]}
    (out / "rf_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("RF held-out: kappa=%.3f overall=%.3f", metrics["kappa"], metrics["overall_agreement"])
    for name, pc in metrics["per_class"].items():
        if pc["reference_support"]:
            logger.info("  %-6s precision=%s recall=%s support=%d", name,
                        f"{pc['precision']:.3f}" if pc["precision"] is not None else "n/a",
                        f"{pc['recall']:.3f}" if pc["recall"] is not None else "n/a",
                        pc["reference_support"])
    logger.info("Top features: %s", ", ".join(n for n, _ in importance[:5]))


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))