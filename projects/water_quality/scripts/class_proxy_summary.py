"""
Per-class summary of optical proxies over the SFA KMeans water classes (from stage 10).

Reports per-class water fraction and median NDTI / NDCI / NIR-Red, plus a relative
turbidity tier consistent with the NDTI ordering used to label the classes. Writes a CSV
and a rendered table PNG under outputs/water_features/sfa_kmeans/.

NOTE: medians are optical proxies on TOA reflectance, not validated concentrations.
"""
from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

from wyvernhsi import indices, io
from wyvernhsi.config import Config, load_config
from wyvernhsi.logging_setup import configure_logging
from wyvernhsi.masks import load_valid_mask, load_water_mask
from wyvernhsi.paths import project_dir_of, repo_root, resolve_scene

logger = logging.getLogger(__name__)

# Stage 10 already orders classes by median NDTI (0 = clearest, K-1 = most turbid).
TURBIDITY_TIERS = {
    0: "clearest (lowest NDTI)",
    1: "low turbidity",
    2: "moderate turbidity",
    3: "turbid",
    4: "most turbid (highest NDTI)",
}


def _render_table(df, out_png, title):
    plt.figure(figsize=(12, 2.0 + 0.35 * len(df)))
    plt.axis("off")
    plt.title(title)
    d = df.copy()
    for col in ("water_fraction", "ndti_median", "ndci_median", "nir_red_median"):
        if col in d.columns:
            d[col] = d[col].map(lambda x: f"{x:.3f}")
    if "n" in d.columns:
        d["n"] = d["n"].map(lambda x: f"{int(x):,}")
    table = plt.table(cellText=d.values, colLabels=list(d.columns), cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    logger.info("Wrote: %s", out_png)


def main(config: Config) -> None:
    s = config.sfa
    k = config.clustering.k
    scene = resolve_scene(config.project_dir)
    in_dir = scene.outputs_dir / "water_features" / "sfa_kmeans"
    labels_tif = in_dir / f"sfa_kmeans_K{k}.tif"
    if not labels_tif.exists():
        raise FileNotFoundError(f"Missing SFA KMeans labels: {labels_tif}. Run stage 10 first.")

    valid = load_valid_mask(scene.mask)
    water = load_water_mask(scene.outputs_dir / "masks" / "water_mask.tif")
    use = valid & water

    with rasterio.open(labels_tif) as ds_lab:
        lab = ds_lab.read(1).astype(np.int16)

    with rasterio.open(scene.reflectance) as ds:
        ndti = indices.ndti(io.read_band_nm(ds, s.ndti_red_nm), io.read_band_nm(ds, s.ndti_green_nm))
        ndci = indices.ndci(io.read_band_nm(ds, s.ndci_red_edge_nm), io.read_band_nm(ds, s.ndci_red_nm))
        nir_red = indices.ratio(io.read_band_nm(ds, s.nir_red_nir_nm), io.read_band_nm(ds, s.nir_red_red_nm))

    for arr in (ndti, ndci, nir_red):
        arr[~use] = np.nan

    m = use & (lab >= 0) & np.isfinite(ndti) & np.isfinite(ndci) & np.isfinite(nir_red)
    if not np.any(m):
        raise RuntimeError("No overlapping pixels between water mask and class raster.")

    water_n = int(use.sum())
    rows = []
    for c in sorted(np.unique(lab[m]).tolist()):
        mk = m & (lab == c)
        n = int(mk.sum())
        rows.append({
            "class": int(c), "n": n, "water_fraction": float(n / water_n),
            "ndti_median": float(np.median(ndti[mk])),
            "ndci_median": float(np.median(ndci[mk])),
            "nir_red_median": float(np.median(nir_red[mk])),
            "tier": TURBIDITY_TIERS.get(int(c), f"class {int(c)}"),
        })

    df = pd.DataFrame(rows).sort_values("class").reset_index(drop=True)
    out_csv = in_dir / f"sfa_kmeans_K{k}_class_summary.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote: %s", out_csv)

    _render_table(df, in_dir / f"sfa_kmeans_K{k}_class_summary.png",
                  "SFA KMeans class summary (water-only medians)")


if __name__ == "__main__":
    configure_logging()
    main(load_config(repo_root() / "configs" / f"{project_dir_of(__file__).name}.yaml"))