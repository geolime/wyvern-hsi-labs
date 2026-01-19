from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, ACTIVE_WYVERN_MASK, OUTPUTS_DIR
from wyvernhsi.masks import load_valid_mask, load_water_mask


# ---------------- Settings ----------------
# Bands (1-based)
B570 = 6
B660 = 12
B669 = 13
B711 = 17
B764 = 21

EPS = 1e-6

IN_DIR = OUTPUTS_DIR / "water_features" / "sfa_kmeans"
LABELS_TIF = IN_DIR / "sfa_kmeans_K5.tif"  # from script 10

OUT_CSV = IN_DIR / "sfa_kmeans_K5_class_summary.csv"
OUT_PNG = IN_DIR / "sfa_kmeans_K5_class_summary.png"


# ---------------- Helpers ----------------
def nd(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float32)
    denom = a + b
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(denom) > EPS)
    out[ok] = (a[ok] - b[ok]) / denom[ok]
    return out


def ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float32)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > EPS)
    out[ok] = a[ok] / b[ok]
    return out


def read_band(ds: rasterio.DatasetReader, b1: int) -> np.ndarray:
    x = ds.read(b1).astype(np.float32)
    if ds.nodata is not None:
        x[x == ds.nodata] = np.nan
    return x


def tag_from_medians(ndti_med: float, ndci_med: float, nir_red_med: float) -> str:
    # These thresholds are heuristic and scene-dependent; keep labels gentle.
    # NDTI tends to be more negative in your scene; "less negative" => clearer.
    if ndti_med <= -0.22:
        return "sediment-dominant"
    if ndti_med <= -0.12 and (ndci_med > -0.12 or nir_red_med > 0.55):
        return "mixed (sediment + bio)"
    if (ndci_med > -0.10) or (nir_red_med > 0.60):
        return "bio-enhanced / clearer"
    return "mixed"


def render_table_png(df: pd.DataFrame, out_png: Path, title: str) -> None:
    plt.figure(figsize=(12, 2.0 + 0.35 * len(df)))
    plt.axis("off")
    plt.title(title)

    # Format a display copy
    d = df.copy()
    for col in ["water_fraction", "ndti_median", "ndci_median", "nir_red_median"]:
        if col in d.columns:
            d[col] = d[col].map(lambda x: f"{x:.3f}")
    if "n" in d.columns:
        d["n"] = d["n"].map(lambda x: f"{int(x):,}")

    table = plt.table(
        cellText=d.values,
        colLabels=d.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print("Wrote:", out_png)


# ---------------- Main ----------------
def main() -> None:
    if not LABELS_TIF.exists():
        raise FileNotFoundError(f"Missing SFA KMeans labels: {LABELS_TIF}")

    valid = load_valid_mask(ACTIVE_WYVERN_MASK).astype(bool)
    water = load_water_mask().astype(bool)
    use = valid & water

    with rasterio.open(LABELS_TIF) as ds_lab:
        lab = ds_lab.read(1).astype(np.int16)

    # Compute proxies again (standalone)
    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        r570 = read_band(ds, B570)
        r660 = read_band(ds, B660)
        r669 = read_band(ds, B669)
        r711 = read_band(ds, B711)
        r764 = read_band(ds, B764)

    ndti = nd(r660, r570)
    ndci = nd(r711, r669)
    nir_red = ratio(r764, r669)

    ndti[~use] = np.nan
    ndci[~use] = np.nan
    nir_red[~use] = np.nan

    # Summaries per class (water-only)
    m = use & (lab >= 0) & np.isfinite(ndti) & np.isfinite(ndci) & np.isfinite(nir_red)
    if not np.any(m):
        raise RuntimeError("No overlapping pixels between water mask and class raster.")

    water_n = int(use.sum())

    rows = []
    for k in sorted(np.unique(lab[m]).tolist()):
        mk = m & (lab == k)
        n = int(mk.sum())
        rows.append({
            "class": int(k),
            "n": n,
            "water_fraction": float(n / water_n),
            "ndti_median": float(np.nanmedian(ndti[mk])),
            "ndci_median": float(np.nanmedian(ndci[mk])),
            "nir_red_median": float(np.nanmedian(nir_red[mk])),
        })

    df = pd.DataFrame(rows).sort_values("class").reset_index(drop=True)
    df["tag"] = [
        tag_from_medians(r.ndti_median, r.ndci_median, r.nir_red_median)  # type: ignore
        for r in df.itertuples(index=False)
    ]

    df.to_csv(OUT_CSV, index=False)
    print("Wrote:", OUT_CSV)

    render_table_png(df, OUT_PNG, "SFA KMeans class summary (water-only medians)")
    print("Done.")


if __name__ == "__main__":
    main()
