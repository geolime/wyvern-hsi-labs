import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt

from wyvernhsi.paths import ACTIVE_WYVERN_FILE, OUTPUTS_DIR
from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions, pick_band_index_nearest


KMEANS_TIF = OUTPUTS_DIR / "kmeans_clusters_K5.tif"
SAM_TIF    = OUTPUTS_DIR / "sam_fullscene_class.tif"


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(ACTIVE_WYVERN_FILE) as ds:
        wl = parse_wavelengths_nm_from_descriptions(list(ds.descriptions))

        b_red  = pick_band_index_nearest(wl, 660) + 1
        b_re   = pick_band_index_nearest(wl, 720) + 1
        b_nir  = pick_band_index_nearest(wl, 800) + 1

        red = ds.read(b_red).astype(np.float32)
        re  = ds.read(b_re).astype(np.float32)
        nir = ds.read(b_nir).astype(np.float32)

    # Indices
    ndvi = (nir - red) / (nir + red + 1e-6)
    re_slope = (re - red) / (720 - 660)

    # Load classifications
    with rasterio.open(KMEANS_TIF) as ds:
        km = ds.read(1)

    with rasterio.open(SAM_TIF) as ds:
        sam = ds.read(1)

    results = []

    # ----- KMeans stats -----
    for k in range(5):
        m = (km == k)
        if m.sum() < 1000:
            continue

        results.append({
            "source": "kmeans",
            "class": f"cluster_{k}",
            "ndvi_mean": float(np.nanmean(ndvi[m])),
            "ndvi_std": float(np.nanstd(ndvi[m])),
            "re_slope_mean": float(np.nanmean(re_slope[m])),
            "re_slope_std": float(np.nanstd(re_slope[m])),
            "count": int(m.sum())
        })

    # ----- SAM stats -----
    sam_names = {
        0: "trees",
        1: "vegetation",
        2: "soil"
    }

    for k, name in sam_names.items():
        m = (sam == k)
        if m.sum() < 1000:
            continue

        results.append({
            "source": "sam",
            "class": name,
            "ndvi_mean": float(np.nanmean(ndvi[m])),
            "ndvi_std": float(np.nanstd(ndvi[m])),
            "re_slope_mean": float(np.nanmean(re_slope[m])),
            "re_slope_std": float(np.nanstd(re_slope[m])),
            "count": int(m.sum())
        })

    df = pd.DataFrame(results)
    out_csv = OUTPUTS_DIR / "spectral_index_stats.csv"
    df.to_csv(out_csv, index=False)

    print("Wrote:", out_csv)

    # ----- Boxplots (nice for README) -----
    plt.figure(figsize=(12, 5))

    # NDVI boxplot
    plt.subplot(1, 2, 1)
    data = []
    labels = []

    for k in range(5):
        m = (km == k)
        if m.sum() > 1000:
            data.append(ndvi[m])
            labels.append(f"K{k}")

    plt.boxplot(data, showfliers=False)
    plt.title("NDVI by KMeans cluster")
    plt.xticks(range(1, len(labels) + 1), labels)

    # Red-edge slope boxplot
    plt.subplot(1, 2, 2)
    data = []
    labels = []

    for k in range(5):
        m = (km == k)
        if m.sum() > 1000:
            data.append(re_slope[m])
            labels.append(f"K{k}")

    plt.boxplot(data, showfliers=False)
    plt.title("Red-edge slope by KMeans cluster")
    plt.xticks(range(1, len(labels) + 1), labels)

    plt.tight_layout()
    out_png = OUTPUTS_DIR / "spectral_indices_kmeans.png"
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("Wrote:", out_png)


if __name__ == "__main__":
    main()
