import numpy as np
import rasterio

from wyvernhsi.paths import OUTPUTS_DIR

CLS_TIF = OUTPUTS_DIR / "sam_fullscene_class.tif"

def main():
    with rasterio.open(CLS_TIF) as ds:
        arr = ds.read(1)  # int16
        nodata = ds.nodata

    # Treat nodata as -1 just in case
    if nodata is not None:
        arr = np.where(arr == nodata, -1, arr)

    total = arr.size
    vals, counts = np.unique(arr, return_counts=True)

    print(f"Total pixels: {total}")
    print("Counts / fractions:")
    for v, c in zip(vals.tolist(), counts.tolist()):
        frac = c / total
        print(f"  {v:>3}: {c:>12}  ({frac:.3%})")

    unclassified = counts[vals == -1][0] if np.any(vals == -1) else 0
    print(f"\nUnclassified fraction: {unclassified/total:.3%}")

if __name__ == "__main__":
    main()
