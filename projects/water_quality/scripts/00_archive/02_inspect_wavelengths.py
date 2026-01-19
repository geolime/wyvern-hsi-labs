import numpy as np
import rasterio

from wyvernhsi.wavelengths import parse_wavelengths_nm_from_descriptions

URL = "https://wyvern-prod-public-open-data-program.s3.ca-central-1.amazonaws.com/wyvern_dragonette-001_20250724T071146_fbaa00bd/wyvern_dragonette-001_20250724T071146_fbaa00bd.tiff"

with rasterio.open(URL) as ds:
    print("bands:", ds.count)
    desc = list(ds.descriptions)

    print("\nAll band descriptions:")
    for i, d in enumerate(desc[:], start=1):
        print(f"{i:>3}: {d}")

    wl = parse_wavelengths_nm_from_descriptions(desc)
    n_ok = int(np.isfinite(wl).sum())
    print(f"\nParsed wavelengths (nm): {n_ok}/{len(wl)} bands")

    if n_ok:
        print(f"Range: {np.nanmin(wl):.1f}–{np.nanmax(wl):.1f} nm")

        # Show a few sample pairs
        print("\nSample band -> nm:")
        for i in np.linspace(0, len(wl) - 1, num=min(10, len(wl)), dtype=int):
            print(f"{i+1:>3} -> {wl[i]}")
    else:
        print("\nNo wavelengths found in band descriptions.")
        print("Next step would be checking dataset tags/metadata for wavelengths.")
