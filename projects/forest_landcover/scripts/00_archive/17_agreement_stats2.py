from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio
import matplotlib.pyplot as plt

from wyvernhsi.paths import OUTPUTS_DIR


SAM_TIF = OUTPUTS_DIR / "sam_fullscene_class.tif"
KMEANS_TIF = OUTPUTS_DIR / "kmeans_semantic_K5.tif"

# ---- Map KMeans semantic classes (0..4) into SAM-style classes (0..2) ----
# SAM: 0=dense_trees, 1=bright_veg, 2=low_veg_soil, -1=unclassified
# KMeans semantic IDs: 0 field A, 1 field B, 2 dense forest, 3 bare soil, 4 mixed veg transition
KMEANS_TO_SAM = {
    1: 0,  # dense_forest -> trees
    0: 1,  # crop_transition -> vegetation
    2: 1,  # vegetation_other -> vegetation
    3: 1,  # crop_field -> vegetation
    4: 2,  # soil_or_dry_veg -> soil
}


SAM_LABELS = {-1: "unclassified", 0: "dense_trees", 1: "bright_veg", 2: "low_veg_soil"}
K_LABELS = {-1: "unclassified", 0: "field_A", 1: "field_B", 2: "dense_forest", 3: "bare_soil", 4: "mixed_transition"}


def ensure_same_grid(a: rasterio.DatasetReader, b: rasterio.DatasetReader):
    if (a.width, a.height) != (b.width, b.height):
        raise ValueError(f"Shape mismatch: SAM {a.height}x{a.width} vs KMeans {b.height}x{b.width}")
    if a.transform != b.transform:
        raise ValueError("Transform mismatch: rasters are not on the same grid.")
    if a.crs != b.crs:
        raise ValueError("CRS mismatch: rasters are not in the same CRS.")


def remap_kmeans_to_sam(k: np.ndarray) -> np.ndarray:
    out = np.full_like(k, -1, dtype=np.int16)
    for src, dst in KMEANS_TO_SAM.items():
        out[k == src] = np.int16(dst)
    out[k == -1] = -1
    return out


def confusion_matrix(a: np.ndarray, b: np.ndarray, a_vals: list[int], b_vals: list[int]) -> np.ndarray:
    """
    Count occurrences of (a==ai, b==bi).
    Returns matrix shape (len(a_vals), len(b_vals))
    """
    mat = np.zeros((len(a_vals), len(b_vals)), dtype=np.int64)
    for i, av in enumerate(a_vals):
        am = (a == av)
        if not np.any(am):
            continue
        for j, bv in enumerate(b_vals):
            mat[i, j] = int(np.sum(am & (b == bv)))
    return mat


def save_heatmap(mat: np.ndarray, row_labels: list[str], col_labels: list[str], out_png: Path, title: str):
    plt.figure(figsize=(10, 7))
    plt.imshow(mat)
    plt.title(title)
    plt.xlabel("KMeans (mapped)")
    plt.ylabel("SAM")
    plt.xticks(np.arange(len(col_labels)), col_labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(row_labels)), row_labels)

    # annotate counts
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    out_csv = OUTPUTS_DIR / "agreement_confusion_matrix.csv"
    out_png = OUTPUTS_DIR / "agreement_confusion_matrix.png"

    with rasterio.open(SAM_TIF) as ds_sam, rasterio.open(KMEANS_TIF) as ds_km:
        ensure_same_grid(ds_sam, ds_km)
        sam = ds_sam.read(1).astype(np.int16)
        km = ds_km.read(1).astype(np.int16)

    # Remap kmeans into SAM-like 3 classes
    km_as_sam = remap_kmeans_to_sam(km)

    # Masks
    valid_both = (sam != -1) & (km_as_sam != -1)

    total = sam.size
    nodata_or_unclassified = int(np.sum(~valid_both))
    overlap = int(np.sum(valid_both))

    # Agreement among valid pixels
    agree = int(np.sum((sam == km_as_sam) & valid_both))
    agree_rate = (agree / overlap) if overlap > 0 else float("nan")

    print("=== Agreement stats (SAM vs KMeans->SAM mapping) ===")
    print(f"Total pixels: {total}")
    print(f"Valid overlap (both classified): {overlap} ({overlap/total:.2%})")
    print(f"Excluded (either unclassified): {nodata_or_unclassified} ({nodata_or_unclassified/total:.2%})")
    print(f"Agreement on valid overlap: {agree} / {overlap} = {agree_rate:.2%}")

    # Confusion matrix on valid pixels only
    sam_vals = [0, 1, 2]
    km_vals = [0, 1, 2]  # mapped into SAM space
    mat = confusion_matrix(sam[valid_both], km_as_sam[valid_both], sam_vals, km_vals)

    row_labels = [SAM_LABELS[v] for v in sam_vals]
    col_labels = [SAM_LABELS[v] for v in km_vals]

    # Save CSV
    header = ["SAM \\ KMeans_mapped"] + col_labels
    lines = [",".join(header)]
    for i, rlab in enumerate(row_labels):
        row = [rlab] + [str(mat[i, j]) for j in range(mat.shape[1])]
        lines.append(",".join(row))
    out_csv.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote:", out_csv)

    # Heatmap PNG
    save_heatmap(mat, row_labels, col_labels, out_png, title="Confusion matrix (valid overlap only)")
    print("Wrote:", out_png)

    # Also show per-class agreement rates
    print("\nPer-class precision-ish (KMeans mapped -> SAM):")
    # column-wise: of pixels predicted as class c by kmeans_mapped, how many match sam
    for j, c in enumerate(km_vals):
        col_sum = mat[:, j].sum()
        correct = mat[j, j]  # diagonal aligns since same label set
        if col_sum > 0:
            print(f"  {SAM_LABELS[c]}: {correct}/{col_sum} = {correct/col_sum:.2%}")

    print("\nPer-class recall-ish (SAM -> KMeans mapped):")
    for i, c in enumerate(sam_vals):
        row_sum = mat[i, :].sum()
        correct = mat[i, i]
        if row_sum > 0:
            print(f"  {SAM_LABELS[c]}: {correct}/{row_sum} = {correct/row_sum:.2%}")


if __name__ == "__main__":
    main()
