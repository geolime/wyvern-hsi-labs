import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects

from wyvernhsi.paths import OUTPUTS_DIR

# Remove connected regions smaller than this many pixels
MIN_SIZE = 200  # try 50, 200, 500 depending on resolution/speckle size

def main():
    in_path = OUTPUTS_DIR / "sam_masked_class.npy"
    out_png = OUTPUTS_DIR / f"sam_masked_class_min{MIN_SIZE}.png"
    out_npy = OUTPUTS_DIR / f"sam_masked_class_min{MIN_SIZE}.npy"

    cls = np.load(in_path).astype(np.int16)

    cleaned = cls.copy()

    # Process each class label separately (ignore -1)
    labels = sorted([int(x) for x in np.unique(cls) if x != -1])
    for lab in labels:
        mask = (cls == lab)
        # Newer skimage uses max_size; to keep behavior similar, use MIN_SIZE-1
        mask_clean = remove_small_objects(mask, max_size=MIN_SIZE - 1)

        # wherever we removed this label, set to -1 (unclassified)
        cleaned[mask & ~mask_clean] = -1

    np.save(out_npy, cleaned)

    plt.figure(figsize=(10, 8))
    plt.imshow(cleaned)
    plt.title(f"SAM masked class map — remove regions < {MIN_SIZE}px")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print("Wrote:")
    print(" -", out_png)
    print(" -", out_npy)

if __name__ == "__main__":
    main()
