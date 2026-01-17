import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import modal
from skimage.morphology import footprint_rectangle


from wyvernhsi.paths import OUTPUTS_DIR

# Choose 3 or 5. Start with 3.
KERNEL_SIZE = 5

def main():
    masked_path = OUTPUTS_DIR / "sam_masked_class.npy"
    out_png = OUTPUTS_DIR / f"sam_masked_class_mode{KERNEL_SIZE}.png"
    out_npy = OUTPUTS_DIR / f"sam_masked_class_mode{KERNEL_SIZE}.npy"

    cls = np.load(masked_path)  # int16 with -1 for unclassified

    # rank filters need non-negative integers; shift so -1 becomes 0
    cls_shift = (cls + 1).astype(np.uint16)

    # mode/majority filter
    footprint = footprint_rectangle((KERNEL_SIZE, KERNEL_SIZE))
    cls_filt_shift = modal(cls_shift, footprint)


    # shift back to original labels
    cls_filt = cls_filt_shift.astype(np.int16) - 1

    np.save(out_npy, cls_filt)

    plt.figure(figsize=(10, 8))
    plt.imshow(cls_filt)
    plt.title(f"SAM masked class map — majority filter {KERNEL_SIZE}x{KERNEL_SIZE}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()

    print("Wrote:")
    print(" -", out_png)
    print(" -", out_npy)

if __name__ == "__main__":
    main()
