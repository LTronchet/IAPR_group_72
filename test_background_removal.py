"""
Usage:
    python test_background_removal.py [IMAGE_ID]

Default IMAGE_ID: L1000988  (from test_images/)
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.background_removal import remove_background, _white_mask, _SAT_HI, _VAL_LO, _MIN_AREA

DATA_DIR = "iapr-26-uno-vision-challenge/test_images"


def run(image_id: str = "L1000988") -> None:
    path = os.path.join(DATA_DIR, f"{image_id}.jpg")
    img = cv2.imread(path)
    assert img is not None, f"Cannot load {path}"
    H, W = img.shape[:2]
    print(f"Image: {image_id}  ({W}x{H})  SAT_HI={_SAT_HI}  VAL_LO={_VAL_LO}")

    # --- Intermediate steps for visualization (mirrors remove_background internals) ---
    mask = _white_mask(img)

    ks_open = max(int(min(H, W) * 0.002) | 1, 8)
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_open, ks_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    ks = max(int(min(H, W) * 0.005) | 1, 25)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= _MIN_AREA]
    print(f"Contours kept (area >= {_MIN_AREA}): {len(contours)}")

    filled_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(filled_mask, contours, -1, 255, cv2.FILLED)

    # Opening to remove large background blobs that slipped through
    ks_post = max(int(min(H, W) * 0.065) | 1, 21)
    k_post  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_post, ks_post))
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, k_post)

    # Draw contours on a copy of the image
    vis = img.copy()
    cv2.drawContours(vis, contours, -1, (0, 255, 0), 3)
    for i, cnt in enumerate(contours):
        x, y, *_ = cv2.boundingRect(cnt)
        cv2.putText(vis, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # --- Final result from the module ---
    img_masked = remove_background(img)

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask_closed, cmap="gray")
    axes[1].set_title(f"White mask (sat<{_SAT_HI}, val>{_VAL_LO})")
    axes[1].axis("off")

    axes[2].imshow(filled_mask, cmap="gray")
    axes[2].set_title(f"Filled cards ({len(contours)})")
    axes[2].axis("off")

    axes[3].imshow(cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB))
    axes[3].set_title("Background removed")
    axes[3].axis("off")

    fig.suptitle(image_id, fontsize=13)
    plt.tight_layout()
    plt.show()


def run_all(start_id: str = "L1000922") -> None:
    image_ids = sorted(f[:-4] for f in os.listdir(DATA_DIR) if f.lower().endswith(".jpg"))
    image_ids = [i for i in image_ids if i >= start_id]
    print(f"Running on {len(image_ids)} images (>= {start_id}) in {DATA_DIR!r}")
    for image_id in image_ids:
        run(image_id)


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 1:
        run(sys.argv[1])
    else:
        run_all()
