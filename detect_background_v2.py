import cv2
import numpy as np

from pathlib import Path


# =========================================================
# RESIZE KEEP RATIO
# =========================================================

def resize_keep_ratio(img, max_dim=1600):

    h, w = img.shape[:2]

    scale = max_dim / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    return resized


# =========================================================
# BACKGROUND REMOVAL
# =========================================================

def remove_background_grabcut(image_path, debug=False):

    img = cv2.imread(str(image_path))

    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img = resize_keep_ratio(img)

    original = img.copy()

    h, w = img.shape[:2]

    # =====================================================
    # MASK
    # =====================================================

    mask = np.zeros((h, w), np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # =====================================================
    # FOREGROUND RECTANGLE
    # =====================================================

    margin_x = int(w * 0.02)
    margin_y = int(h * 0.02)

    rect = (
        margin_x,
        margin_y,
        w - 2 * margin_x,
        h - 2 * margin_y
    )
    #rect = (1, 1, w-2, h-2)

    # =====================================================
    # RUN GRABCUT
    # =====================================================

    cv2.grabCut(
        img,
        mask,
        rect,
        bgdModel,
        fgdModel,
        8,
        cv2.GC_INIT_WITH_RECT
    )

    # =====================================================
    # FINAL MASK
    # =====================================================

    final_mask = np.where(
        (mask == 2) | (mask == 0),
        0,
        1
    ).astype("uint8")

    # Smooth cleanup
    kernel = np.ones((7, 7), np.uint8)

    # Fill small holes inside cards
    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Remove tiny noise
    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Remove tiny connected components

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        final_mask,
        connectivity=8
    )

    clean_mask = np.zeros_like(final_mask)

    clean_mask = cv2.GaussianBlur(
        clean_mask.astype(np.float32),
        (5,5),
        0
    )

    clean_mask = (clean_mask > 0.5).astype("uint8")

    MIN_AREA = 10000

    for i in range(1, num_labels):

        area = stats[i, cv2.CC_STAT_AREA]

        if area > MIN_AREA:

            clean_mask[labels == i] = 1

    final_mask = clean_mask

    # Shrink mask slightly to remove border artifacts

    kernel_erode = np.ones((3,3), np.uint8)

    final_mask = cv2.erode(
        final_mask,
        kernel_erode,
        iterations=1
    )

    # =====================================================
    # MORPHOLOGICAL CLEANUP
    # =====================================================

    kernel = np.ones((5, 5), np.uint8)

    final_mask = cv2.morphologyEx(
        final_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # =====================================================
    # APPLY MASK
    # =====================================================

    result = original.copy()

    result[final_mask == 0] = [255, 255, 255]

    # =====================================================
    # DEBUG
    # =====================================================

    if debug:

        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

        cv2.imshow("Original", original)
        cv2.imshow("Mask", final_mask * 255)
        cv2.imshow("Result", result)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    INPUT_DIR = Path("train_images")

    OUTPUT_DIR = Path("outputs")

    OUTPUT_DIR.mkdir(exist_ok=True)

    image_paths = list(INPUT_DIR.glob("*"))

    print(f"Found {len(image_paths)} images\n")

    for image_path in image_paths:

        print(f"Processing: {image_path.name}")

        try:

            result = remove_background_grabcut(
                image_path,
                debug=False
            )

            output_path = OUTPUT_DIR / image_path.name

            cv2.imwrite(str(output_path), result)

            print(f"Saved: {output_path}\n")

        except Exception as e:

            print(f"ERROR with {image_path.name}")
            print(e)
            print()