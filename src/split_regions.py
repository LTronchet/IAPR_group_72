import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict


def extract_cluster_crops(
    img,
    min_area=3000,
    eps=180,
    ratio_limit=6,
    pad_ratio=0.2
):
    """
    Detects regions in an image and returns crops by cluster DBSCAN.

    Returns:
        dict:
            {
                label: {
                    "image": crop (np.array),
                    "bbox": (x1, y1, x2, y2),
                }
            }
    """

    original = img.copy()

    # =========================================================
    # MASK (simple placeholder)
    # =========================================================

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, final_mask = cv2.threshold(
        gray,
        240,
        255,
        cv2.THRESH_BINARY_INV
    )

    kernel = np.ones((5, 5), np.uint8)

    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    # =========================================================
    # CONTOURS
    # =========================================================

    contours, _ = cv2.findContours(
        final_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    points = []
    boxes = []

    for cnt in contours:

        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if min(w, h) == 0:
            continue

        ratio = max(w, h) / min(w, h)
        if ratio > ratio_limit:
            continue

        cx = x + w // 2
        cy = y + h // 2

        points.append([cx, cy])
        boxes.append((x, y, w, h))

    if len(points) == 0:
        return {}

    points = np.array(points)

    # =========================================================
    # DBSCAN
    # =========================================================

    dbscan = DBSCAN(eps=eps, min_samples=1)
    labels = dbscan.fit_predict(points)

    # =========================================================
    # COMPUTE CLUSTER CENTERS
    # =========================================================

    unique_labels = np.unique(labels)

    cluster_centers = {}

    for label in unique_labels:

        if label == -1:
            continue

        cluster_points = points[labels == label]

        center = np.mean(cluster_points, axis=0)

        cluster_centers[label] = center


    # =========================================================
    # CLASSIFY REGIONS
    # =========================================================

    H, W = img.shape[:2]

    regions = {}

    for label, center in cluster_centers.items():

        cx, cy = center

        # CENTER
        if (
            abs(cx - W / 2) < W * 0.18
            and
            abs(cy - H / 2) < H * 0.18
        ):

            regions[label] = "center"

        # TOP
        elif cy < H * 0.35:

            regions[label] = "p3"

        # BOTTOM
        elif cy > H * 0.65:

            regions[label] = "p1"

        # LEFT
        elif cx < W * 0.5:

            regions[label] = "p4"

        # RIGHT
        else:

            regions[label] = "p2"

    # =========================================================
    # GROUP BOXES BY CLUSTER
    # =========================================================

    cluster_boxes = defaultdict(list)

    for box, label in zip(boxes, labels):
        if label == -1:
            continue
        cluster_boxes[label].append(box)

    def merge_boxes(box_list):
        x_min = min(x for x, y, w, h in box_list)
        y_min = min(y for x, y, w, h in box_list)
        x_max = max(x + w for x, y, w, h in box_list)
        y_max = max(y + h for x, y, w, h in box_list)
        return x_min, y_min, x_max - x_min, y_max - y_min

    def add_padding(x, y, w, h, shape):
        H, W = shape[:2]

        pad_x = int(w * pad_ratio)
        pad_y = int(h * pad_ratio)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(W, x + w + pad_x)
        y2 = min(H, y + h + pad_y)

        return x1, y1, x2, y2

    # =========================================================
    # CREATE CROPS
    # =========================================================

    results = {}

    for label, box_list in cluster_boxes.items():

        x, y, w, h = merge_boxes(box_list)
        x1, y1, x2, y2 = add_padding(x, y, w, h, original.shape)

        crop = original[y1:y2, x1:x2].copy()

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        region_name = regions.get(label, "unknown")

        results[label] = {
            "cluster_id": label,
            "region": region_name,
            "centroid": (cx, cy),
            "bbox": (x1, y1, x2, y2),
            "image": crop
        }

    return results

# ==============================================
# MAIN
# ==============================================

if __name__ == "__main__":

    img = cv2.imread("test2.jpg")

    result = extract_cluster_crops(
        img
    )
    #for label, data in result.items():
    #    cv2.imwrite(f"crop_{label}_{data['region']}.jpg", data["image"])