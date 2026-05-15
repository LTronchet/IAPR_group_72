from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# HSV thresholds — derived from filtered pixel analysis of token/leaf crops
#
# Black token (V<=150 filter to remove white-background leakage):
#   S p95=70 → upper 75   V p95=132 → upper 140
#
# Yellow token vs lime-green leaves (only H separates them):
#   Token H: p10=22, p75=25, p90=33
#   Leaf  H: p1=32  → virtually no leaf pixel has H <= 31
#   H upper = 30 excludes leaves while keeping ~87% of token pixels.
# ---------------------------------------------------------------------------
_BLACK_TOKEN_HSV = {
    'lower': np.array([0,   0,   0]),
    'upper': np.array([180, 75, 140]),
}
_YELLOW_TOKEN_HSV = {
    'lower': np.array([18,  70, 120]),
    'upper': np.array([30, 255, 255]),
}

_MIN_BLOB_AREA   = 2000
_MAX_BLOB_AREA   = 80000
_MIN_SHAPE_SCORE = 0.4

# Opening kernels (erosion → dilation):
#   Circle 25x25: erodes thin skip-card arcs, solid token disk survives.
#   Rect   5x5:   light denoising only.
_KERNEL_CIRCLE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
_KERNEL_RECT   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _find_token_centroid(mask: np.ndarray, shape_mode: str) -> Optional[Tuple[float, float]]:
    """
    Return the centroid of the largest blob that passes area + shape filters.

    Args:
        mask:       Binary mask (uint8, 0/255).
        shape_mode: 'circle' scores by circularity; 'rect' scores by extent.

    Returns:
        (cx, cy) in pixel coordinates, or None if no valid blob found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_area, best_centroid = -1, None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < _MIN_BLOB_AREA or area > _MAX_BLOB_AREA:
            continue

        if shape_mode == 'circle':
            perimeter = cv2.arcLength(cnt, closed=True)
            if perimeter == 0:
                continue
            score = (4 * np.pi * area) / (perimeter ** 2)
        else:
            x, y, bw, bh = cv2.boundingRect(cnt)
            bbox_area = bw * bh
            if bbox_area == 0:
                continue
            score = area / bbox_area

        if score < _MIN_SHAPE_SCORE:
            continue

        if area > best_area:
            best_area = area
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            best_centroid = (M['m10'] / M['m00'], M['m01'] / M['m00'])

    return best_centroid


def _centroid_to_player(cx: float, cy: float, img_w: int, img_h: int) -> str:
    """
    Map a token centroid to one of the four fixed player positions.

    Zones (bottom-left origin convention):
        p1 (bottom): user_y < H/5       and  W/4 < cx < 3W/4
        p3 (top):    user_y > 4H/5      and  W/4 < cx < 3W/4
        p2 (right):  cx > 3.85W/5
        p4 (left):   cx < 1.15W/5
    Falls back to nearest edge for ambiguous centroids.
    """
    user_y    = img_h - cy
    x_centred = (img_w / 4) < cx < (3 * img_w / 4)

    if user_y < img_h / 5 and x_centred:
        return 'p1'
    if user_y > 4 * img_h / 5 and x_centred:
        return 'p3'
    if cx > 3.85 * img_w / 5:
        return 'p2'
    if cx < 1.15 * img_w / 5:
        return 'p4'

    distances = {
        'p1': img_h - cy,
        'p2': img_w - cx,
        'p3': cy,
        'p4': cx,
    }
    return min(distances, key=distances.get)


def detect_active_player(img: np.ndarray, background: str) -> Optional[str]:
    """
    Detect the active player from the token visible in the image.

    Pipeline:
        1. HSV threshold to isolate the token colour.
        2. Morphological opening to remove thin confusers (skip-card arcs).
        3. Select the largest blob with a valid shape score.
        4. Map centroid coordinates to a player zone.

    Args:
        img:        RGB image as a numpy array (H, W, 3).
        background: 'white' (black rectangular token) or
                    'noisy' (yellow circular token).

    Returns:
        One of 'p1', 'p2', 'p3', 'p4', or None if no token detected.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    if background == 'white':
        mask = cv2.inRange(hsv, _BLACK_TOKEN_HSV['lower'], _BLACK_TOKEN_HSV['upper'])
        kernel     = _KERNEL_RECT
        shape_mode = 'rect'
    else:
        mask = cv2.inRange(hsv, _YELLOW_TOKEN_HSV['lower'], _YELLOW_TOKEN_HSV['upper'])
        kernel     = _KERNEL_CIRCLE
        shape_mode = 'circle'

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    centroid = _find_token_centroid(mask, shape_mode)
    if centroid is None:
        return None

    h, w = img.shape[:2]
    return _centroid_to_player(centroid[0], centroid[1], w, h)
