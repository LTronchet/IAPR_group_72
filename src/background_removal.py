import cv2
import numpy as np

_SAT_HI   = 45    # HSV saturation upper bound  (white/light-gray has very low S)
_VAL_LO   = 130   # HSV value lower bound        (shadowed white corners still bright)
_MIN_AREA = 5000  # minimum contour area to keep (filters noise)


def resize_keep_ratio(img: np.ndarray, max_dim: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def _white_mask(img_bgr: np.ndarray) -> np.ndarray:
    """HSV-based mask: pixels with low saturation and high value = white/light border."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 0, _VAL_LO), (179, _SAT_HI, 255))


def remove_background(img_bgr: np.ndarray) -> np.ndarray:
    """
    Remove the background from a UNO game image using HSV white-border detection.

    Detects card regions by their white border (low S, high V in HSV), fills each
    card's interior, and sets all background pixels to white (255, 255, 255).

    Returns a BGR image the same size as the input.
    """
    H, W = img_bgr.shape[:2]

    mask = _white_mask(img_bgr)

    # Small opening to remove noise before closing
    ks_open = max(int(min(H, W) * 0.002) | 1, 8)
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_open, ks_open))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    # Close to fill small gaps inside card borders
    ks = max(int(min(H, W) * 0.005) | 1, 25)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    # Keep only contours large enough to be card regions
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= _MIN_AREA]

    # Fill each contour directly — no convex hull so overlapping cards stay separate
    filled_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(filled_mask, contours, -1, 255, cv2.FILLED)

    # Opening on the filled mask to remove large background blobs that slipped through
    ks_post = max(int(min(H, W) * 0.065) | 1, 21)
    k_post  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks_post, ks_post))
    filled_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_OPEN, k_post)

    result = np.full_like(img_bgr, 255)
    result[filled_mask == 255] = img_bgr[filled_mask == 255]
    return result
