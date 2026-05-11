from __future__ import annotations

import itertools

import numpy as np
import cv2

CARD_WIDTH = 140
CARD_HEIGHT = 218
_CARD_ASPECT = 87.0 / 56.0

# (lo_hsv, hi_hsv, is_dark, color)
# For colored ranges (is_dark=False), sat/val in lo are overridden at runtime
# by sat_lo/val_lo. For the dark range (is_dark=True), bounds are fixed.
# Two red ranges share the same color label so their fragments can be paired.
_RANGES = [
    ((0,   0, 0), (10,  255, 255), False, "red"),     # red-lo
    ((160, 0, 0), (179, 255, 255), False, "red"),     # red-hi
    ((22,  0, 0), (36,  255, 255), False, "yellow"),  # yellow
    ((45,  0, 0), (85,  255, 255), False, "green"),   # green
    ((90,  0, 0), (130, 255, 255), False, "blue"),    # blue
    ((0,   0, 0), (179, 100,  120), True,  "dark"),   # wild/+4 dark body
]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes via flood-fill from the image border."""
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    hp, wp = padded.shape
    ffm = np.zeros((hp + 2, wp + 2), dtype=np.uint8)
    temp = padded.copy()
    cv2.floodFill(temp, ffm, (0, 0), 255)
    return cv2.bitwise_or(padded, cv2.bitwise_not(temp))[1:-1, 1:-1]


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def _quad_dimensions(quad: np.ndarray) -> tuple[float, float]:
    """(width, height) as the mean length of opposite sides."""
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[2] - quad[1]) + np.linalg.norm(quad[3] - quad[0])) / 2
    return float(w), float(h)




def _expand_quad(quad: np.ndarray, frac: float = 0.15) -> np.ndarray:
    """Expand a quad outward from its centroid by frac of each half-diagonal."""
    cx, cy = quad.mean(axis=0)
    return (quad + frac * (quad - np.array([cx, cy]))).astype(np.float32)


def _bbox_iou(quad_a: np.ndarray, quad_b: np.ndarray) -> float:
    """Approximate IoU using axis-aligned bounding boxes."""
    def bb(q):
        return q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
    ax0, ay0, ax1, ay1 = bb(quad_a)
    bx0, by0, bx1, by1 = bb(quad_b)
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _warp_card(region: np.ndarray, quad: np.ndarray,
               skip_tight_crop: bool = False) -> np.ndarray | None:
    """
    Perspective-warp a quad to a canonical (CARD_WIDTH × CARD_HEIGHT) image.
    skip_tight_crop: pass True for dark-background cards (wild/+4) — their black
    body has low sat AND low val so the sat>25 OR val>180 filter would discard it.
    """
    w, h = _quad_dimensions(quad)
    out_w, out_h = (CARD_HEIGHT, CARD_WIDTH) if w > h else (CARD_WIDTH, CARD_HEIGHT)
    dst = np.array([[0, 0], [out_w - 1, 0],
                    [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    try:
        M = cv2.getPerspectiveTransform(_expand_quad(quad, frac=0.25), dst)
        card = cv2.warpPerspective(region, M, (out_w, out_h))
        if out_w > out_h:
            card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)

        if not skip_tight_crop:
            ch, cw = card.shape[:2]
            mx, my = int(cw * 0.075), int(ch * 0.075)
            hsv_c = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
            _, sat_m = cv2.threshold(hsv_c[:, :, 1], 25, 255, cv2.THRESH_BINARY)
            _, val_m = cv2.threshold(hsv_c[:, :, 2], 180, 255, cv2.THRESH_BINARY)
            col = cv2.bitwise_or(sat_m, val_m)
            col[:my, :] = col[ch - my:, :] = col[:, :mx] = col[:, cw - mx:] = 0
            ys, xs = np.where(col > 0)
            if len(xs) > 100:
                x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    card = card[y1:y2 + 1, x1:x2 + 1]

        return cv2.resize(card, (CARD_WIDTH, CARD_HEIGHT))
    except cv2.error:
        return None


# ---------------------------------------------------------------------------
# Pipeline step 1 — preprocessing
# ---------------------------------------------------------------------------

def preprocess_mask(region: np.ndarray,
                    sat_lo: int = 60,
                    val_lo: int = 50) -> list[tuple[np.ndarray, bool]]:
    """
    Build a binary mask for each entry in _RANGES.

    Returns:
        List of (mask, is_dark), one per _RANGES entry, in the same order.
        Kept per-range (not combined) so get_quads can pair bad blobs within
        the same color only, avoiding false cross-color merges.
    """
    h, w = region.shape[:2]
    min_dim = min(h, w)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    close_ks = max(int(min_dim * 0.10) | 1, 7)
    close_k  = cv2.getStructuringElement(cv2.MORPH_RECT, (close_ks, close_ks))
    open_ks  = max(int(min_dim * 0.017) | 1, 3)
    open_k   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
    # Pass 1 — threshold + open (remove small blobs) + close every range.
    closed = []
    for lo, hi, is_dark, color in _RANGES:
        actual_lo = lo if is_dark else (lo[0], sat_lo, val_lo)
        m = cv2.inRange(hsv, actual_lo, hi)
        if is_dark:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, open_k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k)
        closed.append((m, is_dark, color))

    # Build colored_union from closed colored masks so wedge edges are solid.
    colored_union = np.zeros((h, w), dtype=np.uint8)
    for m, is_dark, _ in closed:
        if not is_dark:
            colored_union = cv2.bitwise_or(colored_union, m)

    # adj_ks = max(int(min_dim * 0.02) | 1, 3)
    # adj_k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (adj_ks, adj_ks))

    # Pass 2 — fill holes.
    masks = []
    for m, is_dark, color in closed:
        m = _fill_holes(m)
        masks.append((m, is_dark, color))

    return masks


# ---------------------------------------------------------------------------
# Pipeline step 2 — quad detection
# ---------------------------------------------------------------------------

def get_quads(region: np.ndarray,
              masks: list[tuple[np.ndarray, bool, str]]) -> list[tuple[np.ndarray, bool]]:
    """
    Find card quadrilaterals from pre-computed masks.

    Args:
        masks: output of preprocess_mask — list of (mask, is_dark, color).

    Returns:
        List of (quad, is_dark) where quad is a 4×2 float32 array of corners
        and is_dark=True for wild/+4 cards.
    """
    h, w = region.shape[:2]
    card_min_area = (min(h, w) * 0.15) ** 2
    TIGHT_TOL = 0.30

    candidates: list[tuple[np.ndarray, bool]] = []
    # bad blobs grouped by color label for cross-range pairing
    bad_by_color: dict[str, list[np.ndarray]] = {}

    frag_min_area = card_min_area / 4  # fragments can be 1/4 of a full card

    for mask, is_dark, color in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Full-size threshold for direct detection; fragment threshold for pairing pool.
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < frag_min_area:
                continue
            ww, hh = cv2.minAreaRect(cnt)[1]
            ratio = max(ww, hh) / max(min(ww, hh), 1e-6)
            is_good = area >= card_min_area and abs(ratio - _CARD_ASPECT) < TIGHT_TOL * _CARD_ASPECT
            if is_good:
                candidates.append((_order_quad(cv2.boxPoints(cv2.minAreaRect(cnt))), is_dark))
            # Dark blobs always join the pairing pool — their mask is inherently
            # fragmented (colored wedges break it) so individual detection is unreliable.
            # NMS resolves any duplicate if the single-blob and paired detections overlap.
            if not is_good or is_dark:
                bad_by_color.setdefault(color, []).append(cnt)

    # Reference area from directly detected cards — used to rank grouped candidates.
    ref_area = float(np.median([
        cv2.minAreaRect(q.reshape(-1, 1, 2).astype(np.float32))[1][0] *
        cv2.minAreaRect(q.reshape(-1, 1, 2).astype(np.float32))[1][1]
        for q, _ in candidates
    ])) if candidates else None

    # Pair bad blobs within the same color group — cross-range pairing enabled.
    # Dark: try all subset sizes (card can split into 3+ fragments).
    # Colored: pairs only (larger combos risk false cross-card merges).
    # Greedy selection: closest area to good candidates first; each fragment used at most once.
    for color, bad in bad_by_color.items():
        is_dark = color == "dark"
        sizes = range(2, len(bad) + 1) if is_dark else [2]

        valid: list[tuple[float, tuple[int, ...], np.ndarray]] = []
        for r in sizes:
            for indices in itertools.combinations(range(len(bad)), r):
                pts_col = np.concatenate(
                    [bad[i].reshape(-1, 2) for i in indices]
                ).reshape(-1, 1, 2).astype(np.float32)
                _, (ww, hh), _ = cv2.minAreaRect(pts_col)
                area = ww * hh
                if area < card_min_area:
                    continue
                ratio = max(ww, hh) / max(min(ww, hh), 1e-6)
                if abs(ratio - _CARD_ASPECT) < 0.40 * _CARD_ASPECT:
                    score = abs(area - ref_area) if ref_area else area
                    valid.append((score, indices, pts_col))

        valid.sort(key=lambda x: x[0])
        used: set[int] = set()
        for _, indices, pts_col in valid:
            if used.intersection(indices):
                continue
            candidates.append((_order_quad(cv2.boxPoints(cv2.minAreaRect(pts_col))), is_dark))
            used.update(indices)

    return candidates


# ---------------------------------------------------------------------------
# Pipeline step 3 — card image extraction
# ---------------------------------------------------------------------------

def get_card_images(region: np.ndarray,
                    quads: list[tuple[np.ndarray, bool]]) -> list[np.ndarray]:
    """
    Warp each quad to a canonical card image, then apply NMS.

    Args:
        region: original BGR region.
        quads:  output of get_quads — list of (quad, is_dark).

    Returns:
        List of (CARD_HEIGHT × CARD_WIDTH) BGR images.
    """
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for quad, is_dark in quads:
        img = _warp_card(region, quad, skip_tight_crop=is_dark)
        if img is not None:
            candidates.append((quad, img))

    # NMS: largest first; suppress if centre inside a kept bbox or IoU > 0.3.
    candidates.sort(
        key=lambda x: (x[0][:, 0].max() - x[0][:, 0].min()) *
                      (x[0][:, 1].max() - x[0][:, 1].min()),
        reverse=True,
    )
    kept: list[tuple[np.ndarray, np.ndarray]] = []
    for quad, img in candidates:
        cx, cy = quad.mean(axis=0)
        if not any(
            _bbox_iou(quad, kq) > 0.3 or (
                kq[:, 0].min() <= cx <= kq[:, 0].max() and
                kq[:, 1].min() <= cy <= kq[:, 1].max()
            )
            for kq, _ in kept
        ):
            kept.append((quad, img))

    return [img for _, img in kept]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_cards(region: np.ndarray,
                 sat_lo: int = 60, val_lo: int = 50) -> list[np.ndarray]:
    """Detect and extract UNO cards from a region image."""
    masks = preprocess_mask(region, sat_lo, val_lo)
    quads = get_quads(region, masks)
    return get_card_images(region, quads)


_RANGE_COLORS_BGR = [
    (  0,   0, 220),  # red-lo
    (  0,   0, 220),  # red-hi
    (  0, 220, 220),  # yellow
    (  0, 180,   0),  # green
    (200,  80,   0),  # blue
    ( 80,  80,  80),  # wild/+4 dark
]


def debug_mask(region: np.ndarray, name: str = "",
               sat_lo: int = 60, val_lo: int = 50) -> np.ndarray:
    """Return a BGR color image with each detection range drawn in its own color."""
    masks = preprocess_mask(region, sat_lo, val_lo)
    vis = np.zeros((region.shape[0], region.shape[1], 3), dtype=np.uint8)
    total_pixels = 0
    for (m, *_), color in zip(masks, _RANGE_COLORS_BGR):
        vis[m > 0] = color
        total_pixels += np.count_nonzero(m)
    total = region.shape[0] * region.shape[1]
    print(f"[{name}] sat_lo={sat_lo} val_lo={val_lo}  "
          f"coverage: {total_pixels / total:.1%}")
    return vis
