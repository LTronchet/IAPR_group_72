import numpy as np
import cv2
from itertools import combinations

# Standard output size for every extracted card (portrait orientation).
# UNO card real dimensions: 56 mm × 87 mm  →  aspect ≈ 1.554
CARD_WIDTH = 140
CARD_HEIGHT = 218
_CARD_ASPECT = 87.0 / 56.0


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------

def _seg_angle_deg(x1, y1, x2, y2):
    """Direction angle of a segment, in [0, 180)."""
    return np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180


def _angle_diff(a, b):
    """Smallest difference between two angles in [0, 180), folded at 90°."""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _seg_to_abc(x1, y1, x2, y2):
    """
    Line through two points as unit-normal form  a*x + b*y + c = 0.
    (a, b) is the unit normal; c is the signed offset.
    """
    dx, dy = x2 - x1, y2 - y1
    length = np.hypot(dx, dy)
    if length < 1e-6:
        return None
    a, b = -dy / length, dx / length
    c = -(a * x1 + b * y1)
    return np.array([a, b, c], dtype=np.float64)


def _abc_point_distance(line, px, py):
    """Signed distance from point (px, py) to line (a,b,c)."""
    return line[0] * px + line[1] * py + line[2]


def _abc_distance(l1, l2):
    """
    Perpendicular distance between two nearly-parallel lines.
    Projects the foot of l1 (point closest to origin on l1) onto l2.
    """
    a1, b1, c1 = l1
    # Foot of perpendicular from origin to l1: (-a1*c1, -b1*c1)
    foot_x, foot_y = -a1 * c1, -b1 * c1
    return abs(_abc_point_distance(l2, foot_x, foot_y))


def _abc_intersect(l1, l2):
    """Intersection of two lines (a,b,c).  Returns (x, y) or None if parallel."""
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    denom = a1 * b2 - a2 * b1
    if abs(denom) < 1e-6:
        return None
    x = (b1 * c2 - b2 * c1) / denom
    y = (a2 * c1 - a1 * c2) / denom
    return np.array([x, y], dtype=np.float32)


def _order_quad(pts):
    """Order 4 points as [top-left, top-right, bottom-right, bottom-left]."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(d)],
        pts[np.argmax(s)],
        pts[np.argmax(d)],
    ], dtype=np.float32)


def _quad_dimensions(quad):
    """(width, height) of a quad as the mean length of opposite sides."""
    w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    h = (np.linalg.norm(quad[2] - quad[1]) + np.linalg.norm(quad[3] - quad[0])) / 2
    return float(w), float(h)


# ---------------------------------------------------------------------------
# Contour grouping
# ---------------------------------------------------------------------------

def _group_nearby_contours(contours, pad, mask_shape):
    """
    Group contours that are within `pad` pixels of each other, measured from
    actual contour edges (not bounding boxes). Works correctly for tilted cards
    whose AABBs are artificially inflated.

    Implementation: fill each contour on a blank canvas, dilate by `pad` pixels,
    then label connected components — contours in the same component are grouped.
    """
    if not contours:
        return []
    rh, rw = mask_shape[:2]
    drawn = np.zeros((rh, rw), dtype=np.uint8)
    for cnt in contours:
        cv2.drawContours(drawn, [cnt], -1, 255, cv2.FILLED)

    if pad > 0:
        k = 2 * pad + 1
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dilated = cv2.dilate(drawn, ker)
    else:
        dilated = drawn

    _, label_map = cv2.connectedComponents(dilated)

    groups: dict[int, list[int]] = {}
    for i, cnt in enumerate(contours):
        pt = cnt[0][0]
        label = int(label_map[int(pt[1]), int(pt[0])])
        groups.setdefault(label, []).append(i)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Line clustering and merging
# ---------------------------------------------------------------------------

def _cluster_by_angle(segments, tol=12):
    """
    Cluster segments by similar direction angle.
    Segments are sorted by angle before clustering so that the greedy
    scan is stable and does not produce spurious near-duplicate groups.
    Returns a list of groups, each group being a list of (x1,y1,x2,y2) tuples.
    """
    sorted_segs = sorted(segments, key=lambda s: _seg_angle_deg(*s))
    clusters = []  # each entry: [mean_angle, [segments]]
    for seg in sorted_segs:
        ang = _seg_angle_deg(*seg)
        if clusters and _angle_diff(ang, clusters[-1][0]) < tol:
            clusters[-1][1].append(seg)
            clusters[-1][0] = float(np.mean([_seg_angle_deg(*s) for s in clusters[-1][1]]))
        else:
            clusters.append([ang, [seg]])
    return [c[1] for c in clusters]


def _fit_line_abc(segments):
    """Fit a single line (a,b,c) through all endpoints of a list of segments."""
    pts = []
    for x1, y1, x2, y2 in segments:
        pts.extend([[x1, y1], [x2, y2]])
    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = float(vx), float(vy), float(x0), float(y0)
    length = np.hypot(vx, vy)
    a, b = -vy / length, vx / length
    c = -(a * x0 + b * y0)
    return np.array([a, b, c], dtype=np.float64)


def _merge_collinear(segments, dist_tol=8, top_k=8):
    """
    Within a group of roughly parallel segments, merge those that lie on the
    same line (midpoint-to-line distance < dist_tol).
    Returns the top_k representative lines sorted by total segment support
    (sum of constituent segment lengths), in (a,b,c) form.
    Keeping only the strongest lines avoids combinatorial explosion from
    internal card features (oval borders, number strokes) that produce many
    short, weakly-supported lines.
    """
    if not segments:
        return []
    used = [False] * len(segments)
    abc = [_seg_to_abc(*s) for s in segments]
    result = []  # (line_abc, total_support_length)
    for i in range(len(segments)):
        if used[i] or abc[i] is None:
            continue
        group = [segments[i]]
        used[i] = True
        li = abc[i]
        for j in range(i + 1, len(segments)):
            if used[j] or abc[j] is None:
                continue
            mx = (segments[j][0] + segments[j][2]) / 2.0
            my = (segments[j][1] + segments[j][3]) / 2.0
            if abs(_abc_point_distance(li, mx, my)) < dist_tol:
                group.append(segments[j])
                used[j] = True
        support = sum(np.hypot(x2 - x1, y2 - y1) for x1, y1, x2, y2 in group)
        result.append((_fit_line_abc(group), support))
    result.sort(key=lambda x: -x[1])
    return [line for line, _ in result[:top_k]]


# ---------------------------------------------------------------------------
# Card quad validation and extraction
# ---------------------------------------------------------------------------

def _valid_card_quad(quad, aspect=_CARD_ASPECT, tol=0.25):
    """Return True if the quad has a card-like aspect ratio and is large enough."""
    w, h = _quad_dimensions(quad)
    if min(w, h) < 20:
        return False
    ratio = max(w, h) / max(min(w, h), 1e-6)
    return abs(ratio - aspect) < tol * aspect


def _extract_card(region, quad):
    """
    Perspective-warp the quadrilateral to a canonical (CARD_WIDTH × CARD_HEIGHT)
    image. If the card is wider than tall, it is rotated before warping.
    """
    w, h = _quad_dimensions(quad)
    out_w, out_h = (CARD_HEIGHT, CARD_WIDTH) if w > h else (CARD_WIDTH, CARD_HEIGHT)
    dst = np.array([[0, 0], [out_w - 1, 0],
                    [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    try:
        M = cv2.getPerspectiveTransform(quad, dst)
        card = cv2.warpPerspective(region, M, (out_w, out_h))
        # Always return portrait
        if out_w > out_h:
            card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
        return card
    except cv2.error:
        return None


def _bbox_iou(quad_a, quad_b):
    """Approximate IoU using axis-aligned bounding boxes of two quads."""
    def bb(q):
        return q[:, 0].min(), q[:, 1].min(), q[:, 0].max(), q[:, 1].max()
    ax0, ay0, ax1, ay1 = bb(quad_a)
    bx0, by0, bx1, by1 = bb(quad_b)
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _edge_density_on_quad(edges_img, quad, thickness=8):
    """
    Fraction of pixels along the 4 edges of the quad that are Canny edge pixels.
    A true card boundary should have high edge density (≥ EDGE_DENSITY_THRESH).
    """
    mask = np.zeros(edges_img.shape[:2], dtype=np.uint8)
    cv2.polylines(mask, [quad.astype(np.int32)], isClosed=True,
                  color=255, thickness=thickness)
    total = np.count_nonzero(mask)
    if total == 0:
        return 0.0
    return np.count_nonzero(edges_img & mask) / total


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect_cards(region: np.ndarray) -> list[np.ndarray]:
    """
    Detect and extract individual UNO cards from a region image.

    Thresholds the HSV saturation channel to isolate colored card interiors,
    then fits a minimum-area bounding rectangle to each connected region.
    This gives accurate oriented quads regardless of card tilt, which the
    perspective transform maps to canonical portrait images.

    Args:
        region: BGR image of a single player's area or the center area.

    Returns:
        List of card images, each of shape (CARD_HEIGHT, CARD_WIDTH, 3),
        oriented in portrait mode.  Returns [] if no cards are detected.
    """
    h, w = region.shape[:2]
    min_dim = min(h, w)

    # 1. Saturation mask — colored card interiors have high saturation;
    #    white borders and white backgrounds do not.
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    blur = cv2.GaussianBlur(sat, (9, 9), 0)
    _, mask = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)

    # 2. Per-color detection: process each UNO card color independently so
    #    adjacent cards of different colors can never be merged by morphology.
    #    Red wraps around hue = 0/180, so it needs two ranges.
    sat_lo = 60      # minimum saturation to be considered a card color
    hue_ranges = [
        ((0,   sat_lo, 50), (10,  255, 255)),   # red (low hue)
        ((160, sat_lo, 50), (179, 255, 255)),   # red (high hue)
        ((15,  sat_lo, 50), (40,  255, 255)),   # yellow
        ((45,  sat_lo, 50), (85,  255, 255)),   # green
        ((90,  sat_lo, 50), (130, 255, 255)),   # blue
    ]

    # Large isotropic close per color: safely bridges the within-card oval gap
    # because fragments of different cards can never be adjacent in the same
    # single-color mask.
    ks = max(int(min_dim * 0.05) | 1, 7)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ks, ks))
    card_min_area = (min_dim * 0.15) ** 2

    candidates = []
    for lo, hi in hue_ranges:
        cmask = cv2.inRange(hsv, lo, hi)
        cmask = cv2.morphologyEx(cmask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < card_min_area:
                continue
            hull = cv2.convexHull(cnt)
            rect = cv2.minAreaRect(hull)
            box = cv2.boxPoints(rect)
            quad = _order_quad(box.astype(np.float32))
            if not _valid_card_quad(quad, tol=0.55):
                continue
            card_img = _extract_card(region, quad)
            if card_img is not None:
                candidates.append((quad, card_img))

    # 3. Deduplicate (red is detected twice — high/low hue ranges).
    kept = []
    for quad, img in candidates:
        if not any(_bbox_iou(quad, kq) > 0.3 for kq, _ in kept):
            kept.append((quad, img))

    return [img for _, img in kept]
