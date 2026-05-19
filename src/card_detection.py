from __future__ import annotations

import itertools

import numpy as np
import cv2

CARD_WIDTH = 140
CARD_HEIGHT = 218
_CARD_ASPECT = 87.0 / 56.0

# Expected card area in the region image, in pixels.
# Tune this constant if cards are consistently over- or under-filtered.
_CARD_REF_AREA_PX = 140_000

# (lo_hsv, hi_hsv, is_dark, color)
# For colored ranges (is_dark=False), sat/val in lo are overridden at runtime
# by sat_lo/val_lo. For the dark range (is_dark=True), bounds are fixed.
# Two red ranges share the same color label so their fragments can be paired.
_RANGES = [
    ((0,   0, 0), ( 8,  255, 255), False, "red"),     # red-lo
    ((165, 0, 0), (179, 255, 255), False, "red"),     # red-hi
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

    # Pass 2 — fill holes, then subtract dark from colored to suppress wedge blobs.
    filled = [(is_dark, color, _fill_holes(m)) for m, is_dark, color in closed]

    # Union of filled dark masks covers the full body of wild/+4 cards,
    # including the colored center wedges that sit inside the dark border.
    dark_filled = np.zeros((h, w), dtype=np.uint8)
    for is_dark, _, fm in filled:
        if is_dark:
            dark_filled = cv2.bitwise_or(dark_filled, fm)

    # Remove pixels inside dark cards from colored masks so their center
    # wedges are not mistaken for fragments of colored cards.
    masks = []
    for is_dark, color, fm in filled:
        if is_dark:
            masks.append((fm, is_dark, color))
        else:
            masks.append((cv2.bitwise_and(fm, cv2.bitwise_not(dark_filled)), is_dark, color))

    return masks


# ---------------------------------------------------------------------------
# Pipeline step 2 — quad detection
# ---------------------------------------------------------------------------

def _vertex_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle in degrees at vertex b between edges b-a and b-c."""
    v1 = a - b
    v2 = c - b
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def _valid_recon_quad(
    quad: np.ndarray,
    img_w: int,
    img_h: int,
) -> bool:
    """Geometric sanity checks before the more expensive blob-coverage test."""
    qw, qh = _quad_dimensions(quad)
    if abs(max(qw, qh) / max(min(qw, qh), 1e-6) - _CARD_ASPECT) > 0.35 * _CARD_ASPECT:
        return False
    if not (0 <= float(quad[:, 0].mean()) <= img_w and
            0 <= float(quad[:, 1].mean()) <= img_h):
        return False
    return True


def _sample_quad_edges(quad: np.ndarray, n_per_side: int = 30) -> np.ndarray:
    """Sample n_per_side points uniformly along each of the 4 edges of a quad."""
    pts = []
    for i in range(4):
        p1 = quad[i]
        p2 = quad[(i + 1) % 4]
        t = np.linspace(0, 1, n_per_side, endpoint=False)
        pts.append(p1[None] + t[:, None] * (p2 - p1)[None])
    return np.vstack(pts)


def _hull_corner_triples(
    all_pts: np.ndarray, angle_tol: float = 20.0
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Return (A, B, C) triplets of consecutive hull vertices where the angle at B is ≈ 90°.
    A and C are the two neighbours; u = BA and v = BC are the two visible card edges.
    """
    pts_int = all_pts.astype(np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts_int)
    peri = cv2.arcLength(hull, True)
    poly = cv2.approxPolyDP(hull, max(2.0, 0.015 * peri), True).reshape(-1, 2).astype(float)
    n = len(poly)
    triples = []
    for j in range(n):
        A, B, C = poly[(j - 1) % n], poly[j], poly[(j + 1) % n]
        if abs(_vertex_angle(A, B, C) - 90.0) < angle_tol:
            triples.append((A, B, C))
    return triples


def _confirm_orientation(
    all_pts: np.ndarray,
    B: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    card_w: float,
    card_h: float,
    angle_tol: float = 20.0,
    dist_tol: float = 0.25,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Scan hull edges for a 3rd side that confirms which direction is width vs height.

    If an edge parallel to u is found at distance ≈ card_h from B (along v),
    then v is the height direction → (width_dir=u, height_dir=v).
    If at distance ≈ card_w, then v is the width direction → (width_dir=v, height_dir=u).
    Symmetric logic for edges parallel to v.

    Returns (width_dir, height_dir) or None if no confirming edge is found.
    """
    pts_int = all_pts.astype(np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(pts_int)
    peri = cv2.arcLength(hull, True)
    poly = cv2.approxPolyDP(hull, max(2.0, 0.015 * peri), True).reshape(-1, 2).astype(float)
    n = len(poly)

    for j in range(n):
        P, Q = poly[j], poly[(j + 1) % n]
        edge = Q - P
        elen = float(np.linalg.norm(edge))
        if elen < 5.0:
            continue
        edir = edge / elen

        cos_u = float(abs(np.dot(edir, u)))
        cos_v = float(abs(np.dot(edir, v)))

        if np.degrees(np.arccos(np.clip(cos_u, 0.0, 1.0))) < angle_tol:
            # Edge ≈ parallel to u → opposite side; distance along v = card height
            d = float(abs(np.dot(P - B, v)))
            if abs(d - card_h) < dist_tol * card_h:
                return u, v   # u=width, v=height
            if abs(d - card_w) < dist_tol * card_w:
                return v, u   # v=width, u=height

        if np.degrees(np.arccos(np.clip(cos_v, 0.0, 1.0))) < angle_tol:
            # Edge ≈ parallel to v → opposite side; distance along u = card width
            d = float(abs(np.dot(P - B, u)))
            if abs(d - card_w) < dist_tol * card_w:
                return u, v   # u=width, v=height
            if abs(d - card_h) < dist_tol * card_h:
                return v, u   # v=width, u=height

    return None


def _quads_from_contours(
    cnts: list[np.ndarray],
    ref_w: float,
    img_w: int,
    img_h: int,
    min_blob_frac: float = 0.90,
) -> list[np.ndarray]:
    """
    Fit a fixed-size card rectangle to the blob using hull corner triples.

    For each hull vertex B with angle ≈ 90°:
    1. Derive orientation directly from the two adjacent edges BA and BC (no grid-search).
    2. Place a card of fixed size ref_w × (ref_w * _CARD_ASPECT) with corner at B.
    3. Keep placements where ≥ min_blob_frac of the blob is inside the card.
    4. Score by best-fitting single edge (min per-edge mean distance to blob edge).
    Returns candidates sorted by best edge alignment, deduped by IoU > 0.5.
    """
    if not cnts:
        return []

    all_pts = np.concatenate([c.reshape(-1, 2) for c in cnts])

    blob_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for c in cnts:
        cv2.fillPoly(blob_mask, [c.reshape(-1, 2).astype(np.int32)], 255)
    blob_area = int(np.count_nonzero(blob_mask))
    if blob_area == 0:
        return []

    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blob_edge = cv2.morphologyEx(blob_mask, cv2.MORPH_GRADIENT, k3)
    dist_from_edge = cv2.distanceTransform(
        cv2.bitwise_not(blob_edge), cv2.DIST_L2, cv2.DIST_MASK_5
    ).astype(np.float32)

    card_w = ref_w
    card_h = ref_w * _CARD_ASPECT

    candidates: list[tuple[float, np.ndarray]] = []

    for A, B, C in _hull_corner_triples(all_pts):
        ba = A - B
        ba_len = np.linalg.norm(ba)
        if ba_len < 1e-6:
            continue
        u = ba / ba_len  # unit vector along edge BA

        # Exact perpendicular: pick the ±90° rotation that points toward C
        v_cw  = np.array([ u[1], -u[0]])
        v_ccw = np.array([-u[1],  u[0]])
        v = v_cw if np.dot(C - B, v_cw) > 0 else v_ccw

        len_u = ba_len
        len_v = np.linalg.norm(C - B)

        # Try to confirm orientation via a 3rd hull edge; fall back to edge-length heuristic.
        confirmed = _confirm_orientation(all_pts, B, u, v, card_w, card_h)
        if confirmed is not None:
            width_dir, height_dir = confirmed
            len_w, len_h = (len_u, len_v) if width_dir is u else (len_v, len_u)
        elif len_u >= len_v:
            width_dir, height_dir, len_w, len_h = v, u, len_v, len_u
        else:
            width_dir, height_dir, len_w, len_h = u, v, len_u, len_v

        # Reject if visible edge is longer than the card dimension (bad triple)
        if len_w > card_w * 1.10 or len_h > card_h * 1.10:
            continue
        quad = _order_quad(np.array([
            B,
            B + width_dir  * card_w,
            B + width_dir  * card_w + height_dir * card_h,
            B + height_dir * card_h,
        ], dtype=np.float32))
        if not _valid_recon_quad(quad, img_w, img_h):
            continue

        quad_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.fillPoly(quad_mask, [quad.astype(np.int32)], 255)
        overlap = int(np.count_nonzero(cv2.bitwise_and(blob_mask, quad_mask)))
        if overlap / blob_area < min_blob_frac:
            continue

        edge_pts = _sample_quad_edges(quad, n_per_side=30)
        xs = edge_pts[:, 0].astype(np.int32).clip(0, img_w - 1)
        ys = edge_pts[:, 1].astype(np.int32).clip(0, img_h - 1)
        per_edge = dist_from_edge[ys, xs].reshape(4, 30).mean(axis=1)
        edge_score = float(per_edge.min())
        candidates.append((edge_score, quad))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0])

    deduped: list[np.ndarray] = []
    for _, q in candidates:
        if not any(_bbox_iou(q, dq) > 0.5 for dq in deduped):
            deduped.append(q)
    return deduped

def get_quads(region: np.ndarray,
              masks: list[tuple[np.ndarray, bool, str]]) -> list[tuple[np.ndarray, bool, str]]:
    """
    Find card quadrilaterals from pre-computed masks.

    Args:
        masks: output of preprocess_mask — list of (mask, is_dark, color).

    Returns:
        List of (quad, is_dark, color) where quad is a 4×2 float32 array of corners
        and is_dark=True for wild/+4 cards.
    """
    h, w = region.shape[:2]
    card_min_area = (min(h, w) * 0.15) ** 2
    TIGHT_TOL = 0.30

    candidates: list[tuple[np.ndarray, bool, str]] = []
    # bad blobs grouped by color label for cross-range pairing
    bad_by_color: dict[str, list[np.ndarray]] = {}

    frag_min_area = card_min_area / 4  # fragments can be 1/4 of a full card

    # Merge same-color masks (red-lo + red-hi → one blob per red card).
    merged_masks: dict[tuple[bool, str], np.ndarray] = {}
    for m, is_dark, color in masks:
        key = (is_dark, color)
        merged_masks[key] = cv2.bitwise_or(merged_masks[key], m) if key in merged_masks else m.copy()
    masks = [(m, is_dark, color) for (is_dark, color), m in merged_masks.items()]

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
            if is_good and not is_dark:
                candidates.append((_order_quad(cv2.boxPoints(cv2.minAreaRect(cnt))), is_dark, color, "M1"))
            # Dark blobs always join the pairing pool — requires ≥2 blobs to be detected.
            if not is_good or is_dark:
                bad_by_color.setdefault(color, []).append(cnt)

    # Reference area from directly detected cards — used to rank grouped candidates.
    ref_area = float(np.median([
        cv2.minAreaRect(q.reshape(-1, 1, 2).astype(np.float32))[1][0] *
        cv2.minAreaRect(q.reshape(-1, 1, 2).astype(np.float32))[1][1]
        for q, *_ in candidates
    ])) if candidates else None

    # Method 2 — pair bad blobs within the same color group.
    # Dark: try all subset sizes (card can split into 3+ fragments).
    # Colored: pairs only (larger combos risk false cross-card merges).
    # Greedy selection: closest area to good candidates first; each fragment used at most once.
    used_per_color: dict[str, set[int]] = {}
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
            candidates.append((_order_quad(cv2.boxPoints(cv2.minAreaRect(pts_col))), is_dark, color, "M2"))
            used.update(indices)
        used_per_color[color] = used

    # Method 3 — edge-based reconstruction for orphan blobs.
    # Re-derive card dimensions from all candidates so far (methods 1+2).
    # For each color, combine ALL orphan contours into one hull and also try
    # each individually — handles cards split into multiple fragments where the
    # combined shape still doesn't look like a full card (partial occlusion).
    if candidates:
        ref_area_m3 = float(np.median([
            cv2.minAreaRect(q.reshape(-1, 1, 2).astype(np.float32))[1][0] *
            cv2.minAreaRect(q.reshape(-1, 1, 2).astype(np.float32))[1][1]
            for q, *_ in candidates
        ]))
        ref_w = float(np.sqrt(ref_area_m3 / _CARD_ASPECT))

        for color, bad in bad_by_color.items():
            is_dark = color == "dark"
            orphan_idxs = set(range(len(bad))) - used_per_color.get(color, set())
            orphans = [bad[i] for i in orphan_idxs
                       if cv2.contourArea(bad[i]) >= frag_min_area]
            if not orphans:
                continue

            # Try all subsets of orphan blobs (size 1 up to max_r).
            # The 90% coverage check in _quads_from_contours naturally rejects
            # cross-card combinations (2 blobs from 2 cards → ~50% coverage).
            # Dark cards can split into many fragments so allow larger subsets.
            max_r = len(orphans) if is_dark else min(len(orphans), 3)
            min_r = 2 if is_dark else 1  # dark cards require ≥2 blobs
            for r in range(max_r, min_r - 1, -1):
                for sub_idxs in itertools.combinations(range(len(orphans)), r):
                    sub_cnts = [orphans[i] for i in sub_idxs]
                    for quad in _quads_from_contours(sub_cnts, ref_w, w, h):
                        candidates.append((quad, is_dark, color, "M3"))

    return candidates


# ---------------------------------------------------------------------------
# Pipeline step 3 — card image extraction
# ---------------------------------------------------------------------------

def _quad_coverage(quad: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of the quad's interior covered by non-zero mask pixels."""
    h, w = mask.shape[:2]
    poly_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [quad.astype(np.int32)], 255)
    quad_area = int(np.count_nonzero(poly_mask))
    if quad_area == 0:
        return 0.0
    return int(np.count_nonzero(cv2.bitwise_and(poly_mask, mask))) / quad_area


def get_card_images(region: np.ndarray,
                    quads: list[tuple],
                    combined_mask: np.ndarray | None = None,
                    iou_thresh: float = 0.40,
                    min_coverage: float = 0.10,
                    size_tol: float = 0.50,
                    ref_area: float = 0.0) -> list[tuple[np.ndarray, str]]:
    """
    Warp each quad to a canonical card image, then apply NMS.

    NMS keeps the largest quad first and suppresses any later quad whose
    axis-aligned bounding-box IoU with a kept quad exceeds iou_thresh.

    If combined_mask (union of all color masks) is provided, quads whose
    interior is less than min_coverage covered by detected pixels are dropped
    before NMS — this rejects reconstructed quads that fall mostly on background.

    ref_area: expected card area in pixels, derived from region dimensions.
    Quads whose area deviates more than size_tol from ref_area are dropped.
    Dark cards use 2× the tolerance — fragmented blobs make area estimates noisier.
    """
    candidates: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    for quad, is_dark, color, *rest in quads:
        method = rest[0] if rest else "?"
        # Dark cards: skip coverage filter — sparse mask (white border, empty center)
        # means legitimate dark quads would be incorrectly rejected.
        if not is_dark and combined_mask is not None and _quad_coverage(quad, combined_mask) < min_coverage:
            continue
        img = _warp_card(region, quad, skip_tight_crop=is_dark)
        if img is not None:
            candidates.append((quad, img, color, method))

    # Size consistency filter anchored on ref_area (from region dimensions).
    # Dark cards use 2× tolerance — fragmented blobs make area estimates noisier.
    if ref_area > 0 and len(candidates) >= 2:
        def _area(q): return _quad_dimensions(q)[0] * _quad_dimensions(q)[1]
        candidates = [
            c for c in candidates
            if abs(_area(c[0]) - ref_area) / ref_area <= (size_tol * 2 if c[2] == "dark" else size_tol)
        ]

    candidates.sort(
        key=lambda x: (x[0][:, 0].max() - x[0][:, 0].min()) *
                      (x[0][:, 1].max() - x[0][:, 1].min()),
        reverse=True,
    )
    kept: list[tuple[np.ndarray, np.ndarray, str, str]] = []
    for quad, img, color, method in candidates:
        # Same-color dedup.
        if any(_bbox_iou(quad, kq) > iou_thresh
               for kq, _, kcolor, _ in kept if kcolor == color):
            continue
        # Proactive: suppress any colored quad whose centroid falls inside a kept dark quad.
        if color != "dark":
            cx, cy = float(quad[:, 0].mean()), float(quad[:, 1].mean())
            if any(cv2.pointPolygonTest(kq.astype(np.float32), (cx, cy), False) >= 0
                   for kq, _, kcolor, _ in kept if kcolor == "dark"):
                continue
        kept.append((quad, img, color, method))

    return [(img, color) for _, img, color, _ in kept]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_cards(region: np.ndarray,
                 sat_lo: int = 60, val_lo: int = 50) -> list[tuple[np.ndarray, str]]:
    """Detect and extract UNO cards from a region image.

    Returns list of (image, color) where color is one of:
    'red', 'yellow', 'green', 'blue', 'dark' (wild/+4).
    """
    masks = preprocess_mask(region, sat_lo, val_lo)
    combined = np.zeros(region.shape[:2], dtype=np.uint8)
    for m, *_ in masks:
        combined = cv2.bitwise_or(combined, m)
    quads = get_quads(region, masks)
    return get_card_images(region, quads, combined_mask=combined,
                           iou_thresh=0.50, min_coverage=0.60,
                           ref_area=_CARD_REF_AREA_PX)


_RANGE_COLORS_BGR = [
    (  0,   0, 220),  # red-lo
    (  0,   0, 220),  # red-hi
    (  0, 220, 220),  # yellow
    (  0, 180,   0),  # green
    (200,  80,   0),  # blue
    ( 80,  80,  80),  # wild/+4 dark
]

_CARD_COLOR_BGR = {
    "red":    (  0,   0, 220),
    "yellow": (  0, 220, 220),
    "green":  (  0, 180,   0),
    "blue":   (200,  80,   0),
    "dark":   ( 80,  80,  80),
}


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


def debug_orphan_angles(region: np.ndarray,
                        sat_lo: int = 60, val_lo: int = 50,
                        angle_tol: float = 40.0) -> np.ndarray:
    """
    Returns an annotated BGR image for Method 3 debug:
    - Cyan outlines: quads detected by Methods 1+2
    - Colored polygon: convex hull of orphan blobs per card color
    - Green dot + angle: hull vertex with angle ≈ 90° (used as card corner anchor)
    - Red dot + angle:   hull vertex outside the 90° tolerance
    """
    h, w = region.shape[:2]
    card_min_area = (min(h, w) * 0.15) ** 2
    frag_min_area = card_min_area / 4
    TIGHT_TOL = 0.30

    masks = preprocess_mask(region, sat_lo, val_lo)
    # Merge same-color masks (mirrors get_quads).
    merged_masks: dict[tuple[bool, str], np.ndarray] = {}
    for m, is_dark, color in masks:
        key = (is_dark, color)
        merged_masks[key] = cv2.bitwise_or(merged_masks[key], m) if key in merged_masks else m.copy()
    masks = [(m, is_dark, color) for (is_dark, color), m in merged_masks.items()]

    vis = region.copy()

    # Replicate Methods 1+2 from get_quads to collect candidates / orphan state.
    candidates: list[tuple[np.ndarray, bool, str]] = []
    bad_by_color: dict[str, list[np.ndarray]] = {}

    for mask, is_dark, color in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < frag_min_area:
                continue
            ww, hh = cv2.minAreaRect(cnt)[1]
            ratio = max(ww, hh) / max(min(ww, hh), 1e-6)
            is_good = (area >= card_min_area and
                       abs(ratio - _CARD_ASPECT) < TIGHT_TOL * _CARD_ASPECT)
            if is_good:
                candidates.append(
                    (_order_quad(cv2.boxPoints(cv2.minAreaRect(cnt))), is_dark, color))
            if not is_good or is_dark:
                bad_by_color.setdefault(color, []).append(cnt)

    ref_area = float(np.median([
        cv2.minAreaRect(q.reshape(-1, 1, 2).astype(np.float32))[1][0] *
        cv2.minAreaRect(q.reshape(-1, 1, 2).astype(np.float32))[1][1]
        for q, *_ in candidates
    ])) if candidates else None

    used_per_color: dict[str, set[int]] = {}
    for color, bad in bad_by_color.items():
        is_dark = color == "dark"
        sizes = range(2, len(bad) + 1) if is_dark else [2]
        valid: list[tuple[float, tuple, np.ndarray]] = []
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
            candidates.append(
                (_order_quad(cv2.boxPoints(cv2.minAreaRect(pts_col))), is_dark, color))
            used.update(indices)
        used_per_color[color] = used

    # Draw Method 1+2 quads in cyan.
    for quad, *_ in candidates:
        cv2.polylines(vis, [quad.reshape(-1, 1, 2).astype(np.int32)], True, (255, 255, 0), 2)

    # Draw orphan hull polygons with per-vertex angle annotations.
    for color, bad in bad_by_color.items():
        orphan_idxs = set(range(len(bad))) - used_per_color.get(color, set())
        orphans = [bad[i] for i in orphan_idxs if cv2.contourArea(bad[i]) >= frag_min_area]
        if not orphans:
            continue

        edge_bgr = _CARD_COLOR_BGR.get(color, (128, 128, 128))
        all_pts = np.concatenate([c.reshape(-1, 2) for c in orphans])
        combined = all_pts.reshape(-1, 1, 2).astype(np.int32)
        hull = cv2.convexHull(combined)
        peri = cv2.arcLength(hull, True)
        poly = cv2.approxPolyDP(hull, max(2.0, 0.015 * peri), True).reshape(-1, 2).astype(float)
        n_pts = len(poly)

        cv2.polylines(vis, [poly.astype(np.int32).reshape(-1, 1, 2)], True, edge_bgr, 2)

        for j in range(n_pts):
            p_prev = poly[(j - 1) % n_pts]
            p_cur  = poly[j]
            p_next = poly[(j + 1) % n_pts]
            angle = _vertex_angle(p_prev, p_cur, p_next)
            is_corner = abs(angle - 90.0) < angle_tol
            dot_bgr = (0, 200, 0) if is_corner else (0, 0, 200)
            cx, cy = int(p_cur[0]), int(p_cur[1])
            cv2.circle(vis, (cx, cy), 6, dot_bgr, -1)
            cv2.putText(vis, f"{angle:.0f}",
                        (cx + 8, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, dot_bgr, 1, cv2.LINE_AA)

    return vis


def debug_quads(region: np.ndarray,
                sat_lo: int = 60, val_lo: int = 50) -> np.ndarray:
    """
    Returns an annotated BGR image showing every quad from get_quads.
    Outline color = card color (full brightness if kept, dimmed if suppressed).
    Label: "<color> <M1|M2|M3> [kept]" or "[IoU=X.XX]" / "[cov=X.XX]".
    NMS mirrors get_card_images: color-aware, iou_thresh=0.40, min_coverage=0.15.
    """
    masks = preprocess_mask(region, sat_lo, val_lo)
    combined = np.zeros(region.shape[:2], dtype=np.uint8)
    for m, *_ in masks:
        combined = cv2.bitwise_or(combined, m)
    quads = get_quads(region, masks)
    vis = region.copy()

    entries = []  # (quad, color, method, pre_reason)
    for quad, is_dark, color, method in quads:
        cov = _quad_coverage(quad, combined)
        if cov < 0.15:
            entries.append((quad, color, method, f"cov={cov:.2f}"))
            continue
        img = _warp_card(region, quad, skip_tight_crop=is_dark)
        entries.append((quad, color, method, "" if img is not None else "warp-fail"))

    entries.sort(
        key=lambda x: (x[0][:, 0].max() - x[0][:, 0].min()) *
                      (x[0][:, 1].max() - x[0][:, 1].min()),
        reverse=True,
    )

    kept_per_color: dict[str, list[np.ndarray]] = {}
    decisions = []  # (quad, color, method, reason)
    for quad, color, method, pre_reason in entries:
        if pre_reason:
            decisions.append((quad, color, method, pre_reason))
            continue
        reason = ""
        for kq in kept_per_color.get(color, []):
            iou = _bbox_iou(quad, kq)
            if iou > 0.40:
                reason = f"IoU={iou:.2f}"
                break
        if not reason:
            kept_per_color.setdefault(color, []).append(quad)
        decisions.append((quad, color, method, reason))

    for quad, color, method, reason in decisions:
        suppressed = bool(reason)
        base_bgr = _CARD_COLOR_BGR.get(color, (128, 128, 128))
        outline_bgr = tuple(int(c * 0.45) for c in base_bgr) if suppressed else base_bgr
        thickness = 2 if suppressed else 4
        pts = quad.reshape(-1, 1, 2).astype(np.int32)
        if not suppressed:
            cv2.polylines(vis, [pts], True, (255, 255, 255), thickness + 4)  # white border
        cv2.polylines(vis, [pts], True, outline_bgr, thickness)
        qcx = int(quad.mean(axis=0)[0])
        qcy = int(quad.mean(axis=0)[1])
        status = f"[{reason}]" if suppressed else "[kept]"
        label = f"{color} {method} {status}"
        cv2.putText(vis, label, (qcx - 40, qcy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, outline_bgr, 1, cv2.LINE_AA)

    return vis
