from __future__ import annotations

import csv
import os
from typing import Optional

import cv2
import numpy as np

from src.card_detection import detect_cards
from src.card_classification import _corner_patch, _white_mask

_CSV_KEYS = {
    'p1': 'player_1_cards',
    'p2': 'player_2_cards',
    'p3': 'player_3_cards',
    'p4': 'player_4_cards',
    'center': 'center_card',
}

_REGIONS = {
    'p3':     lambda H, W: (slice(0,          H // 3),       slice(W // 4,     3 * W // 4)),
    'p4':     lambda H, W: (slice(H // 4,     3 * H // 4),   slice(0,          W // 4)),
    'p2':     lambda H, W: (slice(H // 4,     3 * H // 4),   slice(3 * W // 4, W)),
    'p1':     lambda H, W: (slice(2 * H // 3, H),            slice(W // 4,     3 * W // 4)),
    'center': lambda H, W: (slice(H // 3,     2 * H // 3),   slice(W // 3,     2 * W // 3)),
}


def _gt_value(label: str) -> Optional[str]:
    """'r_5' -> '5',  'b_skip' -> 'skip',  'wild' -> 'wild',  'draw_4' -> 'draw_4'."""
    if not label or label == 'EMPTY':
        return None
    if '_' in label:
        return label.split('_', 1)[1]
    return label


def build_templates(
    csv_path: str,
    train_images_dir: str,
    sat_lo: int = 80,
    val_lo: int = 120,
    raw_dir: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Build averaged white-corner templates from labeled training data.

    Only processes regions with exactly 1 GT card AND exactly 1 detected card
    to guarantee unambiguous label -> image association without spatial sorting.

    Args:
        raw_dir: if set, saves each BGR corner patch as a PNG under
                 raw_dir/<value>/<image_id>_<region>_<TL|BR>.png for manual curation.

    Returns:
        dict mapping value label -> (48, 33) float32 averaged mask.
    """
    raw: dict[str, list[np.ndarray]] = {}

    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        image_id = row['image_id']
        path = os.path.join(train_images_dir, f'{image_id}.jpg')
        img = cv2.imread(path)
        if img is None:
            continue
        H, W = img.shape[:2]

        for name, crop_fn in _REGIONS.items():
            csv_key = _CSV_KEYS[name]
            gt_str = row.get(csv_key, '')
            if not gt_str or gt_str == 'EMPTY':
                continue
            gt_labels = [s.strip() for s in gt_str.split(';') if s.strip()]
            if len(gt_labels) != 1:
                continue

            value = _gt_value(gt_labels[0])
            if value is None:
                continue

            row_sl, col_sl = crop_fn(H, W)
            region = img[row_sl, col_sl]
            cards = detect_cards(region, sat_lo=sat_lo, val_lo=val_lo)
            if len(cards) != 1:
                continue

            card_img, _ = cards[0]
            for c in ('TL', 'BR'):
                patch = _corner_patch(card_img, c)
                raw.setdefault(value, []).append(_white_mask(patch))
                if raw_dir is not None:
                    out_dir = os.path.join(raw_dir, value)
                    os.makedirs(out_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(out_dir, f'{image_id}_{name}_{c}.png'), patch)

    templates = {v: np.mean(masks, axis=0) for v, masks in raw.items()}
    counts = {v: len(raw[v]) for v in templates}
    print('[build_templates] Built templates: '
          + ', '.join(f'{v}({n})' for v, n in sorted(counts.items())))
    return templates


def build_templates_from_raw(raw_dir: str) -> dict[str, np.ndarray]:
    """
    Rebuild averaged templates from curated PNG patches in raw_dir/<value>/*.png.
    Delete bad patches from the directory before calling this to exclude them.
    """
    templates = {}
    counts = {}
    for value in sorted(os.listdir(raw_dir)):
        value_dir = os.path.join(raw_dir, value)
        if not os.path.isdir(value_dir):
            continue
        masks = []
        for fname in sorted(os.listdir(value_dir)):
            if fname.endswith('.png'):
                patch = cv2.imread(os.path.join(value_dir, fname))
                if patch is not None:
                    masks.append(_white_mask(patch))
        if masks:
            templates[value] = np.mean(masks, axis=0)
            counts[value] = len(masks)
    print('[build_templates_from_raw] Rebuilt templates: '
          + ', '.join(f'{v}({n})' for v, n in sorted(counts.items())))
    return templates
