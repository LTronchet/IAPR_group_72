"""
UNO Vision — main inference script.

Usage:
    python main.py [--test-dir PATH] [--output PATH]

Defaults:
    --test-dir  ../test_images   (relative to this file's directory)
    --output    submission.csv
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

# Add src/ to path so sub-modules resolve correctly regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from src.background_classification import classify_background
from src.active_player_detection import detect_active_player
from src.card_detection import detect_cards
from src.card_classification import (classify_card, load_templates,
                                      build_templates_from_labeled, save_templates)
from src.background_removal import remove_background

_LABELED_DIR   = os.path.join(_HERE, "labeled_cards")
_TEMPLATES_DIR = os.path.join(_HERE, "templates")
_SAT_LO, _VAL_LO = 80, 120

# Fixed-grid region crops (mirrors test_card_classification.py)
_REGIONS = {
    "p3":     lambda H, W: (slice(0,          H // 3),     slice(W // 4,     3 * W // 4)),
    "p4":     lambda H, W: (slice(H // 4,     3 * H // 4), slice(0,          W // 4)),
    "p2":     lambda H, W: (slice(H // 4,     3 * H // 4), slice(3 * W // 4, W)),
    "p1":     lambda H, W: (slice(2 * H // 3, H),          slice(W // 4,     3 * W // 4)),
    "center": lambda H, W: (slice(H // 3,     2 * H // 3), slice(W // 3,     2 * W // 3)),
}


def _load_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def _classify_region(region_bgr: np.ndarray, templates: dict) -> str:
    """Detect and classify all cards in a BGR region. Returns 'EMPTY' if none."""
    cards = detect_cards(region_bgr, sat_lo=_SAT_LO, val_lo=_VAL_LO)
    if not cards:
        return "EMPTY"
    return ";".join(classify_card(img, color, templates) for img, color in cards)


def run(test_dir: str, output_path: str) -> None:
    image_ids = sorted(
        f[:-4] for f in os.listdir(test_dir) if f.lower().endswith(".jpg")
    )
    print(f"Found {len(image_ids)} test images in {test_dir!r}")

    if os.path.isdir(_TEMPLATES_DIR) and any(f.endswith(".npy") for f in os.listdir(_TEMPLATES_DIR)):
        templates = load_templates(_TEMPLATES_DIR)
    else:
        templates = build_templates_from_labeled(_LABELED_DIR)
        save_templates(templates, _TEMPLATES_DIR)
    print(f"Templates loaded: {sorted(templates.keys())}")

    rows = []
    for image_id in image_ids:
        img_bgr = _load_bgr(os.path.join(test_dir, image_id + ".jpg"))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        background = classify_background(img_rgb)
        player = detect_active_player(img_rgb, background)

        clean_bgr = remove_background(img_bgr)
        H, W = clean_bgr.shape[:2]

        regions = {
            name: clean_bgr[row_sl, col_sl]
            for name, crop_fn in _REGIONS.items()
            for row_sl, col_sl in [crop_fn(H, W)]
        }

        rows.append({
            "image_id":       image_id,
            "center_card":    _classify_region(regions["center"], templates),
            "active_player":  player if player is not None else "EMPTY",
            "player_1_cards": _classify_region(regions["p1"],     templates),
            "player_2_cards": _classify_region(regions["p2"],     templates),
            "player_3_cards": _classify_region(regions["p3"],     templates),
            "player_4_cards": _classify_region(regions["p4"],     templates),
        })

    df = pd.DataFrame(rows, columns=[
        "image_id", "center_card", "active_player",
        "player_1_cards", "player_2_cards", "player_3_cards", "player_4_cards",
    ])
    df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path!r}")
    print(df["active_player"].value_counts().to_string())


def main() -> None:
    default_test = os.path.join(_HERE, "..", "test_images")
    parser = argparse.ArgumentParser(description="UNO Vision inference")
    parser.add_argument("--test-dir", default=default_test,
                        help="Directory containing test .jpg images")
    parser.add_argument("--output", default="submission.csv",
                        help="Output CSV path")
    args = parser.parse_args()
    run(os.path.normpath(args.test_dir), args.output)


if __name__ == "__main__":
    main()
