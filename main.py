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


def _load_rgb(path: str) -> np.ndarray:
    """Load an image from disk as an RGB numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run(test_dir: str, output_path: str) -> None:
    image_ids = sorted(
        f[:-4] for f in os.listdir(test_dir) if f.lower().endswith(".jpg")
    )
    print(f"Found {len(image_ids)} test images in {test_dir!r}")

    rows = []
    for image_id in image_ids:
        img = _load_rgb(os.path.join(test_dir, image_id + ".jpg"))
        background = classify_background(img)
        player = detect_active_player(img, background)
        rows.append({
            "image_id":       image_id,
            "center_card":    "EMPTY",
            "active_player":  player if player is not None else "EMPTY",
            "player_1_cards": "EMPTY",
            "player_2_cards": "EMPTY",
            "player_3_cards": "EMPTY",
            "player_4_cards": "EMPTY",
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
