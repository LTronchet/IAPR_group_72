"""
Usage:
    python test_card_detection.py [IMAGE_ID]

Default IMAGE_ID: L1000770
Ground truth loaded from train.csv (EMPTY hands -> 0 cards expected).
Player layout: p1=bottom, p2=right, p3=top, p4=left (fixed for all images).
"""

import sys
import os
import csv
import cv2
import matplotlib.pyplot as plt

from src.card_detection import detect_cards

DATA_DIR  = "iapr-26-uno-vision-challenge/train_images"
CSV_PATH  = "iapr-26-uno-vision-challenge/train.csv"
DEBUG_DIR = "debug_output"

REGIONS = {
    "p3":     lambda H, W: (slice(0,          H // 3),       slice(W // 4,     3 * W // 4)),
    "p4":     lambda H, W: (slice(H // 4,     3 * H // 4),   slice(0,          W // 4)),
    "p2":     lambda H, W: (slice(H // 4,     3 * H // 4),   slice(3 * W // 4, W)),
    "p1":     lambda H, W: (slice(2 * H // 3, H),            slice(W // 4,     3 * W // 4)),
    "center": lambda H, W: (slice(H // 3,     2 * H // 3),   slice(W // 3,     2 * W // 3)),
}
LABELS = {
    "p1": "Player 1 (bottom)",
    "p2": "Player 2 (right)",
    "p3": "Player 3 (top)",
    "p4": "Player 4 (left)",
    "center": "Center",
}
CSV_KEYS = {
    "p1": "player_1_cards",
    "p2": "player_2_cards",
    "p3": "player_3_cards",
    "p4": "player_4_cards",
    "center": "center_card",
}


def _load_gt(image_id):
    if not os.path.exists(CSV_PATH):
        return None
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            if row["image_id"] == image_id:
                return row
    return None


def _gt_count(gt, csv_key):
    if gt is None:
        return None
    val = gt.get(csv_key, "")
    if not val or val == "EMPTY":
        return 0
    return len(val.split(";"))


def _show_region(name, region, cards, gt_count):
    n = len(cards)
    gt_str = f"GT={gt_count}" if gt_count is not None else "GT=?"
    if gt_count is not None:
        result = "OK" if n == gt_count else f"FAIL (detected {n})"
    else:
        result = f"detected {n}"
    title = f"{LABELS[name]} — {gt_str} — {result}"

    ncols = 1 + max(n, 1)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 5))
    axes = list(axes) if ncols > 1 else [axes]

    thumb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    axes[0].imshow(thumb)
    axes[0].set_title("region")
    axes[0].axis("off")

    for i in range(1, ncols):
        if i - 1 < n:
            axes[i].imshow(cv2.cvtColor(cards[i - 1], cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"card {i - 1}")
        else:
            axes[i].axis("off")
        axes[i].axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def _save_debug(name, region, cards):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}_region.jpg"), region)
    for i, card in enumerate(cards):
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}_card_{i}.jpg"), card)


def run(image_id="L1000770"):
    path = os.path.join(DATA_DIR, f"{image_id}.jpg")
    img = cv2.imread(path)
    assert img is not None, f"Cannot load {path}"
    H, W = img.shape[:2]

    gt = _load_gt(image_id)
    print(f"\nImage: {image_id}  ({W}x{H})")
    if gt:
        print(f"  center={gt['center_card']}  active={gt['active_player']}")
        for p in ("player_1_cards", "player_2_cards", "player_3_cards", "player_4_cards"):
            print(f"  {p}: {gt[p]}")

    results = {}
    for name, crop_fn in REGIONS.items():
        row_sl, col_sl = crop_fn(H, W)
        region = img[row_sl, col_sl]
        cards = detect_cards(region)
        gt_count = _gt_count(gt, CSV_KEYS[name])
        results[name] = (region, cards, gt_count)
        _save_debug(name, region, cards)

    print("\nResults:")
    all_ok = True
    for name in ("p1", "p2", "p3", "p4", "center"):
        _, cards, gt_count = results[name]
        n = len(cards)
        if gt_count is not None:
            ok = n == gt_count
            all_ok = all_ok and ok
            status = "OK" if ok else "FAIL"
            print(f"  {LABELS[name]:22s}  detected={n}  expected={gt_count}  [{status}]")
        else:
            print(f"  {LABELS[name]:22s}  detected={n}  expected=?")

    for name in ("p1", "p2", "p3", "p4", "center"):
        region, cards, gt_count = results[name]
        _show_region(name, region, cards, gt_count)

    return all_ok


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    image_id = sys.argv[1] if len(sys.argv) > 1 else "L1000770"
    ok = run(image_id)
    print("\nAll counts match." if ok else "\nSome counts are wrong.")
