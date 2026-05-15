"""
Interactive card labeling tool.

Detects all cards across the training dataset and shows them one by one.
Type the card value in the terminal to label it; the image is saved to
labeled_cards/{value}/{image_id}_{region}_{i}.jpg

The tool remembers which cards were already labeled (by filename) so it can
be interrupted and resumed at any time.

Usage:
    python label_cards.py               # full dataset
    python label_cards.py L1000770      # single image

Terminal commands:
    0-9, skip, reverse, draw_2, draw_4, wild  — label value
    s   — skip this card
    q   — quit (progress is saved)
"""

import os
import sys
import csv
import cv2
import matplotlib.pyplot as plt

from src.card_detection import detect_cards

DATA_DIR    = "iapr-26-uno-vision-challenge/train_images"
CSV_PATH    = "iapr-26-uno-vision-challenge/train.csv"
LABELED_DIR = "labeled_cards"
SAT_LO = 80
VAL_LO = 120

VALID_VALUES = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'skip', 'reverse', 'draw_2', 'draw_4', 'wild',
}

REGIONS = {
    "p1":     lambda H, W: (slice(2 * H // 3, H),       slice(W // 4,     3 * W // 4)),
    "p2":     lambda H, W: (slice(H // 4, 3 * H // 4),  slice(3 * W // 4, W)),
    "p3":     lambda H, W: (slice(0, H // 3),            slice(W // 4,     3 * W // 4)),
    "p4":     lambda H, W: (slice(H // 4, 3 * H // 4),  slice(0,          W // 4)),
    "center": lambda H, W: (slice(H // 3, 2 * H // 3),  slice(W // 3,     2 * W // 3)),
}

CSV_KEYS = {
    "p1": "player_1_cards", "p2": "player_2_cards",
    "p3": "player_3_cards", "p4": "player_4_cards",
    "center": "center_card",
}


def _load_gt():
    gt = {}
    if not os.path.exists(CSV_PATH):
        return gt
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            gt[row["image_id"]] = row
    return gt


def _labeled_ids():
    """Return the set of card IDs already saved to disk."""
    ids = set()
    if not os.path.isdir(LABELED_DIR):
        return ids
    for val_dir in os.scandir(LABELED_DIR):
        if not val_dir.is_dir():
            continue
        for f in os.scandir(val_dir.path):
            if f.name.endswith(".jpg"):
                ids.add(os.path.splitext(f.name)[0])
    return ids


def _collect_cards(image_ids, gt_all):
    """
    Detect cards in every region of every image.
    Returns list of (card_id, card_img, color, gt_hint) where gt_hint is the
    GT label string when unambiguous (1 detected == 1 GT), else None.
    """
    cards = []
    n = len(image_ids)
    for idx, image_id in enumerate(image_ids):
        path = os.path.join(DATA_DIR, f"{image_id}.jpg")
        img = cv2.imread(path)
        if img is None:
            print(f"  [SKIP] {image_id} — cannot load")
            continue
        H, W = img.shape[:2]
        gt_row = gt_all.get(image_id, {})
        print(f"  [{idx+1}/{n}] {image_id}", end="\r", flush=True)

        for name, crop_fn in REGIONS.items():
            row_sl, col_sl = crop_fn(H, W)
            region = img[row_sl, col_sl]
            detected = detect_cards(region, sat_lo=SAT_LO, val_lo=VAL_LO)

            gt_raw = gt_row.get(CSV_KEYS[name], "")
            gt_labels = [] if not gt_raw or gt_raw == "EMPTY" else gt_raw.split(";")

            for i, (card_img, color) in enumerate(detected):
                card_id = f"{image_id}_{name}_{i}"
                gt_hint = (gt_labels[i]
                           if len(detected) == len(gt_labels) and i < len(gt_labels)
                           else None)
                cards.append((card_id, card_img, color, gt_hint))

    print()
    return cards


def _print_summary():
    if not os.path.isdir(LABELED_DIR):
        return
    print("\nLabeled card counts:")
    total = 0
    for entry in sorted(os.scandir(LABELED_DIR), key=lambda e: e.name):
        if entry.is_dir():
            count = sum(1 for f in os.scandir(entry.path) if f.name.endswith(".jpg"))
            print(f"  {entry.name:12s}: {count}")
            total += count
    print(f"  {'TOTAL':12s}: {total}")


def label_cards(image_ids=None):
    if image_ids is None:
        with open(CSV_PATH, newline="") as f:
            image_ids = [row["image_id"] for row in csv.DictReader(f)]

    already = _labeled_ids()
    print(f"Already labeled: {len(already)} cards")

    gt_all = _load_gt()
    print(f"Collecting cards from {len(image_ids)} image(s)...")
    all_cards = _collect_cards(image_ids, gt_all)

    pending = [(cid, img, color, gt)
               for cid, img, color, gt in all_cards
               if cid not in already]

    print(f"Total detected: {len(all_cards)}   Pending: {len(pending)}\n")
    if not pending:
        print("Nothing to label.")
        _print_summary()
        return

    os.makedirs(LABELED_DIR, exist_ok=True)

    plt.ion()
    fig, ax = plt.subplots(figsize=(4, 6))
    fig.tight_layout()

    labeled_this_session = 0

    for card_idx, (card_id, card_img, color, gt_hint) in enumerate(pending):
        # --- display ---
        ax.clear()
        ax.imshow(cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB))
        title = f"{card_id}\ncolor={color}   [{card_idx+1}/{len(pending)}]"
        if gt_hint:
            title += f"\nGT hint: {gt_hint}"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.05)

        # --- label prompt ---
        hint_str = f" [enter={gt_hint}]" if gt_hint else ""
        prompt = f"Value{hint_str} (s=skip, q=quit): "

        while True:
            raw = input(prompt).strip().lower()

            if raw == "" and gt_hint:
                raw = gt_hint.split("_")[-1] if "_" in gt_hint else gt_hint

            if raw == "q":
                plt.close()
                print(f"\nStopped. Labeled {labeled_this_session} cards this session.")
                _print_summary()
                return

            if raw == "s":
                break

            if raw in VALID_VALUES:
                dest_dir = os.path.join(LABELED_DIR, raw)
                os.makedirs(dest_dir, exist_ok=True)
                cv2.imwrite(os.path.join(dest_dir, f"{card_id}.jpg"), card_img)
                labeled_this_session += 1
                print(f"  → {raw}/{card_id}.jpg")
                break

            print(f"  Unknown: '{raw}'. Valid values: {sorted(VALID_VALUES)}")

    plt.close()
    print(f"\nAll done. Labeled {labeled_this_session} cards this session.")
    _print_summary()


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 1:
        label_cards([sys.argv[1]])
    else:
        label_cards()
