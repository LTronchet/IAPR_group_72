"""
Usage:
    python test_card_classification.py [IMAGE_ID]

Default IMAGE_ID: L1000770
Templates are built automatically on first run and cached in templates/.
"""

import sys
import os
import csv
import cv2
import matplotlib.pyplot as plt

from src.card_detection import detect_cards, debug_mask
from src.card_classification import (classify_card, load_templates,
                                      save_templates, build_templates_from_labeled)

DATA_DIR      = "iapr-26-uno-vision-challenge/train_images"
CSV_PATH      = "iapr-26-uno-vision-challenge/train.csv"
DEBUG_DIR     = "debug_output"
TEMPLATES_DIR = "templates"
LABELED_DIR   = "labeled_cards"

SAT_LO = 80
VAL_LO = 120

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


def _get_templates():
    if os.path.isdir(TEMPLATES_DIR) and os.listdir(TEMPLATES_DIR):
        templates = load_templates(TEMPLATES_DIR)
        print(f"Templates loaded from {TEMPLATES_DIR!r}: {sorted(templates.keys())}")
    else:
        print(f"Building templates from {LABELED_DIR!r}...")
        templates = build_templates_from_labeled(LABELED_DIR)
        save_templates(templates, TEMPLATES_DIR)
        print(f"Templates saved to {TEMPLATES_DIR!r}: {sorted(templates.keys())}")
    return templates


def _show_region(name, region, cards, gt_count, templates):
    n = len(cards)
    gt_str = f"GT={gt_count}" if gt_count is not None else "GT=?"
    if gt_count is not None:
        result = "OK" if n == gt_count else f"FAIL (detected {n})"
    else:
        result = f"detected {n}"
    title = f"{LABELS[name]} — {gt_str} — {result}  [sat_lo={SAT_LO} val_lo={VAL_LO}]"

    mask = debug_mask(region, name=name, sat_lo=SAT_LO, val_lo=VAL_LO)

    ncols = 2 + max(n, 1)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 5))
    axes = list(axes)

    axes[0].imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
    axes[0].set_title("region")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    axes[1].set_title("mask")
    axes[1].axis("off")

    for i in range(2, ncols):
        if i - 2 < n:
            card_img, card_color = cards[i - 2]
            full_label = classify_card(card_img, card_color, templates)
            axes[i].imshow(cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"card {i - 2} {full_label}")
        else:
            axes[i].axis("off")
        axes[i].axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def _save_debug(name, region, cards, templates):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}_region.jpg"), region)
    for i, (card, color) in enumerate(cards):
        full_label = classify_card(card, color, templates)
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}_card_{i}_{full_label}.jpg"), card)


def run(image_id="L1000770"):
    path = os.path.join(DATA_DIR, f"{image_id}.jpg")
    img = cv2.imread(path)
    assert img is not None, f"Cannot load {path}"
    H, W = img.shape[:2]

    templates = _get_templates()

    gt = _load_gt(image_id)
    print(f"\nImage: {image_id}  ({W}x{H})  sat_lo={SAT_LO}  val_lo={VAL_LO}")
    if gt:
        print(f"  center={gt['center_card']}  active={gt['active_player']}")
        for p in ("player_1_cards", "player_2_cards", "player_3_cards", "player_4_cards"):
            print(f"  {p}: {gt[p]}")

    results = {}
    for name, crop_fn in REGIONS.items():
        row_sl, col_sl = crop_fn(H, W)
        region = img[row_sl, col_sl]
        cards = detect_cards(region, sat_lo=SAT_LO, val_lo=VAL_LO)
        gt_count = _gt_count(gt, CSV_KEYS[name])
        results[name] = (region, cards, gt_count)
        _save_debug(name, region, cards, templates)

    print("\nResults:")
    all_ok = True
    for name in ("p1", "p2", "p3", "p4", "center"):
        _, cards, gt_count = results[name]
        n = len(cards)
        labels = [classify_card(c, col, templates) for c, col in cards]
        gt_val = gt.get(CSV_KEYS[name], "?") if gt else "?"
        if gt_count is not None:
            ok = n == gt_count
            all_ok = all_ok and ok
            status = "OK" if ok else "FAIL"
            print(f"  {LABELS[name]:22s}  detected={labels}  GT={gt_val}  [{status}]")
        else:
            print(f"  {LABELS[name]:22s}  detected={labels}  GT=?")

    for name in ("p1", "p2", "p3", "p4", "center"):
        region, cards, gt_count = results[name]
        _show_region(name, region, cards, gt_count, templates)

    return all_ok


def _show_image_summary(image_id, results, templates):
    region_order = ("p1", "p2", "p3", "p4", "center")
    max_cards = max(len(cards) for _, cards, _, _ in results.values())
    ncols = 2 + max(max_cards, 1)
    nrows = len(region_order)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 4 * nrows))

    for row_idx, name in enumerate(region_order):
        region, cards, gt_labels, gt_count = results[name]
        labels = [classify_card(c, col, templates) for c, col in cards]
        n = len(labels)
        axs = axes[row_idx]

        if gt_count is not None:
            count_ok = n == gt_count
            label_ok = count_ok and sorted(labels) == sorted(gt_labels)
            row_color = "green" if label_ok else "red"
            if label_ok:
                status = "OK"
            elif not count_ok:
                status = f"FAIL count (detected {n}, GT={gt_count})"
            else:
                status = f"FAIL labels"
        else:
            row_color = "black"
            status = f"detected {n}  GT=?"

        mask = debug_mask(region, name=name, sat_lo=SAT_LO, val_lo=VAL_LO)

        axs[0].imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"{LABELS[name]}\n{status}", color=row_color, fontsize=9, fontweight="bold")
        axs[0].axis("off")

        axs[1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        axs[1].set_title("mask", fontsize=8)
        axs[1].axis("off")

        for col_idx in range(2, ncols):
            card_idx = col_idx - 2
            if card_idx < n:
                card_img, _ = cards[card_idx]
                axs[col_idx].imshow(cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB))
                axs[col_idx].set_title(labels[card_idx], fontsize=8)
            else:
                axs[col_idx].axis("off")
            axs[col_idx].axis("off")

    fig.suptitle(f"{image_id}  [sat_lo={SAT_LO}  val_lo={VAL_LO}]", fontsize=13)
    plt.tight_layout()
    plt.show()


def run_all():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        return

    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    templates = _get_templates()

    REGION_ORDER = ("p1", "p2", "p3", "p4", "center")
    correct_count  = {r: 0 for r in REGION_ORDER}
    correct_labels = {r: 0 for r in REGION_ORDER}
    total          = {r: 0 for r in REGION_ORDER}
    fails = []

    for row in rows:
        image_id = row["image_id"]
        path = os.path.join(DATA_DIR, f"{image_id}.jpg")
        img = cv2.imread(path)
        if img is None:
            print(f"  [SKIP] cannot load {path}")
            continue
        H, W = img.shape[:2]

        img_ok = True
        results = {}
        for name in REGION_ORDER:
            row_sl, col_sl = REGIONS[name](H, W)
            region = img[row_sl, col_sl]
            cards = detect_cards(region, sat_lo=SAT_LO, val_lo=VAL_LO)
            gt_raw = row.get(CSV_KEYS[name], "")
            gt_labels = [] if not gt_raw or gt_raw == "EMPTY" else gt_raw.split(";")
            gt_count = len(gt_labels)
            results[name] = (region, cards, gt_labels, gt_count)

            total[name] += 1
            n = len(cards)
            if n == gt_count:
                correct_count[name] += 1
                pred = sorted(classify_card(c, col, templates) for c, col in cards)
                if pred == sorted(gt_labels):
                    correct_labels[name] += 1
                else:
                    img_ok = False
                    fails.append((image_id, name, pred, gt_labels))
            else:
                img_ok = False
                fails.append((image_id, name,
                               [classify_card(c, col, templates) for c, col in cards],
                               gt_labels))

        status = "OK  " if img_ok else "FAIL"
        print(f"[{status}] {image_id}")
        if not img_ok:
            _show_image_summary(image_id, results, templates)

    print("\n--- Per-region accuracy (labels) ---")
    all_cl = sum(correct_labels.values())
    all_t  = sum(total.values())
    for name in REGION_ORDER:
        t  = total[name]
        cc = correct_count[name]
        cl = correct_labels[name]
        pct = 100.0 * cl / t if t else float("nan")
        print(f"  {LABELS[name]:22s}  count={cc:3d}/{t}  labels={cl:3d}/{t}  ({pct:.1f}%)")
    pct_all = 100.0 * all_cl / all_t if all_t else float("nan")
    print(f"  {'Overall':22s}  labels={all_cl:3d}/{all_t}  ({pct_all:.1f}%)")

    if fails:
        print(f"\n--- Failures ({len(fails)}) ---")
        for image_id, name, pred, gt in sorted(fails):
            print(f"  {image_id}  {LABELS[name]:22s}  pred={pred}  GT={gt}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 1:
        ok = run(sys.argv[1])
        print("\nAll counts match." if ok else "\nSome counts are wrong.")
    else:
        run_all()
