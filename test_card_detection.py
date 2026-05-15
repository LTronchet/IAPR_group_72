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

from src.card_detection import detect_cards, debug_mask, debug_orphan_angles, debug_quads

DATA_DIR  = "iapr-26-uno-vision-challenge/train_images"
CSV_PATH  = "iapr-26-uno-vision-challenge/train.csv"
DEBUG_DIR = "debug_output"

# --- Tunable detection thresholds (change here to test) ---
SAT_LO = 80    # HSV saturation lower bound  (raise to exclude low-sat shadows)
VAL_LO = 120   # HSV value lower bound        (raise to exclude dark shadows)

# Set True to show Method 3 orphan hull polygons with per-vertex angle labels.
# Green dot = angle ≈ 90° (used as card corner anchor); red dot = other angles.
SHOW_ANGLES = False

# Set True to show all quads from get_quads colour-coded by NMS outcome.
# Green = kept, Red = suppressed (label shows IoU or center-inside reason).
SHOW_QUADS = True
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


def _show_image_summary(image_id, results):
    """Single figure: 5 rows (one per region), each showing region | mask | cards."""
    region_order = ("p1", "p2", "p3", "p4", "center")
    max_cards = max(len(cards) for _, cards, _ in results.values())
    ncols = 2 + max(max_cards, 1)
    nrows = len(region_order)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 4 * nrows))

    for row_idx, name in enumerate(region_order):
        region, cards, gt_count = results[name]
        n = len(cards)
        axs = axes[row_idx]

        if gt_count is not None:
            ok = n == gt_count
            row_color = "green" if ok else "red"
            status = "OK" if ok else f"FAIL (detected {n}, GT={gt_count})"
        else:
            row_color = "black"
            status = f"detected {n}  GT=?"

        mask = debug_mask(region, name=name, sat_lo=SAT_LO, val_lo=VAL_LO)

        axs[0].imshow(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
        axs[0].set_title(f"{LABELS[name]}\n{status}", color=row_color,
                         fontsize=9, fontweight="bold")
        axs[0].axis("off")

        axs[1].imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
        axs[1].set_title("mask", fontsize=8)
        axs[1].axis("off")

        for col_idx in range(2, ncols):
            card_idx = col_idx - 2
            if card_idx < n:
                card_img, card_color = cards[card_idx]
                axs[col_idx].imshow(cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB))
                axs[col_idx].set_title(f"card {card_idx}\n({card_color})", fontsize=8)
            else:
                axs[col_idx].axis("off")
            axs[col_idx].axis("off")

    fig.suptitle(f"{image_id}  [sat_lo={SAT_LO}  val_lo={VAL_LO}]", fontsize=13)
    plt.tight_layout()
    plt.show()


def _show_region(name, region, cards, gt_count):
    n = len(cards)
    gt_str = f"GT={gt_count}" if gt_count is not None else "GT=?"
    if gt_count is not None:
        result = "OK" if n == gt_count else f"FAIL (detected {n})"
    else:
        result = f"detected {n}"
    title = f"{LABELS[name]} — {gt_str} — {result}  [sat_lo={SAT_LO} val_lo={VAL_LO}]"

    mask = debug_mask(region, name=name, sat_lo=SAT_LO, val_lo=VAL_LO)

    # region | mask | detected cards
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
            axes[i].imshow(cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB))
            axes[i].set_title(f"card {i - 2} ({card_color})")
        else:
            axes[i].axis("off")
        axes[i].axis("off")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def _show_quads_debug(name, region):
    vis = debug_quads(region, sat_lo=SAT_LO, val_lo=VAL_LO)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(
        f"{LABELS[name]} — all quads NMS  "
        f"(green=kept, red=suppressed)  [sat_lo={SAT_LO} val_lo={VAL_LO}]",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def _show_angle_debug(name, region):
    vis = debug_orphan_angles(region, sat_lo=SAT_LO, val_lo=VAL_LO)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    ax.set_title(
        f"{LABELS[name]} — orphan hull angles  "
        f"(green=~90° corner, red=other)  [sat_lo={SAT_LO} val_lo={VAL_LO}]",
        fontsize=11,
    )
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def _save_debug(name, region, cards):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}_region.jpg"), region)
    for i, (card, color) in enumerate(cards):
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}_card_{i}_{color}.jpg"), card)


def run(image_id="L1000770"):
    path = os.path.join(DATA_DIR, f"{image_id}.jpg")
    img = cv2.imread(path)
    assert img is not None, f"Cannot load {path}"
    H, W = img.shape[:2]

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

    if SHOW_QUADS:
        for name in ("p1", "p2", "p3", "p4", "center"):
            region, _, _ = results[name]
            _show_quads_debug(name, region)

    if SHOW_ANGLES:
        for name in ("p1", "p2", "p3", "p4", "center"):
            region, _, _ = results[name]
            _show_angle_debug(name, region)

    return all_ok


def run_all():
    if not os.path.exists(CSV_PATH):
        print(f"CSV not found: {CSV_PATH}")
        return

    with open(CSV_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    REGION_ORDER = ("p1", "p2", "p3", "p4", "center")
    correct = {r: 0 for r in REGION_ORDER}
    total   = {r: 0 for r in REGION_ORDER}
    fails   = []  # (image_id, name, detected, expected)

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
            gt_count = _gt_count(row, CSV_KEYS[name])
            results[name] = (region, cards, gt_count)
            if gt_count is None:
                continue
            total[name] += 1
            n = len(cards)
            if n == gt_count:
                correct[name] += 1
            else:
                img_ok = False
                fails.append((image_id, name, n, gt_count))

        status = "OK  " if img_ok else "FAIL"
        print(f"[{status}] {image_id}")
        if not img_ok:
            _show_image_summary(image_id, results)

    print("\n--- Per-region accuracy ---")
    all_correct = sum(correct.values())
    all_total   = sum(total.values())
    for name in REGION_ORDER:
        t = total[name]
        c = correct[name]
        pct = 100.0 * c / t if t else float("nan")
        print(f"  {LABELS[name]:22s}  {c:3d}/{t:3d}  ({pct:.1f}%)")
    pct_all = 100.0 * all_correct / all_total if all_total else float("nan")
    print(f"  {'Overall':22s}  {all_correct:3d}/{all_total:3d}  ({pct_all:.1f}%)")

    if fails:
        print(f"\n--- Failures ({len(fails)}) ---")
        for image_id, name, det, exp in sorted(fails):
            print(f"  {image_id}  {LABELS[name]:22s}  detected={det}  expected={exp}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) > 1:
        ok = run(sys.argv[1])
        print("\nAll counts match." if ok else "\nSome counts are wrong.")
    else:
        run_all()
