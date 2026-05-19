"""
Evaluate a predictions CSV against the training ground truth.

Usage:
    python score.py --pred <predictions.csv> [--gt <train.csv>]

Default --gt: iapr-26-uno-vision-challenge/train.csv

The script compares predictions to ground truth for every image_id that
appears in both files.  For multi-card fields the order of labels is
ignored (sorted before comparison).

Metrics reported:
  - Per-field accuracy (center_card, active_player, player_1_cards, …)
  - Overall field accuracy (all fields across all images)
  - Image-level accuracy (fraction of images where every field is correct)
"""
from __future__ import annotations

import argparse
import csv
import os

_FIELDS = [
    "center_card",
    "active_player",
    "player_1_cards",
    "player_2_cards",
    "player_3_cards",
    "player_4_cards",
]

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_GT = os.path.join(
    _HERE, "iapr-26-uno-vision-challenge", "train.csv"
)


def _normalise(value: str) -> str:
    """Sort semicolon-separated card labels; strip whitespace."""
    value = value.strip()
    if not value or value == "EMPTY":
        return "EMPTY"
    return ";".join(sorted(value.split(";")))


def _load(path: str) -> dict[str, dict]:
    with open(path, newline="") as f:
        return {row["image_id"]: row for row in csv.DictReader(f)}


def score(pred_path: str, gt_path: str) -> None:
    pred = _load(pred_path)
    gt   = _load(gt_path)

    common = sorted(set(pred) & set(gt))
    if not common:
        print("No overlapping image_ids between prediction and ground truth.")
        return

    print(f"Scoring {len(common)} image(s) (pred={len(pred)}, gt={len(gt)})\n")

    field_correct = {f: 0 for f in _FIELDS}
    field_total   = {f: 0 for f in _FIELDS}
    image_correct = 0
    failures: list[tuple[str, str, str, str]] = []

    for image_id in common:
        p_row = pred[image_id]
        g_row = gt[image_id]
        img_ok = True

        for field in _FIELDS:
            p_val = _normalise(p_row.get(field, ""))
            g_val = _normalise(g_row.get(field, ""))
            field_total[field] += 1
            if p_val == g_val:
                field_correct[field] += 1
            else:
                img_ok = False
                failures.append((image_id, field, p_val, g_val))

        if img_ok:
            image_correct += 1

    # ── Per-field accuracy ──────────────────────────────────────────────────
    print("Per-field accuracy:")
    total_correct = total_fields = 0
    for field in _FIELDS:
        c, t = field_correct[field], field_total[field]
        pct = 100.0 * c / t if t else float("nan")
        print(f"  {field:20s}  {c:3d}/{t}  ({pct:.1f} %)")
        total_correct += c
        total_fields  += t

    # ── Overall ─────────────────────────────────────────────────────────────
    overall_pct = 100.0 * total_correct / total_fields if total_fields else float("nan")
    image_pct   = 100.0 * image_correct / len(common)
    print(f"\n{'Overall field accuracy':20s}  {total_correct}/{total_fields}  ({overall_pct:.1f} %)")
    print(f"{'Image-level accuracy':20s}  {image_correct}/{len(common)}  ({image_pct:.1f} %)")

    # ── Failures ────────────────────────────────────────────────────────────
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for image_id, field, pred_val, gt_val in sorted(failures):
            print(f"  {image_id}  {field:20s}  pred={pred_val!r:30s}  gt={gt_val!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a UNO Vision prediction CSV")
    parser.add_argument("--pred", required=True, help="Predictions CSV path")
    parser.add_argument("--gt", default=_DEFAULT_GT, help="Ground-truth CSV path")
    args = parser.parse_args()
    score(args.pred, args.gt)


if __name__ == "__main__":
    os.chdir(_HERE)
    main()
