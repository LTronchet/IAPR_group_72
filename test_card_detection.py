import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from src.card_detection import detect_cards, CARD_WIDTH, CARD_HEIGHT

DATA_DIR = "iapr-26-uno-vision-challenge/train_images"
DEBUG_DIR = "debug_output"

# ── Ground truth for L1000770 (no overlapping cards) ─────────────────────────
# center: y_1   active: p3
# p1: EMPTY     p2: y_4;g_2;r_3 (3 cards)
# p3: b_5;y_2;r_skip (3 cards)   p4: EMPTY
#
# Player layout (fixed): p1=bottom, p2=right, p3=top, p4=left
# Image size: 4000 × 2662
#
# Crop coordinates determined by visual inspection with a grid overlay:
#   - Player 3 (top):   rows [0, H//3],        cols [W//4, 3*W//4]
#   - Player 2 (right): rows [H//4, 3*H//4],   cols [3*W//4, W]
#   - Center:           rows [H//3, 2*H//3],   cols [W//3, 2*W//3]

IMAGE_FILE = "L1000770.jpg"
GT_P2_COUNT = 3   # y_4, g_2, r_3
GT_P3_COUNT = 3   # b_5, y_2, r_skip


def _load():
    path = os.path.join(DATA_DIR, IMAGE_FILE)
    img = cv2.imread(path)
    assert img is not None, f"Could not load {path}"
    return img


def _crop_p3(img):
    """Player 3 region (top): 3 non-overlapping cards."""
    H, W = img.shape[:2]
    return img[0 : H // 3, W // 4 : 3 * W // 4]


def _crop_p2(img):
    """Player 2 region (right): 3 non-overlapping cards."""
    H, W = img.shape[:2]
    return img[H // 4 : 3 * H // 4, 3 * W // 4 : W]


def _save_debug(name, region, cards):
    """Save the region and each detected card as images in DEBUG_DIR."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}_region.jpg"), region)
    for i, card in enumerate(cards):
        cv2.imwrite(os.path.join(DEBUG_DIR, f"{name}_card_{i}.jpg"), card)


def _show_cards(title, cards):
    """Display detected cards side by side with matplotlib."""
    if not cards:
        print(f"  [{title}] No cards to display.")
        return
    fig, axes = plt.subplots(1, len(cards), figsize=(4 * len(cards), 5))
    if len(cards) == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=14)
    for ax, card in zip(axes, cards):
        ax.imshow(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_returns_list():
    """detect_cards returns a list on a real player region."""
    img = _load()
    region = _crop_p3(img)
    result = detect_cards(region)
    assert isinstance(result, list), (
        f"Expected list, got {type(result)}"
    )
    print(f"  [returns_list] OK — got {len(result)} card(s)")


def test_card_shape():
    """Every returned card has shape (CARD_HEIGHT, CARD_WIDTH, 3)."""
    img = _load()
    region = _crop_p3(img)
    cards = detect_cards(region)
    assert len(cards) > 0, "No cards detected — cannot check shape"
    for i, card in enumerate(cards):
        assert card.shape == (CARD_HEIGHT, CARD_WIDTH, 3), (
            f"Card {i} has shape {card.shape}, "
            f"expected ({CARD_HEIGHT}, {CARD_WIDTH}, 3)"
        )
    print(f"  [card_shape] OK — all {len(cards)} card(s) are "
          f"{CARD_WIDTH}×{CARD_HEIGHT}")


def test_count_matches_ground_truth():
    """Detected card count matches the annotation for both player regions."""
    img = _load()

    # Player 3 — top region
    region_p3 = _crop_p3(img)
    cards_p3 = detect_cards(region_p3)
    _save_debug("p3", region_p3, cards_p3)
    assert len(cards_p3) == GT_P3_COUNT, (
        f"Player 3: expected {GT_P3_COUNT} cards, detected {len(cards_p3)}"
    )
    print(f"  [count p3] OK — {len(cards_p3)}/{GT_P3_COUNT} cards")
    _show_cards("Player 3 — b_5 · y_2 · r_skip", cards_p3)

    # Player 2 — right region
    region_p2 = _crop_p2(img)
    cards_p2 = detect_cards(region_p2)
    _save_debug("p2", region_p2, cards_p2)
    assert len(cards_p2) == GT_P2_COUNT, (
        f"Player 2: expected {GT_P2_COUNT} cards, detected {len(cards_p2)}"
    )
    print(f"  [count p2] OK — {len(cards_p2)}/{GT_P2_COUNT} cards")
    _show_cards("Player 2 — y_4 · g_2 · r_3", cards_p2)


def test_empty_region():
    """Returns empty list on a blank white image (no cards present)."""
    blank = np.ones((300, 400, 3), dtype=np.uint8) * 255
    result = detect_cards(blank)
    assert result == [], f"Expected [], got {len(result)} card(s)"
    print("  [empty_region] OK — returned []")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    tests = [
        test_returns_list,
        test_card_shape,
        test_count_matches_ground_truth,
        test_empty_region,
    ]

    passed = 0
    for test in tests:
        print(f"\n{test.__name__}")
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL — {e}")
        except Exception as e:
            print(f"  ERROR — {type(e).__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed.")
    print(f"Debug images saved in '{DEBUG_DIR}/'")
