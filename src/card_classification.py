from __future__ import annotations

import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CORNER_H_OUTER = 55   # rows 0:55 of the warped card
_CORNER_W_OUTER = 40   # cols 0:40 of the warped card
_BORDER = 7            # skip the white card border
# Effective template size: (48, 33)
_CORNER_H = _CORNER_H_OUTER - _BORDER
_CORNER_W = _CORNER_W_OUTER - _BORDER

_WHITE_THRESH = 210    # all B,G,R channels must exceed this to count as white

_COLOR_PREFIX = {'red': 'r', 'yellow': 'y', 'green': 'g', 'blue': 'b'}
_DARK_VALUES = {'wild', 'draw_4'}


# ---------------------------------------------------------------------------
# Corner extraction helpers (also used by template_builder)
# ---------------------------------------------------------------------------

def _white_mask(patch: np.ndarray) -> np.ndarray:
    """BGR patch -> float32 binary mask (1.0 where all channels > _WHITE_THRESH)."""
    above = (patch[:, :, 0] > _WHITE_THRESH) & \
            (patch[:, :, 1] > _WHITE_THRESH) & \
            (patch[:, :, 2] > _WHITE_THRESH)
    return above.astype(np.float32)


def _corner_patch(card_img: np.ndarray, corner: str = 'TL') -> np.ndarray:
    """Extract the raw BGR patch for TL or BR corner (before binarization)."""
    H, W = card_img.shape[:2]
    if corner == 'TL':
        patch = card_img[_BORDER:_CORNER_H_OUTER, _BORDER:_CORNER_W_OUTER]
    else:
        patch = card_img[H - _CORNER_H_OUTER:H - _BORDER,
                         W - _CORNER_W_OUTER:W - _BORDER]
        patch = cv2.rotate(patch, cv2.ROTATE_180)
    return patch.copy()


def _extract_corner(card_img: np.ndarray, corner: str = 'TL') -> np.ndarray:
    """Return a (_CORNER_H, _CORNER_W) float32 binary white-pixel mask."""
    return _white_mask(_corner_patch(card_img, corner))


# ---------------------------------------------------------------------------
# Template persistence
# ---------------------------------------------------------------------------

def save_templates(templates: dict[str, np.ndarray], directory: str) -> None:
    """Save each template as <directory>/<value>.npy."""
    os.makedirs(directory, exist_ok=True)
    for value, tmpl in templates.items():
        np.save(os.path.join(directory, f'{value}.npy'), tmpl)


def load_templates(directory: str) -> dict[str, np.ndarray]:
    """Load all .npy files from directory. Filename stem becomes the value key."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f'Templates directory not found: {directory}')
    templates = {}
    for fname in os.listdir(directory):
        if fname.endswith('.npy'):
            value = fname[:-4]
            templates[value] = np.load(os.path.join(directory, fname))
    return templates


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _match_score(mask: np.ndarray, template: np.ndarray) -> float:
    """Normalized cross-correlation between two same-size float32 masks."""
    result = cv2.matchTemplate(mask, template, cv2.TM_CCOEFF_NORMED)
    return float(result[0, 0])


def classify_card(
    card_img: np.ndarray,
    color: str,
    templates: dict[str, np.ndarray],
) -> str:
    """
    Classify the value of a detected UNO card and return the full label.

    Args:
        card_img:  BGR image from detect_cards (CARD_HEIGHT x CARD_WIDTH).
        color:     color string from detect_cards ('red','yellow','green','blue','dark').
        templates: dict from load_templates().

    Returns:
        Full label string: 'r_5', 'b_skip', 'y_draw_2', 'wild', 'draw_4', etc.
    """
    mask_tl = _extract_corner(card_img, 'TL')

    if color == 'dark':
        dark_tmpls = {v: t for v, t in templates.items() if v in _DARK_VALUES}
        if not dark_tmpls:
            # Heuristic: "+4" text is denser than the wild circle
            return 'draw_4' if mask_tl.mean() > 0.10 else 'wild'
        return max(dark_tmpls, key=lambda v: _match_score(mask_tl, dark_tmpls[v]))

    mask_br = _extract_corner(card_img, 'BR')
    colored_tmpls = {v: t for v, t in templates.items() if v not in _DARK_VALUES}
    if not colored_tmpls:
        return _COLOR_PREFIX.get(color, '?')

    best_value = max(
        colored_tmpls,
        key=lambda v: max(
            _match_score(mask_tl, colored_tmpls[v]),
            _match_score(mask_br, colored_tmpls[v]),
        ),
    )
    return f'{_COLOR_PREFIX[color]}_{best_value}'
