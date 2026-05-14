import numpy as np


def classify_background(img: np.ndarray, patch_size: int = 50, threshold: float = 200.0) -> str:
    """
    Classify the background of an UNO game image as 'white' or 'noisy'.

    Samples 50x50 patches from the four corners of the image and computes
    their mean brightness. A white background has mean close to 255; a noisy
    (patterned) background is significantly darker.

    Args:
        img:        RGB image as a numpy array (H, W, 3).
        patch_size: Side length of the corner patch in pixels.
        threshold:  Mean brightness threshold separating white from noisy.

    Returns:
        'white' or 'noisy'.
    """
    h, w = img.shape[:2]
    ps = patch_size
    corners = [
        img[:ps, :ps],
        img[:ps, w - ps:],
        img[h - ps:, :ps],
        img[h - ps:, w - ps:],
    ]
    mean_brightness = np.mean([patch.mean() for patch in corners])
    return 'white' if mean_brightness >= threshold else 'noisy'
