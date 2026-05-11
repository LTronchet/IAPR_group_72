"""
Usage:
    python src/print_templates.py [TEMPLATES_DIR]

Default TEMPLATES_DIR: templates/
Displays all corner template masks sorted by value label.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

TEMPLATES_DIR = sys.argv[1] if len(sys.argv) > 1 else "templates"

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    files = sorted(f for f in os.listdir(TEMPLATES_DIR) if f.endswith(".npy"))
    if not files:
        print(f"No templates found in {TEMPLATES_DIR!r}")
        sys.exit(1)

    ORDER = ["0","1","2","3","4","5","6","7","8","9","skip","reverse","draw_2","draw_4","wild"]
    def sort_key(fname):
        v = fname[:-4]
        return ORDER.index(v) if v in ORDER else len(ORDER)

    files = sorted(files, key=sort_key)
    n = len(files)
    ncols = min(n, 5)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, fname in zip(axes, files):
        value = fname[:-4]
        tmpl = np.load(os.path.join(TEMPLATES_DIR, fname))
        ax.imshow(tmpl, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(value, fontsize=13, fontweight="bold")
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Corner templates ({TEMPLATES_DIR}) — white=symbol pixel", fontsize=14)
    plt.tight_layout()
    plt.show()
