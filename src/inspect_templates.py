"""
Affiche tous les patches bruts par valeur pour identifier les mauvais échantillons.

Usage:
    python src/inspect_templates.py [RAW_DIR]

Default RAW_DIR: templates/raw/

Workflow de curation:
  1. python inspect_templates.py          # identifier les mauvais patches
  2. rm templates/raw/<value>/<fichier>   # supprimer les mauvais
  3. python rebuild_templates.py          # reconstruire les templates
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

RAW_DIR = sys.argv[1] if len(sys.argv) > 1 else "templates/raw"
ORDER = ["0","1","2","3","4","5","6","7","8","9","skip","reverse","draw_2","draw_4","wild"]


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    if not os.path.isdir(RAW_DIR):
        print(f"Répertoire introuvable: {RAW_DIR!r}")
        print("Lance d'abord: python rebuild_templates.py --save-raw")
        sys.exit(1)

    values = [v for v in ORDER if os.path.isdir(os.path.join(RAW_DIR, v))]
    values += sorted(v for v in os.listdir(RAW_DIR)
                     if os.path.isdir(os.path.join(RAW_DIR, v)) and v not in ORDER)

    for value in values:
        value_dir = os.path.join(RAW_DIR, value)
        files = sorted(f for f in os.listdir(value_dir) if f.endswith('.png'))
        if not files:
            continue

        n = len(files)
        ncols = min(n, 8)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 3 * nrows))
        axes = np.array(axes).reshape(-1)

        for ax, fname in zip(axes, files):
            patch = cv2.imread(os.path.join(value_dir, fname))
            ax.imshow(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
            # Nom court: image_id + région (sans l'extension)
            short = fname[:-4].replace('_TL', ' TL').replace('_BR', ' BR')
            ax.set_title(short, fontsize=7)
            ax.axis("off")

        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle(f"[{value}] — {n} patches  ({value_dir})", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()
