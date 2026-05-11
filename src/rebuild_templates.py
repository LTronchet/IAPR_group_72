"""
Reconstruit les templates depuis les patches bruts (après curation manuelle).

Usage:
    python src/rebuild_templates.py              # rebuild depuis templates/raw/
    python src/rebuild_templates.py --save-raw   # rebuild ET re-génère les patches bruts

Pour la première fois (générer les patches):
    python src/rebuild_templates.py --save-raw
"""

import sys
import os

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

from src.template_builder import build_templates, build_templates_from_raw
from src.card_classification import save_templates

DATA_DIR      = "iapr-26-uno-vision-challenge/train_images"
CSV_PATH      = "iapr-26-uno-vision-challenge/train.csv"
TEMPLATES_DIR = "templates"
RAW_DIR       = "templates/raw"
SAT_LO        = 80
VAL_LO        = 120

if __name__ == "__main__":
    save_raw = "--save-raw" in sys.argv

    if save_raw or not os.path.isdir(RAW_DIR) or not os.listdir(RAW_DIR):
        print("Construction des templates + sauvegarde des patches bruts...")
        templates = build_templates(
            CSV_PATH, DATA_DIR,
            sat_lo=SAT_LO, val_lo=VAL_LO,
            raw_dir=RAW_DIR,
        )
    else:
        print(f"Reconstruction depuis les patches curés dans {RAW_DIR!r}...")
        templates = build_templates_from_raw(RAW_DIR)

    save_templates(templates, TEMPLATES_DIR)
    print(f"Templates sauvegardés dans {TEMPLATES_DIR!r}: {sorted(templates.keys())}")
