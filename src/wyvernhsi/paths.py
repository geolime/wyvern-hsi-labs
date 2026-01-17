#from pathlib import Path

#DATA_DIR = Path("data")
#OUTPUTS_DIR = Path("outputs")

# ---- Available datasets ----

#WYVERN_BOLIVIA_TREEFARM = DATA_DIR / "wyvern_dragonette-004_20250927T145218_edb4d3af.tiff"
#WYVERN_OTHER_SCENE = DATA_DIR / "wyvern_dragonette-001_20240930T070744_08fd7f5a.tiff"

# ---- Active dataset (change only this line) ----

#ACTIVE_WYVERN_FILE = WYVERN_BOLIVIA_TREEFARM
#ACTIVE_WYVERN_MASK = ACTIVE_WYVERN_FILE.with_name(ACTIVE_WYVERN_FILE.stem + "_data_mask.tiff")



from pathlib import Path

# This file lives in: src/wyvernhsi/paths.py
# We want: projects/forest_landcover/

REPO_ROOT = Path(__file__).resolve().parents[2]

PROJECT_DIR = REPO_ROOT / "projects" / "forest_landcover"

DATA_DIR = PROJECT_DIR / "data"
OUTPUTS_DIR = PROJECT_DIR / "outputs"

# ---- Active dataset ----

ACTIVE_WYVERN_FILE = DATA_DIR / "wyvern_dragonette-004_20250927T145218_edb4d3af.tiff"
ACTIVE_WYVERN_MASK = DATA_DIR / (ACTIVE_WYVERN_FILE.stem + "_data_mask.tiff")




