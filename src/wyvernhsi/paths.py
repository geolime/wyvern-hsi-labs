from pathlib import Path
import sys

# This file lives in: src/wyvernhsi/paths.py
# Repo layout:
#   wyvern-hsi-labs/
#     src/wyvernhsi/paths.py
#     projects/<project_name>/scripts/*.py

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_active_project_dir() -> Path:
    """
    Infer active project from script execution path.
    Works when running:
      python projects/<project>/scripts/script.py
    """

    # Absolute path of the running script
    script_path = Path(sys.argv[0]).resolve()

    # Walk up until we find ".../projects/<name>"
    for parent in script_path.parents:
        if parent.name == "projects":
            # Next element up is repo root — not what we want
            continue

        if parent.parent.name == "projects":
            return parent

    raise RuntimeError(
        "Could not determine active project directory. "
        "Run scripts from: projects/<project_name>/scripts/"
    )


PROJECT_DIR = find_active_project_dir()

DATA_DIR = PROJECT_DIR / "data"
OUTPUTS_DIR = PROJECT_DIR / "outputs"


# ------------------------
# Automatic dataset selection
# ------------------------

def find_wyvern_scene(data_dir: Path) -> Path:
    """
    Returns the first GeoTIFF that is NOT a data_mask.
    Enforces exactly one active scene per project folder.
    """

    tiffs = sorted(data_dir.glob("*.tif*"))

    scene_files = [
        p for p in tiffs
        if not p.name.endswith("_data_mask.tiff")
    ]

    if len(scene_files) == 0:
        raise FileNotFoundError(
            f"No Wyvern scene found in {data_dir} "
            "(expected *.tif not ending with _data_mask.tiff)"
        )

    if len(scene_files) > 1:
        raise RuntimeError(
            f"Multiple Wyvern scenes found in {data_dir}. "
            "Keep exactly one active scene per project."
        )

    return scene_files[0]


ACTIVE_WYVERN_FILE = find_wyvern_scene(DATA_DIR)
ACTIVE_WYVERN_MASK = DATA_DIR / (ACTIVE_WYVERN_FILE.stem + "_data_mask.tiff")


# Optional sanity check
if not ACTIVE_WYVERN_MASK.exists():
    raise FileNotFoundError(
        f"Expected mask not found:\n{ACTIVE_WYVERN_MASK}"
    )
