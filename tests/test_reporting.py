import numpy as np

from wyvernhsi import reporting
from wyvernhsi.config import Config


def _make_project(tmp_path, write_raster):
    (tmp_path / "data").mkdir()
    write_raster(tmp_path / "data" / "scene.tif", np.zeros((2, 2), dtype=np.float32))
    out = tmp_path / "outputs" / "masks"
    out.mkdir(parents=True)
    (tmp_path / "outputs" / "clusters.png").write_bytes(b"x")
    (tmp_path / "outputs" / "stats.csv").write_text("a,b\n1,2\n")
    return Config(name="t", project_dir=tmp_path, random_seed=42,
                  stages=["x"], clustering=None, features=None, masking=None)


def test_build_manifest(tmp_path, write_raster):
    cfg = _make_project(tmp_path, write_raster)
    m = reporting.build_manifest(cfg, "run1")
    assert m["project"] == "t"
    assert m["scene"]["reflectance"] == "scene_toa_reflectance.tif"
    assert "clusters.png" in m["outputs"]["figures"]
    assert "stats.csv" in m["outputs"]["tables"]
    assert m["git_sha"]  # "unknown" outside a repo, but always a string


def test_write_report(tmp_path):
    manifest = {
        "project": "t", "run_id": "r", "generated_utc": "now", "git_sha": "abc",
        "scene": {"radiance": "scene.tif"},
        "outputs": {"figures": ["masks/water_mask.png"], "tables": ["stats.csv"]},
    }
    rp = reporting.write_report(manifest, tmp_path / "report.md")
    text = rp.read_text()
    assert "![masks/water_mask.png](masks/water_mask.png)" in text
    assert "stats.csv" in text