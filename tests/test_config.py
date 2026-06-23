import textwrap

import pytest

from wyvernhsi.config import load_config

FOREST_YAML = textwrap.dedent("""
    project:
      name: forest_landcover
      random_seed: 42
    pipeline:
      stages:
        - 01_convert_to_toa_reflectance.py
        - 15_unsupervised_kmeans.py
    clustering:
      k: 5
      pca_components: 8
      n_samples: 50000
      tile_size: 512
      use_subset: true
      subset_pad: 1200
      subset_points:
        - [896, 2163]
""")


def test_load_config_happy(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(FOREST_YAML)
    c = load_config(p)
    assert c.name == "forest_landcover"
    assert c.random_seed == 42
    assert c.clustering.k == 5
    assert len(c.stages) == 2
    assert c.masking is None  # absent section -> None


def test_missing_key_fails_fast(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(FOREST_YAML.replace("  random_seed: 42\n", ""))
    with pytest.raises(KeyError):
        load_config(p)