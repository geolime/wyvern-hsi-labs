import numpy as np

from wyvernhsi import validation

CLASSES = ["trees", "grass", "water"]
MAP = {"trees": [3, 4], "grass": [10], "water": [33]}


def test_crosswalk_and_ignore():
    ref = np.array([[3, 4, 10], [33, 99, 0]])
    out = validation.crosswalk(ref, MAP, CLASSES)
    assert out[0, 0] == 0 and out[0, 2] == 1 and out[1, 0] == 2
    assert out[1, 1] == -1 and out[1, 2] == -1   # unmapped codes


def test_crosstab_and_majority():
    cluster = np.array([[0, 0, 1], [1, 1, 0]])
    ref_class = np.array([[0, 0, 1], [1, 1, -1]])
    tab = validation.crosstab(cluster, ref_class, 2, CLASSES)
    assert tab.loc["cluster_0", "trees"] == 2   # cluster 0 mostly trees
    assert validation.majority_labels(tab) == {0: "trees", 1: "grass"}


def test_score_support_and_none_for_absent():
    pred = np.array([0, 0, 1])
    ref = np.array([0, 1, 1])
    _, m = validation.score(pred, ref, CLASSES)        # CLASSES = trees, grass, water
    assert m["per_class"]["trees"]["reference_support"] == 1
    assert m["per_class"]["grass"]["reference_support"] == 2
    assert m["per_class"]["water"]["reference_support"] == 0   # ref has no water
    assert m["per_class"]["water"]["recall"] is None           # not a fake 0.0


def test_score_perfect_and_offdiagonal():
    pred = np.array([0, 0, 1, 2])
    ref = np.array([0, 1, 1, 2])
    cm, m = validation.score(pred, ref, CLASSES)
    assert m["n_pixels"] == 4
    assert np.isclose(m["overall_agreement"], 0.75)   # 3 of 4 on diagonal
    assert m["per_class"]["trees"]["precision"] == 0.5  # pred trees: 1 of 2 correct