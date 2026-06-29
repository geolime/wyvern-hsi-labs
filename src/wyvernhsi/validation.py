"""Land-cover validation: crosswalk reference codes to a target scheme, cross-tabulate
unsupervised clusters against it, and score agreement (confusion matrix + metrics).

Reported numbers are AGREEMENT with an independent reference map (itself a model with its
own error), not absolute accuracy."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def crosswalk(ref: np.ndarray, mapping: dict, classes: list) -> np.ndarray:
    """Map reference codes to target class indices (order = `classes`); -1 for unmapped/ignored."""
    out = np.full(ref.shape, -1, dtype=np.int16)
    for name, codes in mapping.items():
        ci = classes.index(name)
        out[np.isin(ref, codes)] = ci
    return out


def crosstab(cluster: np.ndarray, ref_class: np.ndarray, k: int, classes: list) -> pd.DataFrame:
    """Rows = clusters 0..k-1, cols = target classes; counts of co-occurring valid pixels."""
    valid = (cluster >= 0) & (ref_class >= 0)
    cl, rf = cluster[valid], ref_class[valid]
    tab = np.zeros((k, len(classes)), dtype=np.int64)
    for c in range(k):
        m = cl == c
        if np.any(m):
            idx, cnt = np.unique(rf[m], return_counts=True)
            tab[c, idx] = cnt
    return pd.DataFrame(tab, index=[f"cluster_{c}" for c in range(k)], columns=classes)


def majority_labels(tab: pd.DataFrame) -> dict:
    """Each cluster -> the target class it most overlaps (its evidence-based label)."""
    return {int(r.split("_")[1]): tab.columns[int(np.argmax(tab.loc[r].values))]
            for r in tab.index if tab.loc[r].sum() > 0}


def score(pred_class: np.ndarray, ref_class: np.ndarray, classes: list) -> tuple[pd.DataFrame, dict]:
    """Confusion matrix (rows = predicted, cols = reference) + agreement / per-class / kappa.

    Per-class precision/recall are None where there is no support (so absent classes don't
    report fake 0.0/1.0). Overall agreement is reported but is inflated under class imbalance;
    prefer kappa as the headline."""
    valid = (pred_class >= 0) & (ref_class >= 0)
    p, r = pred_class[valid], ref_class[valid]
    labels = list(range(len(classes)))
    cm = confusion_matrix(p, r, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"pred_{c}" for c in classes], columns=[f"ref_{c}" for c in classes])

    total = int(cm.sum())
    metrics = {
        "overall_agreement": float(np.trace(cm) / total) if total else 0.0,
        "overall_agreement_note": "inflated under class imbalance; prefer kappa",
        "kappa": float(cohen_kappa_score(p, r)) if total else 0.0,
        "n_pixels": total,
        "per_class": {},
    }
    for i, name in enumerate(classes):
        tp = int(cm[i, i])
        pred_n, ref_n = int(cm[i, :].sum()), int(cm[:, i].sum())
        metrics["per_class"][name] = {
            "precision": (tp / pred_n) if pred_n else None,
            "recall": (tp / ref_n) if ref_n else None,
            "reference_support": ref_n,
        }
    return cm_df, metrics