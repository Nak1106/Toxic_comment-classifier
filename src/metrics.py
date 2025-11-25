from typing import Dict, Any

import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    f1_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    label_names=None,
) -> Dict[str, Any]:
    """
    Compute multi label metrics from true labels and predicted probabilities.

    y_true and y_prob are arrays of shape (N, L).
    """
    if label_names is None:
        label_names = [f"label_{i}" for i in range(y_true.shape[1])]

    y_pred = (y_prob >= threshold).astype(int)

    per_label = {}
    for i, name in enumerate(label_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_t, y_p, average="binary", zero_division=0
        )
        try:
            roc = roc_auc_score(y_t, y_prob[:, i])
        except ValueError:
            roc = float("nan")
        try:
            pr_auc = average_precision_score(y_t, y_prob[:, i])
        except ValueError:
            pr_auc = float("nan")

        per_label[name] = {
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "roc_auc": float(roc),
            "pr_auc": float(pr_auc),
        }

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    metrics = {
        "per_label": per_label,
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
    }
    return metrics

