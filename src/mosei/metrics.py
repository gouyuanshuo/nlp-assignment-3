"""Binary metrics with a fixed class order: negative (0), positive (1)."""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(records: list[dict]) -> dict:
    if not records:
        raise ValueError("Cannot score an empty prediction set")
    labels = np.array([r["label"] for r in records], dtype=int)
    predictions = np.array([r["prediction"] for r in records], dtype=int)
    if not np.isin(labels, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError("Binary labels and predictions must be 0 or 1")
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],
        zero_division=0,
    )
    result = {
        "count": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(["negative", "positive"])
        },
    }
    if all("loss" in r for r in records):
        result["loss"] = float(np.mean([r["loss"] for r in records]))
    return result
