"""Regression and classification metrics for model evaluation."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def regression_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, float]:
    """Return MAE, RMSE, and bias."""
    error = prediction - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------

def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)


def classification_metrics(
    actual_label: Iterable[int],
    predicted_label: Iterable[int],
) -> dict[str, Any]:
    """Return a full set of binary classification metrics."""
    actual = np.asarray(actual_label, dtype=np.int8)
    predicted = np.asarray(predicted_label, dtype=np.int8)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError(
            "Actual and predicted labels must be aligned one-dimensional arrays"
        )
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    return {
        "row_count": len(actual),
        "actual_positive_count": int(actual.sum()),
        "actual_negative_count": int(len(actual) - actual.sum()),
        "actual_positive_rate": float(actual.mean()),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": _safe_divide(tp + tn, len(actual)),
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": _safe_divide(fp, fp + tn),
        "false_negative_rate": _safe_divide(fn, fn + tp),
        "predicted_positive_rate": float(predicted.mean()),
    }


def binary_classification_metrics(
    actual_values: Any,
    predicted_values: Any,
    threshold: int,
    *,
    predictions_are_rounded: bool = False,
) -> dict[str, int | float]:
    """Calculate binary classification metrics at one passenger threshold.

    Raw predictions are rounded and lower-clamped for classification unless
    ``predictions_are_rounded`` is true.
    """
    actual = np.asarray(actual_values, dtype=np.float64)
    prediction = np.asarray(predicted_values, dtype=np.float64)
    if len(actual) != len(prediction):
        raise ValueError(
            "actual_values and predicted_values must have equal lengths."
        )
    if predictions_are_rounded:
        rounded = prediction.astype(np.int64)
    else:
        rounded = np.maximum(0.0, np.rint(prediction)).astype(np.int64)

    actual_positive = actual >= threshold
    predicted_positive = rounded >= threshold
    tp = int(np.count_nonzero(actual_positive & predicted_positive))
    tn = int(np.count_nonzero(~actual_positive & ~predicted_positive))
    fp = int(np.count_nonzero(~actual_positive & predicted_positive))
    fn = int(np.count_nonzero(actual_positive & ~predicted_positive))
    row_count = len(actual)
    return {
        "threshold": threshold,
        "row_count": row_count,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "accuracy": _safe_divide(tp + tn, row_count),
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, tp + fn),
        "specificity": _safe_divide(tn, tn + fp),
        "f1_score": _safe_divide(
            2 * tp, 2 * tp + fp + fn
        ),
        "false_positive_rate": _safe_divide(fp, fp + tn),
        "false_negative_rate": _safe_divide(fn, fn + tp),
        "predicted_positive_rate": _safe_divide(
            tp + fp, row_count
        ),
        "actual_positive_rate": _safe_divide(
            tp + fn, row_count
        ),
    }


def probability_metrics(
    actual_label: Iterable[int],
    probability: Iterable[float],
) -> dict[str, float]:
    """Return ROC-AUC, PR-AUC, log-loss, and Brier score."""
    from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

    actual = np.asarray(actual_label, dtype=np.int8)
    probabilities = np.asarray(probability, dtype=np.float64)
    if actual.shape != probabilities.shape:
        raise ValueError("Actual labels and probabilities must align")
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities < 0) | (probabilities > 1)
    ):
        raise ValueError("Probabilities must be finite and inside [0, 1]")
    if np.unique(actual).size != 2:
        raise ValueError("ROC-AUC and PR-AUC require both classes")
    clipped = np.clip(probabilities, 1e-15, 1 - 1e-15)
    return {
        "roc_auc": float(roc_auc_score(actual, probabilities)),
        "pr_auc": float(average_precision_score(actual, probabilities)),
        "log_loss": float(log_loss(actual, clipped, labels=[0, 1])),
        "brier_score": float(np.mean(np.square(probabilities - actual))),
        "average_predicted_probability": float(probabilities.mean()),
    }


def target_group_table(
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    bins: list[int] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Return per-group MAE/RMSE/bias for one or more prediction columns."""
    if bins is None:
        bins = [0, 10, 20, 30, 40, 60, 100, 300]
    if labels is None:
        labels = ["1-10", "11-20", "21-30", "31-40", "41-60", "61-100", "101-300"]

    data = pd.DataFrame({"actual": actual})
    for name, values in predictions.items():
        data[name] = values
    data["target_group"] = pd.cut(
        data["actual"], bins=bins, labels=labels, include_lowest=True
    )

    records = []
    for group in labels:
        subset = data[data["target_group"] == group]
        if subset.empty:
            continue
        record: dict[str, Any] = {
            "target_group": group,
            "row_count": len(subset),
            "average_actual": float(subset["actual"].mean()),
        }
        for name in predictions:
            metrics = regression_metrics(
                subset["actual"].to_numpy(),
                subset[name].to_numpy(),
            )
            record[f"{name}_mae"] = metrics["mae"]
            record[f"{name}_rmse"] = metrics["rmse"]
            record[f"{name}_bias"] = metrics["bias"]
            record[f"average_{name}"] = float(subset[name].mean())
        records.append(record)
    return pd.DataFrame(records)
