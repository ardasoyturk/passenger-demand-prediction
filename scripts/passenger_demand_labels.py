"""Reusable passenger-demand labels and classification metrics.

The labels in this module describe passenger demand only. They do not imply
profitability, because fare and operating-cost data are not part of the model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


MINIMUM_SUCCESS_THRESHOLD = 10
LIKELY_VIABLE_THRESHOLD = 20
STRONG_DEMAND_THRESHOLD = 30
CAPACITY_PRESSURE_THRESHOLD = 43

BUSINESS_THRESHOLDS = (
    MINIMUM_SUCCESS_THRESHOLD,
    LIKELY_VIABLE_THRESHOLD,
    STRONG_DEMAND_THRESHOLD,
    CAPACITY_PRESSURE_THRESHOLD,
)

DEMAND_LABELS = (
    "CLEAR_FAILURE",
    "WEAK_DEMAND",
    "MODERATE_DEMAND",
    "STRONG_DEMAND",
    "CAPACITY_PRESSURE",
)

INDICATOR_THRESHOLDS = {
    "is_minimum_success": MINIMUM_SUCCESS_THRESHOLD,
    "is_likely_viable": LIKELY_VIABLE_THRESHOLD,
    "is_strong_demand": STRONG_DEMAND_THRESHOLD,
    "is_capacity_pressure": CAPACITY_PRESSURE_THRESHOLD,
}


def _validated_numeric_array(values: Any, name: str) -> np.ndarray:
    """Return a finite one-dimensional float array or raise a clear error."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; received {array.ndim} dimensions.")
    if not np.isfinite(array).all():
        bad_count = int((~np.isfinite(array)).sum())
        raise ValueError(f"{name} contains {bad_count:,} missing or non-finite values.")
    return array


def _labels_for_array(values: np.ndarray) -> np.ndarray:
    """Assign ordered demand labels to an already validated numeric array."""

    return np.select(
        [
            values < MINIMUM_SUCCESS_THRESHOLD,
            values < LIKELY_VIABLE_THRESHOLD,
            values < STRONG_DEMAND_THRESHOLD,
            values < CAPACITY_PRESSURE_THRESHOLD,
        ],
        DEMAND_LABELS[:-1],
        default=DEMAND_LABELS[-1],
    )


def assign_demand_label(value: float | int) -> str:
    """Return the business demand label for one passenger-count value."""

    return str(assign_demand_labels([value])[0])


def assign_demand_labels(values: Any) -> np.ndarray:
    """Vectorize demand-label assignment for finite passenger-count values."""

    numeric = _validated_numeric_array(values, "values")
    return _labels_for_array(numeric)


def rounded_nonnegative_predictions(values: Any) -> np.ndarray:
    """Round predictions to nearest-even integers and clamp only below zero.

    NumPy ``rint`` is vectorized and has the same tie-to-even behavior as
    Python's ``round`` for finite values. No upper clamp is applied.
    """

    numeric = _validated_numeric_array(values, "prediction")
    return np.maximum(0.0, np.rint(numeric)).astype(np.int64)


def add_passenger_label_columns(
    dataframe: pd.DataFrame,
    prediction_columns: Mapping[str, str] | Sequence[str],
    *,
    actual_column: str = "actual",
    copy: bool = True,
) -> pd.DataFrame:
    """Add actual and prediction demand-label/indicator columns.

    ``prediction_columns`` may map output prefixes to source columns, for
    example ``{"v4_2_hybrid": "hybrid_prediction"}``. A sequence uses each
    source column name as its output prefix.
    """

    if isinstance(prediction_columns, Mapping):
        prefix_to_column = dict(prediction_columns)
    else:
        prefix_to_column = {column: column for column in prediction_columns}

    required = [actual_column, *prefix_to_column.values()]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required passenger-count columns: {missing}")
    if len(set(prefix_to_column)) != len(prefix_to_column):
        raise ValueError("Prediction output prefixes must be unique.")

    result = dataframe.copy() if copy else dataframe
    actual = _validated_numeric_array(result[actual_column], actual_column)
    result["actual_demand_label"] = _labels_for_array(actual)
    for suffix, threshold in INDICATOR_THRESHOLDS.items():
        result[f"actual_{suffix}"] = (actual >= threshold).astype(np.uint8)

    for prefix, column in prefix_to_column.items():
        if not prefix or not isinstance(prefix, str):
            raise ValueError("Every prediction output prefix must be a non-empty string.")
        rounded = rounded_nonnegative_predictions(result[column])
        result[f"{prefix}_rounded_prediction"] = rounded
        result[f"{prefix}_demand_label"] = _labels_for_array(rounded)
        for suffix, threshold in INDICATOR_THRESHOLDS.items():
            result[f"{prefix}_{suffix}"] = (rounded >= threshold).astype(np.uint8)
    return result


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    """Divide safely, returning zero when the denominator is zero."""

    return 0.0 if denominator == 0 else float(numerator / denominator)


def calculate_binary_classification_metrics(
    actual_values: Any,
    predicted_values: Any,
    threshold: int,
    *,
    predictions_are_rounded: bool = False,
) -> dict[str, int | float]:
    """Calculate binary classification metrics at one passenger threshold.

    Raw predictions are rounded and lower-clamped for classification unless
    ``predictions_are_rounded`` is true. Actual passenger counts are not
    rounded. All divisions are safe and return ``0.0`` for an empty divisor.
    """

    if threshold not in BUSINESS_THRESHOLDS:
        raise ValueError(
            f"threshold must be one of {BUSINESS_THRESHOLDS}; received {threshold}."
        )
    actual = _validated_numeric_array(actual_values, "actual_values")
    prediction = _validated_numeric_array(predicted_values, "predicted_values")
    if len(actual) != len(prediction):
        raise ValueError("actual_values and predicted_values must have equal lengths.")
    rounded = prediction.astype(np.int64) if predictions_are_rounded else rounded_nonnegative_predictions(prediction)
    actual_positive = actual >= threshold
    predicted_positive = rounded >= threshold
    true_positive = int(np.count_nonzero(actual_positive & predicted_positive))
    true_negative = int(np.count_nonzero(~actual_positive & ~predicted_positive))
    false_positive = int(np.count_nonzero(~actual_positive & predicted_positive))
    false_negative = int(np.count_nonzero(actual_positive & ~predicted_positive))
    row_count = len(actual)
    return {
        "threshold": threshold,
        "row_count": row_count,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "accuracy": _safe_divide(true_positive + true_negative, row_count),
        "precision": _safe_divide(true_positive, true_positive + false_positive),
        "recall": _safe_divide(true_positive, true_positive + false_negative),
        "specificity": _safe_divide(true_negative, true_negative + false_positive),
        "f1_score": _safe_divide(2 * true_positive, 2 * true_positive + false_positive + false_negative),
        "false_positive_rate": _safe_divide(false_positive, false_positive + true_negative),
        "false_negative_rate": _safe_divide(false_negative, false_negative + true_positive),
        "predicted_positive_rate": _safe_divide(true_positive + false_positive, row_count),
        "actual_positive_rate": _safe_divide(true_positive + false_negative, row_count),
    }


def calculate_multiclass_confusion_matrix(
    actual_values: Any,
    predicted_values: Any,
    *,
    predictions_are_rounded: bool = False,
) -> pd.DataFrame:
    """Return the fixed-order five-class confusion matrix with totals."""

    actual = _validated_numeric_array(actual_values, "actual_values")
    prediction = _validated_numeric_array(predicted_values, "predicted_values")
    if len(actual) != len(prediction):
        raise ValueError("actual_values and predicted_values must have equal lengths.")
    rounded = prediction.astype(np.int64) if predictions_are_rounded else rounded_nonnegative_predictions(prediction)
    actual_labels = pd.Categorical(_labels_for_array(actual), categories=DEMAND_LABELS, ordered=True)
    predicted_labels = pd.Categorical(_labels_for_array(rounded), categories=DEMAND_LABELS, ordered=True)
    matrix = pd.crosstab(actual_labels, predicted_labels, dropna=False).reindex(
        index=DEMAND_LABELS, columns=DEMAND_LABELS, fill_value=0
    )
    matrix.index.name = "actual_label"
    matrix.columns.name = "predicted_label"
    matrix["ROW_TOTAL"] = matrix.sum(axis=1)
    column_totals = matrix.sum(axis=0)
    matrix.loc["COLUMN_TOTAL"] = column_totals
    return matrix.astype(np.int64)
