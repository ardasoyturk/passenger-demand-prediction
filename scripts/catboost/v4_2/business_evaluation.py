"""Evaluate frozen CatBoost v4.2 predictions with demand business labels.

This script reads existing row-level prediction CSVs only. It never loads or
runs a CatBoost model, and it refuses to overwrite any requested output.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(PROJECT_DIR))
from scripts.passenger_demand_labels import (
    BUSINESS_THRESHOLDS,
    DEMAND_LABELS,
    add_passenger_label_columns,
    assign_demand_labels,
    calculate_binary_classification_metrics,
    calculate_multiclass_confusion_matrix,
    rounded_nonnegative_predictions,
)
RESULTS_DIR = PROJECT_DIR / "results"
CHUNK_SIZE = 200_000

MODEL_COLUMNS = {
    "CatBoost v3": "v3_prediction",
    "CatBoost v4.1": "v4_1_prediction",
    "Weekday baseline": "baseline_prediction",
    "CatBoost v4.2 hybrid": "hybrid_prediction",
}
LABEL_PREFIXES = {
    "v3": "v3_prediction",
    "v4_1": "v4_1_prediction",
    "weekday_baseline": "baseline_prediction",
    "v4_2_hybrid": "hybrid_prediction",
}
HIGH_DEMAND_THRESHOLDS = (10, 20, 30, 43, 60, 100)
OUTPUT_KINDS = (
    "business_binary_metrics",
    "business_label_metrics",
    "business_confusion_matrices",
    "business_regression_by_label",
    "business_decision_distance",
    "business_high_demand_metrics",
)


@dataclass(frozen=True)
class PeriodConfig:
    """Input and expected row contract for one chronological period."""

    key: str
    label: str
    expected_rows: int

    @property
    def input_path(self) -> Path:
        return RESULTS_DIR / f"catboost_v4_2_{self.key}_predictions.csv"

    @property
    def labeled_path(self) -> Path:
        return RESULTS_DIR / f"catboost_v4_2_{self.key}_predictions_labeled.csv"

    def metric_path(self, kind: str) -> Path:
        return RESULTS_DIR / f"catboost_v4_2_{self.key}_{kind}.csv"


PERIODS = (
    PeriodConfig("validation_2025_h1", "2025 H1 validation", 1_420_182),
    PeriodConfig("test_2025_h2", "2025 H2 test", 1_517_010),
    PeriodConfig("final_2026", "2026 final", 780_865),
)
SUMMARY_PATH = RESULTS_DIR / "catboost_v4_2_business_summary.csv"


def _validate_inputs() -> None:
    """Fail once with every missing input path listed."""

    missing = [config.input_path for config in PERIODS if not config.input_path.exists()]
    if missing:
        paths = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Required existing row-level prediction files are missing:\n" + paths
        )


def _all_output_paths() -> list[Path]:
    paths = [SUMMARY_PATH]
    for config in PERIODS:
        paths.append(config.labeled_path)
        paths.extend(config.metric_path(kind) for kind in OUTPUT_KINDS)
    return paths


def _validate_no_output_collisions() -> None:
    """Protect all existing prediction and metric artifacts from overwrite."""

    collisions = [path for path in _all_output_paths() if path.exists()]
    if collisions:
        paths = "\n".join(f"  - {path}" for path in collisions)
        raise FileExistsError("Refusing to overwrite existing output files:\n" + paths)


def _load_numeric_predictions(config: PeriodConfig) -> pd.DataFrame:
    """Load and validate only numeric columns needed for evaluation."""

    required = ["actual", *MODEL_COLUMNS.values()]
    try:
        frame = pd.read_csv(config.input_path, usecols=required)
    except ValueError as error:
        header = pd.read_csv(config.input_path, nrows=0).columns.tolist()
        missing = [column for column in required if column not in header]
        raise ValueError(
            f"{config.input_path} is missing required columns: {missing}. "
            f"Available columns: {header}"
        ) from error
    if len(frame) != config.expected_rows:
        raise RuntimeError(
            f"{config.input_path} has {len(frame):,} rows; "
            f"expected {config.expected_rows:,}."
        )
    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{config.input_path}: {column} contains non-finite values.")
    return frame


def _multiclass_metrics(matrix: pd.DataFrame) -> dict[str, float]:
    """Calculate aggregate metrics from a five-class confusion matrix."""

    core = matrix.loc[list(DEMAND_LABELS), list(DEMAND_LABELS)].to_numpy(dtype=np.int64)
    support = core.sum(axis=1)
    predicted = core.sum(axis=0)
    true_positive = np.diag(core)
    precision = np.divide(true_positive, predicted, out=np.zeros(5), where=predicted != 0)
    recall = np.divide(true_positive, support, out=np.zeros(5), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(5), where=(precision + recall) != 0)
    row_count = int(core.sum())
    return {
        "overall_label_accuracy": float(true_positive.sum() / row_count) if row_count else 0.0,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=support)) if row_count else 0.0,
    }


def _regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, int | float]:
    """Return deterministic regression metrics for one non-empty subset."""

    error = prediction - actual
    absolute_error = np.abs(error)
    return {
        "row_count": len(actual),
        "average_actual": float(actual.mean()) if len(actual) else 0.0,
        "average_prediction": float(prediction.mean()) if len(actual) else 0.0,
        "mae": float(absolute_error.mean()) if len(actual) else 0.0,
        "rmse": float(np.sqrt(np.mean(np.square(error)))) if len(actual) else 0.0,
        "bias": float(error.mean()) if len(actual) else 0.0,
        "median_absolute_error": float(np.median(absolute_error)) if len(actual) else 0.0,
    }


def _evaluate_period(config: PeriodConfig, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build all six required metric tables for a single period."""

    actual = frame["actual"].to_numpy(dtype=np.float64)
    actual_labels = assign_demand_labels(actual)
    binary_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    confusion_rows: list[pd.DataFrame] = []
    regression_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    high_demand_rows: list[dict[str, Any]] = []

    for model, column in MODEL_COLUMNS.items():
        prediction = frame[column].to_numpy(dtype=np.float64)
        rounded = rounded_nonnegative_predictions(prediction)
        for threshold in BUSINESS_THRESHOLDS:
            binary_rows.append({
                "period": config.key,
                "model": model,
                **calculate_binary_classification_metrics(
                    actual, rounded, threshold, predictions_are_rounded=True
                ),
            })
            actual_positive = actual >= threshold
            predicted_positive = rounded >= threshold
            false_positive = ~actual_positive & predicted_positive
            false_negative = actual_positive & ~predicted_positive
            distance_rows.append({
                "period": config.key,
                "threshold": threshold,
                "model": model,
                "row_count": len(actual),
                "false_positive_count": int(false_positive.sum()),
                "average_prediction_for_false_positives": float(prediction[false_positive].mean()) if false_positive.any() else 0.0,
                "average_actual_for_false_positives": float(actual[false_positive].mean()) if false_positive.any() else 0.0,
                "average_distance_above_threshold_for_false_positives": float((rounded[false_positive] - threshold).mean()) if false_positive.any() else 0.0,
                "false_negative_count": int(false_negative.sum()),
                "average_prediction_for_false_negatives": float(prediction[false_negative].mean()) if false_negative.any() else 0.0,
                "average_actual_for_false_negatives": float(actual[false_negative].mean()) if false_negative.any() else 0.0,
                "average_distance_below_threshold_for_false_negatives": float((threshold - rounded[false_negative]).mean()) if false_negative.any() else 0.0,
            })

        matrix = calculate_multiclass_confusion_matrix(
            actual, rounded, predictions_are_rounded=True
        )
        label_rows.append({
            "period": config.key,
            "model": model,
            "row_count": len(actual),
            **_multiclass_metrics(matrix),
        })
        exported_matrix = matrix.reset_index().rename(columns={"index": "actual_label"})
        exported_matrix.insert(0, "model", model)
        exported_matrix.insert(0, "period", config.key)
        confusion_rows.append(exported_matrix)

        for label in DEMAND_LABELS:
            mask = actual_labels == label
            regression_rows.append({
                "period": config.key,
                "actual_demand_label": label,
                "model": model,
                **_regression_metrics(actual[mask], prediction[mask]),
            })
        for threshold in HIGH_DEMAND_THRESHOLDS:
            mask = actual >= threshold
            metrics = _regression_metrics(actual[mask], prediction[mask])
            high_demand_rows.append({
                "period": config.key,
                "actual_threshold": threshold,
                "model": model,
                **{key: metrics[key] for key in ["row_count", "mae", "rmse", "bias", "average_actual", "average_prediction"]},
            })

    return {
        "business_binary_metrics": pd.DataFrame(binary_rows),
        "business_label_metrics": pd.DataFrame(label_rows),
        "business_confusion_matrices": pd.concat(confusion_rows, ignore_index=True),
        "business_regression_by_label": pd.DataFrame(regression_rows),
        "business_decision_distance": pd.DataFrame(distance_rows),
        "business_high_demand_metrics": pd.DataFrame(high_demand_rows),
    }


def _write_labeled_copy(config: PeriodConfig, temporary_path: Path) -> int:
    """Create a labeled copy in vectorized chunks and return its row count."""

    row_count = 0
    first_chunk = True
    try:
        chunks = pd.read_csv(config.input_path, chunksize=CHUNK_SIZE)
        for chunk_number, chunk in enumerate(chunks, start=1):
            labeled = add_passenger_label_columns(
                chunk, LABEL_PREFIXES, actual_column="actual", copy=False
            )
            labeled.to_csv(
                temporary_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )
            first_chunk = False
            row_count += len(labeled)
            print(f"  labeled chunk {chunk_number}: {row_count:,} rows", flush=True)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    if row_count != config.expected_rows:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Labeled output for {config.key} has {row_count:,} rows; "
            f"expected {config.expected_rows:,}."
        )
    return row_count


def _summary_table(period_results: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """Combine classification and high-demand comparisons in one tidy CSV."""

    parts: list[pd.DataFrame] = []
    for results in period_results.values():
        binary = results["business_binary_metrics"].copy()
        binary.insert(1, "evaluation_type", "binary_threshold")
        binary = binary.rename(columns={"threshold": "actual_threshold"})
        parts.append(binary)
        high = results["business_high_demand_metrics"].copy()
        high.insert(1, "evaluation_type", "high_demand_regression")
        parts.append(high)
        labels = results["business_label_metrics"].copy()
        labels.insert(1, "evaluation_type", "five_class_labels")
        labels["actual_threshold"] = pd.NA
        parts.append(labels)
    preferred = ["period", "evaluation_type", "actual_threshold", "model", "row_count"]
    combined = pd.concat(parts, ignore_index=True, sort=False)
    return combined[preferred + [column for column in combined.columns if column not in preferred]]


def _print_console_report(config: PeriodConfig, results: dict[str, pd.DataFrame]) -> None:
    """Print the requested best-model and v4.2 error-rate report."""

    binary = results["business_binary_metrics"]
    high = results["business_high_demand_metrics"]
    print(f"\n{config.label}")
    for threshold in BUSINESS_THRESHOLDS:
        rows = binary[binary["threshold"] == threshold]
        winner = rows.sort_values(["f1_score", "model"], ascending=[False, True], kind="stable").iloc[0]
        print(f"  Best model by >={threshold} F1: {winner['model']} ({winner['f1_score']:.6f})")
    at_20 = high[high["actual_threshold"] == 20].sort_values(["mae", "model"], kind="stable").iloc[0]
    at_43 = high[high["actual_threshold"] == 43].sort_values(["rmse", "model"], kind="stable").iloc[0]
    print(f"  Best model by MAE for actual >=20: {at_20['model']} ({at_20['mae']:.6f})")
    print(f"  Best model by RMSE for actual >=43: {at_43['model']} ({at_43['rmse']:.6f})")
    hybrid = binary[binary["model"] == "CatBoost v4.2 hybrid"].set_index("threshold")
    for threshold in (10, 20):
        print(
            f"  v4.2 at >={threshold}: false-positive rate "
            f"{hybrid.loc[threshold, 'false_positive_rate']:.6%}; "
            f"false-negative rate {hybrid.loc[threshold, 'false_negative_rate']:.6%}"
        )


def main() -> None:
    """Run all evaluations and atomically publish new artifacts."""

    _validate_inputs()
    _validate_no_output_collisions()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary_to_final: dict[Path, Path] = {}
    all_results: dict[str, dict[str, pd.DataFrame]] = {}
    try:
        for config in PERIODS:
            print(f"\nEvaluating {config.label} from {config.input_path.name}")
            frame = _load_numeric_predictions(config)
            results = _evaluate_period(config, frame)
            all_results[config.key] = results
            for kind, table in results.items():
                final_path = config.metric_path(kind)
                temporary_path = final_path.with_name(f".{final_path.name}.{token}.tmp")
                table.to_csv(temporary_path, index=False)
                temporary_to_final[temporary_path] = final_path
            labeled_temp = config.labeled_path.with_name(
                f".{config.labeled_path.name}.{token}.tmp"
            )
            _write_labeled_copy(config, labeled_temp)
            temporary_to_final[labeled_temp] = config.labeled_path
            del frame

        summary = _summary_table(all_results)
        summary_temp = SUMMARY_PATH.with_name(f".{SUMMARY_PATH.name}.{token}.tmp")
        summary.to_csv(summary_temp, index=False)
        temporary_to_final[summary_temp] = SUMMARY_PATH

        # Outputs were collision-checked before work; replace gives atomic publication.
        for temporary_path, final_path in temporary_to_final.items():
            if final_path.exists():
                raise FileExistsError(f"Output appeared during evaluation: {final_path}")
            os.replace(temporary_path, final_path)
    except Exception:
        for temporary_path in temporary_to_final:
            temporary_path.unlink(missing_ok=True)
        raise

    for config in PERIODS:
        _print_console_report(config, all_results[config.key])
    print(f"\nGenerated and verified {len(_all_output_paths())} new output files.")


if __name__ == "__main__":
    main()
