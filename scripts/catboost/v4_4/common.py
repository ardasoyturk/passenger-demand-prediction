"""Shared infrastructure for the CatBoost v4.4 demand-threshold classifiers.

The classifiers reuse the exact 72-feature v4.3 DuckDB pipeline.  Cutoffs are
selected on 2025 H1 validation only and are immutable during test/final runs.
No function in this module trains a model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

PROJECT_DIR = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(PROJECT_DIR))
from scripts.catboost.v4_3 import common as features
DB_PATH = PROJECT_DIR / "analysis.duckdb"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
DEFAULT_THRESHOLDS = (10, 20, 30, 43)
TARGET_NAMES = {threshold: f"target_ge_{threshold}" for threshold in DEFAULT_THRESHOLDS}

TRAIN_CONFIG = features.PeriodConfig(
    "train_2024",
    "2024 leakage-safe supervised training",
    "2023-01-01",
    "2024-01-01",
    "2024-01-01",
    "2025-01-01",
    2_797_931,
    False,
)
PERIODS = {
    "validation": features.PERIODS["validation"],
    "test": features.PERIODS["test"],
    "final": features.PeriodConfig(
        "final_2026",
        "2026 final reporting (through 2026-04-30; data currently ends 2026-04-14)",
        "2023-01-01",
        "2026-01-01",
        "2026-01-01",
        "2026-05-01",
        780_865,
        False,
    ),
}

COMPARISON_COLUMNS = {
    "CatBoost v3 regression": "v3_prediction",
    "CatBoost v4.1 regression": "v4_1_prediction",
    "Weekday baseline": "weekday_baseline_prediction",
    "CatBoost v4.2 hybrid": "v4_2_hybrid_prediction",
    "CatBoost v4.3 regression": "v4_3_prediction",
    "CatBoost v4.3 + frozen hybrid": "v4_3_hybrid_prediction",
}

CALIBRATION_BUCKETS = [f"{index / 10:.1f}-{(index + 1) / 10:.1f}" for index in range(10)]


@dataclass(frozen=True)
class OutputPaths:
    metrics: Path
    predictions: Path
    calibration: Path
    summary: Path


def class_weight_suffix(class_weight_mode: str) -> str:
    if class_weight_mode == "balanced":
        return ""
    if class_weight_mode == "none":
        return "_class_weights_none"
    raise ValueError("class_weight_mode must be 'balanced' or 'none'")


def output_paths(period: str, class_weight_mode: str = "balanced") -> OutputPaths:
    names = {
        "validation": "catboost_v4_4_classifier_validation",
        "test": "catboost_v4_4_classifier_test_2025_h2",
        "final": "catboost_v4_4_classifier_final_2026",
    }
    suffix = class_weight_suffix(class_weight_mode)
    base = names[period]
    stem = RESULTS_DIR / base.replace("_validation", f"{suffix}_validation").replace(
        "_test_2025_h2", f"{suffix}_test_2025_h2"
    ).replace("_final_2026", f"{suffix}_final_2026")
    return OutputPaths(
        Path(f"{stem}_metrics.csv"),
        Path(f"{stem}_predictions.csv"),
        Path(f"{stem}_calibration.csv"),
        Path(f"{stem}_summary.csv"),
    )


def model_path(threshold: int, class_weight_mode: str = "balanced") -> Path:
    suffix = class_weight_suffix(class_weight_mode)
    return MODELS_DIR / f"catboost_demand_model_v4_4_classifier_ge_{threshold}{suffix}.cbm"


def metadata_path(threshold: int, class_weight_mode: str = "balanced") -> Path:
    suffix = class_weight_suffix(class_weight_mode)
    return MODELS_DIR / f"catboost_demand_model_v4_4_classifier_ge_{threshold}{suffix}_metadata.json"


def cutoff_path(threshold: int, class_weight_mode: str = "balanced") -> Path:
    suffix = class_weight_suffix(class_weight_mode)
    return RESULTS_DIR / f"catboost_v4_4_classifier_ge_{threshold}{suffix}_validation_cutoffs.csv"


def parse_thresholds(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError("--thresholds must be a comma-separated list of integers") from error
    if not parsed:
        raise ValueError("At least one threshold is required")
    invalid = sorted(set(parsed) - set(DEFAULT_THRESHOLDS))
    if invalid:
        raise ValueError(f"Unsupported thresholds {invalid}; allowed: {list(DEFAULT_THRESHOLDS)}")
    if len(parsed) != len(set(parsed)):
        raise ValueError("Thresholds must not be repeated")
    return parsed


def target_name(threshold: int) -> str:
    return TARGET_NAMES[threshold]


def ensure_directories() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def validate_database_schema() -> list[str]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    required = set(features.SOURCE_COLUMNS)
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        if "model_data_base" not in tables:
            raise RuntimeError("Required relation model_data_base is missing")
        columns = [row[0] for row in conn.execute("DESCRIBE model_data_base").fetchall()]
    finally:
        conn.close()
    missing = sorted(required - set(columns))
    if missing:
        raise RuntimeError(f"model_data_base is missing required columns: {missing}")
    if len(features.FEATURE_COLUMNS) != 72 or len(set(features.FEATURE_COLUMNS)) != 72:
        raise RuntimeError("Expected exactly 72 unique v4.3 feature columns")
    return columns


def inspect_split_counts() -> tuple[pd.DataFrame, pd.DataFrame]:
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        splits = conn.execute("""
            SELECT split, COUNT(*)::BIGINT AS row_count,
                   MIN(SEFER_TARIHI) AS minimum_date, MAX(SEFER_TARIHI) AS maximum_date
            FROM (
                SELECT CASE
                    WHEN SEFER_TARIHI >= DATE '2023-01-01' AND SEFER_TARIHI < DATE '2025-01-01' THEN 'training_source_2023_2024'
                    WHEN SEFER_TARIHI >= DATE '2025-01-01' AND SEFER_TARIHI < DATE '2025-07-01' THEN 'validation_2025_h1'
                    WHEN SEFER_TARIHI >= DATE '2025-07-01' AND SEFER_TARIHI < DATE '2026-01-01' THEN 'test_2025_h2'
                    WHEN SEFER_TARIHI >= DATE '2026-01-01' AND SEFER_TARIHI < DATE '2026-05-01' THEN 'final_2026_reporting'
                END AS split, SEFER_TARIHI
                FROM model_data_base
                WHERE SEFER_TARIHI >= DATE '2023-01-01' AND SEFER_TARIHI < DATE '2026-05-01'
            )
            GROUP BY split ORDER BY minimum_date
        """).fetchdf()
        distributions = conn.execute("""
            SELECT threshold,
                   SUM(CASE WHEN target >= threshold THEN 1 ELSE 0 END)::BIGINT AS positive_count,
                   SUM(CASE WHEN target < threshold THEN 1 ELSE 0 END)::BIGINT AS negative_count,
                   AVG(CASE WHEN target >= threshold THEN 1.0 ELSE 0.0 END)::DOUBLE AS positive_rate,
                   AVG(CASE WHEN target < threshold THEN 1.0 ELSE 0.0 END)::DOUBLE AS negative_rate
            FROM model_data_base
            CROSS JOIN (VALUES (10), (20), (30), (43)) AS limits(threshold)
            WHERE SEFER_TARIHI >= DATE '2024-01-01' AND SEFER_TARIHI < DATE '2025-01-01'
            GROUP BY threshold ORDER BY threshold
        """).fetchdf()
    finally:
        conn.close()
    return splits, distributions


def build_feature_matrix(config: features.PeriodConfig, *, training: bool) -> pd.DataFrame:
    """Build one matrix through the shared v4.3 SQL, without model inference."""
    conn = features.connect()
    try:
        features.create_source_tables(conn, config, training=training)
        recent_history_table = "v4_3_history"
        if training:
            conn.execute("""
                CREATE OR REPLACE TEMP TABLE v4_3_recent_history AS
                SELECT * FROM v4_3_history
                UNION ALL
                SELECT * FROM v4_3_target
            """)
            recent_history_table = "v4_3_recent_history"
        features.create_long_term_statistics(conn)
        features.create_recent_statistics(conn, config, history_table=recent_history_table)
        features.create_feature_table(conn)
        columns = [
            "SEFER_ID", "SEFER_TARIHI", "FIRMA_ID", "GUZERGAH_KODU",
            *features.FEATURE_COLUMNS, "target", "baseline_prediction", "baseline_source",
        ]
        # Keep the requested identifiers once even though some are also model features.
        columns = list(dict.fromkeys(columns))
        frame = conn.execute(
            f"SELECT {features.sql_identifier_list(columns)} FROM v4_3_features"
        ).fetchdf()
    finally:
        conn.close()
    features.validate_feature_frame(frame)
    if len(frame) != config.expected_rows:
        raise RuntimeError(f"Expected {config.expected_rows:,} {config.key} rows; found {len(frame):,}")
    return frame


def class_distribution(actual: Iterable[int], threshold: int) -> dict[str, Any]:
    labels = np.asarray(actual) >= threshold
    positive = int(labels.sum())
    negative = int(len(labels) - positive)
    return {
        "positive_count": positive,
        "negative_count": negative,
        "positive_rate": positive / len(labels),
        "negative_rate": negative / len(labels),
    }


def _safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def classification_metrics(actual_label: Iterable[int], predicted_label: Iterable[int]) -> dict[str, Any]:
    actual = np.asarray(actual_label, dtype=np.int8)
    predicted = np.asarray(predicted_label, dtype=np.int8)
    if actual.shape != predicted.shape or actual.ndim != 1:
        raise ValueError("Actual and predicted labels must be aligned one-dimensional arrays")
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


def probability_metrics(actual_label: Iterable[int], probability: Iterable[float]) -> dict[str, float]:
    actual = np.asarray(actual_label, dtype=np.int8)
    probabilities = np.asarray(probability, dtype=np.float64)
    if actual.shape != probabilities.shape:
        raise ValueError("Actual labels and probabilities must align")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
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


def cutoff_candidates(actual_label: Iterable[int], probability: Iterable[float]) -> pd.DataFrame:
    actual = np.asarray(actual_label, dtype=np.int8)
    probabilities = np.asarray(probability, dtype=np.float64)
    records = []
    for cutoff in np.round(np.arange(0.05, 0.951, 0.01), 2):
        record = {"probability_cutoff": float(cutoff)}
        record.update(classification_metrics(actual, probabilities >= cutoff))
        records.append(record)
    columns = [
        "probability_cutoff", "true_positive", "true_negative", "false_positive",
        "false_negative", "precision", "recall", "specificity", "f1",
        "balanced_accuracy", "false_positive_rate", "false_negative_rate",
        "predicted_positive_rate",
    ]
    return pd.DataFrame(records)[columns]


def select_cutoff(candidates: pd.DataFrame) -> pd.Series:
    """Maximum F1; then lower FNR, higher precision, closest to 0.50."""
    ranked = candidates.assign(
        cutoff_distance=(candidates["probability_cutoff"] - 0.5).abs()
    ).sort_values(
        ["f1", "false_negative_rate", "precision", "cutoff_distance", "probability_cutoff"],
        ascending=[False, True, False, True, True],
        kind="stable",
    )
    return ranked.iloc[0].drop(labels="cutoff_distance")


def calibration_table(actual_label: Iterable[int], probability: Iterable[float], *, threshold: int, period: str) -> pd.DataFrame:
    actual = np.asarray(actual_label, dtype=np.int8)
    probabilities = np.asarray(probability, dtype=np.float64)
    bucket_index = np.minimum((probabilities * 10).astype(np.int8), 9)
    records = []
    for index, label in enumerate(CALIBRATION_BUCKETS):
        mask = bucket_index == index
        count = int(mask.sum())
        average = float(probabilities[mask].mean()) if count else np.nan
        actual_rate = float(actual[mask].mean()) if count else np.nan
        records.append({
            "period": period,
            "threshold": threshold,
            "probability_bucket": label,
            "row_count": count,
            "average_predicted_probability": average,
            "actual_positive_rate": actual_rate,
            "calibration_error": actual_rate - average if count else np.nan,
        })
    return pd.DataFrame(records)


def comparison_artifact_path(period: str) -> Path:
    key = PERIODS[period].key
    return RESULTS_DIR / f"catboost_v4_3_{key}_predictions.csv"


def merge_comparison_predictions(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    path = comparison_artifact_path(period)
    if not path.exists():
        raise FileNotFoundError(
            f"Required frozen comparison artifact is missing: {path}. "
            "Reproduce the frozen v4.3 period output before evaluating v4.4."
        )
    required = [
        "SEFER_ID", "target", "v3_prediction", "v4_1_prediction",
        "v4_2_hybrid_prediction", "v4_3_prediction", "v4_3_hybrid_prediction",
        "baseline_prediction",
    ]
    comparison = pd.read_csv(path, usecols=required).rename(
        columns={"target": "comparison_target", "baseline_prediction": "weekday_baseline_prediction"}
    )
    if comparison["SEFER_ID"].duplicated().any() or frame["SEFER_ID"].duplicated().any():
        raise RuntimeError("SEFER_ID must be unique for comparison alignment")
    merged = frame.merge(comparison, on="SEFER_ID", how="left", validate="one_to_one")
    if len(merged) != len(frame) or merged[list(COMPARISON_COLUMNS.values())].isna().any().any():
        raise RuntimeError("Frozen comparison predictions did not align to the v4.4 matrix")
    if not np.array_equal(merged["target"].to_numpy(), merged["comparison_target"].to_numpy()):
        raise RuntimeError("Actual passenger counts disagree with frozen comparison artifacts")
    return merged.drop(columns="comparison_target")


def load_classifier_and_metadata(
    threshold: int,
    class_weight_mode: str = "balanced",
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    model_file = model_path(threshold, class_weight_mode)
    metadata_file = metadata_path(threshold, class_weight_mode)
    missing = [path for path in (model_file, metadata_file) if not path.exists()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing trained v4.4 artifact(s): {rendered}. Run the manual v4.4 training command first."
        )
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("threshold") != threshold:
        raise RuntimeError(f"Metadata threshold mismatch in {metadata_file}")
    if metadata.get("class_weight_mode") != class_weight_mode:
        raise RuntimeError(f"Class-weight mode mismatch in {metadata_file}")
    if metadata.get("feature_names") != features.FEATURE_COLUMNS:
        raise RuntimeError(f"Metadata feature contract mismatch in {metadata_file}")
    cutoff = metadata.get("frozen_probability_cutoff")
    if cutoff is None or not 0.05 <= float(cutoff) <= 0.95:
        raise RuntimeError(f"Invalid frozen cutoff in {metadata_file}")
    model = CatBoostClassifier()
    model.load_model(str(model_file))
    if list(model.feature_names_) != features.FEATURE_COLUMNS:
        raise RuntimeError(f"Model feature contract mismatch in {model_file}")
    return model, metadata


def metric_rows(frame: pd.DataFrame, probability: np.ndarray, cutoff: float, threshold: int, period: str) -> pd.DataFrame:
    actual = (frame["target"].to_numpy() >= threshold).astype(np.int8)
    records: list[dict[str, Any]] = []
    for model_name, column in COMPARISON_COLUMNS.items():
        record = {"period": period, "threshold": threshold, "model": model_name, "probability_cutoff": np.nan}
        record.update(classification_metrics(actual, frame[column].to_numpy() >= threshold))
        record.update({key: np.nan for key in ["roc_auc", "pr_auc", "log_loss", "brier_score", "average_predicted_probability"]})
        records.append(record)
    v44 = {"period": period, "threshold": threshold, "model": "CatBoost v4.4 classifier", "probability_cutoff": cutoff}
    v44.update(classification_metrics(actual, probability >= cutoff))
    v44.update(probability_metrics(actual, probability))
    records.append(v44)
    return pd.DataFrame(records)


def prediction_rows(frame: pd.DataFrame, probability: np.ndarray, cutoff: float, threshold: int) -> pd.DataFrame:
    return pd.DataFrame({
        "SEFER_ID": frame["SEFER_ID"].to_numpy(),
        "SEFER_TARIHI": frame["SEFER_TARIHI"].to_numpy(),
        "FIRMA_ID": frame["FIRMA_ID"].to_numpy(),
        "GUZERGAH_KODU": frame["GUZERGAH_KODU"].to_numpy(),
        "actual_passenger_count": frame["target"].to_numpy(),
        "threshold": threshold,
        "actual_label": (frame["target"].to_numpy() >= threshold).astype(np.int8),
        "v4_4_probability": probability,
        "frozen_probability_cutoff": cutoff,
        "v4_4_predicted_label": (probability >= cutoff).astype(np.int8),
        "v3_prediction": frame["v3_prediction"].to_numpy(),
        "v4_1_prediction": frame["v4_1_prediction"].to_numpy(),
        "weekday_baseline_prediction": frame["weekday_baseline_prediction"].to_numpy(),
        "v4_2_hybrid_prediction": frame["v4_2_hybrid_prediction"].to_numpy(),
        "v4_3_prediction": frame["v4_3_prediction"].to_numpy(),
        "v4_3_hybrid_prediction": frame["v4_3_hybrid_prediction"].to_numpy(),
    })


def summary_row(actual: np.ndarray, probability: np.ndarray, cutoff: float, threshold: int, period: str) -> dict[str, Any]:
    selected = classification_metrics(actual, probability >= cutoff)
    standard = classification_metrics(actual, probability >= 0.5)
    probabilities = probability_metrics(actual, probability)
    row: dict[str, Any] = {
        "period": period,
        "threshold": threshold,
        "target_name": target_name(threshold),
        "frozen_probability_cutoff": cutoff,
    }
    row.update({f"selected_{key}": value for key, value in selected.items()})
    row.update({f"cutoff_0_50_{key}": value for key, value in standard.items()})
    row.update(probabilities)
    return row


def refuse_overwrite(paths: Iterable[Path]) -> None:
    collisions = [path for path in paths if path.exists()]
    if collisions:
        rendered = "\n".join(f"  - {path}" for path in collisions)
        raise FileExistsError("Refusing to overwrite v4.4 artifact(s):\n" + rendered)


def run_frozen_evaluation(
    period: str,
    thresholds: list[int],
    class_weight_mode: str = "balanced",
) -> None:
    """Load existing classifiers and frozen cutoffs; never fit or tune."""
    if period not in ("test", "final"):
        raise ValueError("Frozen evaluation is restricted to test or final")
    paths = output_paths(period, class_weight_mode)
    refuse_overwrite(paths.__dict__.values())
    loaded = {
        threshold: load_classifier_and_metadata(threshold, class_weight_mode)
        for threshold in thresholds
    }
    frame = build_feature_matrix(PERIODS[period], training=False)
    frame = merge_comparison_predictions(frame, period)
    metric_frames = []
    prediction_frames = []
    calibration_frames = []
    summaries = []
    for threshold, (model, metadata) in loaded.items():
        cutoff = float(metadata["frozen_probability_cutoff"])
        probability = model.predict_proba(frame[features.FEATURE_COLUMNS])[:, 1]
        actual = (frame["target"].to_numpy() >= threshold).astype(np.int8)
        metric_frames.append(metric_rows(frame, probability, cutoff, threshold, period))
        prediction_frames.append(prediction_rows(frame, probability, cutoff, threshold))
        calibration_frames.append(calibration_table(actual, probability, threshold=threshold, period=period))
        summaries.append(summary_row(actual, probability, cutoff, threshold, period))
    pd.concat(metric_frames, ignore_index=True).to_csv(paths.metrics, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(paths.predictions, index=False)
    pd.concat(calibration_frames, ignore_index=True).to_csv(paths.calibration, index=False)
    pd.DataFrame(summaries).to_csv(paths.summary, index=False)
    print(
        f"Saved frozen {period} evaluation outputs for class weights "
        f"'{class_weight_mode}' under {RESULTS_DIR}"
    )
