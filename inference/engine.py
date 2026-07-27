"""Shared, inference-only proposed-trip pipeline for frozen v4.2 and v4.4.

The module deliberately contains no CatBoost fitting or artifact-writing code.
Live proposal features are generated with the inference-side DuckDB feature
builder, using history strictly earlier than each proposal date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from inference import features as v43
from inference import hybrid as v42
from inference import classifiers as v44
from inference import batch_features


PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "analysis.duckdb"
MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"
THRESHOLDS = (10, 20, 30, 43)
HISTORY_START = "2023-01-01"

# Final mixed-system classifier selection. The >=10 production decision still
# comes from v4.2 numeric demand; its selected classifier remains audit-only.
CLASSIFIER_VARIANTS = {
    10: "none",
    20: "none",
    30: "none",
    43: "balanced",
}

V4_1_MODEL_PATH = MODELS_DIR / "catboost_demand_model_v4_1_recent_mae_6000.cbm"
HYBRID_RULE_PATH = RESULTS_DIR / "catboost_v4_2_hybrid_rule.json"
CLASSIFIER_MODEL_PATHS = {
    threshold: v44.model_path(threshold, CLASSIFIER_VARIANTS[threshold])
    for threshold in THRESHOLDS
}
CLASSIFIER_METADATA_PATHS = {
    threshold: v44.metadata_path(threshold, CLASSIFIER_VARIANTS[threshold])
    for threshold in THRESHOLDS
}

INPUT_COLUMNS = ["FIRMA_ID", "GUZERGAH_KODU", "SEFER_TARIHI", "SEFER_SAATI"]
OUTPUT_COLUMNS = [
    "FIRMA_ID", "GUZERGAH_KODU", "SEFER_TARIHI", "SEFER_SAATI",
    "canonical_route_id", "v4_1_prediction", "weekday_baseline_prediction",
    "v4_2_hybrid_weight", "v4_2_hybrid_prediction",
    *[f"probability_ge_{threshold}" for threshold in THRESHOLDS],
    *[f"cutoff_ge_{threshold}" for threshold in THRESHOLDS],
    *[f"classifier_variant_ge_{threshold}" for threshold in THRESHOLDS],
    *[f"v4_4_raw_decision_ge_{threshold}" for threshold in THRESHOLDS],
    *[f"mixed_decision_ge_{threshold}" for threshold in THRESHOLDS],
    "mixed_demand_label", "classifier_monotonicity_violation",
    "classifier_monotonicity_correction_applied",
    "prediction_reliability", "reliability_reason",
]
DEBUG_OUTPUT_COLUMNS = [
    "normalized_FIRMA_ID",
    "normalized_GUZERGAH_KODU",
    "normalized_SEFER_SAATI",
    "canonical_mapping_source",
    "departure_30min_bucket",
    "weekday",
    "company_route_time_weekday_count",
    "company_route_time_weekday_mean",
    "company_route_time_count",
    "company_route_time_mean",
    "company_route_count",
    "company_route_mean",
    "canonical_route_time_weekday_count",
    "canonical_route_time_weekday_mean",
    "canonical_route_count",
    "canonical_route_mean",
    "canonical_fallback_count",
    "canonical_fallback_mean",
    "previous_weekday_baseline_prediction",
    "previous_weekday_baseline_source",
    "weekday_baseline_prediction",
    "weekday_baseline_source",
]


@dataclass(frozen=True)
class FrozenArtifacts:
    """Loaded frozen models, metadata, cutoffs, and hybrid rule."""

    v4_1_model: CatBoostRegressor
    hybrid_rule: v42.HybridRule
    classifiers: dict[int, Any]
    metadata: dict[int, dict[str, Any]]
    cutoffs: dict[int, float]
    classifier_variants: dict[int, str]


def required_artifact_paths() -> list[Path]:
    """Return every immutable artifact required by mixed inference."""

    return [
        V4_1_MODEL_PATH,
        HYBRID_RULE_PATH,
        *[CLASSIFIER_MODEL_PATHS[t] for t in THRESHOLDS],
        *[CLASSIFIER_METADATA_PATHS[t] for t in THRESHOLDS],
    ]


def validate_required_artifacts() -> None:
    """Fail once with a complete list of missing frozen artifacts."""

    missing = [path for path in required_artifact_paths() if not path.exists()]
    if missing:
        rendered = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError("Missing required frozen inference artifacts:\n" + rendered)


def load_frozen_artifacts() -> FrozenArtifacts:
    """Load and validate all frozen artifacts without modifying them."""

    validate_required_artifacts()
    v4_1_model = v42._load_model(V4_1_MODEL_PATH, v43.V4_1_FEATURE_COLUMNS)
    hybrid_rule = v42.load_frozen_rule()
    classifiers: dict[int, Any] = {}
    metadata: dict[int, dict[str, Any]] = {}
    cutoffs: dict[int, float] = {}
    for threshold in THRESHOLDS:
        variant = CLASSIFIER_VARIANTS[threshold]
        model, model_metadata = v44.load_classifier_and_metadata(threshold, variant)
        classifiers[threshold] = model
        metadata[threshold] = model_metadata
        cutoffs[threshold] = float(model_metadata["frozen_probability_cutoff"])
    return FrozenArtifacts(
        v4_1_model,
        hybrid_rule,
        classifiers,
        metadata,
        cutoffs,
        CLASSIFIER_VARIANTS.copy(),
    )


def _parse_dates(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any():
        bad = values[parsed.isna()].astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(f"SEFER_TARIHI must use YYYY-MM-DD; invalid value(s): {bad}")
    return parsed.dt.normalize()


def _parse_times(values: pd.Series) -> pd.Series:
    text = values.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%H:%M", errors="coerce")
    seconds = text.str.fullmatch(r"\d{2}:\d{2}:\d{2}")
    if seconds.any():
        parsed.loc[seconds] = pd.to_datetime(text.loc[seconds], format="%H:%M:%S", errors="coerce")
    if parsed.isna().any():
        bad = text[parsed.isna()].drop_duplicates().head(10).tolist()
        raise ValueError(f"SEFER_SAATI must use HH:MM or HH:MM:SS; invalid value(s): {bad}")
    return parsed.dt.strftime("%H:%M:%S")


def validate_proposals(proposals: pd.DataFrame) -> pd.DataFrame:
    """Validate input schema, types, route mapping, and historical availability."""

    missing = [column for column in INPUT_COLUMNS if column not in proposals]
    if missing:
        raise ValueError(f"Missing required proposal column(s): {missing}")
    if proposals.empty:
        raise ValueError("No proposed trips were supplied")
    frame = proposals.copy()
    frame["_proposal_order"] = np.arange(len(frame), dtype=np.int64)
    for column in ("FIRMA_ID", "GUZERGAH_KODU"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or np.any(numeric % 1 != 0):
            raise ValueError(f"{column} must contain integer identifiers")
        frame[column] = numeric.astype("int64")
    frame["SEFER_TARIHI"] = _parse_dates(frame["SEFER_TARIHI"])
    frame["SEFER_SAATI"] = _parse_times(frame["SEFER_SAATI"])
    if frame["SEFER_TARIHI"].dt.date.min() <= date.fromisoformat(HISTORY_START):
        raise ValueError(f"Proposal dates must be later than {HISTORY_START}")

    route_keys = frame[["FIRMA_ID", "GUZERGAH_KODU"]].drop_duplicates()
    with duckdb.connect(str(DB_PATH), read_only=True) as conn:
        conn.register("proposal_route_keys", route_keys)
        mapping = conn.execute("""
            WITH historical_candidates AS (
                SELECT
                    history.FIRMA_ID,
                    history.GUZERGAH_KODU,
                    history.canonical_guzergah_id,
                    COUNT(*) AS historical_row_count,
                    MAX(history.SEFER_TARIHI) AS latest_history_date
                FROM model_data_base AS history
                JOIN proposal_route_keys AS keys
                  ON keys.FIRMA_ID = history.FIRMA_ID
                 AND keys.GUZERGAH_KODU = history.GUZERGAH_KODU
                WHERE history.canonical_guzergah_id IS NOT NULL
                GROUP BY
                    history.FIRMA_ID,
                    history.GUZERGAH_KODU,
                    history.canonical_guzergah_id
            ),
            historical_mapping AS (
                SELECT * EXCLUDE (historical_row_count, latest_history_date)
                FROM historical_candidates
                QUALIFY ROW_NUMBER() OVER (
                    PARTITION BY FIRMA_ID, GUZERGAH_KODU
                    ORDER BY historical_row_count DESC, latest_history_date DESC
                ) = 1
            )
            SELECT
                keys.FIRMA_ID,
                keys.GUZERGAH_KODU,
                COALESCE(
                    historical.canonical_guzergah_id,
                    current_mapping.canonical_guzergah_id
                )::UBIGINT AS canonical_guzergah_id,
                CASE
                    WHEN historical.canonical_guzergah_id IS NOT NULL
                        THEN 'model_data_base_history'
                    WHEN current_mapping.canonical_guzergah_id IS NOT NULL
                        THEN 'guzergah_canonical'
                END AS canonical_mapping_source
            FROM proposal_route_keys AS keys
            LEFT JOIN historical_mapping AS historical
              ON historical.FIRMA_ID = keys.FIRMA_ID
             AND historical.GUZERGAH_KODU = keys.GUZERGAH_KODU
            LEFT JOIN guzergah_canonical AS current_mapping
              ON current_mapping.firma_id = keys.FIRMA_ID
             AND current_mapping.guzergah_kodu = keys.GUZERGAH_KODU
        """).fetchdf()
        earliest = frame["SEFER_TARIHI"].min().date().isoformat()
        history_count = conn.execute(
            "SELECT COUNT(*) FROM model_data_base WHERE SEFER_TARIHI >= ? AND SEFER_TARIHI < ?",
            [HISTORY_START, earliest],
        ).fetchone()[0]
    missing_routes = mapping[mapping["canonical_guzergah_id"].isna()]
    if not missing_routes.empty:
        pairs = missing_routes[["FIRMA_ID", "GUZERGAH_KODU"]].to_dict("records")
        raise ValueError(f"Unknown company-route combination(s) or missing canonical route: {pairs}")
    if not history_count:
        raise ValueError(f"No historical feature rows exist before {earliest}")
    frame = frame.merge(mapping, on=["FIRMA_ID", "GUZERGAH_KODU"], validate="many_to_one")
    return frame


def build_proposal_feature_matrix(
    proposals: pd.DataFrame,
    conn: duckdb.DuckDBPyConnection | None = None,
    *,
    timings: dict[str, float] | None = None,
    log: Any | None = None,
) -> pd.DataFrame:
    """Build one vectorized, strict-earlier feature matrix for the full batch."""

    metrics = timings if timings is not None else {}
    owns_connection = conn is None
    connection = conn or v43.connect()
    try:
        return batch_features.build_feature_matrix(connection, proposals, metrics, log=log)
    finally:
        if owns_connection:
            connection.close()


def _demand_labels(decisions: dict[int, np.ndarray]) -> np.ndarray:
    labels = np.full(len(decisions[10]), "CLEAR_FAILURE", dtype=object)
    labels[decisions[10] == 1] = "WEAK_DEMAND"
    labels[decisions[20] == 1] = "MODERATE_DEMAND"
    labels[decisions[30] == 1] = "STRONG_DEMAND"
    labels[decisions[43] == 1] = "CAPACITY_PRESSURE"
    return labels


def prediction_reliability(
    exact_time_history_count: Iterable[int],
    weekday_time_history_count: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Label prediction reliability independently from predicted demand."""

    exact = np.asarray(exact_time_history_count, dtype=np.int64)
    weekday = np.asarray(weekday_time_history_count, dtype=np.int64)
    if exact.shape != weekday.shape:
        raise ValueError("Reliability history counts must have matching shapes")

    labels = np.select(
        [exact == 0, exact < 10, weekday >= 30, weekday >= 10],
        ["NO_HISTORY", "UNSAFE", "HIGH", "MEDIUM"],
        default="LOW",
    )

    def history_phrase(count: int, qualifier: str) -> str:
        return f"{count} {qualifier} sefer"

    reasons = np.empty(len(exact), dtype=object)
    for index, (exact_count, weekday_count, label) in enumerate(
        zip(exact, weekday, labels, strict=True)
    ):
        if label == "NO_HISTORY":
            reasons[index] = "Tam saat için geçmiş sefer bulunamadı"
        elif label == "UNSAFE":
            reasons[index] = (
                f"Yalnızca {history_phrase(int(exact_count), 'adet tam saat eşleşmeli')} ve "
                f"{history_phrase(int(weekday_count), 'adet aynı hafta günü-saat eşleşmeli')} var"
            )
        elif label == "HIGH":
            reasons[index] = (
                f"{history_phrase(int(exact_count), 'adet tam saat eşleşmeli')} ve "
                f"{history_phrase(int(weekday_count), 'adet aynı hafta günü-saat eşleşmeli')} var"
            )
        elif label == "MEDIUM":
            reasons[index] = (
                f"{history_phrase(int(exact_count), 'adet tam saat eşleşmeli')} ve "
                f"{history_phrase(int(weekday_count), 'adet aynı hafta günü-saat eşleşmeli')} var"
            )
        else:
            reasons[index] = (
                f"Tam saat geçmişi yeterli "
                f"({history_phrase(int(exact_count), 'adet tam saat eşleşmeli')}), ancak yalnızca "
                f"{history_phrase(int(weekday_count), 'adet aynı hafta günü-saat eşleşmeli')} var"
            )
    return labels.astype(object), reasons


def print_reliability_summary(frame: pd.DataFrame) -> None:
    """Print counts for every reliability level, including zero counts."""

    counts = frame["prediction_reliability"].value_counts()
    print("Prediction reliability counts:")
    for level in ("NO_HISTORY", "UNSAFE", "HIGH", "MEDIUM", "LOW"):
        print(f"  {level}: {int(counts.get(level, 0)):,}")


def predict_feature_matrix(
    frame: pd.DataFrame,
    artifacts: FrozenArtifacts | None = None,
    *,
    debug: bool = False,
    log: Any = None,
) -> pd.DataFrame:
    """Run frozen v4.2 numeric and v4.4 probability inference in batches."""

    _log = log or (lambda _msg: None)
    bundle = artifacts or load_frozen_artifacts()
    _log(f"Validating {frame.shape[0]:,}-row feature matrix...")
    v43.validate_feature_frame(frame)
    result = pd.DataFrame(index=frame.index)
    for column in ("SEFER_ID", "FIRMA_ID", "GUZERGAH_KODU", "SEFER_TARIHI"):
        if column in frame:
            result[column] = frame[column].to_numpy()
    result["SEFER_SAATI"] = frame["SEFER_SAATI"].to_numpy() if "SEFER_SAATI" in frame else ""
    result["canonical_route_id"] = frame["canonical_guzergah_id"].to_numpy()
    _log("Predicting v4.1 numeric demand...")
    result["v4_1_prediction"] = bundle.v4_1_model.predict(frame[v43.V4_1_FEATURE_COLUMNS])
    result["weekday_baseline_prediction"] = frame["baseline_prediction"].to_numpy(dtype=np.float64)

    _log("Applying v4.2 frozen hybrid rule...")
    hybrid_input = frame.copy()
    hybrid_input["v4_1_prediction"] = result["v4_1_prediction"].to_numpy()
    hybrid_input["baseline_prediction"] = result["weekday_baseline_prediction"].to_numpy()
    hybrid, weights, _ = v42.apply_hybrid_rule(hybrid_input, bundle.hybrid_rule)
    result["v4_2_hybrid_weight"] = weights
    result["v4_2_hybrid_prediction"] = hybrid

    raw: dict[int, np.ndarray] = {}
    _log(f"Predicting v4.4 threshold probabilities for {len(THRESHOLDS)} classifiers...")
    for threshold in THRESHOLDS:
        probability = bundle.classifiers[threshold].predict_proba(frame[v43.FEATURE_COLUMNS])[:, 1]
        cutoff = bundle.cutoffs[threshold]
        raw[threshold] = (probability >= cutoff).astype(np.int8)
        result[f"probability_ge_{threshold}"] = probability
        result[f"cutoff_ge_{threshold}"] = cutoff
        result[f"classifier_variant_ge_{threshold}"] = bundle.classifier_variants[threshold]
        result[f"v4_4_raw_decision_ge_{threshold}"] = raw[threshold]
    _log("Combining v4.4 decisions into mixed demand labels...")

    mixed: dict[int, np.ndarray] = {}
    mixed[43] = raw[43].copy()
    mixed[30] = np.maximum(raw[30], mixed[43])
    mixed[20] = np.maximum(raw[20], mixed[30])
    numeric_ge_10 = (hybrid >= 10).astype(np.int8)
    mixed[10] = np.maximum(numeric_ge_10, mixed[20])
    for threshold in THRESHOLDS:
        result[f"mixed_decision_ge_{threshold}"] = mixed[threshold]

    violation = (raw[43] > raw[30]) | (raw[30] > raw[20]) | (raw[20] > raw[10])
    correction = ((mixed[30] != raw[30]) | (mixed[20] != raw[20]) | (mixed[10] != numeric_ge_10))
    result["mixed_demand_label"] = _demand_labels(mixed)
    result["classifier_monotonicity_violation"] = violation.astype(np.int8)
    result["classifier_monotonicity_correction_applied"] = correction.astype(np.int8)
    if debug:
        result["normalized_FIRMA_ID"] = frame["FIRMA_ID"].astype("string").to_numpy()
        result["normalized_GUZERGAH_KODU"] = frame["GUZERGAH_KODU"].astype("string").to_numpy()
        result["normalized_SEFER_SAATI"] = frame["SEFER_SAATI"].astype("string").to_numpy()
        result["canonical_mapping_source"] = frame["canonical_mapping_source"].to_numpy()
        result["departure_30min_bucket"] = frame["departure_30min_bucket"].to_numpy(dtype=np.int64)
        result["weekday"] = frame["day_of_week"].to_numpy(dtype=np.int64)
        for column in (
            "company_route_time_weekday_count",
            "company_route_time_weekday_mean",
            "company_route_time_count",
            "company_route_time_mean",
            "company_route_count",
            "company_route_mean",
            "canonical_fallback_count",
            "canonical_fallback_mean",
        ):
            source_column = (
                f"debug_{column}"
                if column.startswith("company_route")
                else column
            )
            result[column] = frame[source_column].to_numpy()
        result["canonical_route_time_weekday_count"] = frame[
            "debug_canonical_route_time_weekday_count"
        ].to_numpy()
        result["canonical_route_time_weekday_mean"] = frame[
            "debug_canonical_route_time_weekday_mean"
        ].to_numpy()
        result["canonical_route_count"] = frame["debug_canonical_route_count"].to_numpy()
        result["canonical_route_mean"] = frame["debug_canonical_route_mean"].to_numpy()
        result["previous_weekday_baseline_prediction"] = frame[
            "previous_weekday_baseline_prediction"
        ].to_numpy()
        result["previous_weekday_baseline_source"] = frame[
            "previous_weekday_baseline_source"
        ].to_numpy()
        result["weekday_baseline_source"] = frame["baseline_source"].to_numpy()
    return result


def predict_proposals(
    proposals: pd.DataFrame,
    artifacts: FrozenArtifacts | None = None,
    *,
    debug: bool = False,
    timings: dict[str, float] | None = None,
    log: Any | None = None,
) -> pd.DataFrame:
    """Validate, feature-build, and predict an arbitrary proposal batch."""

    _log = log or (lambda _msg: None)
    metrics = timings if timings is not None else {}
    if artifacts is None:
        _log("Loading frozen v4.1, v4.2 rule, and v4.4 classifier artifacts...")
        bundle = load_frozen_artifacts()
        _log("Frozen artifacts loaded")
    else:
        bundle = artifacts
    _log("Opening DuckDB connection...")
    conn = v43.connect()
    try:
        features = build_proposal_feature_matrix(proposals, conn, timings=metrics, log=_log)
        started = perf_counter()
        _log(f"Running CatBoost prediction on {features.shape[0]:,} rows...")
        predictions = predict_feature_matrix(features, bundle, debug=debug, log=_log)
        metrics["catboost_prediction"] = perf_counter() - started
        _log(f"CatBoost prediction complete ({metrics['catboost_prediction']:.3f}s)")
    finally:
        conn.close()
    _log("Computing prediction-reliability labels...")
    predictions["SEFER_SAATI"] = features["SEFER_SAATI"].to_numpy()
    reliability, reasons = prediction_reliability(
        features["debug_company_route_time_count"],
        features["debug_company_route_time_weekday_count"],
    )
    predictions["prediction_reliability"] = reliability
    predictions["reliability_reason"] = reasons
    columns = OUTPUT_COLUMNS + (DEBUG_OUTPUT_COLUMNS if debug else [])
    return predictions[list(dict.fromkeys(columns))]
