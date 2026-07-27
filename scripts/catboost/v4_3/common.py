"""Shared leakage-safe feature and business evaluation code for CatBoost v4.3.

v4.3 keeps the complete 64-feature v4.1 contract and appends eight
company-route-time-weekday distribution features.  The already-frozen v4.2
rule and artifacts are read only; applying that rule to v4.3 is a separate
comparison and never changes the production-candidate rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

PROJECT_DIR = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(PROJECT_DIR))
from scripts.passenger_demand_labels import (
    calculate_binary_classification_metrics,
    rounded_nonnegative_predictions,
)
DB_PATH = PROJECT_DIR / "analysis.duckdb"
TEMP_DIR = PROJECT_DIR / "duckdb_temp"
RESULTS_DIR = PROJECT_DIR / "results"
MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "catboost_demand_model_v4_3_business_distribution_mae_6000.cbm"
)
FROZEN_RULE_PATH = RESULTS_DIR / "catboost_v4_2_hybrid_rule.json"

DUCKDB_THREADS = max(1, os.cpu_count() or 4)
RECENT_WINDOWS = (30, 60, 90, 180)

CATEGORICAL_FEATURES = [
    "FIRMA_ID",
    "GUZERGAH_KODU",
    "canonical_guzergah_id",
    "month",
    "day_of_week",
    "departure_30min_bucket",
]
CALENDAR_NUMERIC_FEATURES = ["week_of_year", "departure_minute"]

HISTORICAL_GROUP_DEFINITIONS = [
    (
        "company_route_time_weekday",
        [
            "FIRMA_ID",
            "canonical_guzergah_id",
            "departure_30min_bucket",
            "day_of_week",
        ],
    ),
    (
        "company_route_time",
        ["FIRMA_ID", "canonical_guzergah_id", "departure_30min_bucket"],
    ),
    (
        "canonical_route_time_weekday",
        ["canonical_guzergah_id", "departure_30min_bucket", "day_of_week"],
    ),
    ("canonical_route", ["canonical_guzergah_id"]),
    ("company", ["FIRMA_ID"]),
]
RECENT_GROUP_DEFINITIONS = [
    HISTORICAL_GROUP_DEFINITIONS[0],
    HISTORICAL_GROUP_DEFINITIONS[2],
]

# These are the unchanged v3/v4.1 long-term features.  In particular, v4.1
# already passed median, p90, above-60, above-100 and count into CatBoost.
HISTORICAL_STATISTIC_SUFFIXES = [
    "average",
    "median",
    "std",
    "maximum",
    "p90",
    "above_60_rate",
    "above_100_rate",
    "count",
]
HISTORICAL_FEATURES = [
    f"{prefix}_{suffix}"
    for prefix, _ in HISTORICAL_GROUP_DEFINITIONS
    for suffix in HISTORICAL_STATISTIC_SUFFIXES
]
RECENT_FEATURES = [
    f"{prefix}_recent_{window}d_{suffix}"
    for prefix, _ in RECENT_GROUP_DEFINITIONS
    for window in RECENT_WINDOWS
    for suffix in ("average", "count")
]

DISTRIBUTION_PREFIX = "company_route_time_weekday"
NEW_DISTRIBUTION_SUFFIXES = [
    "p10",
    "p25",
    "p75",
    "below_10_rate",
    "above_10_rate",
    "above_20_rate",
    "above_30_rate",
    "above_40_rate",
]
NEW_DISTRIBUTION_FEATURES = [
    f"{DISTRIBUTION_PREFIX}_{suffix}"
    for suffix in NEW_DISTRIBUTION_SUFFIXES
]

V4_1_FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + CALENDAR_NUMERIC_FEATURES
    + HISTORICAL_FEATURES
    + RECENT_FEATURES
)
FEATURE_COLUMNS = V4_1_FEATURE_COLUMNS + NEW_DISTRIBUTION_FEATURES

SOURCE_COLUMNS = [
    "SEFER_ID",
    "SEFER_TARIHI",
    "FIRMA_ID",
    "GUZERGAH_KODU",
    "canonical_guzergah_id",
    "target",
    "month",
    "week_of_year",
    "day_of_week",
    "departure_minute",
    "departure_30min_bucket",
]

MODEL_COMPARISON_COLUMNS = {
    "CatBoost v3": "v3_prediction",
    "CatBoost v4.1": "v4_1_prediction",
    "v4.2 hybrid": "v4_2_hybrid_prediction",
    "CatBoost v4.3": "v4_3_prediction",
    "v4.3 + frozen hybrid rule": "v4_3_hybrid_prediction",
    "Weekday baseline": "baseline_prediction",
}
BUSINESS_THRESHOLDS = (20, 30, 43)
NEAR_THRESHOLD_NEGATIVE_BANDS = {
    20: (10, 20),
    30: (20, 30),
    43: (30, 43),
}


@dataclass(frozen=True)
class PeriodConfig:
    key: str
    label: str
    history_start: str
    history_end_exclusive: str
    target_start: str
    target_end_exclusive: str
    expected_rows: int
    tuning_allowed: bool


PERIODS = {
    "validation": PeriodConfig(
        "validation_2025_h1",
        "2025 H1 validation",
        "2023-01-01",
        "2025-01-01",
        "2025-01-01",
        "2025-07-01",
        1_420_182,
        True,
    ),
    "test": PeriodConfig(
        "test_2025_h2",
        "2025 H2 test",
        "2023-01-01",
        "2025-07-01",
        "2025-07-01",
        "2026-01-01",
        1_517_010,
        False,
    ),
    "final": PeriodConfig(
        "final_2026",
        "2026 final (reporting only; already observed)",
        "2023-01-01",
        "2026-01-01",
        "2026-01-01",
        "2026-04-15",
        780_865,
        False,
    ),
}


def sql_identifier_list(columns: list[str]) -> str:
    return ", ".join(columns)


def ensure_environment(*, require_model: bool = False) -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    if require_model and not MODEL_PATH.exists():
        raise FileNotFoundError(f"v4.3 model not found: {MODEL_PATH}")
    if not FROZEN_RULE_PATH.exists():
        raise FileNotFoundError(f"Frozen v4.2 rule not found: {FROZEN_RULE_PATH}")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)


def connect() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(str(DB_PATH), read_only=False)
    temp_path = TEMP_DIR.as_posix().replace("'", "''")
    connection.execute(f"SET threads = {DUCKDB_THREADS}")
    connection.execute(f"SET temp_directory = '{temp_path}'")
    return connection


def _join_condition(base_alias: str, aggregate_alias: str, columns: list[str]) -> str:
    return " AND ".join(
        f"{base_alias}.{column} = {aggregate_alias}.{column}"
        for column in columns
    )


def create_source_tables(
    conn: duckdb.DuckDBPyConnection,
    config: PeriodConfig,
    *,
    training: bool,
) -> None:
    columns = sql_identifier_list(SOURCE_COLUMNS)
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE v4_3_history AS
        SELECT {columns}
        FROM model_data_base
        WHERE SEFER_TARIHI >= DATE '{config.history_start}'
          AND SEFER_TARIHI < DATE '{config.history_end_exclusive}'
    """)
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE v4_3_target AS
        SELECT {columns}
        FROM model_data_base
        WHERE SEFER_TARIHI >= DATE '{config.target_start}'
          AND SEFER_TARIHI < DATE '{config.target_end_exclusive}'
    """)
    history_count = conn.execute("SELECT COUNT(*) FROM v4_3_history").fetchone()[0]
    target_count = conn.execute("SELECT COUNT(*) FROM v4_3_target").fetchone()[0]
    if not history_count or not target_count:
        raise RuntimeError("History or target period is empty.")
    if not training and target_count != config.expected_rows:
        raise RuntimeError(
            f"Expected {config.expected_rows:,} rows for {config.key}; "
            f"found {target_count:,}."
        )
    latest_history = conn.execute("SELECT MAX(SEFER_TARIHI) FROM v4_3_history").fetchone()[0]
    earliest_target = conn.execute("SELECT MIN(SEFER_TARIHI) FROM v4_3_target").fetchone()[0]
    if latest_history >= earliest_target:
        raise RuntimeError(
            f"Leakage guard failed: history ends {latest_history}, "
            f"target starts {earliest_target}."
        )
    print(f"History rows: {history_count:,}; target rows: {target_count:,}")
    print(f"Leakage cutoff verified: {latest_history} < {earliest_target}")


def create_long_term_statistics(
    conn: duckdb.DuckDBPyConnection,
    *,
    history_where_sql: str = "TRUE",
) -> None:
    """Create v4.1 statistics plus the eight new v4.3 statistics in DuckDB."""

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE v4_3_global_stats AS
        SELECT
            AVG(target)::DOUBLE AS global_average,
            MEDIAN(target)::DOUBLE AS global_median,
            STDDEV_SAMP(target)::DOUBLE AS global_std,
            MAX(target)::DOUBLE AS global_maximum,
            QUANTILE_CONT(target, 0.10)::DOUBLE AS global_p10,
            QUANTILE_CONT(target, 0.25)::DOUBLE AS global_p25,
            QUANTILE_CONT(target, 0.75)::DOUBLE AS global_p75,
            QUANTILE_CONT(target, 0.90)::DOUBLE AS global_p90,
            AVG(CASE WHEN target < 10 THEN 1.0 ELSE 0.0 END)::DOUBLE
                AS global_below_10_rate,
            AVG(CASE WHEN target > 10 THEN 1.0 ELSE 0.0 END)::DOUBLE
                AS global_above_10_rate,
            AVG(CASE WHEN target > 20 THEN 1.0 ELSE 0.0 END)::DOUBLE
                AS global_above_20_rate,
            AVG(CASE WHEN target > 30 THEN 1.0 ELSE 0.0 END)::DOUBLE
                AS global_above_30_rate,
            AVG(CASE WHEN target > 40 THEN 1.0 ELSE 0.0 END)::DOUBLE
                AS global_above_40_rate,
            AVG(CASE WHEN target > 60 THEN 1.0 ELSE 0.0 END)::DOUBLE
                AS global_above_60_rate,
            AVG(CASE WHEN target > 100 THEN 1.0 ELSE 0.0 END)::DOUBLE
                AS global_above_100_rate
        FROM v4_3_history
        WHERE {history_where_sql}
    """)

    for prefix, group_columns in HISTORICAL_GROUP_DEFINITIONS:
        group_sql = sql_identifier_list(group_columns)
        extra_sql = ""
        if prefix == DISTRIBUTION_PREFIX:
            extra_sql = f""",
                QUANTILE_CONT(target, 0.10)::DOUBLE AS {prefix}_p10,
                QUANTILE_CONT(target, 0.25)::DOUBLE AS {prefix}_p25,
                QUANTILE_CONT(target, 0.75)::DOUBLE AS {prefix}_p75,
                AVG(CASE WHEN target < 10 THEN 1.0 ELSE 0.0 END)::DOUBLE
                    AS {prefix}_below_10_rate,
                AVG(CASE WHEN target > 10 THEN 1.0 ELSE 0.0 END)::DOUBLE
                    AS {prefix}_above_10_rate,
                AVG(CASE WHEN target > 20 THEN 1.0 ELSE 0.0 END)::DOUBLE
                    AS {prefix}_above_20_rate,
                AVG(CASE WHEN target > 30 THEN 1.0 ELSE 0.0 END)::DOUBLE
                    AS {prefix}_above_30_rate,
                AVG(CASE WHEN target > 40 THEN 1.0 ELSE 0.0 END)::DOUBLE
                    AS {prefix}_above_40_rate
            """
        print(f"  Long-term aggregation: {prefix}")
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE v4_3_{prefix}_stats AS
            SELECT
                {group_sql},
                AVG(target)::DOUBLE AS {prefix}_average,
                MEDIAN(target)::DOUBLE AS {prefix}_median,
                STDDEV_SAMP(target)::DOUBLE AS {prefix}_std,
                MAX(target)::DOUBLE AS {prefix}_maximum,
                QUANTILE_CONT(target, 0.90)::DOUBLE AS {prefix}_p90,
                AVG(CASE WHEN target > 60 THEN 1.0 ELSE 0.0 END)::DOUBLE
                    AS {prefix}_above_60_rate,
                AVG(CASE WHEN target > 100 THEN 1.0 ELSE 0.0 END)::DOUBLE
                    AS {prefix}_above_100_rate,
                COUNT(*)::BIGINT AS {prefix}_count
                {extra_sql}
            FROM v4_3_history
            WHERE {history_where_sql}
            GROUP BY {group_sql}
        """)


def create_recent_statistics(
    conn: duckdb.DuckDBPyConnection,
    config: PeriodConfig,
    *,
    history_table: str,
) -> None:
    """Build strict-earlier-date rolling features; same-day rows never interact."""

    for prefix, group_columns in RECENT_GROUP_DEFINITIONS:
        reference_columns = ", ".join(f"source.{column}" for column in group_columns)
        grouped_columns = ", ".join(f"reference.{column}" for column in group_columns)
        join_sql = _join_condition("reference", "history", group_columns)
        for window in RECENT_WINDOWS:
            print(f"  Recent aggregation: {prefix}, {window}d")
            conn.execute(f"""
                CREATE OR REPLACE TEMP TABLE
                    v4_3_{prefix}_recent_{window}d_stats AS
                WITH reference_dates AS (
                    SELECT DISTINCT
                        {reference_columns},
                        source.SEFER_TARIHI AS reference_date
                    FROM v4_3_target AS source
                )
                SELECT
                    {grouped_columns},
                    reference.reference_date,
                    AVG(history.target)::DOUBLE
                        AS {prefix}_recent_{window}d_average,
                    COUNT(history.target)::BIGINT
                        AS {prefix}_recent_{window}d_count
                FROM reference_dates AS reference
                LEFT JOIN {history_table} AS history
                    ON {join_sql}
                    AND history.SEFER_TARIHI
                        >= reference.reference_date - INTERVAL '{window} days'
                    AND history.SEFER_TARIHI < reference.reference_date
                GROUP BY {grouped_columns}, reference.reference_date
            """)


def _long_term_expressions(alias: str, prefix: str) -> list[str]:
    expressions = [
        f"COALESCE({alias}.{prefix}_average, global_stats.global_average) AS {prefix}_average",
        f"COALESCE({alias}.{prefix}_median, global_stats.global_median) AS {prefix}_median",
        f"COALESCE({alias}.{prefix}_std, global_stats.global_std) AS {prefix}_std",
        f"COALESCE({alias}.{prefix}_maximum, global_stats.global_maximum) AS {prefix}_maximum",
        f"COALESCE({alias}.{prefix}_p90, global_stats.global_p90) AS {prefix}_p90",
        f"COALESCE({alias}.{prefix}_above_60_rate, global_stats.global_above_60_rate) AS {prefix}_above_60_rate",
        f"COALESCE({alias}.{prefix}_above_100_rate, global_stats.global_above_100_rate) AS {prefix}_above_100_rate",
        f"COALESCE({alias}.{prefix}_count, 0)::BIGINT AS {prefix}_count",
    ]
    if prefix == DISTRIBUTION_PREFIX:
        expressions.extend(
            [
                f"COALESCE({alias}.{prefix}_p10, global_stats.global_p10) AS {prefix}_p10",
                f"COALESCE({alias}.{prefix}_p25, global_stats.global_p25) AS {prefix}_p25",
                f"COALESCE({alias}.{prefix}_p75, global_stats.global_p75) AS {prefix}_p75",
                f"COALESCE({alias}.{prefix}_below_10_rate, global_stats.global_below_10_rate) AS {prefix}_below_10_rate",
                f"COALESCE({alias}.{prefix}_above_10_rate, global_stats.global_above_10_rate) AS {prefix}_above_10_rate",
                f"COALESCE({alias}.{prefix}_above_20_rate, global_stats.global_above_20_rate) AS {prefix}_above_20_rate",
                f"COALESCE({alias}.{prefix}_above_30_rate, global_stats.global_above_30_rate) AS {prefix}_above_30_rate",
                f"COALESCE({alias}.{prefix}_above_40_rate, global_stats.global_above_40_rate) AS {prefix}_above_40_rate",
            ]
        )
    return expressions


def create_feature_table(conn: duckdb.DuckDBPyConnection) -> None:
    select_expressions = [
        "base.SEFER_ID",
        "base.SEFER_TARIHI",
        "base.FIRMA_ID",
        "base.GUZERGAH_KODU",
        "base.canonical_guzergah_id",
        "base.target",
        "base.month",
        "base.week_of_year",
        "base.day_of_week",
        "base.departure_minute",
        "base.departure_30min_bucket",
    ]
    joins: list[str] = []
    aliases: dict[str, str] = {}
    for index, (prefix, group_columns) in enumerate(HISTORICAL_GROUP_DEFINITIONS):
        alias = f"long_term_{index}"
        aliases[prefix] = alias
        joins.append(
            f"LEFT JOIN v4_3_{prefix}_stats AS {alias} "
            f"ON {_join_condition('base', alias, group_columns)}"
        )
        select_expressions.extend(_long_term_expressions(alias, prefix))

    recent_index = len(HISTORICAL_GROUP_DEFINITIONS)
    for prefix, group_columns in RECENT_GROUP_DEFINITIONS:
        fallback = (
            f"COALESCE({aliases[prefix]}.{prefix}_average, "
            "global_stats.global_average)"
        )
        for window in RECENT_WINDOWS:
            alias = f"recent_{recent_index}"
            join = _join_condition("base", alias, group_columns)
            join += f" AND base.SEFER_TARIHI = {alias}.reference_date"
            joins.append(
                f"LEFT JOIN v4_3_{prefix}_recent_{window}d_stats AS {alias} "
                f"ON {join}"
            )
            select_expressions.extend(
                [
                    f"COALESCE({alias}.{prefix}_recent_{window}d_average, {fallback}) "
                    f"AS {prefix}_recent_{window}d_average",
                    f"COALESCE({alias}.{prefix}_recent_{window}d_count, 0)::BIGINT "
                    f"AS {prefix}_recent_{window}d_count",
                ]
            )
            recent_index += 1

    # The selected weekday baseline remains identical to v3/v4.1/v4.2.
    specific = aliases["company_route_time_weekday"]
    canonical_time = aliases["canonical_route_time_weekday"]
    canonical = aliases["canonical_route"]
    select_expressions.extend(
        [
            f"COALESCE({specific}.company_route_time_weekday_average, "
            f"{canonical_time}.canonical_route_time_weekday_average, "
            f"{canonical}.canonical_route_average, global_stats.global_average) "
            "AS baseline_prediction",
            f"CASE WHEN {specific}.company_route_time_weekday_average IS NOT NULL "
            "THEN 'company_route_time_weekday' "
            f"WHEN {canonical_time}.canonical_route_time_weekday_average IS NOT NULL "
            "THEN 'canonical_route_time_weekday' "
            f"WHEN {canonical}.canonical_route_average IS NOT NULL "
            "THEN 'canonical_route' ELSE 'overall_average' END AS baseline_source",
        ]
    )
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE v4_3_features AS
        SELECT {', '.join(select_expressions)}
        FROM v4_3_target AS base
        CROSS JOIN v4_3_global_stats AS global_stats
        {' '.join(joins)}
    """)


def build_evaluation_matrix(config: PeriodConfig) -> pd.DataFrame:
    ensure_environment(require_model=True)
    conn = connect()
    try:
        create_source_tables(conn, config, training=False)
        create_long_term_statistics(conn)
        create_recent_statistics(conn, config, history_table="v4_3_history")
        create_feature_table(conn)
        columns = [
            "SEFER_ID",
            "SEFER_TARIHI",
            *FEATURE_COLUMNS,
            "target",
            "baseline_prediction",
            "baseline_source",
        ]
        frame = conn.execute(
            f"SELECT {sql_identifier_list(columns)} FROM v4_3_features"
        ).fetchdf()
    finally:
        conn.close()
    validate_feature_frame(frame)
    return frame


def validate_feature_frame(frame: pd.DataFrame) -> None:
    if len(FEATURE_COLUMNS) != 72 or len(set(FEATURE_COLUMNS)) != 72:
        raise RuntimeError("v4.3 must have exactly 72 unique feature columns.")
    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame]
    if missing_columns:
        raise RuntimeError(f"Missing v4.3 feature columns: {missing_columns}")
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("string").fillna("missing")
    missing = frame[FEATURE_COLUMNS].isna().sum()
    if int(missing.sum()) != 0:
        raise RuntimeError(
            "v4.3 feature matrix contains missing values:\n"
            + missing[missing > 0].to_string()
        )
    rate_columns = [
        column for column in FEATURE_COLUMNS
        if column.endswith("_rate")
    ]
    invalid_rates = {
        column: int((~frame[column].between(0.0, 1.0)).sum())
        for column in rate_columns
        if not frame[column].between(0.0, 1.0).all()
    }
    if invalid_rates:
        raise RuntimeError(f"Historical rates outside [0, 1]: {invalid_rates}")


def load_v4_3_model() -> CatBoostRegressor:
    ensure_environment(require_model=True)
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    if list(model.feature_names_) != FEATURE_COLUMNS:
        raise RuntimeError("Saved v4.3 model does not match the 72-feature contract.")
    return model


def _load_frozen_rule() -> dict[str, Any]:
    payload = json.loads(FROZEN_RULE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "catboost_v4_2_hybrid_rule_v1":
        raise RuntimeError("Unexpected frozen v4.2 rule schema.")
    if payload.get("frozen_on_period") != "validation_2025_h1":
        raise RuntimeError("v4.2 rule is not the validation-frozen rule.")
    return payload["rule"]


def apply_frozen_hybrid_to_v4_3(frame: pd.DataFrame) -> np.ndarray:
    """Apply the exact frozen v4.2 rule with v4.3 as the underlying model."""

    rule = _load_frozen_rule()
    prediction = frame["v4_3_prediction"].to_numpy(dtype=np.float64)
    baseline = frame["baseline_prediction"].to_numpy(dtype=np.float64)
    prefix = DISTRIBUTION_PREFIX
    supported = frame[f"{prefix}_count"].to_numpy() >= rule["minimum_count"]
    upward_only = baseline > prediction
    moderate = (
        (frame[f"{prefix}_p90"].to_numpy() >= rule["p90_threshold"])
        | (frame[f"{prefix}_above_60_rate"].to_numpy() >= rule["above_60_rate_threshold"])
    )
    above_100 = frame[f"{prefix}_above_100_rate"].to_numpy()
    strong_above_100 = (
        above_100 > 0.0
        if rule["above_100_rate_threshold"] == 0.0
        else above_100 >= rule["above_100_rate_threshold"]
    )
    strong = (
        (frame[f"{prefix}_p90"].to_numpy() >= rule["strong_p90_threshold"])
        | strong_above_100
    )
    eligible = supported & upward_only
    weights = np.zeros(len(frame), dtype=np.float64)
    weights[eligible & moderate & ~strong] = rule["moderate_weight"]
    weights[eligible & strong] = rule["strong_weight"]
    return prediction + weights * (baseline - prediction)


def merge_frozen_comparisons(frame: pd.DataFrame, config: PeriodConfig) -> pd.DataFrame:
    path = RESULTS_DIR / f"catboost_v4_2_{config.key}_predictions.csv"
    required = [
        "SEFER_ID",
        "actual",
        "v3_prediction",
        "v4_1_prediction",
        "hybrid_prediction",
    ]
    frozen = pd.read_csv(path, usecols=required).rename(
        columns={
            "actual": "frozen_actual",
            "hybrid_prediction": "v4_2_hybrid_prediction",
        }
    )
    if frozen["SEFER_ID"].duplicated().any() or frame["SEFER_ID"].duplicated().any():
        raise RuntimeError("SEFER_ID must be unique for comparison alignment.")
    merged = frame.merge(frozen, on="SEFER_ID", how="left", validate="one_to_one")
    comparison_columns = [
        "v3_prediction",
        "v4_1_prediction",
        "v4_2_hybrid_prediction",
    ]
    if merged[comparison_columns].isna().any().any():
        raise RuntimeError("Frozen comparison predictions did not align to v4.3 rows.")
    if not np.array_equal(
        merged["target"].to_numpy(), merged["frozen_actual"].to_numpy()
    ):
        raise RuntimeError("Actual targets disagree with frozen v4.2 artifacts.")
    return merged.drop(columns="frozen_actual")


def _regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def business_metrics(frame: pd.DataFrame, config: PeriodConfig) -> pd.DataFrame:
    """Return metrics in the user-specified decision order for all six models."""

    actual = frame["target"].to_numpy(dtype=np.float64)
    focus = (actual >= 10) & (actual <= 40)
    above_40 = actual >= 40
    records: list[dict[str, Any]] = []
    for model_name, column in MODEL_COMPARISON_COLUMNS.items():
        prediction = frame[column].to_numpy(dtype=np.float64)
        rounded = rounded_nonnegative_predictions(prediction)
        overall = _regression_metrics(actual, prediction)
        record: dict[str, Any] = {
            "period": config.key,
            "tuning_allowed": config.tuning_allowed,
            "model": model_name,
            "row_count": len(frame),
            "mae_actual_10_to_40": _regression_metrics(
                actual[focus], prediction[focus]
            )["mae"],
        }
        for threshold in BUSINESS_THRESHOLDS:
            binary = calculate_binary_classification_metrics(
                actual,
                rounded,
                threshold,
                predictions_are_rounded=True,
            )
            record[f"fnr_at_{threshold}"] = binary["false_negative_rate"]
            record[f"fpr_at_{threshold}"] = binary["false_positive_rate"]
            lower, upper = NEAR_THRESHOLD_NEGATIVE_BANDS[threshold]
            near_negative = (actual >= lower) & (actual < upper)
            record[f"near_threshold_fpr_at_{threshold}"] = float(
                np.mean(rounded[near_negative] >= threshold)
            )
        record.update(
            {
                "overall_mae": overall["mae"],
                "overall_rmse": overall["rmse"],
                "overall_bias": overall["bias"],
                "rmse_actual_40_plus": _regression_metrics(
                    actual[above_40], prediction[above_40]
                )["rmse"],
            }
        )
        records.append(record)
    metrics = pd.DataFrame(records)
    reference = metrics.loc[metrics["model"] == "v4.2 hybrid"].iloc[0]
    for metric in [
        "mae_actual_10_to_40",
        "fnr_at_20",
        "fnr_at_30",
        "fnr_at_43",
        "overall_mae",
        "rmse_actual_40_plus",
        "near_threshold_fpr_at_20",
        "near_threshold_fpr_at_30",
        "near_threshold_fpr_at_43",
    ]:
        metrics[f"delta_vs_v4_2__{metric}"] = metrics[metric] - reference[metric]
    return metrics


def write_comparison_outputs(frame: pd.DataFrame, config: PeriodConfig) -> None:
    stem = RESULTS_DIR / f"catboost_v4_3_{config.key}"
    metrics_path = Path(f"{stem}_business_metrics.csv")
    predictions_path = Path(f"{stem}_predictions.csv")
    metadata_path = Path(f"{stem}_metadata.json")
    for path in (metrics_path, predictions_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite experiment artifact: {path}")
    metrics = business_metrics(frame, config)
    prediction_columns = [
        "SEFER_ID",
        "SEFER_TARIHI",
        "target",
        *MODEL_COMPARISON_COLUMNS.values(),
        "baseline_source",
    ]
    metrics.to_csv(metrics_path, index=False)
    frame[prediction_columns].to_csv(predictions_path, index=False)
    metadata = {
        "schema_version": "catboost_v4_3_business_evaluation_v1",
        "period": config.key,
        "history_start": config.history_start,
        "history_end_exclusive": config.history_end_exclusive,
        "target_start": config.target_start,
        "target_end_exclusive": config.target_end_exclusive,
        "tuning_allowed": config.tuning_allowed,
        "primary_metric": "mae_actual_10_to_40",
        "selection_order": [
            "mae_actual_10_to_40",
            "fnr_at_20/fnr_at_30/fnr_at_43",
            "overall_mae",
            "rmse_actual_40_plus",
            "near_threshold_fpr_at_20/30/43",
        ],
        "near_threshold_negative_bands": {
            str(key): list(value)
            for key, value in NEAR_THRESHOLD_NEGATIVE_BANDS.items()
        },
        "profitability_claimed": False,
        "frozen_v4_2_rule": str(FROZEN_RULE_PATH.relative_to(PROJECT_DIR)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    display_columns = [
        "model",
        "mae_actual_10_to_40",
        "fnr_at_20",
        "fnr_at_30",
        "fnr_at_43",
        "overall_mae",
        "rmse_actual_40_plus",
        "near_threshold_fpr_at_20",
        "near_threshold_fpr_at_30",
        "near_threshold_fpr_at_43",
    ]
    print(f"\n{config.label} business-ordered comparison")
    print(metrics[display_columns].to_string(index=False))
