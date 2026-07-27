"""Leakage-safe 72-feature contract and DuckDB helpers for v4.x inference.

Extracted verbatim from ``scripts/catboost_v4_3_common.py``.  This module is the
self-contained inference-side feature contract: it exposes the constants and
DuckDB plumbing required by the proposed-trip pipeline without importing any
training-only evaluation or artifact-writing code.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_DIR / "analysis.duckdb"
TEMP_DIR = PROJECT_DIR / "duckdb_temp"

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


def sql_identifier_list(columns: list[str]) -> str:
    return ", ".join(columns)


def connect() -> duckdb.DuckDBPyConnection:
    # Inference only creates connection-local TEMP tables. Keeping this
    # read-only also allows it to coexist with API lookup connections, since
    # DuckDB rejects mixed read-only/read-write connections to the same file
    # within one process.
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    temp_path = TEMP_DIR.as_posix().replace("'", "''")
    connection.execute(f"SET threads = {DUCKDB_THREADS}")
    connection.execute(f"SET temp_directory = '{temp_path}'")
    return connection


def _join_condition(base_alias: str, aggregate_alias: str, columns: list[str]) -> str:
    return " AND ".join(
        f"{base_alias}.{column} = {aggregate_alias}.{column}"
        for column in columns
    )


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
