
from __future__ import annotations

"""
Error analysis for the frozen CatBoost v3 MAE model.

This script:
- evaluates 2025 H2 and the already-used 2026 final period;
- rebuilds all v3 historical features in DuckDB without leakage;
- rebuilds the fair weekday baseline for each period;
- verifies the saved model's exact 48-feature contract;
- exports detailed CSV diagnostics, charts, and a Markdown summary;
- never trains or overwrites a model.
"""

from dataclasses import dataclass
from pathlib import Path
import gc
import math
import os
import shutil
from typing import Iterable, Sequence

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


# ============================================================
# Configuration
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_DIR / "analysis.duckdb"
MODEL_PATH = PROJECT_DIR / "models" / "catboost_demand_model_v3_mae_6000.cbm"
TEMP_DIR = PROJECT_DIR / "duckdb_temp"
OUTPUT_DIR = PROJECT_DIR / "error_analysis_v3"

DUCKDB_THREADS = max(1, os.cpu_count() or 4)

MIN_COMPANY_ROWS = 1_000
MIN_ROUTE_ROWS = 500
MIN_COMPANY_ROUTE_ROWS = 250
TOP_ERROR_ROWS = 1_000
TOP_PLOT_GROUPS = 20

TIE_TOLERANCE = 1e-9
METRIC_WARNING_TOLERANCE = 0.02

EXPORT_ALL_ROW_LEVEL_PREDICTIONS = False
CLEAR_PREVIOUS_OUTPUT = True


@dataclass(frozen=True)
class EvaluationPeriod:
    key: str
    label: str
    history_start: str
    history_end_exclusive: str
    target_start: str
    target_end_exclusive: str
    expected_rows: int
    expected_catboost_mae: float
    expected_catboost_rmse: float
    expected_baseline_mae: float
    expected_baseline_rmse: float


PERIODS = (
    EvaluationPeriod(
        key="test_2025_h2",
        label="2025 H2 test",
        history_start="2023-01-01",
        history_end_exclusive="2025-07-01",
        target_start="2025-07-01",
        target_end_exclusive="2026-01-01",
        expected_rows=1_517_010,
        expected_catboost_mae=10.165391574903602,
        expected_catboost_rmse=17.43281807791355,
        expected_baseline_mae=10.244093697542297,
        expected_baseline_rmse=16.999945314248993,
    ),
    EvaluationPeriod(
        key="final_2026",
        label="2026 official final evaluation",
        history_start="2023-01-01",
        history_end_exclusive="2026-01-01",
        target_start="2026-01-01",
        target_end_exclusive="2026-04-15",
        expected_rows=780_865,
        expected_catboost_mae=9.770396935895032,
        expected_catboost_rmse=13.851110926842543,
        expected_baseline_mae=9.86210681757341,
        expected_baseline_rmse=14.361139756955968,
    ),
)


# ============================================================
# Frozen v3 feature contract
# ============================================================

CATEGORICAL_FEATURES = [
    "FIRMA_ID",
    "GUZERGAH_KODU",
    "canonical_guzergah_id",
    "month",
    "day_of_week",
    "departure_30min_bucket",
]

CALENDAR_NUMERIC_FEATURES = [
    "week_of_year",
    "departure_minute",
]

HISTORICAL_GROUP_DEFINITIONS: list[tuple[str, list[str]]] = [
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
        [
            "FIRMA_ID",
            "canonical_guzergah_id",
            "departure_30min_bucket",
        ],
    ),
    (
        "canonical_route_time_weekday",
        [
            "canonical_guzergah_id",
            "departure_30min_bucket",
            "day_of_week",
        ],
    ),
    (
        "canonical_route",
        ["canonical_guzergah_id"],
    ),
    (
        "company",
        ["FIRMA_ID"],
    ),
]

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

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + CALENDAR_NUMERIC_FEATURES
    + HISTORICAL_FEATURES
)

REQUIRED_MODEL_DATA_COLUMNS = {
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
}

OPTIONAL_METADATA_COLUMNS = [
    "SEFER_ID",
    "SEFER_SAATI",
    "departure_hour",
]


# ============================================================
# Analysis bands
# ============================================================

TARGET_BINS = [0, 10, 20, 30, 40, 60, 100, 300]
TARGET_LABELS = [
    "1-10",
    "11-20",
    "21-30",
    "31-40",
    "41-60",
    "61-100",
    "101-300",
]

SUPPORT_BINS = [-np.inf, 0, 5, 20, 50, 100, 500, 1000, np.inf]
SUPPORT_LABELS = [
    "0",
    "1-5",
    "6-20",
    "21-50",
    "51-100",
    "101-500",
    "501-1000",
    "1001+",
]

ERROR_BINS = [-np.inf, 2, 5, 10, 20, 40, np.inf]
ERROR_LABELS = [
    "0-2",
    "2-5",
    "5-10",
    "10-20",
    "20-40",
    "40+",
]

SUPPORT_FEATURES = [
    "company_route_time_weekday_count",
    "canonical_route_time_weekday_count",
    "canonical_route_count",
    "company_count",
]

HIGH_DEMAND_SIGNAL_FEATURES = [
    "company_route_time_weekday_p90",
    "company_route_time_weekday_above_60_rate",
    "company_route_time_weekday_above_100_rate",
    "canonical_route_p90",
    "canonical_route_above_60_rate",
    "canonical_route_above_100_rate",
]


# ============================================================
# General helpers
# ============================================================

def ensure_paths() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"CatBoost model not found: {MODEL_PATH}")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if CLEAR_PREVIOUS_OUTPUT and OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "comparison").mkdir(parents=True, exist_ok=True)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def sql_identifier_list(columns: Iterable[str]) -> str:
    return ", ".join(columns)


def percentage(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return float("nan")
    return float((numerator / denominator) * 100.0)


def check_close(label: str, actual: float, expected: float) -> None:
    difference = abs(actual - expected)
    if difference > METRIC_WARNING_TOLERANCE:
        print(
            f"WARNING: {label} differs from the recorded result by "
            f"{difference:.6f}. Actual={actual:.12f}, "
            f"recorded={expected:.12f}"
        )


def load_model() -> CatBoostRegressor:
    print(f"Loading frozen model: {MODEL_PATH}")

    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))

    saved_feature_names = list(model.feature_names_)

    print(f"Saved feature count: {len(saved_feature_names)}")
    for index, feature_name in enumerate(saved_feature_names, start=1):
        print(f"  {index:02d}. {feature_name}")

    if saved_feature_names != FEATURE_COLUMNS:
        raise RuntimeError(
            "Frozen model feature names/order do not match the expected "
            "v3 contract.\n"
            f"Expected: {FEATURE_COLUMNS}\n"
            f"Saved: {saved_feature_names}"
        )

    return model


def inspect_database(
    conn: duckdb.DuckDBPyConnection,
) -> list[str]:
    tables = {
        row[0]
        for row in conn.execute("SHOW TABLES").fetchall()
    }

    if "model_data_base" not in tables:
        raise RuntimeError(
            "Required DuckDB object 'model_data_base' was not found."
        )

    available_columns = {
        row[0]
        for row in conn.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'model_data_base'
        """).fetchall()
    }

    missing_columns = REQUIRED_MODEL_DATA_COLUMNS - available_columns
    if missing_columns:
        raise RuntimeError(
            "model_data_base is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    optional_columns = [
        column
        for column in OPTIONAL_METADATA_COLUMNS
        if column in available_columns
    ]

    print("Optional metadata columns:", optional_columns or "none")
    return optional_columns


# ============================================================
# DuckDB feature construction
# ============================================================

def create_source_tables(
    conn: duckdb.DuckDBPyConnection,
    period: EvaluationPeriod,
    optional_columns: Sequence[str],
) -> tuple[int, int]:
    selected_columns = [
        *optional_columns,
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

    selected_sql = sql_identifier_list(selected_columns)
    history_table = f"history_{period.key}"
    target_table = f"target_{period.key}"

    print(f"\nCreating source tables for {period.label}...")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE {history_table} AS
        SELECT {selected_sql}
        FROM model_data_base
        WHERE SEFER_TARIHI >= DATE '{period.history_start}'
          AND SEFER_TARIHI < DATE '{period.history_end_exclusive}'
    """)

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE {target_table} AS
        SELECT {selected_sql}
        FROM model_data_base
        WHERE SEFER_TARIHI >= DATE '{period.target_start}'
          AND SEFER_TARIHI < DATE '{period.target_end_exclusive}'
    """)

    history_count = conn.execute(
        f"SELECT COUNT(*) FROM {history_table}"
    ).fetchone()[0]

    target_count = conn.execute(
        f"SELECT COUNT(*) FROM {target_table}"
    ).fetchone()[0]

    if history_count == 0:
        raise RuntimeError(f"{period.label}: historical data is empty.")

    if target_count == 0:
        raise RuntimeError(f"{period.label}: prediction data is empty.")

    print(f"Historical rows: {history_count:,}")
    print(f"Source prediction rows: {target_count:,}")

    if target_count != period.expected_rows:
        print(
            f"WARNING: recorded target count is {period.expected_rows:,}, "
            f"but the live database returned {target_count:,}."
        )

    return history_count, target_count


def create_global_statistics(
    conn: duckdb.DuckDBPyConnection,
    period: EvaluationPeriod,
) -> None:
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE global_stats_{period.key} AS
        SELECT
            AVG(target)::DOUBLE AS global_average,
            MEDIAN(target)::DOUBLE AS global_median,
            STDDEV_SAMP(target)::DOUBLE AS global_std,
            MAX(target)::DOUBLE AS global_maximum,
            QUANTILE_CONT(target, 0.90)::DOUBLE AS global_p90,
            AVG(
                CASE WHEN target > 60 THEN 1.0 ELSE 0.0 END
            )::DOUBLE AS global_above_60_rate,
            AVG(
                CASE WHEN target > 100 THEN 1.0 ELSE 0.0 END
            )::DOUBLE AS global_above_100_rate
        FROM history_{period.key}
    """)


def create_group_statistics(
    conn: duckdb.DuckDBPyConnection,
    period: EvaluationPeriod,
    prefix: str,
    group_columns: list[str],
) -> None:
    group_sql = sql_identifier_list(group_columns)

    print(f"  Aggregating {prefix}")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE {prefix}_{period.key} AS
        SELECT
            {group_sql},
            AVG(target)::DOUBLE AS {prefix}_average,
            MEDIAN(target)::DOUBLE AS {prefix}_median,
            STDDEV_SAMP(target)::DOUBLE AS {prefix}_std,
            MAX(target)::DOUBLE AS {prefix}_maximum,
            QUANTILE_CONT(target, 0.90)::DOUBLE AS {prefix}_p90,
            AVG(
                CASE WHEN target > 60 THEN 1.0 ELSE 0.0 END
            )::DOUBLE AS {prefix}_above_60_rate,
            AVG(
                CASE WHEN target > 100 THEN 1.0 ELSE 0.0 END
            )::DOUBLE AS {prefix}_above_100_rate,
            COUNT(*)::BIGINT AS {prefix}_count
        FROM history_{period.key}
        GROUP BY {group_sql}
    """)


def create_all_statistics(
    conn: duckdb.DuckDBPyConnection,
    period: EvaluationPeriod,
) -> None:
    print(f"Building historical statistics for {period.label}...")

    create_global_statistics(conn, period)

    for prefix, group_columns in HISTORICAL_GROUP_DEFINITIONS:
        create_group_statistics(
            conn,
            period,
            prefix,
            group_columns,
        )


def build_join_condition(
    base_alias: str,
    aggregate_alias: str,
    group_columns: list[str],
) -> str:
    return " AND ".join(
        f"{base_alias}.{column} = {aggregate_alias}.{column}"
        for column in group_columns
    )


def build_feature_expressions(
    aggregate_alias: str,
    prefix: str,
) -> list[str]:
    return [
        (
            f"COALESCE({aggregate_alias}.{prefix}_average, "
            f"global_stats.global_average) AS {prefix}_average"
        ),
        (
            f"COALESCE({aggregate_alias}.{prefix}_median, "
            f"global_stats.global_median) AS {prefix}_median"
        ),
        (
            f"COALESCE({aggregate_alias}.{prefix}_std, "
            f"global_stats.global_std) AS {prefix}_std"
        ),
        (
            f"COALESCE({aggregate_alias}.{prefix}_maximum, "
            f"global_stats.global_maximum) AS {prefix}_maximum"
        ),
        (
            f"COALESCE({aggregate_alias}.{prefix}_p90, "
            f"global_stats.global_p90) AS {prefix}_p90"
        ),
        (
            f"COALESCE({aggregate_alias}.{prefix}_above_60_rate, "
            f"global_stats.global_above_60_rate) "
            f"AS {prefix}_above_60_rate"
        ),
        (
            f"COALESCE({aggregate_alias}.{prefix}_above_100_rate, "
            f"global_stats.global_above_100_rate) "
            f"AS {prefix}_above_100_rate"
        ),
        (
            f"COALESCE({aggregate_alias}.{prefix}_count, 0)::BIGINT "
            f"AS {prefix}_count"
        ),
    ]


def create_feature_table(
    conn: duckdb.DuckDBPyConnection,
    period: EvaluationPeriod,
    optional_columns: Sequence[str],
) -> str:
    target_table = f"target_{period.key}"
    output_table = f"features_{period.key}"

    select_expressions = [
        *(f"base.{column}" for column in optional_columns),
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
        "(base.departure_minute // 60)::INTEGER AS derived_departure_hour",
    ]

    joins: list[str] = []
    aliases: dict[str, str] = {}

    for index, (prefix, group_columns) in enumerate(
        HISTORICAL_GROUP_DEFINITIONS
    ):
        alias = f"aggregate_{index}"
        aliases[prefix] = alias

        joins.append(f"""
            LEFT JOIN {prefix}_{period.key} AS {alias}
                ON {
                    build_join_condition(
                        "base",
                        alias,
                        group_columns,
                    )
                }
        """)

        select_expressions.extend(
            build_feature_expressions(alias, prefix)
        )

    exact_alias = aliases["company_route_time_weekday"]
    canonical_time_alias = aliases["canonical_route_time_weekday"]
    canonical_route_alias = aliases["canonical_route"]

    select_expressions.append(f"""
        COALESCE(
            {exact_alias}.company_route_time_weekday_average,
            {canonical_time_alias}.canonical_route_time_weekday_average,
            {canonical_route_alias}.canonical_route_average,
            global_stats.global_average
        )::DOUBLE AS baseline_prediction
    """)

    select_expressions.append(f"""
        CASE
            WHEN {exact_alias}.company_route_time_weekday_average
                IS NOT NULL
                THEN 'company_route_time_weekday'
            WHEN {canonical_time_alias}.canonical_route_time_weekday_average
                IS NOT NULL
                THEN 'canonical_route_time_weekday'
            WHEN {canonical_route_alias}.canonical_route_average
                IS NOT NULL
                THEN 'canonical_route'
            ELSE 'overall_average'
        END AS baseline_source
    """)

    select_sql = ",\n            ".join(select_expressions)
    joins_sql = "\n".join(joins)

    print(f"Joining completed matrix for {period.label}...")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE {output_table} AS
        SELECT
            {select_sql}
        FROM {target_table} AS base
        CROSS JOIN global_stats_{period.key} AS global_stats
        {joins_sql}
    """)

    source_count = conn.execute(
        f"SELECT COUNT(*) FROM {target_table}"
    ).fetchone()[0]

    feature_count = conn.execute(
        f"SELECT COUNT(*) FROM {output_table}"
    ).fetchone()[0]

    print(f"Completed feature rows: {feature_count:,}")

    if source_count != feature_count:
        raise RuntimeError(
            f"{period.label}: feature joins changed the row count from "
            f"{source_count:,} to {feature_count:,}."
        )

    null_target_count, null_baseline_count = conn.execute(f"""
        SELECT
            COUNT(*) FILTER (WHERE target IS NULL),
            COUNT(*) FILTER (WHERE baseline_prediction IS NULL)
        FROM {output_table}
    """).fetchone()

    if null_target_count:
        raise RuntimeError(
            f"{period.label}: target contains null values."
        )

    if null_baseline_count:
        raise RuntimeError(
            f"{period.label}: baseline predictions contain null values."
        )

    return output_table


def check_granularity(
    conn: duckdb.DuckDBPyConnection,
    period: EvaluationPeriod,
    optional_columns: Sequence[str],
) -> None:
    if "SEFER_ID" not in optional_columns:
        print(
            "SEFER_ID is unavailable; relying on strict row-count "
            "preservation checks."
        )
        return

    total_rows, distinct_ids = conn.execute(f"""
        SELECT
            COUNT(*),
            COUNT(DISTINCT SEFER_ID)
        FROM target_{period.key}
    """).fetchone()

    if total_rows == distinct_ids:
        print("SEFER_ID is unique in this period.")
        return

    duplicate_groups = conn.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT SEFER_ID
            FROM target_{period.key}
            GROUP BY SEFER_ID
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    print(
        f"NOTICE: SEFER_ID alone is not unique. "
        f"Duplicate ID groups: {duplicate_groups:,}. "
        "Strict row-count preservation remains the join-safety check."
    )


def fetch_feature_matrix(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
) -> pd.DataFrame:
    available_columns = [
        row[0]
        for row in conn.execute(
            f"DESCRIBE {table_name}"
        ).fetchall()
    ]

    desired_columns = [
        "SEFER_ID",
        "SEFER_SAATI",
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
        "departure_hour",
        "derived_departure_hour",
        *HISTORICAL_FEATURES,
        "baseline_prediction",
        "baseline_source",
    ]

    fetch_columns = [
        column
        for column in desired_columns
        if column in available_columns
    ]

    print("Fetching completed feature matrix into Pandas...")

    frame = conn.execute(
        f"SELECT {sql_identifier_list(fetch_columns)} "
        f"FROM {table_name}"
    ).fetchdf()

    if "departure_hour" not in frame.columns:
        frame["departure_hour"] = frame["derived_departure_hour"]

    frame.drop(
        columns=["derived_departure_hour"],
        inplace=True,
        errors="ignore",
    )

    return frame


# ============================================================
# Predictions and row-level error columns
# ============================================================

def add_predictions(
    model: CatBoostRegressor,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    missing_features = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in frame.columns
    ]

    if missing_features:
        raise RuntimeError(
            "Completed feature matrix is missing: "
            + ", ".join(missing_features)
        )

    for column in CATEGORICAL_FEATURES:
        frame[column] = (
            frame[column]
            .astype("string")
            .fillna("missing")
        )

    print(f"Predicting {len(frame):,} rows...")

    predictions = model.predict(frame[FEATURE_COLUMNS])

    if np.isnan(predictions).any():
        raise RuntimeError("CatBoost predictions contain NaN values.")

    frame["catboost_prediction"] = predictions.astype("float64")

    frame["catboost_error"] = (
        frame["catboost_prediction"] - frame["target"]
    )
    frame["catboost_absolute_error"] = (
        frame["catboost_error"].abs()
    )
    frame["catboost_squared_error"] = (
        frame["catboost_error"] ** 2
    )
    frame["catboost_bias"] = frame["catboost_error"]

    frame["baseline_error"] = (
        frame["baseline_prediction"] - frame["target"]
    )
    frame["baseline_absolute_error"] = (
        frame["baseline_error"].abs()
    )
    frame["baseline_squared_error"] = (
        frame["baseline_error"] ** 2
    )
    frame["baseline_bias"] = frame["baseline_error"]

    frame["catboost_minus_baseline_absolute_error"] = (
        frame["catboost_absolute_error"]
        - frame["baseline_absolute_error"]
    )

    difference = frame[
        "catboost_minus_baseline_absolute_error"
    ]

    frame["winner"] = np.select(
        [
            difference < -TIE_TOLERANCE,
            difference > TIE_TOLERANCE,
        ],
        [
            "catboost",
            "baseline",
        ],
        default="tie",
    )

    frame["target_group"] = pd.cut(
        frame["target"],
        bins=TARGET_BINS,
        labels=TARGET_LABELS,
        include_lowest=True,
        right=True,
    )

    add_readable_time_labels(frame)
    return frame


def add_readable_time_labels(frame: pd.DataFrame) -> None:
    dates = pd.to_datetime(frame["SEFER_TARIHI"])
    python_weekdays = dates.dt.dayofweek

    mapping = (
        pd.DataFrame(
            {
                "day_of_week": frame["day_of_week"],
                "python_weekday": python_weekdays,
            }
        )
        .drop_duplicates()
        .sort_values("day_of_week")
    )

    if (
        mapping.groupby("day_of_week")["python_weekday"]
        .nunique()
        .max()
        != 1
    ):
        raise RuntimeError(
            "Could not determine a stable weekday-number mapping."
        )

    weekday_names = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }

    number_to_name = {
        row.day_of_week: weekday_names[int(row.python_weekday)]
        for row in mapping.itertuples(index=False)
    }

    frame["weekday_label"] = (
        frame["day_of_week"].map(number_to_name)
    )

    start_minutes = (
        frame["departure_30min_bucket"]
        .astype("int64")
        * 30
    )
    end_minutes = start_minutes + 29

    frame["departure_bucket_label"] = [
        (
            f"{start // 60:02d}:{start % 60:02d}-"
            f"{end // 60:02d}:{end % 60:02d}"
        )
        for start, end in zip(start_minutes, end_minutes)
    ]


# ============================================================
# Metrics
# ============================================================

def overall_metrics(
    frame: pd.DataFrame,
    period: EvaluationPeriod,
) -> pd.DataFrame:
    catboost_mae = float(
        frame["catboost_absolute_error"].mean()
    )
    baseline_mae = float(
        frame["baseline_absolute_error"].mean()
    )
    catboost_rmse = math.sqrt(
        float(frame["catboost_squared_error"].mean())
    )
    baseline_rmse = math.sqrt(
        float(frame["baseline_squared_error"].mean())
    )

    mae_improvement = baseline_mae - catboost_mae
    rmse_improvement = baseline_rmse - catboost_rmse

    winner_rates = (
        frame["winner"].value_counts(normalize=True)
    )

    result = pd.DataFrame(
        [
            {
                "period": period.key,
                "period_label": period.label,
                "row_count": len(frame),
                "average_actual": frame["target"].mean(),
                "average_catboost_prediction": frame[
                    "catboost_prediction"
                ].mean(),
                "average_baseline_prediction": frame[
                    "baseline_prediction"
                ].mean(),
                "catboost_mae": catboost_mae,
                "baseline_mae": baseline_mae,
                "mae_improvement": mae_improvement,
                "mae_improvement_percentage": percentage(
                    mae_improvement,
                    baseline_mae,
                ),
                "catboost_rmse": catboost_rmse,
                "baseline_rmse": baseline_rmse,
                "rmse_improvement": rmse_improvement,
                "average_catboost_bias": frame[
                    "catboost_bias"
                ].mean(),
                "average_baseline_bias": frame[
                    "baseline_bias"
                ].mean(),
                "catboost_win_rate": winner_rates.get(
                    "catboost",
                    0.0,
                ),
                "baseline_win_rate": winner_rates.get(
                    "baseline",
                    0.0,
                ),
                "tie_rate": winner_rates.get("tie", 0.0),
            }
        ]
    )

    check_close(
        f"{period.label} CatBoost MAE",
        catboost_mae,
        period.expected_catboost_mae,
    )
    check_close(
        f"{period.label} CatBoost RMSE",
        catboost_rmse,
        period.expected_catboost_rmse,
    )
    check_close(
        f"{period.label} baseline MAE",
        baseline_mae,
        period.expected_baseline_mae,
    )
    check_close(
        f"{period.label} baseline RMSE",
        baseline_rmse,
        period.expected_baseline_rmse,
    )

    return result


def grouped_metrics(
    frame: pd.DataFrame,
    group_columns: str | Sequence[str],
    *,
    include_percentage: bool = False,
) -> pd.DataFrame:
    if isinstance(group_columns, str):
        group_columns = [group_columns]

    result = (
        frame.groupby(
            list(group_columns),
            observed=True,
            dropna=False,
        )
        .agg(
            row_count=("target", "size"),
            average_actual=("target", "mean"),
            average_catboost_prediction=(
                "catboost_prediction",
                "mean",
            ),
            average_baseline_prediction=(
                "baseline_prediction",
                "mean",
            ),
            catboost_mae=(
                "catboost_absolute_error",
                "mean",
            ),
            baseline_mae=(
                "baseline_absolute_error",
                "mean",
            ),
            catboost_mse=(
                "catboost_squared_error",
                "mean",
            ),
            baseline_mse=(
                "baseline_squared_error",
                "mean",
            ),
            catboost_bias=("catboost_bias", "mean"),
            baseline_bias=("baseline_bias", "mean"),
            catboost_wins=(
                "winner",
                lambda values: int(
                    (values == "catboost").sum()
                ),
            ),
            baseline_wins=(
                "winner",
                lambda values: int(
                    (values == "baseline").sum()
                ),
            ),
            ties=(
                "winner",
                lambda values: int(
                    (values == "tie").sum()
                ),
            ),
        )
        .reset_index()
    )

    result["mae_improvement"] = (
        result["baseline_mae"] - result["catboost_mae"]
    )

    result["catboost_rmse"] = np.sqrt(
        result["catboost_mse"]
    )
    result["baseline_rmse"] = np.sqrt(
        result["baseline_mse"]
    )
    result["rmse_improvement"] = (
        result["baseline_rmse"] - result["catboost_rmse"]
    )

    result["catboost_win_rate"] = (
        result["catboost_wins"] / result["row_count"]
    )
    result["baseline_win_rate"] = (
        result["baseline_wins"] / result["row_count"]
    )
    result["tie_rate"] = (
        result["ties"] / result["row_count"]
    )

    if include_percentage:
        result["row_percentage"] = (
            result["row_count"] / len(frame) * 100.0
        )

    result.drop(
        columns=[
            "catboost_mse",
            "baseline_mse",
            "catboost_wins",
            "baseline_wins",
            "ties",
        ],
        inplace=True,
    )

    ordered_columns = [
        *group_columns,
        "row_count",
    ]

    if include_percentage:
        ordered_columns.append("row_percentage")

    ordered_columns.extend(
        [
            "average_actual",
            "average_catboost_prediction",
            "average_baseline_prediction",
            "catboost_mae",
            "baseline_mae",
            "mae_improvement",
            "catboost_rmse",
            "baseline_rmse",
            "rmse_improvement",
            "catboost_bias",
            "baseline_bias",
            "catboost_win_rate",
            "baseline_win_rate",
            "tie_rate",
        ]
    )

    return result[ordered_columns]


def high_demand_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for threshold in (60, 100):
        subset = frame[frame["target"] > threshold]

        rows.append(
            {
                "threshold": f"target > {threshold}",
                "row_count": len(subset),
                "actual_average": subset["target"].mean(),
                "catboost_prediction_average": subset[
                    "catboost_prediction"
                ].mean(),
                "baseline_prediction_average": subset[
                    "baseline_prediction"
                ].mean(),
                "catboost_mae": subset[
                    "catboost_absolute_error"
                ].mean(),
                "baseline_mae": subset[
                    "baseline_absolute_error"
                ].mean(),
                "catboost_rmse": math.sqrt(
                    subset["catboost_squared_error"].mean()
                ),
                "baseline_rmse": math.sqrt(
                    subset["baseline_squared_error"].mean()
                ),
                "catboost_bias": subset[
                    "catboost_bias"
                ].mean(),
                "baseline_bias": subset[
                    "baseline_bias"
                ].mean(),
            }
        )

    return pd.DataFrame(rows)


def support_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    outputs = []

    for feature in SUPPORT_FEATURES:
        if feature not in frame.columns:
            print(
                f"WARNING: support feature unavailable: {feature}"
            )
            continue

        temporary = frame.copy()
        temporary["support_band"] = pd.cut(
            temporary[feature],
            bins=SUPPORT_BINS,
            labels=SUPPORT_LABELS,
            include_lowest=True,
            right=True,
        )

        metrics = grouped_metrics(
            temporary,
            "support_band",
            include_percentage=True,
        )
        metrics.insert(0, "support_feature", feature)
        outputs.append(metrics)

        del temporary

    if not outputs:
        return pd.DataFrame()

    return pd.concat(outputs, ignore_index=True)


def volatility_metrics(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature = "company_route_time_weekday_std"

    if feature not in frame.columns:
        return pd.DataFrame(), pd.DataFrame()

    quantile_series = frame[feature].quantile(
        [0.0, 0.25, 0.5, 0.75, 1.0]
    )

    boundaries = np.unique(
        quantile_series.to_numpy()
    )

    boundary_frame = pd.DataFrame(
        {
            "quantile": quantile_series.index,
            "boundary": quantile_series.values,
        }
    )

    temporary = frame.copy()

    if len(boundaries) < 3:
        temporary["volatility_band"] = "single_band"
    else:
        labels = [
            f"Q{index + 1}"
            for index in range(len(boundaries) - 1)
        ]

        temporary["volatility_band"] = pd.cut(
            temporary[feature],
            bins=boundaries,
            labels=labels,
            include_lowest=True,
            duplicates="drop",
        )

    metrics = grouped_metrics(
        temporary,
        "volatility_band",
        include_percentage=True,
    )

    return metrics, boundary_frame


def error_band_metrics(
    frame: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    absolute_error_column = (
        f"{model_name}_absolute_error"
    )

    prediction_column = (
        "catboost_prediction"
        if model_name == "catboost"
        else "baseline_prediction"
    )

    bias_column = f"{model_name}_bias"

    temporary = pd.DataFrame(
        {
            "absolute_error": frame[
                absolute_error_column
            ],
            "actual": frame["target"],
            "prediction": frame[
                prediction_column
            ],
            "bias": frame[bias_column],
        }
    )

    temporary["error_band"] = pd.cut(
        temporary["absolute_error"],
        bins=ERROR_BINS,
        labels=ERROR_LABELS,
        include_lowest=True,
        right=False,
    )

    result = (
        temporary.groupby(
            "error_band",
            observed=True,
            dropna=False,
        )
        .agg(
            row_count=("actual", "size"),
            average_actual=("actual", "mean"),
            average_prediction=("prediction", "mean"),
            average_bias=("bias", "mean"),
        )
        .reset_index()
    )

    result["row_percentage"] = (
        result["row_count"] / len(frame) * 100.0
    )

    return result[
        [
            "error_band",
            "row_count",
            "row_percentage",
            "average_actual",
            "average_prediction",
            "average_bias",
        ]
    ]


def high_demand_signal_metrics(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    outputs = []

    for feature in HIGH_DEMAND_SIGNAL_FEATURES:
        if feature not in frame.columns:
            continue

        for threshold in (60, 100):
            subset = frame[
                frame["target"] > threshold
            ].copy()

            if subset.empty:
                continue

            try:
                subset["signal_band"] = pd.qcut(
                    subset[feature],
                    q=5,
                    duplicates="drop",
                )
            except ValueError:
                subset["signal_band"] = "single_band"

            metrics = grouped_metrics(
                subset,
                "signal_band",
                include_percentage=True,
            )

            metrics.insert(
                0,
                "target_threshold",
                f"target > {threshold}",
            )
            metrics.insert(
                0,
                "signal_feature",
                feature,
            )

            outputs.append(metrics)

    if not outputs:
        return pd.DataFrame()

    result = pd.concat(outputs, ignore_index=True)
    result["signal_band"] = (
        result["signal_band"].astype(str)
    )
    return result


# ============================================================
# Exports
# ============================================================

def investigation_columns(
    frame: pd.DataFrame,
) -> list[str]:
    candidates = [
        "SEFER_ID",
        "SEFER_TARIHI",
        "SEFER_SAATI",
        "FIRMA_ID",
        "GUZERGAH_KODU",
        "canonical_guzergah_id",
        "target",
        "month",
        "day_of_week",
        "weekday_label",
        "departure_hour",
        "departure_minute",
        "departure_30min_bucket",
        "departure_bucket_label",
        "catboost_prediction",
        "baseline_prediction",
        "baseline_source",
        "catboost_error",
        "catboost_absolute_error",
        "baseline_error",
        "baseline_absolute_error",
        "catboost_minus_baseline_absolute_error",
        "winner",
        "company_route_time_weekday_average",
        "company_route_time_weekday_median",
        "company_route_time_weekday_std",
        "company_route_time_weekday_p90",
        "company_route_time_weekday_above_60_rate",
        "company_route_time_weekday_above_100_rate",
        "company_route_time_weekday_count",
        "canonical_route_time_weekday_average",
        "canonical_route_time_weekday_p90",
        "canonical_route_time_weekday_count",
        "canonical_route_average",
        "canonical_route_p90",
        "canonical_route_above_60_rate",
        "canonical_route_above_100_rate",
        "canonical_route_count",
        "company_average",
        "company_count",
    ]

    return [
        column
        for column in candidates
        if column in frame.columns
    ]


def export_largest_errors(
    frame: pd.DataFrame,
    period_directory: Path,
) -> None:
    columns = investigation_columns(frame)

    exports = {
        "largest_catboost_errors.csv": frame.nlargest(
            TOP_ERROR_ROWS,
            "catboost_absolute_error",
        ),
        "largest_baseline_errors.csv": frame.nlargest(
            TOP_ERROR_ROWS,
            "baseline_absolute_error",
        ),
        "largest_catboost_underpredictions.csv": frame.nsmallest(
            TOP_ERROR_ROWS,
            "catboost_error",
        ),
        "largest_catboost_overpredictions.csv": frame.nlargest(
            TOP_ERROR_ROWS,
            "catboost_error",
        ),
        "catboost_loses_to_baseline_rows.csv": frame.nlargest(
            TOP_ERROR_ROWS,
            "catboost_minus_baseline_absolute_error",
        ),
        "catboost_beats_baseline_rows.csv": frame.nsmallest(
            TOP_ERROR_ROWS,
            "catboost_minus_baseline_absolute_error",
        ),
    }

    for filename, export_frame in exports.items():
        save_csv(
            export_frame[columns],
            period_directory / filename,
        )


def export_ranked_group_tables(
    metrics: pd.DataFrame,
    period_directory: Path,
    prefix: str,
    minimum_rows: int,
    top_volume_count: int,
    ranking_count: int,
) -> None:
    save_csv(
        metrics,
        period_directory / f"{prefix}_metrics_all.csv",
    )

    save_csv(
        metrics.nlargest(
            top_volume_count,
            "row_count",
        ),
        period_directory
        / f"{prefix}_metrics_top_volume.csv",
    )

    supported = metrics[
        metrics["row_count"] >= minimum_rows
    ].copy()

    save_csv(
        supported.nlargest(
            ranking_count,
            "catboost_mae",
        ),
        period_directory
        / f"{prefix}_metrics_worst_supported.csv",
    )

    save_csv(
        supported.nsmallest(
            ranking_count,
            "catboost_bias",
        ),
        period_directory
        / f"{prefix}_metrics_underpredicted.csv",
    )

    save_csv(
        supported.nlargest(
            ranking_count,
            "catboost_bias",
        ),
        period_directory
        / f"{prefix}_metrics_overpredicted.csv",
    )

    save_csv(
        supported.nlargest(
            ranking_count,
            "mae_improvement",
        ),
        period_directory
        / f"{prefix}_metrics_catboost_best.csv",
    )

    save_csv(
        supported.nsmallest(
            ranking_count,
            "mae_improvement",
        ),
        period_directory
        / f"{prefix}_metrics_baseline_best.csv",
    )


def export_period_analysis(
    frame: pd.DataFrame,
    period: EvaluationPeriod,
) -> dict[str, pd.DataFrame]:
    period_directory = OUTPUT_DIR / period.key
    period_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs: dict[str, pd.DataFrame] = {}

    outputs["overall"] = overall_metrics(
        frame,
        period,
    )
    save_csv(
        outputs["overall"],
        period_directory / "overall_metrics.csv",
    )

    outputs["target_group"] = grouped_metrics(
        frame,
        "target_group",
        include_percentage=True,
    )
    save_csv(
        outputs["target_group"],
        period_directory / "target_group_metrics.csv",
    )

    month_frame = frame.copy()
    month_frame["year"] = pd.to_datetime(
        month_frame["SEFER_TARIHI"]
    ).dt.year

    outputs["monthly"] = grouped_metrics(
        month_frame,
        ["year", "month"],
        include_percentage=True,
    )
    save_csv(
        outputs["monthly"],
        period_directory / "monthly_metrics.csv",
    )

    outputs["weekday"] = grouped_metrics(
        frame,
        [
            "day_of_week",
            "weekday_label",
        ],
        include_percentage=True,
    )
    save_csv(
        outputs["weekday"],
        period_directory / "weekday_metrics.csv",
    )

    outputs["departure_hour"] = grouped_metrics(
        frame,
        "departure_hour",
        include_percentage=True,
    )
    save_csv(
        outputs["departure_hour"],
        period_directory
        / "departure_hour_metrics.csv",
    )

    outputs["departure_bucket"] = grouped_metrics(
        frame,
        [
            "departure_30min_bucket",
            "departure_bucket_label",
        ],
        include_percentage=True,
    )
    save_csv(
        outputs["departure_bucket"],
        period_directory
        / "departure_bucket_metrics.csv",
    )

    outputs["company"] = grouped_metrics(
        frame,
        "FIRMA_ID",
    )
    export_ranked_group_tables(
        outputs["company"],
        period_directory,
        prefix="company",
        minimum_rows=MIN_COMPANY_ROWS,
        top_volume_count=50,
        ranking_count=50,
    )

    outputs["canonical_route"] = grouped_metrics(
        frame,
        "canonical_guzergah_id",
    )
    export_ranked_group_tables(
        outputs["canonical_route"],
        period_directory,
        prefix="canonical_route",
        minimum_rows=MIN_ROUTE_ROWS,
        top_volume_count=100,
        ranking_count=100,
    )

    outputs["company_route"] = grouped_metrics(
        frame,
        [
            "FIRMA_ID",
            "GUZERGAH_KODU",
            "canonical_guzergah_id",
        ],
    )
    export_ranked_group_tables(
        outputs["company_route"],
        period_directory,
        prefix="company_route",
        minimum_rows=MIN_COMPANY_ROUTE_ROWS,
        top_volume_count=100,
        ranking_count=100,
    )

    outputs["historical_support"] = support_metrics(
        frame
    )
    save_csv(
        outputs["historical_support"],
        period_directory
        / "historical_support_metrics.csv",
    )

    outputs["baseline_source"] = grouped_metrics(
        frame,
        "baseline_source",
        include_percentage=True,
    )
    save_csv(
        outputs["baseline_source"],
        period_directory
        / "baseline_source_metrics.csv",
    )

    (
        outputs["volatility"],
        outputs["volatility_boundaries"],
    ) = volatility_metrics(frame)

    save_csv(
        outputs["volatility"],
        period_directory / "volatility_metrics.csv",
    )
    save_csv(
        outputs["volatility_boundaries"],
        period_directory
        / "volatility_quantile_boundaries.csv",
    )

    outputs["catboost_error_bands"] = (
        error_band_metrics(
            frame,
            "catboost",
        )
    )
    save_csv(
        outputs["catboost_error_bands"],
        period_directory
        / "catboost_error_band_metrics.csv",
    )

    outputs["baseline_error_bands"] = (
        error_band_metrics(
            frame,
            "baseline",
        )
    )
    save_csv(
        outputs["baseline_error_bands"],
        period_directory
        / "baseline_error_band_metrics.csv",
    )

    outputs["high_demand"] = (
        high_demand_metrics(frame)
    )
    save_csv(
        outputs["high_demand"],
        period_directory / "high_demand_metrics.csv",
    )

    outputs["high_demand_signals"] = (
        high_demand_signal_metrics(frame)
    )
    save_csv(
        outputs["high_demand_signals"],
        period_directory
        / "high_demand_signal_metrics.csv",
    )

    export_largest_errors(
        frame,
        period_directory,
    )

    if EXPORT_ALL_ROW_LEVEL_PREDICTIONS:
        save_csv(
            frame[investigation_columns(frame)],
            period_directory
            / "all_row_level_predictions.csv",
        )

    del month_frame
    return outputs


# ============================================================
# Charts
# ============================================================

def save_figure(path: Path) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    plt.tight_layout()
    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def plot_mae_comparison(
    metrics: pd.DataFrame,
    x_column: str,
    title: str,
    x_label: str,
    path: Path,
) -> None:
    x_values = metrics[x_column].astype(str)

    plt.figure(figsize=(11, 6))
    plt.plot(
        x_values,
        metrics["catboost_mae"],
        marker="o",
        label="CatBoost",
    )
    plt.plot(
        x_values,
        metrics["baseline_mae"],
        marker="o",
        label="Baseline",
    )
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("MAE")
    plt.xticks(rotation=45, ha="right")
    plt.legend()

    save_figure(path)


def plot_bias_comparison(
    metrics: pd.DataFrame,
    x_column: str,
    title: str,
    path: Path,
) -> None:
    x_positions = np.arange(len(metrics))
    width = 0.4

    plt.figure(figsize=(11, 6))
    plt.bar(
        x_positions - width / 2,
        metrics["catboost_bias"],
        width,
        label="CatBoost",
    )
    plt.bar(
        x_positions + width / 2,
        metrics["baseline_bias"],
        width,
        label="Baseline",
    )
    plt.axhline(0, linewidth=1)
    plt.title(title)
    plt.xlabel(
        x_column.replace("_", " ").title()
    )
    plt.ylabel("Average prediction minus actual")
    plt.xticks(
        x_positions,
        metrics[x_column].astype(str),
        rotation=45,
        ha="right",
    )
    plt.legend()

    save_figure(path)


def generate_charts(
    outputs: dict[str, pd.DataFrame],
    period: EvaluationPeriod,
) -> None:
    chart_directory = (
        OUTPUT_DIR
        / period.key
        / "charts"
    )

    plot_mae_comparison(
        outputs["target_group"],
        "target_group",
        f"{period.label}: MAE by target group",
        "Target group",
        chart_directory
        / "mae_by_target_group.png",
    )

    plot_bias_comparison(
        outputs["target_group"],
        "target_group",
        f"{period.label}: bias by target group",
        chart_directory
        / "bias_by_target_group.png",
    )

    target_metrics = outputs["target_group"]
    x_positions = np.arange(
        len(target_metrics)
    )
    width = 0.27

    plt.figure(figsize=(11, 6))
    plt.bar(
        x_positions - width,
        target_metrics["average_actual"],
        width,
        label="Actual",
    )
    plt.bar(
        x_positions,
        target_metrics[
            "average_catboost_prediction"
        ],
        width,
        label="CatBoost",
    )
    plt.bar(
        x_positions + width,
        target_metrics[
            "average_baseline_prediction"
        ],
        width,
        label="Baseline",
    )
    plt.title(
        f"{period.label}: actual versus predicted "
        "by target group"
    )
    plt.xlabel("Target group")
    plt.ylabel("Average passenger demand")
    plt.xticks(
        x_positions,
        target_metrics["target_group"].astype(str),
        rotation=45,
        ha="right",
    )
    plt.legend()

    save_figure(
        chart_directory
        / "actual_vs_predicted_by_target_group.png"
    )

    monthly = outputs["monthly"].copy()
    monthly["year_month"] = (
        monthly["year"].astype(str)
        + "-"
        + monthly["month"]
        .astype(int)
        .astype(str)
        .str.zfill(2)
    )

    plot_mae_comparison(
        monthly,
        "year_month",
        f"{period.label}: monthly MAE",
        "Month",
        chart_directory / "mae_by_month.png",
    )

    plot_mae_comparison(
        outputs["departure_hour"],
        "departure_hour",
        f"{period.label}: MAE by departure hour",
        "Departure hour",
        chart_directory
        / "mae_by_departure_hour.png",
    )

    plot_mae_comparison(
        outputs["baseline_source"],
        "baseline_source",
        f"{period.label}: MAE by baseline fallback source",
        "Baseline source",
        chart_directory
        / "mae_by_baseline_source.png",
    )

    support = outputs["historical_support"]
    exact_support = support[
        support["support_feature"]
        == "company_route_time_weekday_count"
    ]

    if not exact_support.empty:
        plt.figure(figsize=(11, 6))
        plt.bar(
            exact_support[
                "support_band"
            ].astype(str),
            exact_support["mae_improvement"],
        )
        plt.axhline(0, linewidth=1)
        plt.title(
            f"{period.label}: CatBoost improvement "
            "by exact-history support"
        )
        plt.xlabel("Historical count band")
        plt.ylabel(
            "Baseline MAE minus CatBoost MAE"
        )
        plt.xticks(rotation=45, ha="right")

        save_figure(
            chart_directory
            / "mae_improvement_by_historical_support.png"
        )

    supported_routes = outputs[
        "canonical_route"
    ]
    supported_routes = supported_routes[
        supported_routes["row_count"]
        >= MIN_ROUTE_ROWS
    ].nsmallest(
        TOP_PLOT_GROUPS,
        "catboost_bias",
    )

    if not supported_routes.empty:
        plt.figure(figsize=(12, 7))
        plt.barh(
            supported_routes[
                "canonical_guzergah_id"
            ].astype(str),
            supported_routes["catboost_bias"],
        )
        plt.axvline(0, linewidth=1)
        plt.title(
            f"{period.label}: most underpredicted "
            "supported physical routes"
        )
        plt.xlabel("CatBoost average bias")
        plt.ylabel("Canonical route")

        save_figure(
            chart_directory
            / "top_supported_underpredicted_routes.png"
        )


# ============================================================
# Cross-period comparison and report
# ============================================================

def add_period_columns(
    frame: pd.DataFrame,
    period: EvaluationPeriod,
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "period", period.key)
    result.insert(1, "period_label", period.label)
    return result


def export_comparisons(
    all_outputs: dict[
        str,
        dict[str, pd.DataFrame],
    ],
) -> None:
    comparison_directory = (
        OUTPUT_DIR / "comparison"
    )

    overall = pd.concat(
        [
            all_outputs[period.key]["overall"]
            for period in PERIODS
        ],
        ignore_index=True,
    )
    save_csv(
        overall,
        comparison_directory
        / "period_overall_comparison.csv",
    )

    target_groups = pd.concat(
        [
            add_period_columns(
                all_outputs[period.key][
                    "target_group"
                ],
                period,
            )
            for period in PERIODS
        ],
        ignore_index=True,
    )
    save_csv(
        target_groups,
        comparison_directory
        / "target_group_comparison.csv",
    )

    baseline_sources = pd.concat(
        [
            add_period_columns(
                all_outputs[period.key][
                    "baseline_source"
                ],
                period,
            )
            for period in PERIODS
        ],
        ignore_index=True,
    )
    save_csv(
        baseline_sources,
        comparison_directory
        / "baseline_source_comparison.csv",
    )


def metric_text(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.4f}"


def generate_markdown_summary(
    all_outputs: dict[
        str,
        dict[str, pd.DataFrame],
    ],
) -> None:
    lines = [
        "# CatBoost v3 error analysis",
        "",
        "This report diagnoses the frozen CatBoost v3 MAE model without retraining or changing it.",
        "",
        "**Bias convention:** positive bias means overprediction; negative bias means underprediction.",
        "",
        "The 2026 period is the already-used official final evaluation period. It may be analyzed here, but it is not an untouched tuning set for a changed model.",
        "",
        "## Overall results",
        "",
        "| Period | Rows | CatBoost MAE | Baseline MAE | MAE improvement | CatBoost RMSE | Baseline RMSE | RMSE improvement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for period in PERIODS:
        row = (
            all_outputs[period.key]["overall"]
            .iloc[0]
        )

        lines.append(
            "| "
            f"{period.label} | "
            f"{int(row['row_count']):,} | "
            f"{metric_text(row['catboost_mae'])} | "
            f"{metric_text(row['baseline_mae'])} | "
            f"{metric_text(row['mae_improvement'])} | "
            f"{metric_text(row['catboost_rmse'])} | "
            f"{metric_text(row['baseline_rmse'])} | "
            f"{metric_text(row['rmse_improvement'])} |"
        )

    lines.extend(
        [
            "",
            "## Target-range behavior",
            "",
        ]
    )

    for period in PERIODS:
        target = all_outputs[
            period.key
        ]["target_group"]

        low = target[
            target["target_group"].astype(str)
            == "1-10"
        ]

        high = target[
            target["target_group"].astype(str)
            == "101-300"
        ]

        lines.append(
            f"### {period.label}"
        )
        lines.append("")

        if not low.empty:
            row = low.iloc[0]
            lines.append(
                "- For target 1–10, CatBoost bias is "
                f"{metric_text(row['catboost_bias'])}; "
                "positive values confirm low-demand overprediction."
            )

        if not high.empty:
            row = high.iloc[0]
            lines.append(
                "- For target 101–300, CatBoost bias is "
                f"{metric_text(row['catboost_bias'])}; "
                f"CatBoost MAE is {metric_text(row['catboost_mae'])} "
                f"versus baseline MAE {metric_text(row['baseline_mae'])}."
            )

        lines.append("")

    lines.extend(
        [
            "## Highest-impact target groups",
            "",
            "Impact is approximated as row count multiplied by CatBoost MAE, so large-volume error segments are not hidden by tiny extreme groups.",
            "",
        ]
    )

    for period in PERIODS:
        target = all_outputs[
            period.key
        ]["target_group"].copy()

        target["estimated_error_volume"] = (
            target["row_count"]
            * target["catboost_mae"]
        )

        highest = target.nlargest(
            4,
            "estimated_error_volume",
        )

        lines.append(
            f"### {period.label}"
        )
        lines.append("")

        for row in highest.itertuples(index=False):
            lines.append(
                f"- {row.target_group}: "
                f"{int(row.row_count):,} rows, "
                f"CatBoost MAE {row.catboost_mae:.4f}, "
                f"bias {row.catboost_bias:.4f}, "
                f"MAE improvement {row.mae_improvement:.4f}."
            )

        lines.append("")

    lines.extend(
        [
            "## Historical support and fallbacks",
            "",
        ]
    )

    for period in PERIODS:
        fallback = all_outputs[
            period.key
        ]["baseline_source"]

        lines.append(
            f"### {period.label}"
        )
        lines.append("")

        for row in fallback.itertuples(index=False):
            lines.append(
                f"- {row.baseline_source}: "
                f"{int(row.row_count):,} rows "
                f"({row.row_percentage:.2f}%), "
                f"CatBoost MAE {row.catboost_mae:.4f}, "
                f"baseline MAE {row.baseline_mae:.4f}, "
                f"improvement {row.mae_improvement:.4f}."
            )

        lines.append("")

    lines.extend(
        [
            "## Evidence-based v4 investigation",
            "",
            "The error-analysis exports should be used to decide which feature family deserves a new v4 experiment. The most defensible candidates are:",
            "",
            "1. recent 30/90/180-day route and company-route demand;",
            "2. recent-versus-long-term trend features;",
            "3. public-holiday and holiday-adjacent date features;",
            "4. month- or season-specific route history;",
            "5. competing departures and nearby departure density;",
            "6. route characteristics for sparse or unseen routes.",
            "",
            "Do not assume these features will help merely because they sound useful. Check the monthly, historical-support, fallback-source, route, company-route, and high-demand-signal exports first.",
            "",
            "## Output locations",
            "",
            "- `test_2025_h2/`: detailed 2025 H2 analysis",
            "- `final_2026/`: detailed official-final-period analysis",
            "- `comparison/`: cross-period comparisons",
            "",
        ]
    )

    report_path = (
        OUTPUT_DIR
        / "error_analysis_summary.md"
    )

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# Period pipeline
# ============================================================

def evaluate_period(
    conn: duckdb.DuckDBPyConnection,
    model: CatBoostRegressor,
    period: EvaluationPeriod,
    optional_columns: Sequence[str],
) -> dict[str, pd.DataFrame]:
    create_source_tables(
        conn,
        period,
        optional_columns,
    )

    check_granularity(
        conn,
        period,
        optional_columns,
    )

    create_all_statistics(
        conn,
        period,
    )

    feature_table = create_feature_table(
        conn,
        period,
        optional_columns,
    )

    frame = fetch_feature_matrix(
        conn,
        feature_table,
    )

    print(f"Prediction DataFrame rows: {len(frame):,}")

    if len(frame) == 0:
        raise RuntimeError(
            f"{period.label}: completed matrix is empty."
        )

    frame = add_predictions(
        model,
        frame,
    )

    outputs = export_period_analysis(
        frame,
        period,
    )

    generate_charts(
        outputs,
        period,
    )

    print(
        f"{period.label} complete. "
        f"Outputs: {OUTPUT_DIR / period.key}"
    )

    del frame
    gc.collect()

    return outputs


# ============================================================
# Main
# ============================================================

def main() -> None:
    ensure_paths()

    model = load_model()

    print(f"\nOpening database: {DB_PATH}")

    conn = duckdb.connect(
        str(DB_PATH),
        read_only=False,
    )

    temp_directory_sql = (
        TEMP_DIR
        .as_posix()
        .replace("'", "''")
    )

    conn.execute(
        f"SET threads = {DUCKDB_THREADS}"
    )
    conn.execute(
        f"SET temp_directory = "
        f"'{temp_directory_sql}'"
    )

    all_outputs: dict[
        str,
        dict[str, pd.DataFrame],
    ] = {}

    try:
        optional_columns = inspect_database(conn)

        for period in PERIODS:
            print(
                "\n"
                + "=" * 72
                + f"\nEvaluating {period.label}\n"
                + "=" * 72
            )

            all_outputs[period.key] = evaluate_period(
                conn,
                model,
                period,
                optional_columns,
            )

        export_comparisons(all_outputs)
        generate_markdown_summary(all_outputs)

    finally:
        conn.close()

    print("\nError analysis finished.")
    print(
        "Markdown summary:",
        OUTPUT_DIR / "error_analysis_summary.md",
    )
    print(
        "2025 H2 overall metrics:",
        OUTPUT_DIR
        / "test_2025_h2"
        / "overall_metrics.csv",
    )
    print(
        "2026 overall metrics:",
        OUTPUT_DIR
        / "final_2026"
        / "overall_metrics.csv",
    )
    print(
        "Cross-period comparison:",
        OUTPUT_DIR
        / "comparison"
        / "period_overall_comparison.csv",
    )


if __name__ == "__main__":
    main()
