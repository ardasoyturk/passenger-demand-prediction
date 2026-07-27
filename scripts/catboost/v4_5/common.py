"""Shared leakage-safe feature and evaluation code for CatBoost v4.5.

v4.5 keeps the exact 64-feature CatBoost v4.1 contract and appends only nine
predetermined religious-holiday calendar features.  Passenger demand is never
used to derive those calendar features.  Frozen v4.1 and v4.2 artifacts are
read only and are used only for comparison reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


PROJECT_DIR = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_DIR / "analysis.duckdb"
TEMP_DIR = PROJECT_DIR / "duckdb_temp"
RESULTS_DIR = PROJECT_DIR / "results"
MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "catboost_demand_model_v4_5_religious_holiday_mae_6000.cbm"
)

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
V4_1_FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + CALENDAR_NUMERIC_FEATURES
    + HISTORICAL_FEATURES
    + RECENT_FEATURES
)

HOLIDAY_CATEGORICAL_FEATURES = ["religious_holiday_type"]
HOLIDAY_NUMERIC_FEATURES = [
    "is_arife",
    "is_religious_holiday",
    "religious_holiday_day_index",
    "relative_day_to_religious_holiday",
    "is_pre_holiday_3d",
    "is_pre_holiday_7d",
    "is_post_holiday_3d",
    "is_post_holiday_7d",
]
HOLIDAY_FEATURES = HOLIDAY_CATEGORICAL_FEATURES + HOLIDAY_NUMERIC_FEATURES
FEATURE_COLUMNS = V4_1_FEATURE_COLUMNS + HOLIDAY_FEATURES
MODEL_CATEGORICAL_FEATURES = CATEGORICAL_FEATURES + HOLIDAY_CATEGORICAL_FEATURES

# 99 is outside the valid holiday-relative range and is the explicit neutral
# value.  A positive holiday-relative value can reach 10 for a four-day
# holiday followed by the permitted seven-day post window.
NEUTRAL_RELATIVE_DAY = 99

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

HOLIDAY_ROWS = (
    ("RAMAZAN", "2023-04-20", "2023-04-21", "2023-04-23"),
    ("KURBAN", "2023-06-27", "2023-06-28", "2023-07-01"),
    ("RAMAZAN", "2024-04-09", "2024-04-10", "2024-04-12"),
    ("KURBAN", "2024-06-15", "2024-06-16", "2024-06-19"),
    ("RAMAZAN", "2025-03-29", "2025-03-30", "2025-04-01"),
    ("KURBAN", "2025-06-05", "2025-06-06", "2025-06-09"),
    ("RAMAZAN", "2026-03-19", "2026-03-20", "2026-03-22"),
    ("KURBAN", "2026-05-26", "2026-05-27", "2026-05-30"),
)

FROZEN_PREDICTION_PATHS = {
    "validation_2025_h1": RESULTS_DIR
    / "catboost_v4_2_validation_2025_h1_predictions.csv",
    "test_2025_h2": RESULTS_DIR / "catboost_v4_2_test_2025_h2_predictions.csv",
    "final_2026": RESULTS_DIR / "catboost_v4_2_final_2026_predictions.csv",
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
        "validation_2025_h1", "2025 H1 validation", "2023-01-01",
        "2025-01-01", "2025-01-01", "2025-07-01", 1_420_182, True,
    ),
    "test": PeriodConfig(
        "test_2025_h2", "2025 H2 test", "2023-01-01",
        "2025-07-01", "2025-07-01", "2026-01-01", 1_517_010, False,
    ),
    "final": PeriodConfig(
        "final_2026", "2026 final (reporting only; already observed)",
        "2023-01-01", "2026-01-01", "2026-01-01", "2026-04-15",
        780_865, False,
    ),
}


def sql_identifier_list(columns: list[str]) -> str:
    return ", ".join(columns)


def ensure_environment(*, require_model: bool = False) -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    if require_model and not MODEL_PATH.exists():
        raise FileNotFoundError(f"v4.5 model not found: {MODEL_PATH}")
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


def create_religious_holiday_calendar(conn: duckdb.DuckDBPyConnection) -> None:
    values_sql = ",\n".join(
        f"('{kind}', DATE '{arife}', DATE '{first_day}', DATE '{last_day}')"
        for kind, arife, first_day, last_day in HOLIDAY_ROWS
    )
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE v4_5_religious_holidays AS
        SELECT * FROM (VALUES {values_sql}) AS holiday(
            religious_holiday_type, arife_date, first_day, last_day
        )
    """)


def create_source_tables(
    conn: duckdb.DuckDBPyConnection,
    config: PeriodConfig,
    *,
    training: bool,
) -> None:
    columns = sql_identifier_list(SOURCE_COLUMNS)
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE v4_5_history AS
        SELECT {columns} FROM model_data_base
        WHERE SEFER_TARIHI >= DATE '{config.history_start}'
          AND SEFER_TARIHI < DATE '{config.history_end_exclusive}'
    """)
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE v4_5_target AS
        SELECT {columns} FROM model_data_base
        WHERE SEFER_TARIHI >= DATE '{config.target_start}'
          AND SEFER_TARIHI < DATE '{config.target_end_exclusive}'
    """)
    history_count = conn.execute("SELECT COUNT(*) FROM v4_5_history").fetchone()[0]
    target_count = conn.execute("SELECT COUNT(*) FROM v4_5_target").fetchone()[0]
    if not history_count or not target_count:
        raise RuntimeError("History or target period is empty.")
    if target_count != config.expected_rows:
        raise RuntimeError(
            f"Expected {config.expected_rows:,} rows for {config.key}; "
            f"found {target_count:,}."
        )
    latest_history = conn.execute("SELECT MAX(SEFER_TARIHI) FROM v4_5_history").fetchone()[0]
    earliest_target = conn.execute("SELECT MIN(SEFER_TARIHI) FROM v4_5_target").fetchone()[0]
    if latest_history >= earliest_target:
        raise RuntimeError(
            f"Leakage guard failed: history ends {latest_history}, "
            f"target starts {earliest_target}."
        )
    role = "training" if training else "evaluation"
    print(f"{role.title()} history rows: {history_count:,}; target rows: {target_count:,}")
    print(f"Leakage cutoff verified: {latest_history} < {earliest_target}")


def create_long_term_statistics(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE v4_5_global_stats AS
        SELECT
            AVG(target)::DOUBLE AS global_average,
            MEDIAN(target)::DOUBLE AS global_median,
            STDDEV_SAMP(target)::DOUBLE AS global_std,
            MAX(target)::DOUBLE AS global_maximum,
            QUANTILE_CONT(target, 0.90)::DOUBLE AS global_p90,
            AVG(CASE WHEN target > 60 THEN 1.0 ELSE 0.0 END)::DOUBLE
                AS global_above_60_rate,
            AVG(CASE WHEN target > 100 THEN 1.0 ELSE 0.0 END)::DOUBLE
                AS global_above_100_rate
        FROM v4_5_history
    """)
    for prefix, group_columns in HISTORICAL_GROUP_DEFINITIONS:
        group_sql = sql_identifier_list(group_columns)
        print(f"  Long-term aggregation: {prefix}")
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE v4_5_{prefix}_stats AS
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
            FROM v4_5_history
            GROUP BY {group_sql}
        """)


def create_recent_statistics(
    conn: duckdb.DuckDBPyConnection,
    *,
    history_table: str,
) -> None:
    for prefix, group_columns in RECENT_GROUP_DEFINITIONS:
        reference_columns = ", ".join(f"source.{column}" for column in group_columns)
        grouped_columns = ", ".join(f"reference.{column}" for column in group_columns)
        join_sql = _join_condition("reference", "history", group_columns)
        for window in RECENT_WINDOWS:
            print(f"  Recent aggregation: {prefix}, {window}d")
            conn.execute(f"""
                CREATE OR REPLACE TEMP TABLE
                    v4_5_{prefix}_recent_{window}d_stats AS
                WITH reference_dates AS (
                    SELECT DISTINCT {reference_columns},
                        source.SEFER_TARIHI AS reference_date
                    FROM v4_5_target AS source
                )
                SELECT
                    {grouped_columns}, reference.reference_date,
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
    return [
        f"COALESCE({alias}.{prefix}_average, global_stats.global_average) AS {prefix}_average",
        f"COALESCE({alias}.{prefix}_median, global_stats.global_median) AS {prefix}_median",
        f"COALESCE({alias}.{prefix}_std, global_stats.global_std) AS {prefix}_std",
        f"COALESCE({alias}.{prefix}_maximum, global_stats.global_maximum) AS {prefix}_maximum",
        f"COALESCE({alias}.{prefix}_p90, global_stats.global_p90) AS {prefix}_p90",
        f"COALESCE({alias}.{prefix}_above_60_rate, global_stats.global_above_60_rate) AS {prefix}_above_60_rate",
        f"COALESCE({alias}.{prefix}_above_100_rate, global_stats.global_above_100_rate) AS {prefix}_above_100_rate",
        f"COALESCE({alias}.{prefix}_count, 0)::BIGINT AS {prefix}_count",
    ]


def create_feature_table(conn: duckdb.DuckDBPyConnection) -> None:
    select_expressions = [
        "base.SEFER_ID", "base.SEFER_TARIHI", "base.FIRMA_ID",
        "base.GUZERGAH_KODU", "base.canonical_guzergah_id", "base.target",
        "base.month", "base.week_of_year", "base.day_of_week",
        "base.departure_minute", "base.departure_30min_bucket",
    ]
    joins: list[str] = []
    aliases: dict[str, str] = {}
    for index, (prefix, group_columns) in enumerate(HISTORICAL_GROUP_DEFINITIONS):
        alias = f"long_term_{index}"
        aliases[prefix] = alias
        joins.append(
            f"LEFT JOIN v4_5_{prefix}_stats AS {alias} "
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
                f"LEFT JOIN v4_5_{prefix}_recent_{window}d_stats AS {alias} "
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
            "COALESCE(holiday.religious_holiday_type, 'NONE') "
            "AS religious_holiday_type",
            "CASE WHEN base.SEFER_TARIHI = holiday.arife_date THEN 1 ELSE 0 END::UTINYINT "
            "AS is_arife",
            "CASE WHEN base.SEFER_TARIHI BETWEEN holiday.first_day AND holiday.last_day "
            "THEN 1 ELSE 0 END::UTINYINT AS is_religious_holiday",
            "CASE WHEN base.SEFER_TARIHI BETWEEN holiday.first_day AND holiday.last_day "
            "THEN date_diff('day', holiday.first_day, base.SEFER_TARIHI) + 1 "
            "ELSE 0 END::SMALLINT AS religious_holiday_day_index",
            f"COALESCE(date_diff('day', holiday.first_day, base.SEFER_TARIHI), "
            f"{NEUTRAL_RELATIVE_DAY})::SMALLINT AS relative_day_to_religious_holiday",
            "CASE WHEN base.SEFER_TARIHI >= holiday.first_day - INTERVAL '3 days' "
            "AND base.SEFER_TARIHI < holiday.first_day THEN 1 ELSE 0 END::UTINYINT "
            "AS is_pre_holiday_3d",
            "CASE WHEN base.SEFER_TARIHI >= holiday.first_day - INTERVAL '7 days' "
            "AND base.SEFER_TARIHI < holiday.first_day THEN 1 ELSE 0 END::UTINYINT "
            "AS is_pre_holiday_7d",
            "CASE WHEN base.SEFER_TARIHI > holiday.last_day "
            "AND base.SEFER_TARIHI <= holiday.last_day + INTERVAL '3 days' "
            "THEN 1 ELSE 0 END::UTINYINT AS is_post_holiday_3d",
            "CASE WHEN base.SEFER_TARIHI > holiday.last_day "
            "AND base.SEFER_TARIHI <= holiday.last_day + INTERVAL '7 days' "
            "THEN 1 ELSE 0 END::UTINYINT AS is_post_holiday_7d",
        ]
    )
    joins.append(
        "LEFT JOIN v4_5_religious_holidays AS holiday "
        "ON base.SEFER_TARIHI >= holiday.first_day - INTERVAL '7 days' "
        "AND base.SEFER_TARIHI <= holiday.last_day + INTERVAL '7 days'"
    )
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE v4_5_features AS
        SELECT {', '.join(select_expressions)}
        FROM v4_5_target AS base
        CROSS JOIN v4_5_global_stats AS global_stats
        {' '.join(joins)}
    """)


def build_matrix(config: PeriodConfig, *, training: bool) -> pd.DataFrame:
    # Feature construction is also used to build validation before the new
    # model exists, so model presence is checked only by load_model().
    ensure_environment()
    conn = connect()
    try:
        create_religious_holiday_calendar(conn)
        create_source_tables(conn, config, training=training)
        if training:
            conn.execute("""
                CREATE OR REPLACE TEMP TABLE v4_5_recent_history AS
                SELECT * FROM v4_5_history
                UNION ALL
                SELECT * FROM v4_5_target
            """)
            recent_history_table = "v4_5_recent_history"
        else:
            recent_history_table = "v4_5_history"
        create_long_term_statistics(conn)
        create_recent_statistics(conn, history_table=recent_history_table)
        create_feature_table(conn)
        selected = [
            "SEFER_ID", "SEFER_TARIHI", *FEATURE_COLUMNS, "target",
            "baseline_prediction", "baseline_source",
        ]
        frame = conn.execute(
            f"SELECT {sql_identifier_list(selected)} FROM v4_5_features"
        ).fetchdf()
    finally:
        conn.close()
    validate_feature_frame(frame)
    return frame


def validate_feature_frame(frame: pd.DataFrame) -> None:
    if len(V4_1_FEATURE_COLUMNS) != 64 or len(set(V4_1_FEATURE_COLUMNS)) != 64:
        raise RuntimeError("The inherited v4.1 contract must be 64 unique features.")
    if len(FEATURE_COLUMNS) != 73 or len(set(FEATURE_COLUMNS)) != 73:
        raise RuntimeError("v4.5 must have exactly 73 unique feature columns.")
    missing_columns = [column for column in FEATURE_COLUMNS if column not in frame]
    if missing_columns:
        raise RuntimeError(f"Missing v4.5 feature columns: {missing_columns}")
    for column in MODEL_CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("string").fillna("missing")
    missing = frame[FEATURE_COLUMNS].isna().sum()
    if int(missing.sum()) != 0:
        raise RuntimeError(
            "v4.5 feature matrix contains missing values:\n"
            + missing[missing > 0].to_string()
        )
    holiday_types = set(frame["religious_holiday_type"].unique())
    if not holiday_types <= {"NONE", "RAMAZAN", "KURBAN"}:
        raise RuntimeError(f"Unexpected religious holiday types: {holiday_types}")
    for column in HOLIDAY_NUMERIC_FEATURES:
        if column.startswith("is_") and not frame[column].isin((0, 1)).all():
            raise RuntimeError(f"Non-binary holiday flag: {column}")


def load_model() -> CatBoostRegressor:
    ensure_environment(require_model=True)
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    if list(model.feature_names_) != FEATURE_COLUMNS:
        raise RuntimeError("Saved v4.5 model does not match the 73-feature contract.")
    return model


def merge_frozen_comparisons(frame: pd.DataFrame, config: PeriodConfig) -> pd.DataFrame:
    path = FROZEN_PREDICTION_PATHS[config.key]
    required = [
        "SEFER_ID", "actual", "v4_1_prediction", "hybrid_prediction",
        "baseline_prediction",
    ]
    if not path.exists():
        raise FileNotFoundError(f"Frozen v4.2 comparison artifact not found: {path}")
    frozen = pd.read_csv(path, usecols=required).rename(
        columns={
            "actual": "frozen_actual",
            "hybrid_prediction": "v4_2_hybrid_prediction",
            "baseline_prediction": "frozen_baseline_prediction",
        }
    )
    if frozen["SEFER_ID"].duplicated().any() or frame["SEFER_ID"].duplicated().any():
        raise RuntimeError("SEFER_ID must be unique for frozen comparison alignment.")
    merged = frame.merge(frozen, on="SEFER_ID", how="left", validate="one_to_one")
    required_predictions = [
        "v4_1_prediction", "v4_2_hybrid_prediction",
        "frozen_baseline_prediction",
    ]
    if merged[required_predictions].isna().any().any():
        raise RuntimeError("Frozen predictions did not align to all v4.5 rows.")
    if not np.array_equal(
        merged["target"].to_numpy(), merged["frozen_actual"].to_numpy()
    ):
        raise RuntimeError("Targets disagree with the frozen v4.2 artifact.")
    if not np.allclose(
        merged["baseline_prediction"].to_numpy(),
        merged["frozen_baseline_prediction"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Rebuilt weekday baseline differs from frozen v4.2.")
    return merged.drop(columns=["frozen_actual", "frozen_baseline_prediction"])


def _regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "average_actual": float(np.mean(actual)),
        "average_prediction": float(np.mean(prediction)),
    }


MODEL_COMPARISON_COLUMNS = {
    "CatBoost v4.1": "v4_1_prediction",
    "v4.2 hybrid": "v4_2_hybrid_prediction",
    "CatBoost v4.5": "v4_5_prediction",
    "Weekday baseline": "baseline_prediction",
}


def overall_metrics(frame: pd.DataFrame, config: PeriodConfig) -> pd.DataFrame:
    actual = frame["target"].to_numpy(dtype=np.float64)
    mask_10_40 = (actual >= 10) & (actual <= 40)
    mask_20_40 = (actual >= 20) & (actual <= 40)
    mask_40_plus = actual >= 40
    records = []
    for model, column in MODEL_COMPARISON_COLUMNS.items():
        prediction = frame[column].to_numpy(dtype=np.float64)
        overall = _regression_metrics(actual, prediction)
        records.append(
            {
                "period": config.key,
                "model": model,
                "row_count": len(frame),
                "mae": overall["mae"],
                "rmse": overall["rmse"],
                "bias": overall["bias"],
                "mae_actual_10_to_40": _regression_metrics(
                    actual[mask_10_40], prediction[mask_10_40]
                )["mae"],
                "mae_actual_20_to_40": _regression_metrics(
                    actual[mask_20_40], prediction[mask_20_40]
                )["mae"],
                "rmse_actual_40_plus": _regression_metrics(
                    actual[mask_40_plus], prediction[mask_40_plus]
                )["rmse"],
            }
        )
    return pd.DataFrame(records)


def holiday_window_labels(frame: pd.DataFrame) -> pd.Series:
    relative = frame["relative_day_to_religious_holiday"].to_numpy()
    arife = frame["is_arife"].to_numpy(dtype=bool)
    holiday = frame["is_religious_holiday"].to_numpy(dtype=bool)
    labels = np.full(len(frame), "ordinary days", dtype=object)
    labels[(relative >= -7) & (relative <= -4)] = "7-4 days before"
    labels[(relative >= -3) & (relative <= -1)] = "3-1 days before"
    labels[arife] = "Arife"
    labels[holiday] = "Bayram days"
    # Use flags to avoid relying on holiday length; 1-3 is a subset of 1-7.
    post_7 = frame["is_post_holiday_7d"].to_numpy(dtype=bool)
    post_3 = frame["is_post_holiday_3d"].to_numpy(dtype=bool)
    labels[post_7] = "4-7 days after"
    labels[post_3] = "1-3 days after"
    return pd.Series(labels, index=frame.index, name="holiday_window")


def holiday_window_metrics(frame: pd.DataFrame, config: PeriodConfig) -> pd.DataFrame:
    data = frame.copy()
    data["holiday_window"] = holiday_window_labels(data)
    order = [
        "7-4 days before", "3-1 days before", "Arife", "Bayram days",
        "1-3 days after", "4-7 days after", "ordinary days",
    ]
    records = []
    for window in order:
        subset = data[data["holiday_window"] == window]
        if subset.empty:
            continue
        actual = subset["target"].to_numpy(dtype=np.float64)
        for model, column in MODEL_COMPARISON_COLUMNS.items():
            values = _regression_metrics(
                actual, subset[column].to_numpy(dtype=np.float64)
            )
            records.append(
                {
                    "period": config.key,
                    "holiday_window": window,
                    "model": model,
                    "row_count": len(subset),
                    **values,
                }
            )
    return pd.DataFrame(records)


def output_paths(config: PeriodConfig) -> tuple[Path, Path, Path]:
    stem = RESULTS_DIR / f"catboost_v4_5_{config.key}"
    return (
        Path(f"{stem}_overall_metrics.csv"),
        Path(f"{stem}_holiday_window_metrics.csv"),
        Path(f"{stem}_predictions.csv"),
    )


def refuse_existing_outputs(config: PeriodConfig) -> None:
    collisions = [path for path in output_paths(config) if path.exists()]
    if collisions:
        rendered = "\n".join(f"  - {path}" for path in collisions)
        raise FileExistsError("Refusing to overwrite v4.5 artifacts:\n" + rendered)


def write_evaluation_outputs(frame: pd.DataFrame, config: PeriodConfig) -> None:
    overall_path, holiday_path, predictions_path = output_paths(config)
    refuse_existing_outputs(config)
    overall = overall_metrics(frame, config)
    windows = holiday_window_metrics(frame, config)
    output = frame[
        [
            "SEFER_ID", "SEFER_TARIHI", "target",
            *MODEL_COMPARISON_COLUMNS.values(), "baseline_source",
            *HOLIDAY_FEATURES,
        ]
    ].copy()
    output["holiday_window"] = holiday_window_labels(frame)
    overall.to_csv(overall_path, index=False)
    windows.to_csv(holiday_path, index=False)
    output.to_csv(predictions_path, index=False)
    print(f"\n{config.label} overall comparison")
    print(overall.drop(columns=["period", "row_count"]).to_string(index=False))
    print(f"\n{config.label} religious-holiday windows")
    print(windows.drop(columns="period").to_string(index=False))
