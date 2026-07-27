from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


PROJECT_DIR = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_DIR / "analysis.duckdb"
MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "catboost_demand_model_v4_1_recent_mae_6000.cbm"
)
TEMP_DIR = PROJECT_DIR / "duckdb_temp"
RESULTS_DIR = PROJECT_DIR / "results"

DUCKDB_THREADS = max(1, os.cpu_count() or 4)
RECENT_WINDOWS = [30, 60, 90, 180]

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
    ("canonical_route", ["canonical_guzergah_id"]),
    ("company", ["FIRMA_ID"]),
]

RECENT_GROUP_DEFINITIONS = [
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
        "canonical_route_time_weekday",
        [
            "canonical_guzergah_id",
            "departure_30min_bucket",
            "day_of_week",
        ],
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

RECENT_FEATURES = [
    f"{prefix}_recent_{window}d_{suffix}"
    for prefix, _ in RECENT_GROUP_DEFINITIONS
    for window in RECENT_WINDOWS
    for suffix in ["average", "count"]
]

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + CALENDAR_NUMERIC_FEATURES
    + HISTORICAL_FEATURES
    + RECENT_FEATURES
)

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


@dataclass(frozen=True)
class EvaluationConfig:
    key: str
    label: str
    history_start: str
    history_end_exclusive: str
    target_start: str
    target_end_exclusive: str
    expected_rows: int
    baseline_mae: float
    baseline_rmse: float


def sql_identifier_list(columns: list[str]) -> str:
    return ", ".join(columns)


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
        f"COALESCE({aggregate_alias}.{prefix}_average, global_stats.global_average) AS {prefix}_average",
        f"COALESCE({aggregate_alias}.{prefix}_median, global_stats.global_median) AS {prefix}_median",
        f"COALESCE({aggregate_alias}.{prefix}_std, global_stats.global_std) AS {prefix}_std",
        f"COALESCE({aggregate_alias}.{prefix}_maximum, global_stats.global_maximum) AS {prefix}_maximum",
        f"COALESCE({aggregate_alias}.{prefix}_p90, global_stats.global_p90) AS {prefix}_p90",
        f"COALESCE({aggregate_alias}.{prefix}_above_60_rate, global_stats.global_above_60_rate) AS {prefix}_above_60_rate",
        f"COALESCE({aggregate_alias}.{prefix}_above_100_rate, global_stats.global_above_100_rate) AS {prefix}_above_100_rate",
        f"COALESCE({aggregate_alias}.{prefix}_count, 0)::BIGINT AS {prefix}_count",
    ]


def ensure_environment() -> None:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model() -> CatBoostRegressor:
    print(f"Loading model: {MODEL_PATH}")
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))

    saved_features = list(model.feature_names_)
    if saved_features != FEATURE_COLUMNS:
        raise RuntimeError(
            "Saved model feature contract does not match the expected "
            "64-feature v4.1 order.\n"
            f"Saved: {saved_features}\nExpected: {FEATURE_COLUMNS}"
        )

    print(f"Verified feature count: {len(saved_features)}")
    return model


def create_source_tables(
    conn: duckdb.DuckDBPyConnection,
    config: EvaluationConfig,
) -> None:
    columns_sql = sql_identifier_list(SOURCE_COLUMNS)

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE evaluation_history AS
        SELECT {columns_sql}
        FROM model_data_base
        WHERE SEFER_TARIHI >= DATE '{config.history_start}'
          AND SEFER_TARIHI < DATE '{config.history_end_exclusive}'
    """)

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE evaluation_target AS
        SELECT {columns_sql}
        FROM model_data_base
        WHERE SEFER_TARIHI >= DATE '{config.target_start}'
          AND SEFER_TARIHI < DATE '{config.target_end_exclusive}'
    """)

    history_count = conn.execute(
        "SELECT COUNT(*) FROM evaluation_history"
    ).fetchone()[0]
    target_count = conn.execute(
        "SELECT COUNT(*) FROM evaluation_target"
    ).fetchone()[0]

    print(f"Historical rows: {history_count:,}")
    print(f"Evaluation rows: {target_count:,}")

    if history_count == 0 or target_count == 0:
        raise RuntimeError("History or evaluation target is empty.")

    if target_count != config.expected_rows:
        print(
            f"WARNING: expected {config.expected_rows:,} evaluation rows, "
            f"found {target_count:,}."
        )


def create_long_term_statistics(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE global_stats AS
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
        FROM evaluation_history
    """)

    for prefix, group_columns in HISTORICAL_GROUP_DEFINITIONS:
        group_sql = sql_identifier_list(group_columns)
        print(f"  Long-term aggregation: {prefix}")

        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE {prefix}_stats AS
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
            FROM evaluation_history
            GROUP BY {group_sql}
        """)


def create_recent_statistics(
    conn: duckdb.DuckDBPyConnection,
    config: EvaluationConfig,
) -> None:
    """
    Build recent features for each target group/date using only the frozen
    pre-period history. Evaluation-period targets are never used as history.
    """

    for prefix, group_columns in RECENT_GROUP_DEFINITIONS:
        reference_columns_sql = ", ".join(
            f"source.{column}" for column in group_columns
        )
        grouped_columns_sql = ", ".join(
            f"reference.{column}" for column in group_columns
        )
        join_condition_sql = " AND ".join(
            f"reference.{column} = history.{column}"
            for column in group_columns
        )

        for window in RECENT_WINDOWS:
            print(f"  Recent aggregation: {prefix}, {window}d")

            conn.execute(f"""
                CREATE OR REPLACE TEMP TABLE
                    {prefix}_recent_{window}d_stats AS

                WITH reference_dates AS (
                    SELECT DISTINCT
                        {reference_columns_sql},
                        source.SEFER_TARIHI AS reference_date
                    FROM evaluation_target AS source
                )

                SELECT
                    {grouped_columns_sql},
                    reference.reference_date,
                    AVG(history.target)::DOUBLE
                        AS {prefix}_recent_{window}d_average,
                    COUNT(history.target)::BIGINT
                        AS {prefix}_recent_{window}d_count
                FROM reference_dates AS reference
                LEFT JOIN evaluation_history AS history
                    ON {join_condition_sql}
                    AND history.SEFER_TARIHI
                        >= reference.reference_date - INTERVAL '{window} days'
                    AND history.SEFER_TARIHI
                        < reference.reference_date
                GROUP BY
                    {grouped_columns_sql},
                    reference.reference_date
            """)


def create_feature_table(
    conn: duckdb.DuckDBPyConnection,
) -> None:
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
    long_term_aliases: dict[str, str] = {}

    for index, (prefix, group_columns) in enumerate(
        HISTORICAL_GROUP_DEFINITIONS
    ):
        alias = f"long_term_{index}"
        long_term_aliases[prefix] = alias

        joins.append(f"""
            LEFT JOIN {prefix}_stats AS {alias}
                ON {build_join_condition("base", alias, group_columns)}
        """)

        select_expressions.extend(
            build_feature_expressions(alias, prefix)
        )

    recent_index = 0
    for prefix, group_columns in RECENT_GROUP_DEFINITIONS:
        fallback_alias = long_term_aliases[prefix]
        fallback_average = (
            f"COALESCE({fallback_alias}.{prefix}_average, "
            "global_stats.global_average)"
        )

        for window in RECENT_WINDOWS:
            alias = f"recent_{recent_index}"
            joins.append(f"""
                LEFT JOIN {prefix}_recent_{window}d_stats AS {alias}
                    ON {build_join_condition("base", alias, group_columns)}
                   AND base.SEFER_TARIHI = {alias}.reference_date
            """)

            select_expressions.extend(
                [
                    (
                        f"COALESCE("
                        f"{alias}.{prefix}_recent_{window}d_average, "
                        f"{fallback_average}"
                        f") AS {prefix}_recent_{window}d_average"
                    ),
                    (
                        f"COALESCE("
                        f"{alias}.{prefix}_recent_{window}d_count, 0"
                        f")::BIGINT AS {prefix}_recent_{window}d_count"
                    ),
                ]
            )
            recent_index += 1

    exact_alias = long_term_aliases[
        "company_route_time_weekday"
    ]
    canonical_time_alias = long_term_aliases[
        "canonical_route_time_weekday"
    ]
    canonical_alias = long_term_aliases["canonical_route"]

    select_expressions.extend(
        [
            f"""
            COALESCE(
                {exact_alias}.company_route_time_weekday_average,
                {canonical_time_alias}.canonical_route_time_weekday_average,
                {canonical_alias}.canonical_route_average,
                global_stats.global_average
            )::DOUBLE AS baseline_prediction
            """,
            f"""
            CASE
                WHEN {exact_alias}.company_route_time_weekday_average
                    IS NOT NULL
                    THEN 'company_route_time_weekday'
                WHEN {canonical_time_alias}.canonical_route_time_weekday_average
                    IS NOT NULL
                    THEN 'canonical_route_time_weekday'
                WHEN {canonical_alias}.canonical_route_average
                    IS NOT NULL
                    THEN 'canonical_route'
                ELSE 'overall_average'
            END AS baseline_source
            """,
        ]
    )

    select_sql = ",\n            ".join(select_expressions)
    joins_sql = "\n".join(joins)

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE evaluation_features AS
        SELECT
            {select_sql}
        FROM evaluation_target AS base
        CROSS JOIN global_stats
        {joins_sql}
    """)

    source_count = conn.execute(
        "SELECT COUNT(*) FROM evaluation_target"
    ).fetchone()[0]
    feature_count = conn.execute(
        "SELECT COUNT(*) FROM evaluation_features"
    ).fetchone()[0]

    if source_count != feature_count:
        raise RuntimeError(
            f"Feature joins changed row count: "
            f"{source_count:,} -> {feature_count:,}"
        )


def calculate_target_groups(
    actual: pd.Series,
    predictions: np.ndarray,
    baseline: pd.Series,
) -> pd.DataFrame:
    results = pd.DataFrame(
        {
            "actual": actual.to_numpy(),
            "catboost_prediction": predictions,
            "baseline_prediction": baseline.to_numpy(),
        }
    )

    results["catboost_absolute_error"] = (
        results["actual"] - results["catboost_prediction"]
    ).abs()
    results["baseline_absolute_error"] = (
        results["actual"] - results["baseline_prediction"]
    ).abs()
    results["catboost_squared_error"] = (
        results["actual"] - results["catboost_prediction"]
    ) ** 2
    results["baseline_squared_error"] = (
        results["actual"] - results["baseline_prediction"]
    ) ** 2

    results["target_group"] = pd.cut(
        results["actual"],
        bins=[0, 10, 20, 30, 40, 60, 100, 300],
        labels=[
            "1-10",
            "11-20",
            "21-30",
            "31-40",
            "41-60",
            "61-100",
            "101-300",
        ],
        include_lowest=True,
    )

    grouped = (
        results.groupby("target_group", observed=True)
        .agg(
            row_count=("actual", "size"),
            average_actual=("actual", "mean"),
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
        )
        .reset_index()
    )

    grouped["catboost_rmse"] = np.sqrt(grouped["catboost_mse"])
    grouped["baseline_rmse"] = np.sqrt(grouped["baseline_mse"])
    grouped["catboost_bias"] = (
        grouped["average_catboost_prediction"]
        - grouped["average_actual"]
    )
    grouped["baseline_bias"] = (
        grouped["average_baseline_prediction"]
        - grouped["average_actual"]
    )
    grouped["mae_improvement"] = (
        grouped["baseline_mae"] - grouped["catboost_mae"]
    )

    return grouped.drop(
        columns=["catboost_mse", "baseline_mse"]
    )


def run_evaluation(config: EvaluationConfig) -> None:
    ensure_environment()
    model = load_model()

    print(f"\nEvaluating: {config.label}")
    print(
        "Recent features use only history before the evaluation period; "
        "evaluation targets are never reused as history."
    )

    conn = duckdb.connect(str(DB_PATH), read_only=False)
    temp_path = TEMP_DIR.as_posix().replace("'", "''")

    conn.execute(f"SET threads = {DUCKDB_THREADS}")
    conn.execute(f"SET temp_directory = '{temp_path}'")

    try:
        create_source_tables(conn, config)

        print("\nBuilding long-term statistics...")
        create_long_term_statistics(conn)

        print("\nBuilding recent statistics...")
        create_recent_statistics(conn, config)

        print("\nCreating completed feature matrix...")
        create_feature_table(conn)

        selected_columns = FEATURE_COLUMNS + [
            "target",
            "baseline_prediction",
            "baseline_source",
        ]

        frame = conn.execute(
            f"""
            SELECT {sql_identifier_list(selected_columns)}
            FROM evaluation_features
            """
        ).fetchdf()

    finally:
        conn.close()

    for column in CATEGORICAL_FEATURES:
        frame[column] = (
            frame[column]
            .astype("string")
            .fillna("missing")
        )

    missing = frame[FEATURE_COLUMNS].isna().sum()
    if missing.sum() > 0:
        raise RuntimeError(
            "Feature matrix contains missing values:\n"
            + missing[missing > 0].to_string()
        )

    print(f"\nPredicting {len(frame):,} rows...")
    predictions = model.predict(frame[FEATURE_COLUMNS])

    actual = frame["target"]
    baseline_predictions = frame["baseline_prediction"]

    catboost_mae = mean_absolute_error(actual, predictions)
    catboost_rmse = mean_squared_error(actual, predictions) ** 0.5
    baseline_mae = mean_absolute_error(
        actual,
        baseline_predictions,
    )
    baseline_rmse = mean_squared_error(
        actual,
        baseline_predictions,
    ) ** 0.5

    print(f"\nCatBoost v4.1 {config.label} results")
    print("MAE:", catboost_mae)
    print("RMSE:", catboost_rmse)

    print("\nWeekday baseline results")
    print("MAE:", baseline_mae)
    print("RMSE:", baseline_rmse)

    print("\nComparison")
    print("MAE improvement:", baseline_mae - catboost_mae)
    print(
        "MAE improvement percentage:",
        ((baseline_mae - catboost_mae) / baseline_mae) * 100,
    )
    print("RMSE improvement:", baseline_rmse - catboost_rmse)

    if abs(baseline_mae - config.baseline_mae) > 0.02:
        print(
            "WARNING: reproduced baseline MAE differs from the "
            f"recorded value {config.baseline_mae}."
        )
    if abs(baseline_rmse - config.baseline_rmse) > 0.02:
        print(
            "WARNING: reproduced baseline RMSE differs from the "
            f"recorded value {config.baseline_rmse}."
        )

    target_groups = calculate_target_groups(
        actual,
        predictions,
        baseline_predictions,
    )

    print("\nResults by passenger-count group")
    print(target_groups.to_string(index=False))

    predictions_output = pd.DataFrame(
        {
            "actual": actual.to_numpy(),
            "catboost_prediction": predictions,
            "baseline_prediction": baseline_predictions.to_numpy(),
            "baseline_source": frame["baseline_source"].to_numpy(),
        }
    )
    predictions_output["catboost_absolute_error"] = (
        predictions_output["actual"]
        - predictions_output["catboost_prediction"]
    ).abs()
    predictions_output["baseline_absolute_error"] = (
        predictions_output["actual"]
        - predictions_output["baseline_prediction"]
    ).abs()

    summary = pd.DataFrame(
        [
            {
                "period": config.key,
                "row_count": len(frame),
                "catboost_mae": catboost_mae,
                "catboost_rmse": catboost_rmse,
                "baseline_mae": baseline_mae,
                "baseline_rmse": baseline_rmse,
                "mae_improvement": baseline_mae - catboost_mae,
                "rmse_improvement": baseline_rmse - catboost_rmse,
            }
        ]
    )

    summary_path = (
        RESULTS_DIR
        / f"catboost_v4_1_{config.key}_metrics.csv"
    )
    group_path = (
        RESULTS_DIR
        / f"catboost_v4_1_{config.key}_target_groups.csv"
    )
    prediction_path = (
        RESULTS_DIR
        / f"catboost_v4_1_{config.key}_predictions.csv"
    )

    summary.to_csv(summary_path, index=False)
    target_groups.to_csv(group_path, index=False)
    predictions_output.to_csv(prediction_path, index=False)

    print("\nSaved:")
    print(summary_path)
    print(group_path)
    print(prediction_path)
