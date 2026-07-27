from pathlib import Path
import os

import duckdb
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_DIR / "analysis.duckdb"
TEMP_DIR = PROJECT_DIR / "duckdb_temp"

MODEL_PATH = PROJECT_DIR / "models" / "catboost_demand_model_v3_mae_6000.cbm"

V2_TEST_MAE = 10.228807877656902
V2_TEST_RMSE = 17.483430588957827

DUCKDB_THREADS = max(1, os.cpu_count() or 4)


if not DB_PATH.exists():
    raise FileNotFoundError(
        f"Database not found: {DB_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"CatBoost model not found: {MODEL_PATH}"
    )

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# --------------------------------------------------
# 2. Feature definitions
# --------------------------------------------------

categorical_features = [
    "FIRMA_ID",
    "GUZERGAH_KODU",
    "canonical_guzergah_id",
    "month",
    "day_of_week",
    "departure_30min_bucket",
]

calendar_numeric_features = [
    "week_of_year",
    "departure_minute",
]

historical_group_definitions = [
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
        [
            "canonical_guzergah_id",
        ],
    ),
    (
        "company",
        [
            "FIRMA_ID",
        ],
    ),
]

historical_statistic_suffixes = [
    "average",
    "median",
    "std",
    "maximum",
    "p90",
    "above_60_rate",
    "above_100_rate",
    "count",
]

historical_features = [
    f"{prefix}_{suffix}"
    for prefix, _ in historical_group_definitions
    for suffix in historical_statistic_suffixes
]

feature_columns = (
    categorical_features
    + calendar_numeric_features
    + historical_features
)

source_columns = [
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


# --------------------------------------------------
# 3. SQL helper functions
# --------------------------------------------------

def sql_identifier_list(
    columns: list[str],
) -> str:
    return ", ".join(columns)


def create_source_tables(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """
    Test period:
        2025-07-01 through 2025-12-31

    Historical features use only information available
    before the test period:
        2023-01-01 through 2025-06-30
    """

    columns_sql = sql_identifier_list(
        source_columns
    )

    print("\nCreating DuckDB source tables...")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE
            history_before_test AS

        SELECT
            {columns_sql}

        FROM model_data_base

        WHERE SEFER_TARIHI >= DATE '2023-01-01'
          AND SEFER_TARIHI < DATE '2025-07-01'
    """)

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE
            test_2025_h2 AS

        SELECT
            {columns_sql}

        FROM model_data_base

        WHERE SEFER_TARIHI >= DATE '2025-07-01'
          AND SEFER_TARIHI < DATE '2026-01-01'
    """)


def create_global_statistics_table(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    print("\nCreating global historical statistics...")

    conn.execute("""
        CREATE OR REPLACE TEMP TABLE
            global_stats_test AS

        SELECT
            AVG(target)::DOUBLE
                AS global_average,

            MEDIAN(target)::DOUBLE
                AS global_median,

            STDDEV_SAMP(target)::DOUBLE
                AS global_std,

            MAX(target)::DOUBLE
                AS global_maximum,

            QUANTILE_CONT(target, 0.90)::DOUBLE
                AS global_p90,

            AVG(
                CASE
                    WHEN target > 60 THEN 1.0
                    ELSE 0.0
                END
            )::DOUBLE
                AS global_above_60_rate,

            AVG(
                CASE
                    WHEN target > 100 THEN 1.0
                    ELSE 0.0
                END
            )::DOUBLE
                AS global_above_100_rate

        FROM history_before_test
    """)


def create_group_statistics_table(
    conn: duckdb.DuckDBPyConnection,
    output_table: str,
    prefix: str,
    group_columns: list[str],
) -> None:
    group_columns_sql = sql_identifier_list(
        group_columns
    )

    print(f"  Aggregating {prefix}")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE
            {output_table} AS

        SELECT
            {group_columns_sql},

            AVG(target)::DOUBLE
                AS {prefix}_average,

            MEDIAN(target)::DOUBLE
                AS {prefix}_median,

            STDDEV_SAMP(target)::DOUBLE
                AS {prefix}_std,

            MAX(target)::DOUBLE
                AS {prefix}_maximum,

            QUANTILE_CONT(target, 0.90)::DOUBLE
                AS {prefix}_p90,

            AVG(
                CASE
                    WHEN target > 60 THEN 1.0
                    ELSE 0.0
                END
            )::DOUBLE
                AS {prefix}_above_60_rate,

            AVG(
                CASE
                    WHEN target > 100 THEN 1.0
                    ELSE 0.0
                END
            )::DOUBLE
                AS {prefix}_above_100_rate,

            COUNT(*)::BIGINT
                AS {prefix}_count

        FROM history_before_test

        GROUP BY
            {group_columns_sql}
    """)


def create_all_statistics_tables(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    create_global_statistics_table(conn)

    print("\nCreating grouped historical statistics...")

    for prefix, group_columns in (
        historical_group_definitions
    ):
        create_group_statistics_table(
            conn=conn,
            output_table=f"{prefix}_test",
            prefix=prefix,
            group_columns=group_columns,
        )


def build_join_condition(
    base_alias: str,
    aggregate_alias: str,
    group_columns: list[str],
) -> str:
    return " AND ".join(
        f"{base_alias}.{column} = "
        f"{aggregate_alias}.{column}"
        for column in group_columns
    )


def build_feature_expressions(
    aggregate_alias: str,
    prefix: str,
) -> list[str]:
    """
    These fallback rules match the v3 training file.
    """

    return [
        (
            f"COALESCE("
            f"{aggregate_alias}.{prefix}_average, "
            f"global_stats.global_average"
            f") AS {prefix}_average"
        ),
        (
            f"COALESCE("
            f"{aggregate_alias}.{prefix}_median, "
            f"global_stats.global_median"
            f") AS {prefix}_median"
        ),
        (
            f"COALESCE("
            f"{aggregate_alias}.{prefix}_std, "
            f"global_stats.global_std"
            f") AS {prefix}_std"
        ),
        (
            f"COALESCE("
            f"{aggregate_alias}.{prefix}_maximum, "
            f"global_stats.global_maximum"
            f") AS {prefix}_maximum"
        ),
        (
            f"COALESCE("
            f"{aggregate_alias}.{prefix}_p90, "
            f"global_stats.global_p90"
            f") AS {prefix}_p90"
        ),
        (
            f"COALESCE("
            f"{aggregate_alias}.{prefix}_above_60_rate, "
            f"global_stats.global_above_60_rate"
            f") AS {prefix}_above_60_rate"
        ),
        (
            f"COALESCE("
            f"{aggregate_alias}.{prefix}_above_100_rate, "
            f"global_stats.global_above_100_rate"
            f") AS {prefix}_above_100_rate"
        ),
        (
            f"COALESCE("
            f"{aggregate_alias}.{prefix}_count, 0"
            f")::BIGINT AS {prefix}_count"
        ),
    ]


def create_test_feature_table(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """
    Join all v3 features and the weekday baseline onto
    the test rows entirely inside DuckDB.
    """

    print("\nCreating complete DuckDB test feature table...")

    select_expressions = [
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

    joins = []
    aggregate_aliases: dict[str, str] = {}

    for index, (
        prefix,
        group_columns,
    ) in enumerate(historical_group_definitions):
        aggregate_alias = f"aggregate_{index}"
        aggregate_aliases[prefix] = aggregate_alias

        join_condition = build_join_condition(
            base_alias="base",
            aggregate_alias=aggregate_alias,
            group_columns=group_columns,
        )

        joins.append(
            f"""
            LEFT JOIN {prefix}_test
                AS {aggregate_alias}
                ON {join_condition}
            """
        )

        select_expressions.extend(
            build_feature_expressions(
                aggregate_alias=aggregate_alias,
                prefix=prefix,
            )
        )

    exact_alias = aggregate_aliases[
        "company_route_time_weekday"
    ]

    canonical_time_alias = aggregate_aliases[
        "canonical_route_time_weekday"
    ]

    canonical_route_alias = aggregate_aliases[
        "canonical_route"
    ]

    # Fair weekday baseline:
    # exact company-route-time-weekday
    # -> canonical route-time-weekday
    # -> canonical route
    # -> overall historical average
    select_expressions.append(
        f"""
        COALESCE(
            {exact_alias}.
                company_route_time_weekday_average,

            {canonical_time_alias}.
                canonical_route_time_weekday_average,

            {canonical_route_alias}.
                canonical_route_average,

            global_stats.global_average
        ) AS baseline_prediction
        """
    )

    select_expressions.append(
        f"""
        CASE
            WHEN {exact_alias}.
                company_route_time_weekday_average
                IS NOT NULL
                THEN 'company_route_time_weekday'

            WHEN {canonical_time_alias}.
                canonical_route_time_weekday_average
                IS NOT NULL
                THEN 'canonical_route_time_weekday'

            WHEN {canonical_route_alias}.
                canonical_route_average
                IS NOT NULL
                THEN 'canonical_route'

            ELSE 'overall_average'
        END AS baseline_source
        """
    )

    select_sql = ",\n            ".join(
        select_expressions
    )

    joins_sql = "\n".join(joins)

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE
            test_features_v3 AS

        SELECT
            {select_sql}

        FROM test_2025_h2 AS base

        CROSS JOIN global_stats_test
            AS global_stats

        {joins_sql}
    """)


def print_table_summary(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
) -> None:
    row_count, min_date, max_date = conn.execute(f"""
        SELECT
            COUNT(*),
            MIN(SEFER_TARIHI),
            MAX(SEFER_TARIHI)

        FROM {table_name}
    """).fetchone()

    print(
        f"{table_name}: {row_count:,} rows, "
        f"{min_date} to {max_date}"
    )


# --------------------------------------------------
# 4. Build test features in DuckDB
# --------------------------------------------------

print(f"Opening database: {DB_PATH}")

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


create_source_tables(conn)

print("\nSource summaries")

print_table_summary(
    conn,
    "history_before_test",
)

print_table_summary(
    conn,
    "test_2025_h2",
)


create_all_statistics_tables(conn)
create_test_feature_table(conn)


print("\nFinished feature-table summary")

print_table_summary(
    conn,
    "test_features_v3",
)


# --------------------------------------------------
# 5. Fetch only the completed test matrix
# --------------------------------------------------

columns_to_fetch = [
    "SEFER_TARIHI",
    *feature_columns,
    "target",
    "baseline_prediction",
    "baseline_source",
]

columns_to_fetch_sql = sql_identifier_list(
    columns_to_fetch
)


print("\nLoading completed test matrix into Pandas...")

test_features_df = conn.execute(f"""
    SELECT
        {columns_to_fetch_sql}

    FROM test_features_v3
""").fetchdf()


conn.close()


print("\nLoaded test matrix")
print("Test:", test_features_df.shape)
print("Feature count:", len(feature_columns))


# --------------------------------------------------
# 6. Prepare CatBoost test matrix
# --------------------------------------------------

for column in categorical_features:
    test_features_df[column] = (
        test_features_df[column]
        .astype("string")
        .fillna("missing")
    )


missing_values = (
    test_features_df[feature_columns]
    .isna()
    .sum()
)


if missing_values.sum() > 0:
    print("\nMissing test values")

    print(
        missing_values[
            missing_values > 0
        ].to_string()
    )

    raise ValueError(
        "Test feature matrix contains "
        "missing values."
    )


X_test = test_features_df[
    feature_columns
]

y_test = test_features_df[
    "target"
]


print("\nModel matrix")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# --------------------------------------------------
# 7. Load and validate the selected MAE v3 model
# --------------------------------------------------

print(f"\nLoading model: {MODEL_PATH}")

model = CatBoostRegressor()
model.load_model(
    str(MODEL_PATH)
)


model_feature_names = list(
    model.feature_names_
)


if (
    model_feature_names
    and model_feature_names != feature_columns
):
    missing_from_test = [
        feature
        for feature in model_feature_names
        if feature not in feature_columns
    ]

    extra_in_test = [
        feature
        for feature in feature_columns
        if feature not in model_feature_names
    ]

    raise ValueError(
        "Model feature columns do not match the "
        "test feature columns.\n"
        f"Missing from test: {missing_from_test}\n"
        f"Extra in test: {extra_in_test}"
    )


# --------------------------------------------------
# 8. Create predictions
# --------------------------------------------------

print("Creating CatBoost v3 MAE test predictions...")

catboost_predictions = model.predict(
    X_test
)

baseline_predictions = test_features_df[
    "baseline_prediction"
].to_numpy()


# --------------------------------------------------
# 9. Calculate overall test metrics
# --------------------------------------------------

catboost_test_mae = mean_absolute_error(
    y_test,
    catboost_predictions,
)

catboost_test_rmse = mean_squared_error(
    y_test,
    catboost_predictions,
) ** 0.5


baseline_test_mae = mean_absolute_error(
    y_test,
    baseline_predictions,
)

baseline_test_rmse = mean_squared_error(
    y_test,
    baseline_predictions,
) ** 0.5


baseline_mae_improvement = (
    baseline_test_mae - catboost_test_mae
)

baseline_rmse_improvement = (
    baseline_test_rmse - catboost_test_rmse
)

baseline_mae_improvement_percentage = (
    baseline_mae_improvement
    / baseline_test_mae
) * 100


v2_mae_improvement = (
    V2_TEST_MAE - catboost_test_mae
)

v2_rmse_improvement = (
    V2_TEST_RMSE - catboost_test_rmse
)


print("\nCatBoost v3 MAE test results")
print("Test MAE:", catboost_test_mae)
print("Test RMSE:", catboost_test_rmse)


print("\nWeekday baseline test results")
print("Test MAE:", baseline_test_mae)
print("Test RMSE:", baseline_test_rmse)


print("\nComparison against test baseline")

print("Baseline test MAE:", baseline_test_mae)
print("CatBoost v3 test MAE:", catboost_test_mae)
print("MAE improvement:", baseline_mae_improvement)

print(
    "MAE improvement percentage:",
    baseline_mae_improvement_percentage,
)

print("\nBaseline test RMSE:", baseline_test_rmse)
print("CatBoost v3 test RMSE:", catboost_test_rmse)
print("RMSE improvement:", baseline_rmse_improvement)


print("\nComparison against CatBoost v2 test")

print("CatBoost v2 test MAE:", V2_TEST_MAE)
print("CatBoost v3 test MAE:", catboost_test_mae)
print("v3 MAE improvement over v2:", v2_mae_improvement)

print("\nCatBoost v2 test RMSE:", V2_TEST_RMSE)
print("CatBoost v3 test RMSE:", catboost_test_rmse)
print("v3 RMSE improvement over v2:", v2_rmse_improvement)


# --------------------------------------------------
# 10. Baseline fallback usage
# --------------------------------------------------

baseline_source_percentages = (
    test_features_df["baseline_source"]
    .value_counts(normalize=True)
    .mul(100)
)


print("\nBaseline prediction source percentages")
print(baseline_source_percentages.to_string())


# --------------------------------------------------
# 11. Build results table for analysis
# --------------------------------------------------

results_df = pd.DataFrame({
    "date":
        test_features_df[
            "SEFER_TARIHI"
        ].to_numpy(),

    "actual":
        y_test.to_numpy(),

    "catboost_prediction":
        catboost_predictions,

    "baseline_prediction":
        baseline_predictions,
})


results_df["catboost_absolute_error"] = (
    results_df["actual"]
    - results_df["catboost_prediction"]
).abs()

results_df["baseline_absolute_error"] = (
    results_df["actual"]
    - results_df["baseline_prediction"]
).abs()

results_df["catboost_squared_error"] = (
    results_df["actual"]
    - results_df["catboost_prediction"]
) ** 2

results_df["baseline_squared_error"] = (
    results_df["actual"]
    - results_df["baseline_prediction"]
) ** 2


# --------------------------------------------------
# 12. Absolute-error percentiles
# --------------------------------------------------

print("\nAbsolute-error percentiles")

for percentile in [
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
]:
    catboost_error = (
        results_df[
            "catboost_absolute_error"
        ]
        .quantile(percentile)
    )

    baseline_error = (
        results_df[
            "baseline_absolute_error"
        ]
        .quantile(percentile)
    )

    print(
        f"{percentile:>5.0%} percentile | "
        f"CatBoost: {catboost_error:.3f} | "
        f"Baseline: {baseline_error:.3f}"
    )


# --------------------------------------------------
# 13. Monthly analysis
# --------------------------------------------------

results_df["month"] = (
    pd.to_datetime(
        results_df["date"]
    ).dt.month
)


monthly_results = (
    results_df
    .groupby("month")
    .agg(
        row_count=(
            "actual",
            "size",
        ),

        average_actual=(
            "actual",
            "mean",
        ),

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

        catboost_mean_squared_error=(
            "catboost_squared_error",
            "mean",
        ),

        baseline_mean_squared_error=(
            "baseline_squared_error",
            "mean",
        ),
    )
)


monthly_results["catboost_rmse"] = (
    monthly_results[
        "catboost_mean_squared_error"
    ] ** 0.5
)

monthly_results["baseline_rmse"] = (
    monthly_results[
        "baseline_mean_squared_error"
    ] ** 0.5
)

monthly_results["mae_improvement"] = (
    monthly_results["baseline_mae"]
    - monthly_results["catboost_mae"]
)

monthly_results["catboost_bias"] = (
    monthly_results[
        "average_catboost_prediction"
    ]
    - monthly_results["average_actual"]
)

monthly_results["baseline_bias"] = (
    monthly_results[
        "average_baseline_prediction"
    ]
    - monthly_results["average_actual"]
)

monthly_results = monthly_results.drop(
    columns=[
        "catboost_mean_squared_error",
        "baseline_mean_squared_error",
    ]
)


print("\nMonthly test results")
print(monthly_results.to_string())


monthly_results.to_csv(
    "results/catboost_v3_mae_test_monthly_results.csv"
)


# --------------------------------------------------
# 14. Passenger-count group analysis
# --------------------------------------------------

results_df["target_group"] = pd.cut(
    results_df["actual"],
    bins=[
        0,
        10,
        20,
        30,
        40,
        60,
        100,
        300,
    ],
    labels=[
        "1-10",
        "11-20",
        "21-30",
        "31-40",
        "41-60",
        "61-100",
        "101-300",
    ],
)


target_group_results = (
    results_df
    .groupby(
        "target_group",
        observed=True,
    )
    .agg(
        row_count=(
            "actual",
            "size",
        ),

        average_actual=(
            "actual",
            "mean",
        ),

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

        catboost_mean_squared_error=(
            "catboost_squared_error",
            "mean",
        ),

        baseline_mean_squared_error=(
            "baseline_squared_error",
            "mean",
        ),
    )
)


target_group_results["catboost_rmse"] = (
    target_group_results[
        "catboost_mean_squared_error"
    ] ** 0.5
)

target_group_results["baseline_rmse"] = (
    target_group_results[
        "baseline_mean_squared_error"
    ] ** 0.5
)

target_group_results["mae_improvement"] = (
    target_group_results["baseline_mae"]
    - target_group_results["catboost_mae"]
)

target_group_results["catboost_bias"] = (
    target_group_results[
        "average_catboost_prediction"
    ]
    - target_group_results["average_actual"]
)

target_group_results["baseline_bias"] = (
    target_group_results[
        "average_baseline_prediction"
    ]
    - target_group_results["average_actual"]
)

target_group_results = target_group_results.drop(
    columns=[
        "catboost_mean_squared_error",
        "baseline_mean_squared_error",
    ]
)


print(
    "\nPrediction results by passenger-count group"
)

print(
    target_group_results.to_string()
)


target_group_results.to_csv(
    "results/catboost_v3_mae_test_target_groups.csv"
)


# --------------------------------------------------
# 15. Final summary
# --------------------------------------------------

print("\nFinal test summary")

print(
    f"CatBoost v3 MAE: "
    f"{catboost_test_mae:.6f}"
)

print(
    f"CatBoost v3 RMSE: "
    f"{catboost_test_rmse:.6f}"
)

print(
    f"Baseline MAE: "
    f"{baseline_test_mae:.6f}"
)

print(
    f"Baseline RMSE: "
    f"{baseline_test_rmse:.6f}"
)

print(
    f"MAE improvement over baseline: "
    f"{baseline_mae_improvement_percentage:.3f}%"
)

print(
    f"MAE improvement over v2: "
    f"{v2_mae_improvement:.6f}"
)

print(
    f"RMSE improvement over v2: "
    f"{v2_rmse_improvement:.6f}"
)
