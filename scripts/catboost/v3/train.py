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

LOSS_FUNCTION = "RMSE"
EVAL_METRIC = "RMSE"

MODEL_PATH = PROJECT_DIR / "models" / "catboost_demand_model_v3_rmse_6000.cbm"

BASELINE_MAE = 10.459917606689025
BASELINE_RMSE = 15.195470914044426

ITERATIONS = 6000
EARLY_STOPPING_ROUNDS = 300
DUCKDB_THREADS = max(1, os.cpu_count() or 4)


if not DB_PATH.exists():
    raise FileNotFoundError(
        f"Database not found: {DB_PATH}"
    )

TEMP_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Model feature definitions
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
# 3. SQL helpers
# --------------------------------------------------

def sql_identifier_list(columns: list[str]) -> str:
    return ", ".join(columns)


def create_source_tables(
    conn: duckdb.DuckDBPyConnection,
) -> None:
    """
    Materialize only the chronological source periods
    needed by this experiment.

    All grouped analytics and joins happen in DuckDB.
    """

    columns_sql = sql_identifier_list(source_columns)

    print("\nCreating DuckDB source tables...")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE history_until_2025 AS
        SELECT
            {columns_sql}
        FROM model_data_base
        WHERE SEFER_TARIHI >= DATE '2023-01-01'
          AND SEFER_TARIHI < DATE '2025-01-01'
    """)

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE validation_2025_h1 AS
        SELECT
            {columns_sql}
        FROM model_data_base
        WHERE SEFER_TARIHI >= DATE '2025-01-01'
          AND SEFER_TARIHI < DATE '2025-07-01'
    """)


def create_global_statistics_table(
    conn: duckdb.DuckDBPyConnection,
    output_table: str,
    history_where_sql: str,
) -> None:
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE {output_table} AS
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

        FROM history_until_2025
        WHERE {history_where_sql}
    """)


def create_group_statistics_table(
    conn: duckdb.DuckDBPyConnection,
    output_table: str,
    prefix: str,
    group_columns: list[str],
    history_where_sql: str,
) -> None:
    group_columns_sql = sql_identifier_list(group_columns)

    print(f"  Aggregating {prefix}")

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE {output_table} AS
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

        FROM history_until_2025
        WHERE {history_where_sql}
        GROUP BY
            {group_columns_sql}
    """)


def create_all_statistics_tables(
    conn: duckdb.DuckDBPyConnection,
    suffix: str,
    history_where_sql: str,
) -> None:
    print(f"\nBuilding DuckDB statistics: {suffix}")

    create_global_statistics_table(
        conn=conn,
        output_table=f"global_stats_{suffix}",
        history_where_sql=history_where_sql,
    )

    for prefix, group_columns in historical_group_definitions:
        create_group_statistics_table(
            conn=conn,
            output_table=f"{prefix}_{suffix}",
            prefix=prefix,
            group_columns=group_columns,
            history_where_sql=history_where_sql,
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


def create_feature_table(
    conn: duckdb.DuckDBPyConnection,
    output_table: str,
    target_from_sql: str,
    statistics_suffix: str,
) -> None:
    """
    Join every historical feature onto the target period
    inside DuckDB.
    """

    print(f"\nCreating DuckDB feature table: {output_table}")

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

    join_sql_parts = []

    for index, (
        prefix,
        group_columns,
    ) in enumerate(historical_group_definitions):
        aggregate_alias = f"aggregate_{index}"
        aggregate_table = (
            f"{prefix}_{statistics_suffix}"
        )

        join_condition = build_join_condition(
            base_alias="base",
            aggregate_alias=aggregate_alias,
            group_columns=group_columns,
        )

        join_sql_parts.append(
            f"""
            LEFT JOIN {aggregate_table} AS {aggregate_alias}
                ON {join_condition}
            """
        )

        select_expressions.extend(
            build_feature_expressions(
                aggregate_alias=aggregate_alias,
                prefix=prefix,
            )
        )

    select_sql = ",\n            ".join(
        select_expressions
    )

    joins_sql = "\n".join(join_sql_parts)

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE {output_table} AS
        SELECT
            {select_sql}
        FROM (
            {target_from_sql}
        ) AS base

        CROSS JOIN
            global_stats_{statistics_suffix}
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
# 4. Build all features in DuckDB
# --------------------------------------------------

print(f"Opening database: {DB_PATH}")

conn = duckdb.connect(
    str(DB_PATH),
    read_only=False,
)

temp_directory_sql = TEMP_DIR.as_posix().replace(
    "'",
    "''",
)

conn.execute(
    f"SET threads = {DUCKDB_THREADS}"
)

conn.execute(
    f"SET temp_directory = '{temp_directory_sql}'"
)

create_source_tables(conn)

print("\nSource summaries")
print_table_summary(
    conn,
    "history_until_2025",
)
print_table_summary(
    conn,
    "validation_2025_h1",
)


# Training rows are 2024.
# Their historical features use only 2023.
create_all_statistics_tables(
    conn=conn,
    suffix="for_train",
    history_where_sql=(
        "SEFER_TARIHI >= DATE '2023-01-01' "
        "AND SEFER_TARIHI < DATE '2024-01-01'"
    ),
)


# Validation rows are January-June 2025.
# Their historical features use 2023-2024 only.
create_all_statistics_tables(
    conn=conn,
    suffix="for_validation",
    history_where_sql=(
        "SEFER_TARIHI >= DATE '2023-01-01' "
        "AND SEFER_TARIHI < DATE '2025-01-01'"
    ),
)


create_feature_table(
    conn=conn,
    output_table="train_features_v3",
    target_from_sql="""
        SELECT *
        FROM history_until_2025
        WHERE SEFER_TARIHI >= DATE '2024-01-01'
          AND SEFER_TARIHI < DATE '2025-01-01'
    """,
    statistics_suffix="for_train",
)


create_feature_table(
    conn=conn,
    output_table="validation_features_v3",
    target_from_sql="""
        SELECT *
        FROM validation_2025_h1
    """,
    statistics_suffix="for_validation",
)


print("\nFinished feature-table summaries")
print_table_summary(
    conn,
    "train_features_v3",
)
print_table_summary(
    conn,
    "validation_features_v3",
)


# --------------------------------------------------
# 5. Fetch only completed model matrices
# --------------------------------------------------

model_columns = [
    *feature_columns,
    "target",
]

model_columns_sql = sql_identifier_list(
    model_columns
)


print("\nLoading completed training matrix into Pandas...")

train_features_df = conn.execute(f"""
    SELECT
        {model_columns_sql}
    FROM train_features_v3
""").fetchdf()


print("Loading completed validation matrix into Pandas...")

validation_features_df = conn.execute(f"""
    SELECT
        {model_columns_sql}
    FROM validation_features_v3
""").fetchdf()


conn.close()


print("\nLoaded completed matrices")
print("Training:", train_features_df.shape)
print("Validation:", validation_features_df.shape)
print("Feature count:", len(feature_columns))


# --------------------------------------------------
# 6. Prepare CatBoost matrices
# --------------------------------------------------

for column in categorical_features:
    train_features_df[column] = (
        train_features_df[column]
        .astype("string")
        .fillna("missing")
    )

    validation_features_df[column] = (
        validation_features_df[column]
        .astype("string")
        .fillna("missing")
    )


train_missing = (
    train_features_df[feature_columns]
    .isna()
    .sum()
)

validation_missing = (
    validation_features_df[feature_columns]
    .isna()
    .sum()
)


if train_missing.sum() > 0:
    print("\nMissing training values")
    print(
        train_missing[
            train_missing > 0
        ].to_string()
    )
    raise ValueError(
        "Training features contain missing values."
    )


if validation_missing.sum() > 0:
    print("\nMissing validation values")
    print(
        validation_missing[
            validation_missing > 0
        ].to_string()
    )
    raise ValueError(
        "Validation features contain missing values."
    )


X_train = train_features_df[feature_columns]
y_train = train_features_df["target"]

X_validation = validation_features_df[
    feature_columns
]
y_validation = validation_features_df[
    "target"
]


print("\nModel matrices")
print("X_train:", X_train.shape)
print("X_validation:", X_validation.shape)


del train_features_df
del validation_features_df


# --------------------------------------------------
# 7. Train CatBoost v3 MAE
# --------------------------------------------------

model = CatBoostRegressor(
    loss_function=LOSS_FUNCTION,
    eval_metric=EVAL_METRIC,

    iterations=ITERATIONS,
    learning_rate=0.08,
    depth=8,

    task_type="GPU",
    devices="0",

    random_seed=42,
    verbose=100,
    metric_period=5,

    allow_writing_files=False,
)


print(
    f"\nTraining CatBoost v3 with "
    f"{LOSS_FUNCTION} loss..."
)


model.fit(
    X_train,
    y_train,

    cat_features=categorical_features,

    eval_set=(
        X_validation,
        y_validation,
    ),

    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    use_best_model=True,
)


# --------------------------------------------------
# 8. Validation metrics
# --------------------------------------------------

validation_predictions = model.predict(
    X_validation
)


validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions,
)

validation_rmse = mean_squared_error(
    y_validation,
    validation_predictions,
) ** 0.5


print("\nCatBoost v3 validation results")
print("Loss function:", LOSS_FUNCTION)
print("Best iteration:", model.get_best_iteration())
print("Validation MAE:", validation_mae)
print("Validation RMSE:", validation_rmse)


mae_improvement = (
    BASELINE_MAE - validation_mae
)

rmse_improvement = (
    BASELINE_RMSE - validation_rmse
)

mae_improvement_percentage = (
    mae_improvement / BASELINE_MAE
) * 100


print("\nComparison against weekday baseline")
print("Baseline MAE:", BASELINE_MAE)
print("CatBoost v3 MAE:", validation_mae)
print("MAE improvement:", mae_improvement)
print(
    "MAE improvement percentage:",
    mae_improvement_percentage,
)

print("\nBaseline RMSE:", BASELINE_RMSE)
print("CatBoost v3 RMSE:", validation_rmse)
print("RMSE improvement:", rmse_improvement)


# --------------------------------------------------
# 9. Validation target-group analysis
# --------------------------------------------------

validation_results_df = pd.DataFrame({
    "actual": y_validation.to_numpy(),
    "prediction": validation_predictions,
})

validation_results_df["absolute_error"] = (
    validation_results_df["actual"]
    - validation_results_df["prediction"]
).abs()

validation_results_df["squared_error"] = (
    validation_results_df["actual"]
    - validation_results_df["prediction"]
) ** 2

validation_results_df["target_group"] = pd.cut(
    validation_results_df["actual"],
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
    validation_results_df
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
        average_prediction=(
            "prediction",
            "mean",
        ),
        mae=(
            "absolute_error",
            "mean",
        ),
        mean_squared_error=(
            "squared_error",
            "mean",
        ),
    )
)

target_group_results["rmse"] = (
    target_group_results[
        "mean_squared_error"
    ] ** 0.5
)

target_group_results["bias"] = (
    target_group_results["average_prediction"]
    - target_group_results["average_actual"]
)

target_group_results = (
    target_group_results.drop(
        columns=["mean_squared_error"]
    )
)


print(
    "\nValidation results by passenger-count group"
)
print(target_group_results.to_string())


target_group_results.to_csv(
    "results/catboost_v3_mae_6000_validation_target_groups.csv"
)


# --------------------------------------------------
# 10. Feature importance
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.get_feature_importance(),
}).sort_values(
    "importance",
    ascending=False,
)


print("\nTop 30 features")
print(
    feature_importance
    .head(30)
    .to_string(index=False)
)


feature_importance.to_csv(
    "results/catboost_v3_mae_6000_feature_importance.csv",
    index=False,
)


# --------------------------------------------------
# 11. Save the new model
# --------------------------------------------------

model.save_model(
    str(MODEL_PATH)
)


print(
    f"\nModel saved to: {MODEL_PATH}"
)
