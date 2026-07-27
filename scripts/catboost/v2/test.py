from pathlib import Path

import duckdb
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_DIR / "analysis.duckdb"
MODEL_PATH = PROJECT_DIR / "models" / "catboost_demand_model_v2.cbm"


if not DB_PATH.exists():
    raise FileNotFoundError(
        f"Database not found: {DB_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"CatBoost model not found: {MODEL_PATH}"
    )


# --------------------------------------------------
# 2. Define model columns
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

historical_average_features = [
    "company_route_time_weekday_average",
    "company_route_time_average",
    "canonical_route_time_weekday_average",
    "canonical_route_average",
    "company_average",
]

historical_count_features = [
    "company_route_time_weekday_count",
    "company_route_time_count",
    "canonical_route_time_weekday_count",
    "canonical_route_count",
    "company_count",
]

feature_columns = (
    categorical_features
    + calendar_numeric_features
    + historical_average_features
    + historical_count_features
)


# Columns that must be loaded from DuckDB
selected_columns = [
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

selected_columns_sql = ", ".join(selected_columns)


# --------------------------------------------------
# 3. Load history and test data
# --------------------------------------------------

print(f"Opening database: {DB_PATH}")

conn = duckdb.connect(
    str(DB_PATH),
    read_only=True,
)


# Everything known before the test period.
#
# Test begins on 2025-07-01, so historical features
# may use rows up to 2025-06-30.
history_before_test_df = conn.execute(f"""
    SELECT {selected_columns_sql}
    FROM model_data_base
    WHERE SEFER_TARIHI >= DATE '2023-01-01'
      AND SEFER_TARIHI < DATE '2025-07-01'
""").fetchdf()


# Untouched test period.
test_df = conn.execute(f"""
    SELECT {selected_columns_sql}
    FROM model_data_base
    WHERE SEFER_TARIHI >= DATE '2025-07-01'
      AND SEFER_TARIHI < DATE '2026-01-01'
""").fetchdf()


conn.close()


print("\nLoaded data")
print(
    "Historical rows:",
    len(history_before_test_df),
    history_before_test_df["SEFER_TARIHI"].min(),
    history_before_test_df["SEFER_TARIHI"].max(),
)

print(
    "Test rows:",
    len(test_df),
    test_df["SEFER_TARIHI"].min(),
    test_df["SEFER_TARIHI"].max(),
)


if history_before_test_df.empty:
    raise ValueError(
        "Historical dataset is empty."
    )

if test_df.empty:
    raise ValueError(
        "Test dataset is empty."
    )


# --------------------------------------------------
# 4. Helper: create aggregate table
# --------------------------------------------------

def create_aggregate_table(
    history_df: pd.DataFrame,
    group_columns: list[str],
    average_column_name: str,
    count_column_name: str,
) -> pd.DataFrame:
    """
    Calculate historical average demand and historical
    observation count for the requested grouping.
    """

    aggregate_table = (
        history_df
        .groupby(
            group_columns,
            as_index=False,
        )
        .agg(
            historical_average=(
                "target",
                "mean",
            ),
            historical_count=(
                "target",
                "size",
            ),
        )
        .rename(
            columns={
                "historical_average":
                    average_column_name,

                "historical_count":
                    count_column_name,
            }
        )
    )

    return aggregate_table


# --------------------------------------------------
# 5. Helper: add historical model features
# --------------------------------------------------

def add_historical_features(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach historical demand averages and counts to
    target_df.

    history_df must contain only rows that occurred
    before target_df.
    """

    result_df = target_df.copy()


    # ----------------------------------------------
    # A. Company + canonical route
    #    + time bucket + weekday
    # ----------------------------------------------

    company_route_time_weekday_columns = [
        "FIRMA_ID",
        "canonical_guzergah_id",
        "departure_30min_bucket",
        "day_of_week",
    ]

    company_route_time_weekday_table = (
        create_aggregate_table(
            history_df=history_df,

            group_columns=(
                company_route_time_weekday_columns
            ),

            average_column_name=(
                "company_route_time_weekday_average"
            ),

            count_column_name=(
                "company_route_time_weekday_count"
            ),
        )
    )

    result_df = result_df.merge(
        company_route_time_weekday_table,
        on=company_route_time_weekday_columns,
        how="left",
    )


    # ----------------------------------------------
    # B. Company + canonical route
    #    + time bucket
    # ----------------------------------------------

    company_route_time_columns = [
        "FIRMA_ID",
        "canonical_guzergah_id",
        "departure_30min_bucket",
    ]

    company_route_time_table = (
        create_aggregate_table(
            history_df=history_df,

            group_columns=(
                company_route_time_columns
            ),

            average_column_name=(
                "company_route_time_average"
            ),

            count_column_name=(
                "company_route_time_count"
            ),
        )
    )

    result_df = result_df.merge(
        company_route_time_table,
        on=company_route_time_columns,
        how="left",
    )


    # ----------------------------------------------
    # C. Canonical route
    #    + time bucket + weekday
    # ----------------------------------------------

    canonical_route_time_weekday_columns = [
        "canonical_guzergah_id",
        "departure_30min_bucket",
        "day_of_week",
    ]

    canonical_route_time_weekday_table = (
        create_aggregate_table(
            history_df=history_df,

            group_columns=(
                canonical_route_time_weekday_columns
            ),

            average_column_name=(
                "canonical_route_time_weekday_average"
            ),

            count_column_name=(
                "canonical_route_time_weekday_count"
            ),
        )
    )

    result_df = result_df.merge(
        canonical_route_time_weekday_table,
        on=canonical_route_time_weekday_columns,
        how="left",
    )


    # ----------------------------------------------
    # D. Canonical route only
    # ----------------------------------------------

    canonical_route_columns = [
        "canonical_guzergah_id",
    ]

    canonical_route_table = (
        create_aggregate_table(
            history_df=history_df,

            group_columns=canonical_route_columns,

            average_column_name=(
                "canonical_route_average"
            ),

            count_column_name=(
                "canonical_route_count"
            ),
        )
    )

    result_df = result_df.merge(
        canonical_route_table,
        on=canonical_route_columns,
        how="left",
    )


    # ----------------------------------------------
    # E. Company only
    # ----------------------------------------------

    company_columns = [
        "FIRMA_ID",
    ]

    company_table = create_aggregate_table(
        history_df=history_df,

        group_columns=company_columns,

        average_column_name="company_average",
        count_column_name="company_count",
    )

    result_df = result_df.merge(
        company_table,
        on=company_columns,
        how="left",
    )


    # ----------------------------------------------
    # F. Fill missing values
    # ----------------------------------------------

    overall_average = history_df["target"].mean()

    for column in historical_average_features:
        result_df[column] = (
            result_df[column]
            .fillna(overall_average)
        )

    for column in historical_count_features:
        result_df[column] = (
            result_df[column]
            .fillna(0)
            .astype("int64")
        )

    return result_df


# --------------------------------------------------
# 6. Create CatBoost test features
# --------------------------------------------------

print("\nCreating historical features for test data...")

test_features_df = add_historical_features(
    target_df=test_df,
    history_df=history_before_test_df,
)

print(
    "Test feature dataset shape:",
    test_features_df.shape,
)


# --------------------------------------------------
# 7. Prepare categorical columns
# --------------------------------------------------

for column in categorical_features:
    test_features_df[column] = (
        test_features_df[column]
        .astype("string")
        .fillna("missing")
    )


# --------------------------------------------------
# 8. Prepare model matrix
# --------------------------------------------------

X_test = test_features_df[feature_columns]
y_test = test_features_df["target"]

print("\nModel matrix")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# --------------------------------------------------
# 9. Load saved CatBoost model
# --------------------------------------------------

print(f"\nLoading model: {MODEL_PATH}")

model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))


# --------------------------------------------------
# 10. Predict test rows
# --------------------------------------------------

print("Creating CatBoost predictions...")

catboost_test_predictions = model.predict(
    X_test
)


# --------------------------------------------------
# 11. Calculate CatBoost test metrics
# --------------------------------------------------

catboost_test_mae = mean_absolute_error(
    y_test,
    catboost_test_predictions,
)

catboost_test_rmse = mean_squared_error(
    y_test,
    catboost_test_predictions,
) ** 0.5


print("\nCatBoost v2 test results")
print("Test MAE:", catboost_test_mae)
print("Test RMSE:", catboost_test_rmse)


# --------------------------------------------------
# 12. Build a fair weekday baseline for test
# --------------------------------------------------
#
# We should not compare the test result against the
# old validation baseline score.
#
# Instead, create a new baseline using the same
# history available before the test period.
# --------------------------------------------------

print("\nCreating test-period weekday baseline...")


# Level 1:
# company + route + time bucket + weekday

baseline_company_columns = [
    "FIRMA_ID",
    "canonical_guzergah_id",
    "departure_30min_bucket",
    "day_of_week",
]

baseline_company_table = (
    history_before_test_df
    .groupby(
        baseline_company_columns,
        as_index=False,
    )["target"]
    .mean()
    .rename(
        columns={
            "target":
                "company_route_time_weekday_prediction"
        }
    )
)


# Level 2:
# canonical route + time bucket + weekday

baseline_canonical_time_columns = [
    "canonical_guzergah_id",
    "departure_30min_bucket",
    "day_of_week",
]

baseline_canonical_time_table = (
    history_before_test_df
    .groupby(
        baseline_canonical_time_columns,
        as_index=False,
    )["target"]
    .mean()
    .rename(
        columns={
            "target":
                "canonical_route_time_weekday_prediction"
        }
    )
)


# Level 3:
# canonical route only

baseline_canonical_route_table = (
    history_before_test_df
    .groupby(
        "canonical_guzergah_id",
        as_index=False,
    )["target"]
    .mean()
    .rename(
        columns={
            "target":
                "canonical_route_prediction"
        }
    )
)


# Level 4:
# global historical average

baseline_overall_average = (
    history_before_test_df["target"].mean()
)


# --------------------------------------------------
# 13. Apply baseline to test rows
# --------------------------------------------------

baseline_test_df = test_df.merge(
    baseline_company_table,
    on=baseline_company_columns,
    how="left",
)

baseline_test_df = baseline_test_df.merge(
    baseline_canonical_time_table,
    on=baseline_canonical_time_columns,
    how="left",
)

baseline_test_df = baseline_test_df.merge(
    baseline_canonical_route_table,
    on="canonical_guzergah_id",
    how="left",
)


baseline_test_df["baseline_prediction"] = (
    baseline_test_df[
        "company_route_time_weekday_prediction"
    ]
    .fillna(
        baseline_test_df[
            "canonical_route_time_weekday_prediction"
        ]
    )
    .fillna(
        baseline_test_df[
            "canonical_route_prediction"
        ]
    )
    .fillna(baseline_overall_average)
)


# --------------------------------------------------
# 14. Calculate baseline test metrics
# --------------------------------------------------

baseline_test_mae = mean_absolute_error(
    baseline_test_df["target"],
    baseline_test_df["baseline_prediction"],
)

baseline_test_rmse = mean_squared_error(
    baseline_test_df["target"],
    baseline_test_df["baseline_prediction"],
) ** 0.5


print("\nWeekday baseline test results")
print("Test MAE:", baseline_test_mae)
print("Test RMSE:", baseline_test_rmse)


# --------------------------------------------------
# 15. Check baseline fallback usage
# --------------------------------------------------

baseline_test_df["prediction_source"] = (
    "overall_average"
)

baseline_test_df.loc[
    baseline_test_df[
        "canonical_route_prediction"
    ].notna(),
    "prediction_source",
] = "canonical_route"

baseline_test_df.loc[
    baseline_test_df[
        "canonical_route_time_weekday_prediction"
    ].notna(),
    "prediction_source",
] = "canonical_route_time_weekday"

baseline_test_df.loc[
    baseline_test_df[
        "company_route_time_weekday_prediction"
    ].notna(),
    "prediction_source",
] = "company_route_time_weekday"


baseline_source_percentages = (
    baseline_test_df["prediction_source"]
    .value_counts(normalize=True)
    .mul(100)
)


print("\nBaseline prediction source percentages")
print(baseline_source_percentages)


# --------------------------------------------------
# 16. Compare CatBoost against test baseline
# --------------------------------------------------

mae_improvement = (
    baseline_test_mae - catboost_test_mae
)

rmse_improvement = (
    baseline_test_rmse - catboost_test_rmse
)

mae_improvement_percentage = (
    mae_improvement / baseline_test_mae
) * 100


print("\nCatBoost comparison against test baseline")

print("\nBaseline test MAE:", baseline_test_mae)
print("CatBoost test MAE:", catboost_test_mae)
print("MAE improvement:", mae_improvement)
print(
    "MAE improvement percentage:",
    mae_improvement_percentage,
)

print("\nBaseline test RMSE:", baseline_test_rmse)
print("CatBoost test RMSE:", catboost_test_rmse)
print("RMSE improvement:", rmse_improvement)


# --------------------------------------------------
# 17. Optional: inspect error distribution
# --------------------------------------------------

results_df = pd.DataFrame({
    "actual": y_test.to_numpy(),
    "catboost_prediction": catboost_test_predictions,
    "baseline_prediction":
        baseline_test_df["baseline_prediction"].to_numpy(),
})

results_df["catboost_absolute_error"] = (
    results_df["actual"]
    - results_df["catboost_prediction"]
).abs()

results_df["baseline_absolute_error"] = (
    results_df["actual"]
    - results_df["baseline_prediction"]
).abs()


print("\nAbsolute-error percentiles")

for percentile in [
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
]:
    catboost_error = (
        results_df["catboost_absolute_error"]
        .quantile(percentile)
    )

    baseline_error = (
        results_df["baseline_absolute_error"]
        .quantile(percentile)
    )

    print(
        f"{percentile:>5.0%} percentile | "
        f"CatBoost: {catboost_error:.3f} | "
        f"Baseline: {baseline_error:.3f}"
    )


# --------------------------------------------------
# 18. Final summary
# --------------------------------------------------

print("\nFinal test summary")

print(
    f"CatBoost MAE: {catboost_test_mae:.6f}"
)

print(
    f"Baseline MAE: {baseline_test_mae:.6f}"
)

print(
    f"MAE improvement: "
    f"{mae_improvement_percentage:.3f}%"
)

print(
    f"CatBoost RMSE: {catboost_test_rmse:.6f}"
)

print(
    f"Baseline RMSE: {baseline_test_rmse:.6f}"
)

# analyze
results_df["date"] = test_df[
    "SEFER_TARIHI"
].to_numpy()

results_df["month"] = (
    pd.to_datetime(results_df["date"]).dt.month
)

monthly_results = (
    results_df
    .groupby("month")
    .agg(
        row_count=("actual", "size"),

        catboost_mae=(
            "catboost_absolute_error",
            "mean",
        ),

        baseline_mae=(
            "baseline_absolute_error",
            "mean",
        ),
    )
)

monthly_results["mae_improvement"] = (
    monthly_results["baseline_mae"]
    - monthly_results["catboost_mae"]
)

print("\nMonthly test results")
print(monthly_results)

# --------------------------------------------------
# 20. Analyze prediction bias by passenger-count group
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
    )
)


# Positive means CatBoost is better.
# Negative means the baseline is better.
target_group_results["mae_improvement"] = (
    target_group_results["baseline_mae"]
    - target_group_results["catboost_mae"]
)


# Negative bias means underprediction.
# Positive bias means overprediction.
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


print("\nPrediction results by passenger-count group")
print(target_group_results.to_string())