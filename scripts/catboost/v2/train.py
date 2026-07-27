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

BASELINE_MAE = 10.459917606689025
BASELINE_RMSE = 15.195470914044426


if not DB_PATH.exists():
    raise FileNotFoundError(
        f"Database not found: {DB_PATH}"
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


# --------------------------------------------------
# 3. Load data from DuckDB
# --------------------------------------------------

print(f"Opening database: {DB_PATH}")

conn = duckdb.connect(
    str(DB_PATH),
    read_only=True,
)


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


# Historical data used to create features for 2024
history_2023_df = conn.execute(f"""
    SELECT {selected_columns_sql}
    FROM model_data_base
    WHERE SEFER_TARIHI >= DATE '2023-01-01'
      AND SEFER_TARIHI < DATE '2024-01-01'
""").fetchdf()


# CatBoost training rows
train_2024_df = conn.execute(f"""
    SELECT {selected_columns_sql}
    FROM model_data_base
    WHERE SEFER_TARIHI >= DATE '2024-01-01'
      AND SEFER_TARIHI < DATE '2025-01-01'
""").fetchdf()


# Complete historical data available before validation
history_2023_2024_df = conn.execute(f"""
    SELECT {selected_columns_sql}
    FROM model_data_base
    WHERE SEFER_TARIHI >= DATE '2023-01-01'
      AND SEFER_TARIHI < DATE '2025-01-01'
""").fetchdf()


# Validation period
validation_df = conn.execute(f"""
    SELECT {selected_columns_sql}
    FROM model_data_base
    WHERE SEFER_TARIHI >= DATE '2025-01-01'
      AND SEFER_TARIHI < DATE '2025-07-01'
""").fetchdf()


conn.close()


print("\nLoaded data")
print("2023 history:", history_2023_df.shape)
print("2024 model training:", train_2024_df.shape)
print(
    "2023–2024 validation history:",
    history_2023_2024_df.shape,
)
print("Validation:", validation_df.shape)


# --------------------------------------------------
# 4. Helper: create one aggregate table
# --------------------------------------------------

def create_aggregate_table(
    history_df: pd.DataFrame,
    group_columns: list[str],
    average_column_name: str,
    count_column_name: str,
) -> pd.DataFrame:
    """
    Calculate historical average demand and historical
    observation count for a particular group.
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
# 5. Helper: add all historical features
# --------------------------------------------------

def add_historical_features(
    target_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build historical aggregate tables from history_df,
    then attach them to target_df.

    target_df contains the rows that will be predicted.
    history_df contains only earlier rows.
    """

    result_df = target_df.copy()


    # ----------------------------------------------
    # A. Company + canonical route
    #    + 30-minute bucket + weekday
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
    #    + 30-minute bucket
    # ----------------------------------------------

    company_route_time_columns = [
        "FIRMA_ID",
        "canonical_guzergah_id",
        "departure_30min_bucket",
    ]

    company_route_time_table = (
        create_aggregate_table(
            history_df=history_df,
            group_columns=company_route_time_columns,
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
    #    + 30-minute bucket + weekday
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
    # Fill missing aggregate values
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
# 6. Create leakage-safe model features  
# --------------------------------------------------

print("\nCreating training historical features...")

# 2024 rows receive information from 2023 only.
train_features_df = add_historical_features(
    target_df=train_2024_df,
    history_df=history_2023_df,
)


print("Creating validation historical features...")

# 2025 validation rows receive information
# from all data before 2025.
validation_features_df = add_historical_features(
    target_df=validation_df,
    history_df=history_2023_2024_df,
)


print("\nFeature datasets")
print("Training:", train_features_df.shape)
print("Validation:", validation_features_df.shape)


# --------------------------------------------------
# 7. Prepare categorical columns
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


# --------------------------------------------------
# 8. Separate features and targets
# --------------------------------------------------

X_train = train_features_df[feature_columns]
y_train = train_features_df["target"]

X_validation = validation_features_df[
    feature_columns
]
y_validation = validation_features_df["target"]


print("\nModel matrix shapes")
print("X_train:", X_train.shape)
print("X_validation:", X_validation.shape)


# --------------------------------------------------
# 9. Create CatBoost model
# --------------------------------------------------

model = CatBoostRegressor(
    loss_function="MAE",
    eval_metric="MAE",

    iterations=3000,
    learning_rate=0.08,
    depth=8,

    task_type="GPU",
    devices="0",

    random_seed=42,
    verbose=100,
    metric_period=5,

    allow_writing_files=False,
)


# --------------------------------------------------
# 10. Train CatBoost
# --------------------------------------------------

model.fit(
    X_train,
    y_train,

    cat_features=categorical_features,

    eval_set=(
        X_validation,
        y_validation,
    ),

    early_stopping_rounds=200,
    use_best_model=True,
)


# --------------------------------------------------
# 11. Predict validation rows
# --------------------------------------------------

validation_predictions = model.predict(
    X_validation
)


# --------------------------------------------------
# 12. Calculate validation metrics
# --------------------------------------------------

validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions,
)

validation_rmse = mean_squared_error(
    y_validation,
    validation_predictions,
) ** 0.5


print("\nCatBoost v2 validation results")
print("Validation MAE:", validation_mae)
print("Validation RMSE:", validation_rmse)


# --------------------------------------------------
# 13. Compare against selected baseline
# --------------------------------------------------

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
print("CatBoost MAE:", validation_mae)
print("MAE improvement:", mae_improvement)
print(
    "MAE improvement percentage:",
    mae_improvement_percentage,
)

print("\nBaseline RMSE:", BASELINE_RMSE)
print("CatBoost RMSE:", validation_rmse)
print("RMSE improvement:", rmse_improvement)


# --------------------------------------------------
# 14. Show feature importance
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.get_feature_importance(),
}).sort_values(
    "importance",
    ascending=False,
)


print("\nFeature importance")
print(
    feature_importance.to_string(
        index=False
    )
)


# --------------------------------------------------
# 15. Save model
# --------------------------------------------------

model.save_model(str(MODEL_PATH))

print(f"\nModel saved to: {MODEL_PATH}")