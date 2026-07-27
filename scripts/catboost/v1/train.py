from pathlib import Path

import duckdb
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Connect to DuckDB
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_DIR / "analysis.duckdb"

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

print(f"Opening database: {DB_PATH}")

conn = duckdb.connect(
    str(DB_PATH),
    read_only=True,
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

numeric_features = [
    "year",
    "week_of_year",
    "day_of_month",
    "departure_hour",
    "departure_minute",
]

feature_columns = (
    categorical_features
    + numeric_features
)


# --------------------------------------------------
# 3. Load only training and validation data
# --------------------------------------------------

selected_columns = ", ".join(
    feature_columns + ["target"]
)

train_df = conn.execute(f"""
    SELECT {selected_columns}
    FROM model_data_base
    WHERE SEFER_TARIHI < DATE '2025-01-01'
""").fetchdf()

validation_df = conn.execute(f"""
    SELECT {selected_columns}
    FROM model_data_base
    WHERE SEFER_TARIHI >= DATE '2025-01-01'
      AND SEFER_TARIHI < DATE '2025-07-01'
""").fetchdf()

conn.close()

print("\nTrain shape:", train_df.shape)
print("Validation shape:", validation_df.shape)


# --------------------------------------------------
# 4. Prepare categorical columns
# --------------------------------------------------

# CatBoost should treat identifiers as categories,
# not as meaningful numerical magnitudes.
for column in categorical_features:
    train_df[column] = (
        train_df[column]
        .astype("string")
        .fillna("missing")
    )

    validation_df[column] = (
        validation_df[column]
        .astype("string")
        .fillna("missing")
    )


# --------------------------------------------------
# 5. Separate features and target
# --------------------------------------------------

X_train = train_df[feature_columns]
y_train = train_df["target"]

X_validation = validation_df[feature_columns]
y_validation = validation_df["target"]


# --------------------------------------------------
# 6. Create CatBoost model
# --------------------------------------------------

model = CatBoostRegressor(
    loss_function="MAE",
    eval_metric="MAE",

    iterations=1000,
    learning_rate=0.08,
    depth=8,

    task_type="GPU",
    devices="0",

    random_seed=42,
    verbose=100,

    allow_writing_files=False,
)


# --------------------------------------------------
# 7. Train model
# --------------------------------------------------

model.fit(
    X_train,
    y_train,

    cat_features=categorical_features,

    eval_set=(
        X_validation,
        y_validation,
    ),

    early_stopping_rounds=100,
    use_best_model=True,
)


# --------------------------------------------------
# 8. Predict validation data
# --------------------------------------------------

validation_predictions = model.predict(
    X_validation
)


# --------------------------------------------------
# 9. Calculate metrics
# --------------------------------------------------

validation_mae = mean_absolute_error(
    y_validation,
    validation_predictions,
)

validation_rmse = mean_squared_error(
    y_validation,
    validation_predictions,
) ** 0.5

print("\nCatBoost validation results")
print("Validation MAE:", validation_mae)
print("Validation RMSE:", validation_rmse)


# --------------------------------------------------
# 10. Compare against selected baseline
# --------------------------------------------------

baseline_mae = 10.459917606689025
baseline_rmse = 15.195470914044426

mae_improvement = (
    baseline_mae - validation_mae
)

rmse_improvement = (
    baseline_rmse - validation_rmse
)

mae_improvement_percentage = (
    mae_improvement / baseline_mae
) * 100

print("\nComparison against weekday baseline")
print("Baseline MAE:", baseline_mae)
print("CatBoost MAE:", validation_mae)
print("MAE improvement:", mae_improvement)
print(
    "MAE improvement percentage:",
    mae_improvement_percentage,
)

print("\nBaseline RMSE:", baseline_rmse)
print("CatBoost RMSE:", validation_rmse)
print("RMSE improvement:", rmse_improvement)


# --------------------------------------------------
# 11. Show feature importance
# --------------------------------------------------

feature_importance = pd.DataFrame({
    "feature": feature_columns,
    "importance": model.get_feature_importance(),
}).sort_values(
    "importance",
    ascending=False,
)

print("\nFeature importance")
print(feature_importance.to_string(index=False))


# --------------------------------------------------
# 12. Save model
# --------------------------------------------------

MODEL_PATH = Path(
    "models/catboost_demand_model.cbm"
).resolve()

model.save_model(str(MODEL_PATH))

print(f"\nModel saved to: {MODEL_PATH}")