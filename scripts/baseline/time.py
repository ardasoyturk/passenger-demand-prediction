from pathlib import Path

import duckdb
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------------------------
# 1. Connect to DuckDB
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_DIR / "analysis.duckdb"

if not DB_PATH.exists():
    raise FileNotFoundError(f"Database not found: {DB_PATH}")

print(f"Opening database: {DB_PATH}")

conn = duckdb.connect(str(DB_PATH), read_only=True)


# --------------------------------------------------
# 2. Load model dataset
# --------------------------------------------------

df = conn.execute("""
    SELECT *
    FROM model_data_base
    ORDER BY SEFER_TARIHI, SEFER_SAATI
""").fetchdf()

print("\nDataset shape:", df.shape)
print(df.head())
print(df.dtypes)


# --------------------------------------------------
# 3. Create chronological splits
# --------------------------------------------------

train_df = df[
    df["SEFER_TARIHI"] < "2025-01-01"
].copy()

validation_df = df[
    (df["SEFER_TARIHI"] >= "2025-01-01")
    & (df["SEFER_TARIHI"] < "2025-07-01")
].copy()

test_df = df[
    (df["SEFER_TARIHI"] >= "2025-07-01")
    & (df["SEFER_TARIHI"] < "2026-01-01")
].copy()

final_test_df = df[
    df["SEFER_TARIHI"] >= "2026-01-01"
].copy()


for name, split_df in [
    ("Train", train_df),
    ("Validation", validation_df),
    ("Test", test_df),
    ("Final test", final_test_df),
]:
    print(
        f"{name}:",
        len(split_df),
        split_df["SEFER_TARIHI"].min(),
        split_df["SEFER_TARIHI"].max(),
    )


# --------------------------------------------------
# 4. Create baseline averages from training data
# --------------------------------------------------

company_route_time_columns = [
    "FIRMA_ID",
    "canonical_guzergah_id",
    "departure_30min_bucket",
]

company_route_time_table = (
    train_df
    .groupby(
        company_route_time_columns,
        as_index=False,
    )["target"]
    .mean()
    .rename(
        columns={
            "target": "company_route_time_prediction"
        }
    )
)


canonical_time_columns = [
    "canonical_guzergah_id",
    "departure_30min_bucket",
]

canonical_time_table = (
    train_df
    .groupby(
        canonical_time_columns,
        as_index=False,
    )["target"]
    .mean()
    .rename(
        columns={
            "target": "canonical_time_prediction"
        }
    )
)


canonical_route_table = (
    train_df
    .groupby(
        "canonical_guzergah_id",
        as_index=False,
    )["target"]
    .mean()
    .rename(
        columns={
            "target": "canonical_prediction"
        }
    )
)


overall_average = train_df["target"].mean()


# --------------------------------------------------
# 5. Apply baseline to validation data
# --------------------------------------------------

validation_predictions = validation_df.merge(
    company_route_time_table,
    on=company_route_time_columns,
    how="left",
)

validation_predictions = validation_predictions.merge(
    canonical_time_table,
    on=canonical_time_columns,
    how="left",
)

validation_predictions = validation_predictions.merge(
    canonical_route_table,
    on="canonical_guzergah_id",
    how="left",
)


validation_predictions["prediction"] = (
    validation_predictions["company_route_time_prediction"]
    .fillna(
        validation_predictions["canonical_time_prediction"]
    )
    .fillna(
        validation_predictions["canonical_prediction"]
    )
    .fillna(overall_average)
)


# --------------------------------------------------
# 6. Calculate validation metrics
# --------------------------------------------------

mae = mean_absolute_error(
    validation_predictions["target"],
    validation_predictions["prediction"],
)

rmse = mean_squared_error(
    validation_predictions["target"],
    validation_predictions["prediction"],
) ** 0.5

print("\nBaseline results")
print("Validation MAE:", mae)
print("Validation RMSE:", rmse)


# --------------------------------------------------
# 7. Check fallback usage
# --------------------------------------------------

validation_predictions["prediction_source"] = (
    "overall_average"
)

validation_predictions.loc[
    validation_predictions["canonical_prediction"].notna(),
    "prediction_source",
] = "canonical_route"

validation_predictions.loc[
    validation_predictions[
        "canonical_time_prediction"
    ].notna(),
    "prediction_source",
] = "canonical_route_time"

validation_predictions.loc[
    validation_predictions[
        "company_route_time_prediction"
    ].notna(),
    "prediction_source",
] = "company_route_time"


source_percentages = (
    validation_predictions["prediction_source"]
    .value_counts(normalize=True)
    .mul(100)
)

print("\nPrediction source percentages")
print(source_percentages)


# --------------------------------------------------
# 8. Close database connection
# --------------------------------------------------

conn.close()