from pathlib import Path

import duckdb
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
# 4. Create baseline average tables
# --------------------------------------------------

# Level 1:
# company + route + time + weekday + month

company_route_time_weekday_month_columns = [
    "FIRMA_ID",
    "canonical_guzergah_id",
    "departure_30min_bucket",
    "day_of_week",
    "month",
]

company_route_time_weekday_month_table = (
    train_df
    .groupby(
        company_route_time_weekday_month_columns,
        as_index=False,
    )["target"]
    .mean()
    .rename(
        columns={
            "target":
            "company_route_time_weekday_month_prediction"
        }
    )
)


# Level 2:
# company + route + time + weekday

company_route_time_weekday_columns = [
    "FIRMA_ID",
    "canonical_guzergah_id",
    "departure_30min_bucket",
    "day_of_week",
]

company_route_time_weekday_table = (
    train_df
    .groupby(
        company_route_time_weekday_columns,
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


# Level 3:
# canonical route + time + weekday + month

canonical_route_time_weekday_month_columns = [
    "canonical_guzergah_id",
    "departure_30min_bucket",
    "day_of_week",
    "month",
]

canonical_route_time_weekday_month_table = (
    train_df
    .groupby(
        canonical_route_time_weekday_month_columns,
        as_index=False,
    )["target"]
    .mean()
    .rename(
        columns={
            "target":
            "canonical_route_time_weekday_month_prediction"
        }
    )
)


# Level 4:
# canonical route + time + weekday

canonical_route_time_weekday_columns = [
    "canonical_guzergah_id",
    "departure_30min_bucket",
    "day_of_week",
]

canonical_route_time_weekday_table = (
    train_df
    .groupby(
        canonical_route_time_weekday_columns,
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


# Level 5:
# canonical route only

canonical_route_table = (
    train_df
    .groupby(
        "canonical_guzergah_id",
        as_index=False,
    )["target"]
    .mean()
    .rename(
        columns={
            "target": "canonical_route_prediction"
        }
    )
)


# Level 6:
# overall average

overall_average = train_df["target"].mean()


# --------------------------------------------------
# 5. Apply baseline tables to validation data
# --------------------------------------------------

validation_predictions = validation_df.merge(
    company_route_time_weekday_month_table,
    on=company_route_time_weekday_month_columns,
    how="left",
)

validation_predictions = validation_predictions.merge(
    company_route_time_weekday_table,
    on=company_route_time_weekday_columns,
    how="left",
)

validation_predictions = validation_predictions.merge(
    canonical_route_time_weekday_month_table,
    on=canonical_route_time_weekday_month_columns,
    how="left",
)

validation_predictions = validation_predictions.merge(
    canonical_route_time_weekday_table,
    on=canonical_route_time_weekday_columns,
    how="left",
)

validation_predictions = validation_predictions.merge(
    canonical_route_table,
    on="canonical_guzergah_id",
    how="left",
)


# --------------------------------------------------
# 6. Create final prediction using fallbacks
# --------------------------------------------------

validation_predictions["prediction"] = (
    validation_predictions[
        "company_route_time_weekday_month_prediction"
    ]
    .fillna(
        validation_predictions[
            "company_route_time_weekday_prediction"
        ]
    )
    .fillna(
        validation_predictions[
            "canonical_route_time_weekday_month_prediction"
        ]
    )
    .fillna(
        validation_predictions[
            "canonical_route_time_weekday_prediction"
        ]
    )
    .fillna(
        validation_predictions[
            "canonical_route_prediction"
        ]
    )
    .fillna(overall_average)
)


# --------------------------------------------------
# 7. Calculate validation metrics
# --------------------------------------------------

mae = mean_absolute_error(
    validation_predictions["target"],
    validation_predictions["prediction"],
)

rmse = mean_squared_error(
    validation_predictions["target"],
    validation_predictions["prediction"],
) ** 0.5

print("\nMonth-aware baseline results")
print("Validation MAE:", mae)
print("Validation RMSE:", rmse)


# --------------------------------------------------
# 8. Check fallback usage
# --------------------------------------------------

validation_predictions["prediction_source"] = (
    "overall_average"
)

validation_predictions.loc[
    validation_predictions[
        "canonical_route_prediction"
    ].notna(),
    "prediction_source",
] = "canonical_route"

validation_predictions.loc[
    validation_predictions[
        "canonical_route_time_weekday_prediction"
    ].notna(),
    "prediction_source",
] = "canonical_route_time_weekday"

validation_predictions.loc[
    validation_predictions[
        "canonical_route_time_weekday_month_prediction"
    ].notna(),
    "prediction_source",
] = "canonical_route_time_weekday_month"

validation_predictions.loc[
    validation_predictions[
        "company_route_time_weekday_prediction"
    ].notna(),
    "prediction_source",
] = "company_route_time_weekday"

validation_predictions.loc[
    validation_predictions[
        "company_route_time_weekday_month_prediction"
    ].notna(),
    "prediction_source",
] = "company_route_time_weekday_month"


source_percentages = (
    validation_predictions["prediction_source"]
    .value_counts(normalize=True)
    .mul(100)
)

print("\nPrediction source percentages")
print(source_percentages)


# --------------------------------------------------
# 9. Close database connection
# --------------------------------------------------

conn.close()