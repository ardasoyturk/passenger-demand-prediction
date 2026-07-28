"""Parametric baseline runner for hierarchical mean-lookup baselines.

Replaces the near-identical ``scripts/baseline/time.py``,
``time_dayofweek.py``, and ``time_dayofweek_month.py`` scripts.
"""

from __future__ import annotations

from typing import Sequence

import duckdb
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scripts.shared.paths import DB_PATH


def _build_group_table(
    train_df: pd.DataFrame,
    group_columns: list[str],
    prediction_name: str,
) -> pd.DataFrame:
    return (
        train_df
        .groupby(group_columns, as_index=False)["target"]
        .mean()
        .rename(columns={"target": prediction_name})
    )


def run_baseline(
    grouping_levels: Sequence[tuple[list[str], str]],
    *,
    output_label: str,
    train_end: str = "2025-01-01",
    validation_start: str = "2025-01-01",
    validation_end: str = "2025-07-01",
) -> None:
    """Run a hierarchical baseline evaluation.

    *grouping_levels* is a sequence of ``(group_columns, prediction_name)``
    tuples ordered from most specific to least specific.  The first level's
    prediction is preferred; missing values fall through to subsequent levels
    and finally to the overall average.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    print(f"Opening database: {DB_PATH}")
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        df = conn.execute("""
            SELECT * FROM model_data_base ORDER BY SEFER_TARIHI, SEFER_SAATI
        """).fetchdf()
    finally:
        conn.close()

    print(f"\nDataset shape: {df.shape}")

    train_df = df[df["SEFER_TARIHI"] < train_end].copy()
    validation_df = df[
        (df["SEFER_TARIHI"] >= validation_start)
        & (df["SEFER_TARIHI"] < validation_end)
    ].copy()

    print(f"Train: {len(train_df):,} rows")
    print(f"Validation: {len(validation_df):,} rows")

    # Build lookup tables
    group_tables = []
    for columns, pred_name in grouping_levels:
        group_tables.append(
            (columns, pred_name, _build_group_table(train_df, columns, pred_name))
        )
    overall_average = train_df["target"].mean()

    # Merge into validation
    result = validation_df.copy()
    for columns, _, table in group_tables:
        result = result.merge(table, on=columns, how="left")

    # Fallback chain: most specific first, then overall average
    pred_names = [name for _, name in grouping_levels]
    result["prediction"] = result[pred_names[0]]
    for name in pred_names[1:]:
        result["prediction"] = result["prediction"].fillna(result[name])
    result["prediction"] = result["prediction"].fillna(overall_average)

    # Metrics
    mae = mean_absolute_error(result["target"], result["prediction"])
    rmse = mean_squared_error(result["target"], result["prediction"]) ** 0.5

    print(f"\n{output_label} results")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation RMSE: {rmse:.4f}")

    # Fallback source tracking: iterate from least to most specific
    result["prediction_source"] = "overall_average"
    for _, pred_name, _ in reversed(list(group_tables)):
        source_label = pred_name.replace("_prediction", "")
        mask = result[pred_name].notna()
        result.loc[mask, "prediction_source"] = source_label

    source_percentages = (
        result["prediction_source"]
        .value_counts(normalize=True)
        .mul(100)
    )
    print("\nPrediction source percentages")
    print(source_percentages)
