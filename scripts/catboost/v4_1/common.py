from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from scripts.shared.constants import (
    CATEGORICAL_FEATURES,
    V4_1_FEATURE_COLUMNS as FEATURE_COLUMNS,
)
from scripts.shared.feature_pipeline import (
    connect,
    create_feature_table,
    create_long_term_statistics,
    create_recent_statistics,
    create_source_tables,
    ensure_environment,
    sql_identifier_list,
    validate_feature_frame,
)
from scripts.shared.model_utils import load_catboost_regressor
from scripts.shared.paths import RESULTS_DIR
from scripts.shared.paths import DB_PATH, DUCKDB_THREADS, TEMP_DIR
from scripts.shared import paths as _paths

MODEL_PATH = (
    _paths.MODELS_DIR
    / "catboost_demand_model_v4_1_recent_mae_6000.cbm"
)


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


def load_model():
    print(f"Loading model: {MODEL_PATH}")
    model = load_catboost_regressor(MODEL_PATH, FEATURE_COLUMNS)
    print(f"Verified feature count: {len(FEATURE_COLUMNS)}")
    return model


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
            "1-10", "11-20", "21-30", "31-40", "41-60", "61-100", "101-300",
        ],
        include_lowest=True,
    )

    grouped = (
        results.groupby("target_group", observed=True)
        .agg(
            row_count=("actual", "size"),
            average_actual=("actual", "mean"),
            average_catboost_prediction=("catboost_prediction", "mean"),
            average_baseline_prediction=("baseline_prediction", "mean"),
            catboost_mae=("catboost_absolute_error", "mean"),
            baseline_mae=("baseline_absolute_error", "mean"),
            catboost_mse=("catboost_squared_error", "mean"),
            baseline_mse=("baseline_squared_error", "mean"),
        )
        .reset_index()
    )

    grouped["catboost_rmse"] = np.sqrt(grouped["catboost_mse"])
    grouped["baseline_rmse"] = np.sqrt(grouped["baseline_mse"])
    grouped["catboost_bias"] = (
        grouped["average_catboost_prediction"] - grouped["average_actual"]
    )
    grouped["baseline_bias"] = (
        grouped["average_baseline_prediction"] - grouped["average_actual"]
    )
    grouped["mae_improvement"] = (
        grouped["baseline_mae"] - grouped["catboost_mae"]
    )

    return grouped.drop(columns=["catboost_mse", "baseline_mse"])


def run_evaluation(config: EvaluationConfig) -> None:
    ensure_environment(require_model=True, model_path=MODEL_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()

    print(f"\nEvaluating: {config.label}")
    print(
        "Recent features use only history before the evaluation period; "
        "evaluation targets are never reused as history."
    )

    conn = connect()

    try:
        create_source_tables(conn, config, table_prefix="eval")

        print("\nBuilding long-term statistics...")
        create_long_term_statistics(conn, table_prefix="eval")

        print("\nBuilding recent statistics...")
        create_recent_statistics(
            conn, config, table_prefix="eval", history_table="eval_history",
        )

        print("\nCreating completed feature matrix...")
        create_feature_table(
            conn, table_prefix="eval", feature_columns=FEATURE_COLUMNS,
        )

        selected_columns = list(FEATURE_COLUMNS) + [
            "target",
            "baseline_prediction",
            "baseline_source",
        ]

        frame = conn.execute(
            f"""
            SELECT {sql_identifier_list(selected_columns)}
            FROM eval_features
            """
        ).fetchdf()

    finally:
        conn.close()

    validate_feature_frame(
        frame, feature_columns=FEATURE_COLUMNS,
    )

    print(f"\nPredicting {len(frame):,} rows...")
    predictions = model.predict(frame[FEATURE_COLUMNS])

    actual = frame["target"]
    baseline_predictions = frame["baseline_prediction"]

    catboost_mae = mean_absolute_error(actual, predictions)
    catboost_rmse = mean_squared_error(actual, predictions) ** 0.5
    baseline_mae = mean_absolute_error(actual, baseline_predictions)
    baseline_rmse = mean_squared_error(actual, baseline_predictions) ** 0.5

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
        actual, predictions, baseline_predictions,
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
        predictions_output["actual"] - predictions_output["catboost_prediction"]
    ).abs()
    predictions_output["baseline_absolute_error"] = (
        predictions_output["actual"] - predictions_output["baseline_prediction"]
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

    summary_path = RESULTS_DIR / f"catboost_v4_1_{config.key}_metrics.csv"
    group_path = RESULTS_DIR / f"catboost_v4_1_{config.key}_target_groups.csv"
    prediction_path = RESULTS_DIR / f"catboost_v4_1_{config.key}_predictions.csv"

    summary.to_csv(summary_path, index=False)
    target_groups.to_csv(group_path, index=False)
    predictions_output.to_csv(prediction_path, index=False)

    print("\nSaved:")
    print(summary_path)
    print(group_path)
    print(prediction_path)
