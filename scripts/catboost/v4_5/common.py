"""Shared leakage-safe feature and evaluation code for CatBoost v4.5.

v4.5 keeps the exact 64-feature CatBoost v4.1 contract and appends only nine
predetermined religious-holiday calendar features.  Passenger demand is never
used to derive those calendar features.  Frozen v4.1 and v4.2 artifacts are
read only and are used only for comparison reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from scripts.shared.constants import (
    CATEGORICAL_FEATURES,
    CALENDAR_NUMERIC_FEATURES,
    HISTORICAL_FEATURES,
    RECENT_FEATURES,
    V4_1_FEATURE_COLUMNS,
)
from scripts.shared.feature_pipeline import (
    connect,
    create_feature_table,
    create_long_term_statistics,
    create_recent_statistics,
    create_source_tables,
    ensure_environment,
    join_condition,
    sql_identifier_list,
    validate_feature_frame as _shared_validate_feature_frame,
)
from scripts.shared.metrics import regression_metrics
from scripts.shared.model_utils import load_catboost_regressor
from scripts.shared.paths import DB_PATH, MODELS_DIR, RESULTS_DIR
from scripts.shared.period_config import PeriodConfig, STANDARD_PERIODS

MODEL_PATH = (
    MODELS_DIR
    / "catboost_demand_model_v4_5_religious_holiday_mae_6000.cbm"
)

HOLIDAY_CATEGORICAL_FEATURES = ["religious_holiday_type"]
HOLIDAY_NUMERIC_FEATURES = [
    "is_arife",
    "is_religious_holiday",
    "religious_holiday_day_index",
    "relative_day_to_religious_holiday",
    "is_pre_holiday_3d",
    "is_pre_holiday_7d",
    "is_post_holiday_3d",
    "is_post_holiday_7d",
]
HOLIDAY_FEATURES = HOLIDAY_CATEGORICAL_FEATURES + HOLIDAY_NUMERIC_FEATURES
FEATURE_COLUMNS = list(V4_1_FEATURE_COLUMNS) + HOLIDAY_FEATURES
MODEL_CATEGORICAL_FEATURES = list(CATEGORICAL_FEATURES) + HOLIDAY_CATEGORICAL_FEATURES

NEUTRAL_RELATIVE_DAY = 99

HOLIDAY_ROWS = (
    ("RAMAZAN", "2023-04-20", "2023-04-21", "2023-04-23"),
    ("KURBAN", "2023-06-27", "2023-06-28", "2023-07-01"),
    ("RAMAZAN", "2024-04-09", "2024-04-10", "2024-04-12"),
    ("KURBAN", "2024-06-15", "2024-06-16", "2024-06-19"),
    ("RAMAZAN", "2025-03-29", "2025-03-30", "2025-04-01"),
    ("KURBAN", "2025-06-05", "2025-06-06", "2025-06-09"),
    ("RAMAZAN", "2026-03-19", "2026-03-20", "2026-03-22"),
    ("KURBAN", "2026-05-26", "2026-05-27", "2026-05-30"),
)

FROZEN_PREDICTION_PATHS = {
    "validation_2025_h1": RESULTS_DIR / "catboost_v4_2_validation_2025_h1_predictions.csv",
    "test_2025_h2": RESULTS_DIR / "catboost_v4_2_test_2025_h2_predictions.csv",
    "final_2026": RESULTS_DIR / "catboost_v4_2_final_2026_predictions.csv",
}


PERIODS = {
    "validation": PeriodConfig(**STANDARD_PERIODS["validation"].__dict__),
    "test": PeriodConfig(**STANDARD_PERIODS["test"].__dict__),
    "final": PeriodConfig(
        "final_2026",
        "2026 final (reporting only; already observed)",
        "2023-01-01",
        "2026-01-01",
        "2026-01-01",
        "2026-04-15",
        780_865,
        False,
    ),
}

MODEL_COMPARISON_COLUMNS = {
    "CatBoost v4.1": "v4_1_prediction",
    "v4.2 hybrid": "v4_2_hybrid_prediction",
    "CatBoost v4.5": "v4_5_prediction",
    "Weekday baseline": "baseline_prediction",
}


def ensure_v4_5_environment(*, require_model: bool = False) -> None:
    ensure_environment(require_model=require_model, model_path=MODEL_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def create_religious_holiday_calendar(conn) -> None:
    values_sql = ",\n".join(
        f"('{kind}', DATE '{arife}', DATE '{first_day}', DATE '{last_day}')"
        for kind, arife, first_day, last_day in HOLIDAY_ROWS
    )
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE v4_5_religious_holidays AS
        SELECT * FROM (VALUES {values_sql}) AS holiday(
            religious_holiday_type, arife_date, first_day, last_day
        )
    """)


def _holiday_extra_selects() -> list[str]:
    return [
        "COALESCE(holiday.religious_holiday_type, 'NONE') AS religious_holiday_type",
        "CASE WHEN base.SEFER_TARIHI = holiday.arife_date THEN 1 ELSE 0 END::UTINYINT AS is_arife",
        "CASE WHEN base.SEFER_TARIHI BETWEEN holiday.first_day AND holiday.last_day "
        "THEN 1 ELSE 0 END::UTINYINT AS is_religious_holiday",
        "CASE WHEN base.SEFER_TARIHI BETWEEN holiday.first_day AND holiday.last_day "
        "THEN date_diff('day', holiday.first_day, base.SEFER_TARIHI) + 1 "
        "ELSE 0 END::SMALLINT AS religious_holiday_day_index",
        f"COALESCE(date_diff('day', holiday.first_day, base.SEFER_TARIHI), "
        f"{NEUTRAL_RELATIVE_DAY})::SMALLINT AS relative_day_to_religious_holiday",
        "CASE WHEN base.SEFER_TARIHI >= holiday.first_day - INTERVAL '3 days' "
        "AND base.SEFER_TARIHI < holiday.first_day THEN 1 ELSE 0 END::UTINYINT AS is_pre_holiday_3d",
        "CASE WHEN base.SEFER_TARIHI >= holiday.first_day - INTERVAL '7 days' "
        "AND base.SEFER_TARIHI < holiday.first_day THEN 1 ELSE 0 END::UTINYINT AS is_pre_holiday_7d",
        "CASE WHEN base.SEFER_TARIHI > holiday.last_day "
        "AND base.SEFER_TARIHI <= holiday.last_day + INTERVAL '3 days' "
        "THEN 1 ELSE 0 END::UTINYINT AS is_post_holiday_3d",
        "CASE WHEN base.SEFER_TARIHI > holiday.last_day "
        "AND base.SEFER_TARIHI <= holiday.last_day + INTERVAL '7 days' "
        "THEN 1 ELSE 0 END::UTINYINT AS is_post_holiday_7d",
    ]


def _holiday_extra_joins() -> list[str]:
    return [
        "LEFT JOIN v4_5_religious_holidays AS holiday "
        "ON base.SEFER_TARIHI >= holiday.first_day - INTERVAL '7 days' "
        "AND base.SEFER_TARIHI <= holiday.last_day + INTERVAL '7 days'"
    ]


def build_matrix(config: PeriodConfig, *, training: bool) -> pd.DataFrame:
    ensure_v4_5_environment()
    conn = connect()
    try:
        create_religious_holiday_calendar(conn)
        create_source_tables(conn, config, table_prefix="v4_5")
        if training:
            conn.execute("""
                CREATE OR REPLACE TEMP TABLE v4_5_recent_history AS
                SELECT * FROM v4_5_history
                UNION ALL
                SELECT * FROM v4_5_target
            """)
            recent_history_table = "v4_5_recent_history"
        else:
            recent_history_table = "v4_5_history"
        create_long_term_statistics(conn, table_prefix="v4_5")
        create_recent_statistics(
            conn, config, table_prefix="v4_5", history_table=recent_history_table,
        )
        create_feature_table(
            conn,
            table_prefix="v4_5",
            feature_columns=FEATURE_COLUMNS,
            extra_selects=_holiday_extra_selects(),
            extra_joins=_holiday_extra_joins(),
        )
        selected = [
            "SEFER_ID", "SEFER_TARIHI", *FEATURE_COLUMNS,
            "target", "baseline_prediction", "baseline_source",
        ]
        frame = conn.execute(
            f"SELECT {sql_identifier_list(selected)} FROM v4_5_features"
        ).fetchdf()
    finally:
        conn.close()
    validate_feature_frame(frame)
    return frame


def validate_feature_frame(frame: pd.DataFrame) -> None:
    if len(V4_1_FEATURE_COLUMNS) != 64 or len(set(V4_1_FEATURE_COLUMNS)) != 64:
        raise RuntimeError("The inherited v4.1 contract must be 64 unique features.")
    if len(FEATURE_COLUMNS) != 73 or len(set(FEATURE_COLUMNS)) != 73:
        raise RuntimeError("v4.5 must have exactly 73 unique feature columns.")
    _shared_validate_feature_frame(
        frame,
        feature_columns=FEATURE_COLUMNS,
        categorical_columns=MODEL_CATEGORICAL_FEATURES,
    )
    holiday_types = set(frame["religious_holiday_type"].unique())
    if not holiday_types <= {"NONE", "RAMAZAN", "KURBAN"}:
        raise RuntimeError(f"Unexpected religious holiday types: {holiday_types}")
    for column in HOLIDAY_NUMERIC_FEATURES:
        if column.startswith("is_") and not frame[column].isin((0, 1)).all():
            raise RuntimeError(f"Non-binary holiday flag: {column}")


def load_model() -> CatBoostRegressor:
    ensure_v4_5_environment(require_model=True)
    return load_catboost_regressor(MODEL_PATH, FEATURE_COLUMNS)


def merge_frozen_comparisons(frame: pd.DataFrame, config: PeriodConfig) -> pd.DataFrame:
    path = FROZEN_PREDICTION_PATHS[config.key]
    required = [
        "SEFER_ID", "actual", "v4_1_prediction",
        "hybrid_prediction", "baseline_prediction",
    ]
    if not path.exists():
        raise FileNotFoundError(f"Frozen v4.2 comparison artifact not found: {path}")
    frozen = pd.read_csv(path, usecols=required).rename(
        columns={
            "actual": "frozen_actual",
            "hybrid_prediction": "v4_2_hybrid_prediction",
            "baseline_prediction": "frozen_baseline_prediction",
        }
    )
    if frozen["SEFER_ID"].duplicated().any() or frame["SEFER_ID"].duplicated().any():
        raise RuntimeError("SEFER_ID must be unique for frozen comparison alignment.")
    merged = frame.merge(frozen, on="SEFER_ID", how="left", validate="one_to_one")
    required_predictions = [
        "v4_1_prediction", "v4_2_hybrid_prediction", "frozen_baseline_prediction",
    ]
    if merged[required_predictions].isna().any().any():
        raise RuntimeError("Frozen predictions did not align to all v4.5 rows.")
    if not np.array_equal(
        merged["target"].to_numpy(), merged["frozen_actual"].to_numpy()
    ):
        raise RuntimeError("Targets disagree with the frozen v4.2 artifact.")
    if not np.allclose(
        merged["baseline_prediction"].to_numpy(),
        merged["frozen_baseline_prediction"].to_numpy(),
        rtol=0.0, atol=1e-12,
    ):
        raise RuntimeError("Rebuilt weekday baseline differs from frozen v4.2.")
    return merged.drop(columns=["frozen_actual", "frozen_baseline_prediction"])


def overall_metrics(frame: pd.DataFrame, config: PeriodConfig) -> pd.DataFrame:
    actual = frame["target"].to_numpy(dtype=np.float64)
    mask_10_40 = (actual >= 10) & (actual <= 40)
    mask_20_40 = (actual >= 20) & (actual <= 40)
    mask_40_plus = actual >= 40
    records = []
    for model, column in MODEL_COMPARISON_COLUMNS.items():
        prediction = frame[column].to_numpy(dtype=np.float64)
        overall = regression_metrics(actual, prediction)
        records.append({
            "period": config.key,
            "model": model,
            "row_count": len(frame),
            "mae": overall["mae"],
            "rmse": overall["rmse"],
            "bias": overall["bias"],
            "mae_actual_10_to_40": regression_metrics(actual[mask_10_40], prediction[mask_10_40])["mae"],
            "mae_actual_20_to_40": regression_metrics(actual[mask_20_40], prediction[mask_20_40])["mae"],
            "rmse_actual_40_plus": regression_metrics(actual[mask_40_plus], prediction[mask_40_plus])["rmse"],
        })
    return pd.DataFrame(records)


def holiday_window_labels(frame: pd.DataFrame) -> pd.Series:
    relative = frame["relative_day_to_religious_holiday"].to_numpy()
    arife = frame["is_arife"].to_numpy(dtype=bool)
    holiday = frame["is_religious_holiday"].to_numpy(dtype=bool)
    labels = np.full(len(frame), "ordinary days", dtype=object)
    labels[(relative >= -7) & (relative <= -4)] = "7-4 days before"
    labels[(relative >= -3) & (relative <= -1)] = "3-1 days before"
    labels[arife] = "Arife"
    labels[holiday] = "Bayram days"
    post_7 = frame["is_post_holiday_7d"].to_numpy(dtype=bool)
    post_3 = frame["is_post_holiday_3d"].to_numpy(dtype=bool)
    labels[post_7] = "4-7 days after"
    labels[post_3] = "1-3 days after"
    return pd.Series(labels, index=frame.index, name="holiday_window")


def holiday_window_metrics(frame: pd.DataFrame, config: PeriodConfig) -> pd.DataFrame:
    data = frame.copy()
    data["holiday_window"] = holiday_window_labels(data)
    order = [
        "7-4 days before", "3-1 days before", "Arife", "Bayram days",
        "1-3 days after", "4-7 days after", "ordinary days",
    ]
    records = []
    for window in order:
        subset = data[data["holiday_window"] == window]
        if subset.empty:
            continue
        actual = subset["target"].to_numpy(dtype=np.float64)
        for model, column in MODEL_COMPARISON_COLUMNS.items():
            values = regression_metrics(actual, subset[column].to_numpy(dtype=np.float64))
            records.append({
                "period": config.key,
                "holiday_window": window,
                "model": model,
                "row_count": len(subset),
                **values,
            })
    return pd.DataFrame(records)


def output_paths(config: PeriodConfig) -> tuple[Path, Path, Path]:
    stem = RESULTS_DIR / f"catboost_v4_5_{config.key}"
    return (
        Path(f"{stem}_overall_metrics.csv"),
        Path(f"{stem}_holiday_window_metrics.csv"),
        Path(f"{stem}_predictions.csv"),
    )


def refuse_existing_outputs(config: PeriodConfig) -> None:
    collisions = [path for path in output_paths(config) if path.exists()]
    if collisions:
        rendered = "\n".join(f"  - {path}" for path in collisions)
        raise FileExistsError("Refusing to overwrite v4.5 artifacts:\n" + rendered)


def write_evaluation_outputs(frame: pd.DataFrame, config: PeriodConfig) -> None:
    overall_path, holiday_path, predictions_path = output_paths(config)
    refuse_existing_outputs(config)
    overall = overall_metrics(frame, config)
    windows = holiday_window_metrics(frame, config)
    output = frame[
        [
            "SEFER_ID", "SEFER_TARIHI", "target",
            *MODEL_COMPARISON_COLUMNS.values(), "baseline_source",
            *HOLIDAY_FEATURES,
        ]
    ].copy()
    output["holiday_window"] = holiday_window_labels(frame)
    overall.to_csv(overall_path, index=False)
    windows.to_csv(holiday_path, index=False)
    output.to_csv(predictions_path, index=False)
    print(f"\n{config.label} overall comparison")
    print(overall.drop(columns=["period", "row_count"]).to_string(index=False))
    print(f"\n{config.label} religious-holiday windows")
    print(windows.drop(columns="period").to_string(index=False))
