"""Shared leakage-safe feature and business evaluation code for CatBoost v4.3.

v4.3 keeps the complete 64-feature v4.1 contract and appends eight
company-route-time-weekday distribution features.  The already-frozen v4.2
rule and artifacts are read only; applying that rule to v4.3 is a separate
comparison and never changes the production-candidate rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    long_term_expressions,
    join_condition,
    sql_identifier_list,
    validate_feature_frame as _shared_validate_feature_frame,
)
from scripts.shared.metrics import regression_metrics, binary_classification_metrics
from scripts.shared.model_utils import load_catboost_regressor
from scripts.shared.paths import DB_PATH, MODELS_DIR, RESULTS_DIR
from scripts.shared.period_config import STANDARD_PERIODS

MODEL_PATH = (
    MODELS_DIR
    / "catboost_demand_model_v4_3_business_distribution_mae_6000.cbm"
)
FROZEN_RULE_PATH = RESULTS_DIR / "catboost_v4_2_hybrid_rule.json"

DISTRIBUTION_PREFIX = "company_route_time_weekday"
NEW_DISTRIBUTION_SUFFIXES = [
    "p10", "p25", "p75",
    "below_10_rate", "above_10_rate", "above_20_rate",
    "above_30_rate", "above_40_rate",
]
NEW_DISTRIBUTION_FEATURES = [
    f"{DISTRIBUTION_PREFIX}_{suffix}" for suffix in NEW_DISTRIBUTION_SUFFIXES
]

V4_1_FEATURE_COLUMNS_LOCAL = list(V4_1_FEATURE_COLUMNS)
FEATURE_COLUMNS = V4_1_FEATURE_COLUMNS_LOCAL + NEW_DISTRIBUTION_FEATURES

MODEL_COMPARISON_COLUMNS = {
    "CatBoost v3": "v3_prediction",
    "CatBoost v4.1": "v4_1_prediction",
    "v4.2 hybrid": "v4_2_hybrid_prediction",
    "CatBoost v4.3": "v4_3_prediction",
    "v4.3 + frozen hybrid rule": "v4_3_hybrid_prediction",
    "Weekday baseline": "baseline_prediction",
}
BUSINESS_THRESHOLDS = (20, 30, 43)
NEAR_THRESHOLD_NEGATIVE_BANDS = {
    20: (10, 20),
    30: (20, 30),
    43: (30, 43),
}


@dataclass(frozen=True)
class PeriodConfig:
    key: str
    label: str
    history_start: str
    history_end_exclusive: str
    target_start: str
    target_end_exclusive: str
    expected_rows: int
    tuning_allowed: bool


PERIODS = {
    "validation": PeriodConfig(**{
        **STANDARD_PERIODS["validation"].__dict__,
    }),
    "test": PeriodConfig(**{
        **STANDARD_PERIODS["test"].__dict__,
    }),
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

# Extra SQL for v4.3 distribution features
_EXTRA_GLOBAL_SQL = """
    QUANTILE_CONT(target, 0.10)::DOUBLE AS global_p10,
    QUANTILE_CONT(target, 0.25)::DOUBLE AS global_p25,
    QUANTILE_CONT(target, 0.75)::DOUBLE AS global_p75,
    AVG(CASE WHEN target < 10 THEN 1.0 ELSE 0.0 END)::DOUBLE
        AS global_below_10_rate,
    AVG(CASE WHEN target > 10 THEN 1.0 ELSE 0.0 END)::DOUBLE
        AS global_above_10_rate,
    AVG(CASE WHEN target > 20 THEN 1.0 ELSE 0.0 END)::DOUBLE
        AS global_above_20_rate,
    AVG(CASE WHEN target > 30 THEN 1.0 ELSE 0.0 END)::DOUBLE
        AS global_above_30_rate,
    AVG(CASE WHEN target > 40 THEN 1.0 ELSE 0.0 END)::DOUBLE
        AS global_above_40_rate
""".strip()

_EXTRA_GROUP_SQL_COMPANY_ROUTE_TIME_WEEKDAY = """
    QUANTILE_CONT(target, 0.10)::DOUBLE
        AS company_route_time_weekday_p10,
    QUANTILE_CONT(target, 0.25)::DOUBLE
        AS company_route_time_weekday_p25,
    QUANTILE_CONT(target, 0.75)::DOUBLE
        AS company_route_time_weekday_p75,
    AVG(CASE WHEN target < 10 THEN 1.0 ELSE 0.0 END)::DOUBLE
        AS company_route_time_weekday_below_10_rate,
    AVG(CASE WHEN target > 10 THEN 1.0 ELSE 0.0 END)::DOUBLE
        AS company_route_time_weekday_above_10_rate,
    AVG(CASE WHEN target > 20 THEN 1.0 ELSE 0.0 END)::DOUBLE
        AS company_route_time_weekday_above_20_rate,
    AVG(CASE WHEN target > 30 THEN 1.0 ELSE 0.0 END)::DOUBLE
        AS company_route_time_weekday_above_30_rate,
    AVG(CASE WHEN target > 40 THEN 1.0 ELSE 0.0 END)::DOUBLE
        AS company_route_time_weekday_above_40_rate
""".strip()

EXTRA_GROUP_SQLS = {
    DISTRIBUTION_PREFIX: _EXTRA_GROUP_SQL_COMPANY_ROUTE_TIME_WEEKDAY,
}


def _distribution_extra_long_term_expressions(alias: str, prefix: str) -> list[str]:
    """Return the 8 distribution COALESCE expressions for the distribution prefix."""
    if prefix != DISTRIBUTION_PREFIX:
        return []
    return [
        f"COALESCE({alias}.{prefix}_p10, global_stats.global_p10) AS {prefix}_p10",
        f"COALESCE({alias}.{prefix}_p25, global_stats.global_p25) AS {prefix}_p25",
        f"COALESCE({alias}.{prefix}_p75, global_stats.global_p75) AS {prefix}_p75",
        f"COALESCE({alias}.{prefix}_below_10_rate, global_stats.global_below_10_rate) AS {prefix}_below_10_rate",
        f"COALESCE({alias}.{prefix}_above_10_rate, global_stats.global_above_10_rate) AS {prefix}_above_10_rate",
        f"COALESCE({alias}.{prefix}_above_20_rate, global_stats.global_above_20_rate) AS {prefix}_above_20_rate",
        f"COALESCE({alias}.{prefix}_above_30_rate, global_stats.global_above_30_rate) AS {prefix}_above_30_rate",
        f"COALESCE({alias}.{prefix}_above_40_rate, global_stats.global_above_40_rate) AS {prefix}_above_40_rate",
    ]


def _build_v4_3_feature_table(conn, *, table_prefix: str) -> None:
    """Assemble the feature table with v4.3-specific distribution extensions."""
    from scripts.shared.constants import (
        HISTORICAL_GROUP_DEFINITIONS,
        RECENT_GROUP_DEFINITIONS,
        RECENT_WINDOWS,
    )

    select_expressions = [
        "base.SEFER_ID",
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
    joins: list[str] = []
    aliases: dict[str, str] = {}

    for index, (prefix, group_columns) in enumerate(HISTORICAL_GROUP_DEFINITIONS):
        alias = f"long_term_{index}"
        aliases[prefix] = alias
        joins.append(
            f"LEFT JOIN {table_prefix}_{prefix}_stats AS {alias} "
            f"ON {join_condition('base', alias, group_columns)}"
        )
        select_expressions.extend(long_term_expressions(alias, prefix))
        select_expressions.extend(
            _distribution_extra_long_term_expressions(alias, prefix)
        )

    recent_index = len(HISTORICAL_GROUP_DEFINITIONS)
    for prefix, group_columns in RECENT_GROUP_DEFINITIONS:
        fallback = (
            f"COALESCE({aliases[prefix]}.{prefix}_average, "
            "global_stats.global_average)"
        )
        for window in RECENT_WINDOWS:
            alias = f"recent_{recent_index}"
            join = join_condition("base", alias, group_columns)
            join += f" AND base.SEFER_TARIHI = {alias}.reference_date"
            joins.append(
                f"LEFT JOIN {table_prefix}_{prefix}_recent_{window}d_stats "
                f"AS {alias} ON {join}"
            )
            select_expressions.extend(
                [
                    f"COALESCE({alias}.{prefix}_recent_{window}d_average, "
                    f"{fallback}) AS {prefix}_recent_{window}d_average",
                    f"COALESCE({alias}.{prefix}_recent_{window}d_count, "
                    f"0)::BIGINT AS {prefix}_recent_{window}d_count",
                ]
            )
            recent_index += 1

    specific = aliases["company_route_time_weekday"]
    canonical_time = aliases["canonical_route_time_weekday"]
    canonical = aliases["canonical_route"]
    select_expressions.extend(
        [
            f"COALESCE({specific}.company_route_time_weekday_average, "
            f"{canonical_time}.canonical_route_time_weekday_average, "
            f"{canonical}.canonical_route_average, "
            "global_stats.global_average) AS baseline_prediction",
            f"CASE WHEN {specific}.company_route_time_weekday_average "
            "IS NOT NULL THEN 'company_route_time_weekday' "
            f"WHEN {canonical_time}.canonical_route_time_weekday_average "
            "IS NOT NULL THEN 'canonical_route_time_weekday' "
            f"WHEN {canonical}.canonical_route_average IS NOT NULL "
            "THEN 'canonical_route' ELSE 'overall_average' END AS baseline_source",
        ]
    )

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table_prefix}_features AS
        SELECT {', '.join(select_expressions)}
        FROM {table_prefix}_target AS base
        CROSS JOIN {table_prefix}_global_stats AS global_stats
        {' '.join(joins)}
    """)


def ensure_v4_3_environment(*, require_model: bool = False) -> None:
    extra = [(FROZEN_RULE_PATH, "Frozen v4.2 rule not found")]
    ensure_environment(
        require_model=require_model,
        model_path=MODEL_PATH,
        extra_paths=extra,
    )


def connect_v4_3():
    return connect()


def create_v4_3_source_tables(conn, config, *, training: bool) -> None:
    create_source_tables(conn, config, table_prefix="v4_3")
    if not training and hasattr(config, 'expected_rows'):
        target_count = conn.execute(
            "SELECT COUNT(*) FROM v4_3_target"
        ).fetchone()[0]
        if target_count != config.expected_rows:
            raise RuntimeError(
                f"Expected {config.expected_rows:,} rows for {config.key}; "
                f"found {target_count:,}."
            )


def create_v4_3_long_term_statistics(conn, *, history_where_sql: str = "TRUE") -> None:
    create_long_term_statistics(
        conn,
        table_prefix="v4_3",
        history_where_sql=history_where_sql,
        extra_global_sql=_EXTRA_GLOBAL_SQL,
        extra_group_sqls=EXTRA_GROUP_SQLS,
    )


def create_v4_3_recent_statistics(conn, config, *, history_table: str) -> None:
    create_recent_statistics(
        conn, config, table_prefix="v4_3", history_table=history_table,
    )


def create_v4_3_feature_table(conn) -> None:
    _build_v4_3_feature_table(conn, table_prefix="v4_3")


def validate_feature_frame(frame: pd.DataFrame) -> None:
    if len(FEATURE_COLUMNS) != 72 or len(set(FEATURE_COLUMNS)) != 72:
        raise RuntimeError("v4.3 must have exactly 72 unique feature columns.")
    _shared_validate_feature_frame(frame, feature_columns=FEATURE_COLUMNS)


def build_evaluation_matrix(config: PeriodConfig) -> pd.DataFrame:
    ensure_v4_3_environment(require_model=True)
    conn = connect_v4_3()
    try:
        create_v4_3_source_tables(conn, config, training=False)
        create_v4_3_long_term_statistics(conn)
        create_v4_3_recent_statistics(conn, config, history_table="v4_3_history")
        create_v4_3_feature_table(conn)
        columns = [
            "SEFER_ID", "SEFER_TARIHI", *FEATURE_COLUMNS,
            "target", "baseline_prediction", "baseline_source",
        ]
        frame = conn.execute(
            f"SELECT {sql_identifier_list(columns)} FROM v4_3_features"
        ).fetchdf()
    finally:
        conn.close()
    validate_feature_frame(frame)
    return frame


def load_v4_3_model():
    ensure_v4_3_environment(require_model=True)
    return load_catboost_regressor(MODEL_PATH, FEATURE_COLUMNS)


def _load_frozen_rule() -> dict[str, Any]:
    payload = json.loads(FROZEN_RULE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "catboost_v4_2_hybrid_rule_v1":
        raise RuntimeError("Unexpected frozen v4.2 rule schema.")
    if payload.get("frozen_on_period") != "validation_2025_h1":
        raise RuntimeError("v4.2 rule is not the validation-frozen rule.")
    return payload["rule"]


def apply_frozen_hybrid_to_v4_3(frame: pd.DataFrame) -> np.ndarray:
    """Apply the exact frozen v4.2 rule with v4.3 as the underlying model."""
    rule = _load_frozen_rule()
    prediction = frame["v4_3_prediction"].to_numpy(dtype=np.float64)
    baseline = frame["baseline_prediction"].to_numpy(dtype=np.float64)
    prefix = DISTRIBUTION_PREFIX
    supported = frame[f"{prefix}_count"].to_numpy() >= rule["minimum_count"]
    upward_only = baseline > prediction
    moderate = (
        (frame[f"{prefix}_p90"].to_numpy() >= rule["p90_threshold"])
        | (frame[f"{prefix}_above_60_rate"].to_numpy() >= rule["above_60_rate_threshold"])
    )
    above_100 = frame[f"{prefix}_above_100_rate"].to_numpy()
    strong_above_100 = (
        above_100 > 0.0
        if rule["above_100_rate_threshold"] == 0.0
        else above_100 >= rule["above_100_rate_threshold"]
    )
    strong = (
        (frame[f"{prefix}_p90"].to_numpy() >= rule["strong_p90_threshold"])
        | strong_above_100
    )
    eligible = supported & upward_only
    weights = np.zeros(len(frame), dtype=np.float64)
    weights[eligible & moderate & ~strong] = rule["moderate_weight"]
    weights[eligible & strong] = rule["strong_weight"]
    return prediction + weights * (baseline - prediction)


def merge_frozen_comparisons(frame: pd.DataFrame, config: PeriodConfig) -> pd.DataFrame:
    path = RESULTS_DIR / f"catboost_v4_2_{config.key}_predictions.csv"
    required = [
        "SEFER_ID", "actual", "v3_prediction",
        "v4_1_prediction", "hybrid_prediction",
    ]
    frozen = pd.read_csv(path, usecols=required).rename(
        columns={
            "actual": "frozen_actual",
            "hybrid_prediction": "v4_2_hybrid_prediction",
        }
    )
    if frozen["SEFER_ID"].duplicated().any() or frame["SEFER_ID"].duplicated().any():
        raise RuntimeError("SEFER_ID must be unique for comparison alignment.")
    merged = frame.merge(frozen, on="SEFER_ID", how="left", validate="one_to_one")
    comparison_columns = [
        "v3_prediction", "v4_1_prediction", "v4_2_hybrid_prediction",
    ]
    if merged[comparison_columns].isna().any().any():
        raise RuntimeError("Frozen comparison predictions did not align to v4.3 rows.")
    if not np.array_equal(
        merged["target"].to_numpy(), merged["frozen_actual"].to_numpy()
    ):
        raise RuntimeError("Actual targets disagree with frozen v4.2 artifacts.")
    return merged.drop(columns="frozen_actual")


def business_metrics(frame: pd.DataFrame, config: PeriodConfig) -> pd.DataFrame:
    """Return metrics in the user-specified decision order for all six models."""
    actual = frame["target"].to_numpy(dtype=np.float64)
    focus = (actual >= 10) & (actual <= 40)
    above_40 = actual >= 40
    records: list[dict[str, Any]] = []
    for model_name, column in MODEL_COMPARISON_COLUMNS.items():
        prediction = frame[column].to_numpy(dtype=np.float64)
        rounded = np.maximum(0.0, np.rint(prediction)).astype(np.int64)
        overall = regression_metrics(actual, prediction)
        record: dict[str, Any] = {
            "period": config.key,
            "tuning_allowed": config.tuning_allowed,
            "model": model_name,
            "row_count": len(frame),
            "mae_actual_10_to_40": regression_metrics(
                actual[focus], prediction[focus]
            )["mae"],
        }
        for threshold in BUSINESS_THRESHOLDS:
            binary = binary_classification_metrics(
                actual, rounded, threshold, predictions_are_rounded=True,
            )
            record[f"fnr_at_{threshold}"] = binary["false_negative_rate"]
            record[f"fpr_at_{threshold}"] = binary["false_positive_rate"]
            lower, upper = NEAR_THRESHOLD_NEGATIVE_BANDS[threshold]
            near_negative = (actual >= lower) & (actual < upper)
            record[f"near_threshold_fpr_at_{threshold}"] = float(
                np.mean(rounded[near_negative] >= threshold)
            )
        record.update(
            {
                "overall_mae": overall["mae"],
                "overall_rmse": overall["rmse"],
                "overall_bias": overall["bias"],
                "rmse_actual_40_plus": regression_metrics(
                    actual[above_40], prediction[above_40]
                )["rmse"],
            }
        )
        records.append(record)
    metrics = pd.DataFrame(records)
    reference = metrics.loc[metrics["model"] == "v4.2 hybrid"].iloc[0]
    for metric in [
        "mae_actual_10_to_40", "fnr_at_20", "fnr_at_30", "fnr_at_43",
        "overall_mae", "rmse_actual_40_plus",
        "near_threshold_fpr_at_20", "near_threshold_fpr_at_30", "near_threshold_fpr_at_43",
    ]:
        metrics[f"delta_vs_v4_2__{metric}"] = metrics[metric] - reference[metric]
    return metrics


def write_comparison_outputs(frame: pd.DataFrame, config: PeriodConfig) -> None:
    stem = RESULTS_DIR / f"catboost_v4_3_{config.key}"
    metrics_path = Path(f"{stem}_business_metrics.csv")
    predictions_path = Path(f"{stem}_predictions.csv")
    metadata_path = Path(f"{stem}_metadata.json")
    for path in (metrics_path, predictions_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite experiment artifact: {path}")
    metrics = business_metrics(frame, config)
    prediction_columns = [
        "SEFER_ID", "SEFER_TARIHI", "target",
        *MODEL_COMPARISON_COLUMNS.values(), "baseline_source",
    ]
    metrics.to_csv(metrics_path, index=False)
    frame[prediction_columns].to_csv(predictions_path, index=False)
    metadata = {
        "schema_version": "catboost_v4_3_business_evaluation_v1",
        "period": config.key,
        "history_start": config.history_start,
        "history_end_exclusive": config.history_end_exclusive,
        "target_start": config.target_start,
        "target_end_exclusive": config.target_end_exclusive,
        "tuning_allowed": config.tuning_allowed,
        "primary_metric": "mae_actual_10_to_40",
        "selection_order": [
            "mae_actual_10_to_40",
            "fnr_at_20/fnr_at_30/fnr_at_43",
            "overall_mae",
            "rmse_actual_40_plus",
            "near_threshold_fpr_at_20/30/43",
        ],
        "near_threshold_negative_bands": {
            str(key): list(value)
            for key, value in NEAR_THRESHOLD_NEGATIVE_BANDS.items()
        },
        "profitability_claimed": False,
        "frozen_v4_2_rule": str(FROZEN_RULE_PATH.relative_to(DB_PATH.parent)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    display_columns = [
        "model", "mae_actual_10_to_40", "fnr_at_20", "fnr_at_30", "fnr_at_43",
        "overall_mae", "rmse_actual_40_plus",
        "near_threshold_fpr_at_20", "near_threshold_fpr_at_30", "near_threshold_fpr_at_43",
    ]
    print(f"\n{config.label} business-ordered comparison")
    print(metrics[display_columns].to_string(index=False))

