from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

PROJECT_DIR = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(PROJECT_DIR))
from scripts.catboost.v4_1 import common as v4_1
RESULTS_DIR = PROJECT_DIR / "results"
V3_MODEL_PATH = (
    PROJECT_DIR / "models" / "catboost_demand_model_v3_mae_6000.cbm"
)
V4_1_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "catboost_demand_model_v4_1_recent_mae_6000.cbm"
)
RULE_PATH = RESULTS_DIR / "catboost_v4_2_hybrid_rule.json"

V3_FEATURE_COLUMNS = (
    v4_1.CATEGORICAL_FEATURES
    + v4_1.CALENDAR_NUMERIC_FEATURES
    + v4_1.HISTORICAL_FEATURES
)

SIGNAL_PREFIX = "company_route_time_weekday"
P90_COLUMN = f"{SIGNAL_PREFIX}_p90"
ABOVE_60_COLUMN = f"{SIGNAL_PREFIX}_above_60_rate"
ABOVE_100_COLUMN = f"{SIGNAL_PREFIX}_above_100_rate"
COUNT_COLUMN = f"{SIGNAL_PREFIX}_count"

MINIMUM_COUNTS = [10, 20, 50, 100]
P90_THRESHOLDS = [50.0, 60.0, 70.0, 80.0, 90.0]
ABOVE_60_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.30]
ABOVE_100_THRESHOLDS = [0.00, 0.01, 0.02, 0.05, 0.10]
MODERATE_WEIGHTS = [0.10, 0.20, 0.30, 0.40]
STRONG_WEIGHTS = [0.40, 0.50, 0.60, 0.75, 1.00]
STRONG_P90_OFFSET = 20.0
MAE_TOLERANCE = 0.01
METRIC_EQUALITY_TOLERANCE = 1e-6

TARGET_GROUP_BINS = [0, 10, 20, 30, 40, 60, 100, 300]
TARGET_GROUP_LABELS = [
    "1-10",
    "11-20",
    "21-30",
    "31-40",
    "41-60",
    "61-100",
    "101-300",
]


@dataclass(frozen=True)
class PeriodConfig:
    key: str
    label: str
    history_start: str
    history_end_exclusive: str
    target_start: str
    target_end_exclusive: str
    expected_rows: int


@dataclass(frozen=True)
class HybridRule:
    minimum_count: int
    p90_threshold: float
    strong_p90_threshold: float
    above_60_rate_threshold: float
    above_100_rate_threshold: float
    moderate_weight: float
    strong_weight: float


def _metric_values(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - actual
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
    }


def _load_model(path: Path, expected_features: list[str]) -> CatBoostRegressor:
    if not path.exists():
        raise FileNotFoundError(f"Frozen model not found: {path}")

    model = CatBoostRegressor()
    model.load_model(str(path))
    saved_features = list(model.feature_names_)
    if saved_features != expected_features:
        raise RuntimeError(
            f"Feature contract mismatch for {path.name}.\n"
            f"Saved: {saved_features}\nExpected: {expected_features}"
        )
    return model


def ensure_environment(require_rule: bool = False) -> None:
    if not v4_1.DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {v4_1.DB_PATH}")
    if require_rule and not RULE_PATH.exists():
        raise FileNotFoundError(
            f"Frozen v4.2 rule not found: {RULE_PATH}. "
            "Run scripts/catboost/v4_2/validation.py first."
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    v4_1.TEMP_DIR.mkdir(parents=True, exist_ok=True)


def build_prediction_frame(config: PeriodConfig) -> pd.DataFrame:
    """Build leakage-safe v3/v4.1 features in DuckDB, then predict."""

    ensure_environment()
    v3_model = _load_model(V3_MODEL_PATH, V3_FEATURE_COLUMNS)
    v4_1_model = _load_model(V4_1_MODEL_PATH, v4_1.FEATURE_COLUMNS)

    evaluation_config = v4_1.EvaluationConfig(
        key=config.key,
        label=config.label,
        history_start=config.history_start,
        history_end_exclusive=config.history_end_exclusive,
        target_start=config.target_start,
        target_end_exclusive=config.target_end_exclusive,
        expected_rows=config.expected_rows,
        baseline_mae=0.0,
        baseline_rmse=0.0,
    )

    print(f"\nBuilding frozen-period features for {config.label}")
    print(
        f"History: [{config.history_start}, {config.history_end_exclusive}); "
        f"target: [{config.target_start}, {config.target_end_exclusive})"
    )

    conn = duckdb.connect(str(v4_1.DB_PATH), read_only=False)
    temp_path = v4_1.TEMP_DIR.as_posix().replace("'", "''")
    conn.execute(f"SET threads = {v4_1.DUCKDB_THREADS}")
    conn.execute(f"SET temp_directory = '{temp_path}'")

    try:
        v4_1.create_source_tables(conn, evaluation_config)
        v4_1.create_long_term_statistics(conn)
        v4_1.create_recent_statistics(conn, evaluation_config)
        v4_1.create_feature_table(conn)

        selected_columns = [
            "SEFER_ID",
            "SEFER_TARIHI",
            *v4_1.FEATURE_COLUMNS,
            "target",
            "baseline_prediction",
            "baseline_source",
        ]
        frame = conn.execute(
            f"SELECT {v4_1.sql_identifier_list(selected_columns)} "
            "FROM evaluation_features"
        ).fetchdf()
    finally:
        conn.close()

    if len(frame) != config.expected_rows:
        raise RuntimeError(
            f"Expected {config.expected_rows:,} rows for {config.key}, "
            f"received {len(frame):,}."
        )

    for column in v4_1.CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype("string").fillna("missing")

    missing = frame[v4_1.FEATURE_COLUMNS].isna().sum()
    if int(missing.sum()) != 0:
        raise RuntimeError(
            "Feature matrix contains missing values:\n"
            + missing[missing > 0].to_string()
        )

    print(f"Predicting {len(frame):,} rows with frozen v3 and v4.1 models")
    frame["v3_prediction"] = v3_model.predict(frame[V3_FEATURE_COLUMNS])
    frame["v4_1_prediction"] = v4_1_model.predict(
        frame[v4_1.FEATURE_COLUMNS]
    )
    return frame


def _above_100_signal(values: np.ndarray, threshold: float) -> np.ndarray:
    # A zero threshold means "any observed >100 event", not the vacuous >= 0.
    if threshold == 0.0:
        return values > 0.0
    return values >= threshold


def apply_hybrid_rule(
    frame: pd.DataFrame,
    rule: HybridRule,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v4_prediction = frame["v4_1_prediction"].to_numpy(dtype=np.float64)
    baseline = frame["baseline_prediction"].to_numpy(dtype=np.float64)
    supported = frame[COUNT_COLUMN].to_numpy() >= rule.minimum_count
    upward_only = baseline > v4_prediction

    moderate_signal = (
        (frame[P90_COLUMN].to_numpy(dtype=np.float64) >= rule.p90_threshold)
        | (
            frame[ABOVE_60_COLUMN].to_numpy(dtype=np.float64)
            >= rule.above_60_rate_threshold
        )
    )
    strong_signal = (
        (
            frame[P90_COLUMN].to_numpy(dtype=np.float64)
            >= rule.strong_p90_threshold
        )
        | _above_100_signal(
            frame[ABOVE_100_COLUMN].to_numpy(dtype=np.float64),
            rule.above_100_rate_threshold,
        )
    )

    eligible = supported & upward_only
    strong = eligible & strong_signal
    moderate = eligible & moderate_signal & ~strong
    weights = np.zeros(len(frame), dtype=np.float64)
    weights[moderate] = rule.moderate_weight
    weights[strong] = rule.strong_weight
    hybrid = v4_prediction + weights * (baseline - v4_prediction)
    activation_level = np.full(len(frame), "none", dtype=object)
    activation_level[moderate] = "moderate"
    activation_level[strong] = "strong"
    return hybrid, weights, activation_level


def comparison_metrics(
    frame: pd.DataFrame,
    hybrid_prediction: np.ndarray,
) -> pd.DataFrame:
    actual = frame["target"].to_numpy(dtype=np.float64)
    models = [
        ("CatBoost v3", frame["v3_prediction"].to_numpy(dtype=np.float64)),
        ("CatBoost v4.1", frame["v4_1_prediction"].to_numpy(dtype=np.float64)),
        (
            "Weekday baseline",
            frame["baseline_prediction"].to_numpy(dtype=np.float64),
        ),
        ("v4.2 hybrid", hybrid_prediction),
    ]
    return pd.DataFrame(
        [{"model": name, **_metric_values(actual, prediction)} for name, prediction in models]
    )


def target_group_metrics(
    frame: pd.DataFrame,
    hybrid_prediction: np.ndarray,
    hybrid_weight: np.ndarray,
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "actual": frame["target"].to_numpy(dtype=np.float64),
            "v3": frame["v3_prediction"].to_numpy(dtype=np.float64),
            "v4_1": frame["v4_1_prediction"].to_numpy(dtype=np.float64),
            "baseline": frame["baseline_prediction"].to_numpy(dtype=np.float64),
            "hybrid": hybrid_prediction,
            "hybrid_weight": hybrid_weight,
        }
    )
    data["target_group"] = pd.cut(
        data["actual"],
        bins=TARGET_GROUP_BINS,
        labels=TARGET_GROUP_LABELS,
        include_lowest=True,
    )
    for name in ["v3", "v4_1", "baseline", "hybrid"]:
        error = data[name] - data["actual"]
        data[f"{name}_absolute_error"] = error.abs()
        data[f"{name}_squared_error"] = np.square(error)
    data["hybrid_active"] = data["hybrid_weight"] > 0.0

    grouped = data.groupby("target_group", observed=False).agg(
        row_count=("actual", "size"),
        average_actual=("actual", "mean"),
        average_v3_prediction=("v3", "mean"),
        average_v4_1_prediction=("v4_1", "mean"),
        average_baseline_prediction=("baseline", "mean"),
        average_hybrid_prediction=("hybrid", "mean"),
        v3_mae=("v3_absolute_error", "mean"),
        v4_1_mae=("v4_1_absolute_error", "mean"),
        baseline_mae=("baseline_absolute_error", "mean"),
        hybrid_mae=("hybrid_absolute_error", "mean"),
        v3_mse=("v3_squared_error", "mean"),
        v4_1_mse=("v4_1_squared_error", "mean"),
        baseline_mse=("baseline_squared_error", "mean"),
        hybrid_mse=("hybrid_squared_error", "mean"),
        hybrid_bias_sum=("hybrid", "sum"),
        actual_sum=("actual", "sum"),
        average_hybrid_weight=("hybrid_weight", "mean"),
        hybrid_activation_rate=("hybrid_active", "mean"),
    ).reset_index()

    for name in ["v3", "v4_1", "baseline", "hybrid"]:
        grouped[f"{name}_rmse"] = np.sqrt(grouped.pop(f"{name}_mse"))
    grouped["hybrid_bias"] = (
        grouped.pop("hybrid_bias_sum") - grouped.pop("actual_sum")
    ) / grouped["row_count"]

    required_order = [
        "target_group",
        "row_count",
        "average_actual",
        "average_v3_prediction",
        "average_v4_1_prediction",
        "average_baseline_prediction",
        "average_hybrid_prediction",
        "v3_mae",
        "v4_1_mae",
        "baseline_mae",
        "hybrid_mae",
        "v3_rmse",
        "v4_1_rmse",
        "baseline_rmse",
        "hybrid_rmse",
        "hybrid_bias",
        "average_hybrid_weight",
        "hybrid_activation_rate",
    ]
    return grouped[required_order]


def export_period_results(
    config: PeriodConfig,
    frame: pd.DataFrame,
    rule: HybridRule,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    hybrid, weights, levels = apply_hybrid_rule(frame, rule)
    metrics = comparison_metrics(frame, hybrid)
    metrics.insert(0, "period", config.key)
    metrics.insert(2, "row_count", len(frame))
    groups = target_group_metrics(frame, hybrid, weights)
    groups.insert(0, "period", config.key)

    predictions = pd.DataFrame(
        {
            "SEFER_ID": frame["SEFER_ID"].to_numpy(),
            "SEFER_TARIHI": frame["SEFER_TARIHI"].to_numpy(),
            "actual": frame["target"].to_numpy(),
            "v3_prediction": frame["v3_prediction"].to_numpy(),
            "v4_1_prediction": frame["v4_1_prediction"].to_numpy(),
            "baseline_prediction": frame["baseline_prediction"].to_numpy(),
            "baseline_source": frame["baseline_source"].to_numpy(),
            "hybrid_prediction": hybrid,
            "hybrid_weight": weights,
            "hybrid_activation_level": levels,
            P90_COLUMN: frame[P90_COLUMN].to_numpy(),
            ABOVE_60_COLUMN: frame[ABOVE_60_COLUMN].to_numpy(),
            ABOVE_100_COLUMN: frame[ABOVE_100_COLUMN].to_numpy(),
            COUNT_COLUMN: frame[COUNT_COLUMN].to_numpy(),
        }
    )

    stem = RESULTS_DIR / f"catboost_v4_2_{config.key}"
    metrics.to_csv(f"{stem}_metrics.csv", index=False)
    groups.to_csv(f"{stem}_target_groups.csv", index=False)
    predictions.to_csv(f"{stem}_predictions.csv", index=False)

    print(f"\n{config.label} comparison")
    print(metrics.drop(columns=["period", "row_count"]).to_string(index=False))
    print("\nTarget groups")
    print(groups.drop(columns=["period"]).to_string(index=False))
    print(f"\nHybrid activation rate: {float(np.mean(weights > 0.0)):.6%}")
    print(f"Average hybrid weight: {float(np.mean(weights)):.8f}")
    return metrics, groups


def _threshold_mask(values: np.ndarray, thresholds: list[float], *, positive_zero: bool = False) -> np.ndarray:
    mask = np.zeros(len(values), dtype=np.uint8)
    for index, threshold in enumerate(thresholds):
        condition = values > 0.0 if positive_zero and threshold == 0.0 else values >= threshold
        mask |= condition.astype(np.uint8) << index
    return mask


def _build_search_cells(frame: pd.DataFrame) -> pd.DataFrame:
    actual = frame["target"].to_numpy(dtype=np.float64)
    v4_prediction = frame["v4_1_prediction"].to_numpy(dtype=np.float64)
    baseline = frame["baseline_prediction"].to_numpy(dtype=np.float64)
    error = actual - v4_prediction
    delta = baseline - v4_prediction
    strong_p90_thresholds = [value + STRONG_P90_OFFSET for value in P90_THRESHOLDS]

    cells = pd.DataFrame(
        {
            "upward_only": delta > 0.0,
            "count_mask": _threshold_mask(frame[COUNT_COLUMN].to_numpy(), MINIMUM_COUNTS),
            "p90_mask": _threshold_mask(frame[P90_COLUMN].to_numpy(dtype=np.float64), P90_THRESHOLDS),
            "strong_p90_mask": _threshold_mask(frame[P90_COLUMN].to_numpy(dtype=np.float64), strong_p90_thresholds),
            "above_60_mask": _threshold_mask(frame[ABOVE_60_COLUMN].to_numpy(dtype=np.float64), ABOVE_60_THRESHOLDS),
            "above_100_mask": _threshold_mask(frame[ABOVE_100_COLUMN].to_numpy(dtype=np.float64), ABOVE_100_THRESHOLDS, positive_zero=True),
            "row_count": np.ones(len(frame), dtype=np.int64),
            "sum_error": error,
            "sum_delta": delta,
            "sum_error_delta": error * delta,
            "sum_delta_squared": np.square(delta),
            "sum_absolute_error": np.abs(error),
            "sum_squared_error": np.square(error),
        }
    )
    all_weights = sorted(set(MODERATE_WEIGHTS + STRONG_WEIGHTS))
    for weight in all_weights:
        cells[f"absolute_delta_{weight:g}"] = (
            np.abs(error - weight * delta) - np.abs(error)
        )

    group_columns = [
        "upward_only",
        "count_mask",
        "p90_mask",
        "strong_p90_mask",
        "above_60_mask",
        "above_100_mask",
    ]
    return cells.groupby(group_columns, observed=True, sort=False).sum().reset_index()


def grid_search_hybrid(frame: pd.DataFrame) -> tuple[pd.DataFrame, HybridRule]:
    """Evaluate the complete 10,000-candidate grid exactly on compressed cells."""

    cells = _build_search_cells(frame)
    total_rows = int(cells["row_count"].sum())
    base_abs = float(cells["sum_absolute_error"].sum())
    base_squared = float(cells["sum_squared_error"].sum())
    base_bias_sum = -float(cells["sum_error"].sum())
    v4_1_mae = base_abs / total_rows

    records: list[dict[str, Any]] = []
    for count_index, p90_index, above_60_index, above_100_index in product(
        range(len(MINIMUM_COUNTS)),
        range(len(P90_THRESHOLDS)),
        range(len(ABOVE_60_THRESHOLDS)),
        range(len(ABOVE_100_THRESHOLDS)),
    ):
        bit_count = np.uint8(1 << count_index)
        bit_p90 = np.uint8(1 << p90_index)
        bit_60 = np.uint8(1 << above_60_index)
        bit_100 = np.uint8(1 << above_100_index)
        eligible = cells["upward_only"].to_numpy() & (
            (cells["count_mask"].to_numpy(dtype=np.uint8) & bit_count) != 0
        )
        strong_signal = (
            (cells["strong_p90_mask"].to_numpy(dtype=np.uint8) & bit_p90) != 0
        ) | ((cells["above_100_mask"].to_numpy(dtype=np.uint8) & bit_100) != 0)
        moderate_signal = (
            (cells["p90_mask"].to_numpy(dtype=np.uint8) & bit_p90) != 0
        ) | ((cells["above_60_mask"].to_numpy(dtype=np.uint8) & bit_60) != 0)
        strong = eligible & strong_signal
        moderate = eligible & moderate_signal & ~strong

        moderate_count = int(cells.loc[moderate, "row_count"].sum())
        strong_count = int(cells.loc[strong, "row_count"].sum())
        mod_ed = float(cells.loc[moderate, "sum_error_delta"].sum())
        mod_d2 = float(cells.loc[moderate, "sum_delta_squared"].sum())
        mod_d = float(cells.loc[moderate, "sum_delta"].sum())
        strong_ed = float(cells.loc[strong, "sum_error_delta"].sum())
        strong_d2 = float(cells.loc[strong, "sum_delta_squared"].sum())
        strong_d = float(cells.loc[strong, "sum_delta"].sum())

        for moderate_weight, strong_weight in product(MODERATE_WEIGHTS, STRONG_WEIGHTS):
            absolute_sum = (
                base_abs
                + float(cells.loc[moderate, f"absolute_delta_{moderate_weight:g}"].sum())
                + float(cells.loc[strong, f"absolute_delta_{strong_weight:g}"].sum())
            )
            squared_sum = (
                base_squared
                - 2.0 * moderate_weight * mod_ed
                + moderate_weight**2 * mod_d2
                - 2.0 * strong_weight * strong_ed
                + strong_weight**2 * strong_d2
            )
            bias_sum = (
                base_bias_sum
                + moderate_weight * mod_d
                + strong_weight * strong_d
            )
            active_count = moderate_count + strong_count
            records.append(
                {
                    "minimum_count": MINIMUM_COUNTS[count_index],
                    "p90_threshold": P90_THRESHOLDS[p90_index],
                    "strong_p90_threshold": P90_THRESHOLDS[p90_index] + STRONG_P90_OFFSET,
                    "above_60_rate_threshold": ABOVE_60_THRESHOLDS[above_60_index],
                    "above_100_rate_threshold": ABOVE_100_THRESHOLDS[above_100_index],
                    "moderate_weight": moderate_weight,
                    "strong_weight": strong_weight,
                    "mae": absolute_sum / total_rows,
                    "rmse": float(np.sqrt(max(0.0, squared_sum / total_rows))),
                    "bias": bias_sum / total_rows,
                    "moderate_row_count": moderate_count,
                    "strong_row_count": strong_count,
                    "activation_rate": active_count / total_rows,
                    "average_weight": (
                        moderate_count * moderate_weight + strong_count * strong_weight
                    ) / total_rows,
                }
            )

    candidates = pd.DataFrame.from_records(records)
    candidates["mae_allowed"] = candidates["mae"] <= v4_1_mae + MAE_TOLERANCE
    allowed = candidates[candidates["mae_allowed"]].copy()
    if allowed.empty:
        raise RuntimeError("No hybrid candidate satisfies the validation MAE safeguard.")

    # Six-decimal equivalence implements the documented simpler/lower-weight tie-break.
    allowed["rmse_equivalence"] = (
        allowed["rmse"] / METRIC_EQUALITY_TOLERANCE
    ).round().astype(np.int64)
    allowed["mae_equivalence"] = (
        allowed["mae"] / METRIC_EQUALITY_TOLERANCE
    ).round().astype(np.int64)
    selected_index = allowed.sort_values(
        by=[
            "rmse_equivalence",
            "mae_equivalence",
            "strong_weight",
            "moderate_weight",
            "activation_rate",
            "minimum_count",
            "p90_threshold",
            "above_60_rate_threshold",
            "above_100_rate_threshold",
        ],
        ascending=[True, True, True, True, True, False, False, False, False],
        kind="stable",
    ).index[0]
    candidates["selected"] = candidates.index == selected_index
    winner = candidates.loc[selected_index]
    rule = HybridRule(
        minimum_count=int(winner["minimum_count"]),
        p90_threshold=float(winner["p90_threshold"]),
        strong_p90_threshold=float(winner["strong_p90_threshold"]),
        above_60_rate_threshold=float(winner["above_60_rate_threshold"]),
        above_100_rate_threshold=float(winner["above_100_rate_threshold"]),
        moderate_weight=float(winner["moderate_weight"]),
        strong_weight=float(winner["strong_weight"]),
    )
    return candidates, rule


def save_frozen_rule(
    rule: HybridRule,
    validation_config: PeriodConfig,
    candidates: pd.DataFrame,
) -> None:
    selected = candidates.loc[candidates["selected"]].iloc[0]
    payload = {
        "schema_version": "catboost_v4_2_hybrid_rule_v1",
        "frozen_on_period": validation_config.key,
        "validation_history_end_exclusive": validation_config.history_end_exclusive,
        "validation_target_end_exclusive": validation_config.target_end_exclusive,
        "v3_model": str(V3_MODEL_PATH.relative_to(PROJECT_DIR)),
        "v4_1_model": str(V4_1_MODEL_PATH.relative_to(PROJECT_DIR)),
        "selection": {
            "mae_tolerance_above_v4_1": MAE_TOLERANCE,
            "metric_equality_tolerance": METRIC_EQUALITY_TOLERANCE,
            "primary_metric": "rmse",
            "tie_breakers": ["mae", "lower_weights", "lower_activation"],
            "candidate_count": int(len(candidates)),
            "selected_metrics": {
                key: float(selected[key])
                for key in ["mae", "rmse", "bias", "activation_rate", "average_weight"]
            },
        },
        "semantics": {
            "blend": "v4_1 + weight * (baseline - v4_1)",
            "upward_only": True,
            "moderate": "p90 >= threshold OR above_60_rate >= threshold",
            "strong": "p90 >= strong_p90_threshold OR above_100_rate meets threshold",
            "zero_above_100_threshold_means": "above_100_rate > 0",
            "strong_p90_offset": STRONG_P90_OFFSET,
        },
        "rule": asdict(rule),
    }
    RULE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_frozen_rule() -> HybridRule:
    ensure_environment(require_rule=True)
    payload = json.loads(RULE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "catboost_v4_2_hybrid_rule_v1":
        raise RuntimeError(f"Unsupported hybrid rule schema in {RULE_PATH}")
    if payload.get("frozen_on_period") != "validation_2025_h1":
        raise RuntimeError("The v4.2 rule was not frozen on 2025 H1 validation.")
    return HybridRule(**payload["rule"])


def run_validation(config: PeriodConfig) -> None:
    frame = build_prediction_frame(config)
    print("\nSearching the complete 10,000-candidate hybrid grid...")
    candidates, rule = grid_search_hybrid(frame)
    candidate_path = RESULTS_DIR / "catboost_v4_2_hybrid_validation_candidates.csv"
    candidates.to_csv(candidate_path, index=False)
    save_frozen_rule(rule, config, candidates)
    print(f"Selected frozen rule: {rule}")
    print(f"Saved candidates: {candidate_path}")
    print(f"Saved frozen rule: {RULE_PATH}")
    export_period_results(config, frame, rule)


def run_frozen_evaluation(config: PeriodConfig) -> None:
    rule = load_frozen_rule()
    print(f"Loaded frozen validation rule: {rule}")
    frame = build_prediction_frame(config)
    export_period_results(config, frame, rule)
