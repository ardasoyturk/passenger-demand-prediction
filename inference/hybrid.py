"""Frozen v4.2 hybrid-rule loading and application for inference.

Extracted from ``scripts/catboost_v4_2_hybrid_common.py``.  This module holds
only the inference-side pieces of the v4.2 hybrid: the rule dataclass, the
frozen-rule loader, the applier, and a generic CatBoost model loader.  Grid
search, validation, and frozen-rule authoring remain in the training archive.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
V4_1_MODEL_PATH = (
    PROJECT_DIR
    / "models"
    / "catboost_demand_model_v4_1_recent_mae_6000.cbm"
)
RULE_PATH = RESULTS_DIR / "catboost_v4_2_hybrid_rule.json"

SIGNAL_PREFIX = "company_route_time_weekday"
P90_COLUMN = f"{SIGNAL_PREFIX}_p90"
ABOVE_60_COLUMN = f"{SIGNAL_PREFIX}_above_60_rate"
ABOVE_100_COLUMN = f"{SIGNAL_PREFIX}_above_100_rate"
COUNT_COLUMN = f"{SIGNAL_PREFIX}_count"


@dataclass(frozen=True)
class HybridRule:
    minimum_count: int
    p90_threshold: float
    strong_p90_threshold: float
    above_60_rate_threshold: float
    above_100_rate_threshold: float
    moderate_weight: float
    strong_weight: float


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


def load_frozen_rule() -> HybridRule:
    if not RULE_PATH.exists():
        raise FileNotFoundError(
            f"Frozen v4.2 rule not found: {RULE_PATH}. "
            "Run catboost_v4_2_hybrid_validation.py first."
        )
    payload = json.loads(RULE_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "catboost_v4_2_hybrid_rule_v1":
        raise RuntimeError(f"Unsupported hybrid rule schema in {RULE_PATH}")
    if payload.get("frozen_on_period") != "validation_2025_h1":
        raise RuntimeError("The v4.2 rule was not frozen on 2025 H1 validation.")
    return HybridRule(**payload["rule"])