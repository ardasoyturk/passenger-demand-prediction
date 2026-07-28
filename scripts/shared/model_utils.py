"""CatBoost model loading with feature-contract validation."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from catboost import CatBoostClassifier, CatBoostRegressor


def load_catboost_regressor(
    path: Path,
    expected_features: Sequence[str],
) -> CatBoostRegressor:
    """Load a CatBoostRegressor and verify its feature contract."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = CatBoostRegressor()
    model.load_model(str(path))
    saved = list(model.feature_names_)
    if saved != list(expected_features):
        raise RuntimeError(
            f"Feature contract mismatch for {path.name}.\n"
            f"Saved: {saved}\nExpected: {list(expected_features)}"
        )
    return model


def load_catboost_classifier(
    path: Path,
    expected_features: Sequence[str],
) -> CatBoostClassifier:
    """Load a CatBoostClassifier and verify its feature contract."""
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = CatBoostClassifier()
    model.load_model(str(path))
    saved = list(model.feature_names_)
    if saved != list(expected_features):
        raise RuntimeError(
            f"Feature contract mismatch for {path.name}.\n"
            f"Saved: {saved}\nExpected: {list(expected_features)}"
        )
    return model
