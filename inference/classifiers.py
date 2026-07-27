"""Frozen v4.4 demand-threshold classifier loading for inference.

Extracted from ``scripts/catboost_v4_4_classifier_common.py``.  Only the
artifact-loading pieces needed by the proposed-trip pipeline are kept here:
path conventions, class-weight suffixing, and the loader that validates the
frozen model and metadata contract.  Cutoff search, calibration, and frozen
evaluation remain in the training archive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier

from inference.features import FEATURE_COLUMNS


PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
DEFAULT_THRESHOLDS = (10, 20, 30, 43)


def class_weight_suffix(class_weight_mode: str) -> str:
    if class_weight_mode == "balanced":
        return ""
    if class_weight_mode == "none":
        return "_class_weights_none"
    raise ValueError("class_weight_mode must be 'balanced' or 'none'")


def model_path(threshold: int, class_weight_mode: str = "balanced") -> Path:
    suffix = class_weight_suffix(class_weight_mode)
    return MODELS_DIR / f"catboost_demand_model_v4_4_classifier_ge_{threshold}{suffix}.cbm"


def metadata_path(threshold: int, class_weight_mode: str = "balanced") -> Path:
    suffix = class_weight_suffix(class_weight_mode)
    return MODELS_DIR / f"catboost_demand_model_v4_4_classifier_ge_{threshold}{suffix}_metadata.json"


def load_classifier_and_metadata(
    threshold: int,
    class_weight_mode: str = "balanced",
) -> tuple[CatBoostClassifier, dict[str, Any]]:
    model_file = model_path(threshold, class_weight_mode)
    metadata_file = metadata_path(threshold, class_weight_mode)
    missing = [path for path in (model_file, metadata_file) if not path.exists()]
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Missing trained v4.4 artifact(s): {rendered}. Run the manual v4.4 training command first."
        )
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("threshold") != threshold:
        raise RuntimeError(f"Metadata threshold mismatch in {metadata_file}")
    if metadata.get("class_weight_mode") != class_weight_mode:
        raise RuntimeError(f"Class-weight mode mismatch in {metadata_file}")
    if metadata.get("feature_names") != FEATURE_COLUMNS:
        raise RuntimeError(f"Metadata feature contract mismatch in {metadata_file}")
    cutoff = metadata.get("frozen_probability_cutoff")
    if cutoff is None or not 0.05 <= float(cutoff) <= 0.95:
        raise RuntimeError(f"Invalid frozen cutoff in {metadata_file}")
    model = CatBoostClassifier()
    model.load_model(str(model_file))
    if list(model.feature_names_) != FEATURE_COLUMNS:
        raise RuntimeError(f"Model feature contract mismatch in {model_file}")
    return model, metadata