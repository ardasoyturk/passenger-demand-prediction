"""Paths and immutable contracts for stop-addition inference."""

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_DIR / "models" / "stop_addition"

MODEL_PATH = MODEL_DIR / "stop_addition_demand_selected.cbm"
MODEL_SUMMARY_PATH = MODEL_DIR / "stop_addition_demand_model_summary.json"

EXPECTED_FEATURE_COUNT = 131

