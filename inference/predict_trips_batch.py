"""Vectorized CSV CLI for mixed v4.2/v4.4 proposed-trip inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from inference.engine import (
    CLASSIFIER_VARIANTS,
    INPUT_COLUMNS,
    PROJECT_DIR,
    print_reliability_summary,
    predict_proposals,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict passenger demand for a CSV of proposed trips.")
    parser.add_argument("--input", required=True, type=Path, help="Input CSV path")
    parser.add_argument("--output", required=True, type=Path, help="Output CSV path")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Append normalized lookup keys, history matches, and baseline source",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_DIR / path


def main() -> None:
    total_started = perf_counter()
    args = parse_args()
    input_path = _resolve(args.input)
    output_path = _resolve(args.output)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")
    load_started = perf_counter()
    proposals = pd.read_csv(
        input_path,
        dtype={column: "string" for column in INPUT_COLUMNS},
    )
    timings = {"loading_proposals": perf_counter() - load_started}
    print(f"Processing {len(proposals):,} proposed trips...")
    missing_columns = [column for column in INPUT_COLUMNS if column not in proposals]
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required column(s): "
            + ", ".join(missing_columns)
        )
    predictions = predict_proposals(proposals, debug=args.debug, timings=timings, log=print)
    print(f"Writing {len(predictions):,} predictions to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    timings["total_runtime"] = perf_counter() - total_started
    print(f"Predicted {len(predictions):,} proposed trips")
    print_reliability_summary(predictions)
    print(f"Classifier variants: {CLASSIFIER_VARIANTS}")
    if args.debug:
        print("Debug columns: enabled")
    print("Timing:")
    for key, label in (
        ("loading_proposals", "loading proposals"),
        ("canonical_mapping", "canonical mapping"),
        ("long_term_aggregations", "long-term aggregations"),
        ("recent_aggregations", "recent aggregations"),
        ("feature_joins", "feature joins"),
        ("catboost_prediction", "CatBoost prediction"),
        ("total_runtime", "total runtime"),
    ):
        print(f"  {label}: {timings[key]:.3f}s")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()