"""End-to-end batch evaluation for stop-addition proposals."""

from __future__ import annotations

import argparse
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

from inference.stop_addition.contracts import (
    MODEL_PATH as FROZEN_MODEL,
    MODEL_SUMMARY_PATH as FROZEN_MODEL_SUMMARY,
)

PREDICTION_MODULE = "inference.stop_addition.proposed_route"
BUSINESS_RULE_MODULE = "inference.stop_addition.business_rules"
DEFAULT_OUTPUT = Path("stop_addition_evaluation.csv")

RULE_INPUT_COLUMNS = {
    "training_scenario",
    "has_same_company_exact_proposed_history",
    "has_any_exact_proposed_history",
    "has_similar_route_history",
    "proposed_route_prediction",
    "predicted_uplift",
    "detour_ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen stop-addition inference, current-route inference, "
            "uplift calculation, and frozen business rules."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--database", default=str(REPO_ROOT / "analysis.duckdb"))
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype="string")
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format {path.suffix!r}; use CSV or Parquet.")


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(
            f"Unsupported output format {path.suffix!r}; use CSV or Parquet."
        )


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def validate_frozen_files(input_path: Path) -> None:
    required = (
        FROZEN_MODEL,
        FROZEN_MODEL_SUMMARY,
        input_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inference files: {missing}")


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")

    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    database_path = resolve_path(args.database)
    validate_frozen_files(input_path)
    if input_path.suffix.lower() not in {".csv", ".parquet", ".pq"}:
        raise ValueError("Input must be CSV or Parquet.")
    if output_path.suffix.lower() not in {".csv", ".parquet", ".pq"}:
        raise ValueError("Output must be CSV or Parquet.")

    temp_root = REPO_ROOT / "duckdb_temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex
    prediction_path = temp_root / f"stop_addition_predictions_{run_id}.csv"
    decision_path = temp_root / f"stop_addition_decisions_{run_id}.csv"
    try:
        prediction_command = [
            sys.executable,
            "-m",
            PREDICTION_MODULE,
            "--input",
            str(input_path),
            "--output",
            str(prediction_path),
            "--database",
            str(database_path),
            "--model",
            str(FROZEN_MODEL),
            "--model-summary",
            str(FROZEN_MODEL_SUMMARY),
        ]
        if args.limit is not None:
            prediction_command.extend(["--limit", str(args.limit)])
        run(prediction_command)

        predictions = read_frame(prediction_path)
        for column in RULE_INPUT_COLUMNS - set(predictions.columns):
            predictions[column] = pd.NA
        predictions.to_csv(prediction_path, index=False)

        run(
            [
                sys.executable,
                "-m",
                BUSINESS_RULE_MODULE,
                "--input",
                str(prediction_path),
                "--output",
                str(decision_path),
                "--database",
                str(database_path),
            ]
        )
        output = read_frame(decision_path)
    finally:
        prediction_path.unlink(missing_ok=True)
        decision_path.unlink(missing_ok=True)

    write_frame(output, output_path)
    successful = output["prediction_status"].eq("SUCCESS")
    counts = output["business_decision"].value_counts()
    print(f"Input rows: {len(output)}")
    print(f"Successful rows: {int(successful.sum())}")
    print(f"Failed rows: {int((~successful).sum())}")
    for decision in ("APPROVE", "REVIEW", "REJECT"):
        print(f"{decision}: {int(counts.get(decision, 0))}")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
