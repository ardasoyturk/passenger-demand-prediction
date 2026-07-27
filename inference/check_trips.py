"""Audit selected proposed trips against strict-earlier historical demand.

This is an inference-only entry point. It reuses the final mixed v4.2/v4.4
pipeline and performs all large historical aggregations and joins in DuckDB.
No model training, cutoff selection, or artifact mutation is performed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import numpy as np
import pandas as pd

from inference.engine import (
    DB_PATH,
    INPUT_COLUMNS,
    PROJECT_DIR,
    print_reliability_summary,
    predict_proposals,
    validate_proposals,
)


STATISTICS = ("count", "average", "median", "p90", "minimum", "maximum")
HISTORY_LEVELS = ("route", "exact_time", "weekday_time")
FLAG_COLUMNS = [
    "flag_weekday_time_count_below_20",
    "flag_no_exact_time_history",
    "flag_no_weekday_time_history",
    "flag_prediction_above_weekday_time_p90_by_more_than_10",
    "flag_prediction_below_weekday_time_median_by_more_than_10",
    "flag_weekday_time_average_difference_greater_than_10",
    "flag_expected_below_10_but_label_moderate_or_higher",
    "flag_expected_above_43_but_label_below_strong",
    "flag_monotonicity_correction_applied",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run final mixed inference and strict-earlier historical checks "
            "for selected proposed trips."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Proposed-trip CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/proposed_trips_historical_check.csv"),
        help="Output CSV (default: results/proposed_trips_historical_check.csv)",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    """Resolve CLI paths relative to the training directory."""

    return path if path.is_absolute() else PROJECT_DIR / path


def read_proposals(path: Path) -> pd.DataFrame:
    """Read and explicitly validate the required four-column CSV schema."""

    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    proposals = pd.read_csv(
        path,
        dtype={column: "string" for column in INPUT_COLUMNS},
    )
    missing = [column for column in INPUT_COLUMNS if column not in proposals]
    if missing:
        raise ValueError("Input CSV is missing required column(s): " + ", ".join(missing))
    return proposals[INPUT_COLUMNS].copy()


def _aggregate_sql(prefix: str, extra_conditions: str) -> str:
    """Return one proposal-keyed DuckDB aggregation for a history level."""

    return f"""
        CREATE OR REPLACE TEMP TABLE {prefix}_history_stats AS
        SELECT
            proposals.proposal_id,
            COUNT(history.target)::BIGINT AS {prefix}_history_count,
            AVG(history.target)::DOUBLE AS {prefix}_history_average,
            MEDIAN(history.target)::DOUBLE AS {prefix}_history_median,
            QUANTILE_CONT(history.target, 0.90)::DOUBLE AS {prefix}_history_p90,
            MIN(history.target)::DOUBLE AS {prefix}_history_minimum,
            MAX(history.target)::DOUBLE AS {prefix}_history_maximum
        FROM selected_proposals AS proposals
        LEFT JOIN model_data_base AS history
          ON history.FIRMA_ID = proposals.FIRMA_ID
         AND history.GUZERGAH_KODU = proposals.GUZERGAH_KODU
         AND history.SEFER_TARIHI < proposals.SEFER_TARIHI
         {extra_conditions}
        GROUP BY proposals.proposal_id
    """


def historical_statistics(validated: pd.DataFrame) -> pd.DataFrame:
    """Calculate all three historical levels with vectorized DuckDB joins.

    Every aggregate uses only database rows dated strictly before its proposal.
    Proposal rows are never added to ``model_data_base`` or to one another's
    history.
    """

    proposal_table = pd.DataFrame({
        "proposal_id": validated["_proposal_order"].to_numpy(dtype=np.int64),
        "FIRMA_ID": validated["FIRMA_ID"].to_numpy(dtype=np.int64),
        "GUZERGAH_KODU": validated["GUZERGAH_KODU"].to_numpy(dtype=np.int64),
        "SEFER_TARIHI": validated["SEFER_TARIHI"].dt.date,
        "SEFER_SAATI": validated["SEFER_SAATI"].to_numpy(),
        "proposal_weekday": (validated["SEFER_TARIHI"].dt.dayofweek + 1).to_numpy(),
    })

    connection = duckdb.connect(str(DB_PATH), read_only=False)
    try:
        connection.register("selected_proposals_input", proposal_table)
        connection.execute("""
            CREATE OR REPLACE TEMP TABLE selected_proposals AS
            SELECT
                proposal_id,
                FIRMA_ID,
                GUZERGAH_KODU,
                CAST(SEFER_TARIHI AS DATE) AS SEFER_TARIHI,
                CAST(SEFER_SAATI AS TIME) AS SEFER_SAATI,
                proposal_weekday
            FROM selected_proposals_input
        """)
        connection.execute(_aggregate_sql("route", ""))
        connection.execute(_aggregate_sql(
            "exact_time",
            "AND history.SEFER_SAATI = proposals.SEFER_SAATI",
        ))
        connection.execute(_aggregate_sql(
            "weekday_time",
            """AND history.SEFER_SAATI = proposals.SEFER_SAATI
               AND history.day_of_week = proposals.proposal_weekday""",
        ))
        statistic_columns = [
            f"{level}_history_{statistic}"
            for level in HISTORY_LEVELS
            for statistic in STATISTICS
        ]
        result = connection.execute(f"""
            SELECT
                proposals.proposal_id,
                {', '.join(f'{level}.{level}_history_{statistic}' for level in HISTORY_LEVELS for statistic in STATISTICS)}
            FROM selected_proposals AS proposals
            JOIN route_history_stats AS route USING (proposal_id)
            JOIN exact_time_history_stats AS exact_time USING (proposal_id)
            JOIN weekday_time_history_stats AS weekday_time USING (proposal_id)
            ORDER BY proposals.proposal_id
        """).fetchdf()
    finally:
        connection.close()

    if len(result) != len(validated) or result["proposal_id"].duplicated().any():
        raise RuntimeError("Historical aggregation did not preserve one row per proposal")
    for level in HISTORY_LEVELS:
        result[f"{level}_history_count"] = result[f"{level}_history_count"].fillna(0).astype("int64")
    return result[["proposal_id", *statistic_columns]]


def add_review_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the specified automatic review flags without changing predictions."""

    result = frame.copy()
    prediction = result["v4_2_hybrid_prediction"].to_numpy(dtype=np.float64)
    weekday_count = result["weekday_time_history_count"].to_numpy(dtype=np.int64)
    exact_count = result["exact_time_history_count"].to_numpy(dtype=np.int64)
    weekday_p90 = result["weekday_time_history_p90"].to_numpy(dtype=np.float64)
    weekday_median = result["weekday_time_history_median"].to_numpy(dtype=np.float64)
    weekday_average = result["weekday_time_history_average"].to_numpy(dtype=np.float64)
    labels = result["mixed_demand_label"].astype(str).to_numpy()

    result["flag_weekday_time_count_below_20"] = weekday_count < 20
    result["flag_no_exact_time_history"] = exact_count == 0
    result["flag_no_weekday_time_history"] = weekday_count == 0
    result["flag_prediction_above_weekday_time_p90_by_more_than_10"] = (
        np.isfinite(weekday_p90) & (prediction > weekday_p90 + 10)
    )
    result["flag_prediction_below_weekday_time_median_by_more_than_10"] = (
        np.isfinite(weekday_median) & (prediction < weekday_median - 10)
    )
    result["absolute_difference_from_weekday_time_average"] = np.where(
        np.isfinite(weekday_average), np.abs(prediction - weekday_average), np.nan
    )
    result["flag_weekday_time_average_difference_greater_than_10"] = (
        result["absolute_difference_from_weekday_time_average"] > 10
    )
    moderate_or_higher = np.isin(
        labels, ["MODERATE_DEMAND", "STRONG_DEMAND", "CAPACITY_PRESSURE"]
    )
    below_strong = np.isin(labels, ["CLEAR_FAILURE", "WEAK_DEMAND", "MODERATE_DEMAND"])
    result["flag_expected_below_10_but_label_moderate_or_higher"] = (
        (prediction < 10) & moderate_or_higher
    )
    result["flag_expected_above_43_but_label_below_strong"] = (
        (prediction > 43) & below_strong
    )
    result["flag_monotonicity_correction_applied"] = (
        result["classifier_monotonicity_correction_applied"].astype(bool)
    )
    result["any_review_flag"] = result[FLAG_COLUMNS].any(axis=1)
    return result


def build_historical_check(proposals: pd.DataFrame) -> pd.DataFrame:
    """Run shared mixed inference and join proposal-level history statistics."""

    validated = validate_proposals(proposals)
    predictions = predict_proposals(proposals, debug=True)
    statistics = historical_statistics(validated)
    base = validated.sort_values("_proposal_order")[INPUT_COLUMNS].reset_index(drop=True)
    base["SEFER_TARIHI"] = base["SEFER_TARIHI"].dt.strftime("%Y-%m-%d")
    base["SEFER_SAATI"] = base["SEFER_SAATI"].str.slice(0, 5)
    selected_outputs = predictions[[
        "v4_2_hybrid_prediction",
        "probability_ge_20",
        "probability_ge_30",
        "probability_ge_43",
        "mixed_demand_label",
        "prediction_reliability",
        "reliability_reason",
        "classifier_monotonicity_correction_applied",
        "weekday_baseline_prediction",
        "weekday_baseline_source",
    ]].reset_index(drop=True)
    statistics = statistics.sort_values("proposal_id").drop(columns="proposal_id").reset_index(drop=True)
    combined = pd.concat([base, statistics, selected_outputs], axis=1)
    return add_review_flags(combined)


def print_summary(frame: pd.DataFrame) -> None:
    """Print the required compact audit summary."""

    missing_history = frame["flag_no_exact_time_history"] | frame["flag_no_weekday_time_history"]
    average_difference = frame["absolute_difference_from_weekday_time_average"].mean()
    rendered_difference = "not available" if pd.isna(average_difference) else f"{average_difference:.6f}"
    print(f"Total trips: {len(frame):,}")
    print(f"Flagged trips: {int(frame['any_review_flag'].sum()):,}")
    print(f"Low-history trips: {int(frame['flag_weekday_time_count_below_20'].sum()):,}")
    print(f"Missing-history trips: {int(missing_history.sum()):,}")
    print(f"Average absolute difference from weekday-time average: {rendered_difference}")
    print_reliability_summary(frame)


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    proposals = read_proposals(input_path)
    result = build_historical_check(proposals)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print_summary(result)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()