"""Stop-addition decision scoring, warnings, and business overrides."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

# Demand thresholds.
MIN_CURRENT_ROUTE_PREDICTION = 10.0
MIN_PROPOSED_ROUTE_PREDICTION = 10.0
LOW_PROPOSED_ROUTE_PREDICTION = 15.0
HIGH_PROPOSED_ROUTE_PREDICTION = 30.0

# Uplift thresholds.
STRONG_POSITIVE_UPLIFT = 5.0
POSITIVE_UPLIFT = 2.0
MATERIAL_NEGATIVE_UPLIFT = -2.0
STRONG_NEGATIVE_UPLIFT = -5.0

# Detour thresholds (fractions of the current route length).
SMALL_DETOUR_RATIO = 0.03
ACCEPTABLE_DETOUR_RATIO = 0.08
LARGE_DETOUR_RATIO = 0.12
CLEARLY_UNREASONABLE_DETOUR_RATIO = 0.20

# Final score thresholds.
APPROVE_SCORE = 4.0
REJECT_SCORE = -3.0

# Score contributions.
SCORE_STRONG_POSITIVE_UPLIFT = 3.0
SCORE_POSITIVE_UPLIFT = 2.0
SCORE_NON_NEGATIVE_UPLIFT = 0.5
SCORE_NEGATIVE_UPLIFT = -1.5
SCORE_STRONG_NEGATIVE_UPLIFT = -3.0
SCORE_SMALL_DETOUR = 2.0
SCORE_ACCEPTABLE_DETOUR = 0.5
SCORE_LARGE_DETOUR = -2.0
SCORE_VERY_LARGE_DETOUR = -3.0
SCORE_SAME_COMPANY_EXACT_HISTORY = 2.0
SCORE_ANY_EXACT_HISTORY = 1.0
SCORE_SIMILAR_HISTORY = 0.5
SCORE_NO_HISTORY = -1.5
SCORE_SAME_COMPANY_SCENARIO = 1.0
SCORE_OTHER_COMPANY_SCENARIO = 0.5
SCORE_COLD_START_SCENARIO = -1.0
SCORE_HIGH_PROPOSED_DEMAND = 0.5
SCORE_LOW_PROPOSED_DEMAND = -1.0

REQUIRED_COLUMNS = {
    "FIRMA_ID",
    "CANDIDATE_STOP_UETDS_YER_ID",
    "prediction_status",
    "current_route_prediction_status",
    "current_route_prediction",
    "proposed_route_prediction",
    "predicted_uplift",
    "detour_ratio",
    "training_scenario",
    "has_same_company_exact_proposed_history",
    "has_any_exact_proposed_history",
    "has_similar_route_history",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply tunable business rules to stop-addition predictions."
    )
    parser.add_argument(
        "--input",
        required=True,
    )
    parser.add_argument(
        "--output",
        default=(
            "stop_addition_decisions.csv"
        ),
    )
    parser.add_argument("--database", default="analysis.duckdb")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def finite_float(value: Any) -> float | None:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric):
        return None
    result = float(numeric)
    return result if math.isfinite(result) else None


def load_city_pairs(
    frame: pd.DataFrame, database_path: Path
) -> dict[tuple[int, int], tuple[int | None, int | None]]:
    company_ids = sorted(
        {
            int(value)
            for value in pd.to_numeric(frame["FIRMA_ID"], errors="coerce").dropna()
        }
    )
    stop_ids = sorted(
        {
            int(value)
            for value in pd.to_numeric(
                frame["CANDIDATE_STOP_UETDS_YER_ID"], errors="coerce"
            ).dropna()
        }
    )
    if not company_ids or not stop_ids:
        return {}

    con = duckdb.connect(str(database_path), read_only=True)
    try:
        companies = dict(
            con.execute(
                "SELECT firma_id, faaliyet_il_id FROM ats_firma "
                "WHERE firma_id = ANY(?)",
                [company_ids],
            ).fetchall()
        )
        stops = dict(
            con.execute(
                "SELECT id, il_id FROM ats_yer WHERE id = ANY(?)", [stop_ids]
            ).fetchall()
        )
    finally:
        con.close()

    return {
        (company_id, stop_id): (companies.get(company_id), stops.get(stop_id))
        for company_id in company_ids
        for stop_id in stop_ids
    }


def score_prediction(row: pd.Series) -> tuple[float, list[str]]:
    uplift = finite_float(row["predicted_uplift"])
    detour = finite_float(row["detour_ratio"])
    proposed = finite_float(row["proposed_route_prediction"])
    assert uplift is not None and detour is not None and proposed is not None

    score = 0.0
    evidence: list[str] = []

    if uplift >= STRONG_POSITIVE_UPLIFT:
        score += SCORE_STRONG_POSITIVE_UPLIFT
        evidence.append("strong positive uplift")
    elif uplift >= POSITIVE_UPLIFT:
        score += SCORE_POSITIVE_UPLIFT
        evidence.append("positive uplift")
    elif uplift >= 0:
        score += SCORE_NON_NEGATIVE_UPLIFT
        evidence.append("marginal non-negative uplift")
    elif uplift <= STRONG_NEGATIVE_UPLIFT:
        score += SCORE_STRONG_NEGATIVE_UPLIFT
        evidence.append("strong negative uplift")
    else:
        score += SCORE_NEGATIVE_UPLIFT
        evidence.append("negative uplift")

    if detour <= SMALL_DETOUR_RATIO:
        score += SCORE_SMALL_DETOUR
        evidence.append("small detour")
    elif detour <= ACCEPTABLE_DETOUR_RATIO:
        score += SCORE_ACCEPTABLE_DETOUR
        evidence.append("acceptable detour")
    elif detour < LARGE_DETOUR_RATIO:
        score += SCORE_LARGE_DETOUR
        evidence.append("substantial detour")
    else:
        score += SCORE_VERY_LARGE_DETOUR
        evidence.append("large detour")

    same_company = parse_bool(row["has_same_company_exact_proposed_history"])
    any_exact = parse_bool(row["has_any_exact_proposed_history"])
    similar = parse_bool(row["has_similar_route_history"])
    if same_company:
        score += SCORE_SAME_COMPANY_EXACT_HISTORY
        evidence.append("same-company exact history")
    elif any_exact:
        score += SCORE_ANY_EXACT_HISTORY
        evidence.append("other exact history")
    elif similar:
        score += SCORE_SIMILAR_HISTORY
        evidence.append("similar-route history")
    else:
        score += SCORE_NO_HISTORY
        evidence.append("no exact or similar history")

    scenario = str(row["training_scenario"]).strip().upper()
    if scenario == "SAME_COMPANY_EXACT":
        score += SCORE_SAME_COMPANY_SCENARIO
    elif scenario == "OTHER_COMPANY_EXACT":
        score += SCORE_OTHER_COMPANY_SCENARIO
    elif scenario == "COLD_START":
        score += SCORE_COLD_START_SCENARIO
        evidence.append("cold-start scenario")

    if proposed >= HIGH_PROPOSED_ROUTE_PREDICTION:
        score += SCORE_HIGH_PROPOSED_DEMAND
        evidence.append("high proposed-route demand")
    elif proposed < LOW_PROPOSED_ROUTE_PREDICTION:
        score += SCORE_LOW_PROPOSED_DEMAND
        evidence.append("low proposed-route demand")

    return score, evidence


def decision_warnings(row: pd.Series) -> list[str]:
    warnings: list[str] = []
    uplift = finite_float(row["predicted_uplift"])
    proposed = finite_float(row["proposed_route_prediction"])
    detour = finite_float(row["detour_ratio"])
    scenario = str(row["training_scenario"]).strip().upper()
    has_support = any(
        parse_bool(row[column])
        for column in (
            "has_same_company_exact_proposed_history",
            "has_any_exact_proposed_history",
            "has_similar_route_history",
        )
    )

    if uplift is not None and uplift < 0:
        warnings.append("NEGATIVE_UPLIFT")
    if proposed is not None and proposed < LOW_PROPOSED_ROUTE_PREDICTION:
        warnings.append("LOW_PROPOSED_DEMAND")
    if detour is not None and detour >= LARGE_DETOUR_RATIO:
        warnings.append("LARGE_DETOUR")
    if scenario == "COLD_START" or not has_support:
        warnings.append("LOW_EVIDENCE")
    return warnings


def decide(
    row: pd.Series,
    city_pair: tuple[int | None, int | None] | None,
) -> tuple[str, str, float, float | None, str | None]:
    current = finite_float(row["current_route_prediction"])
    proposed = finite_float(row["proposed_route_prediction"])
    uplift = finite_float(row["predicted_uplift"])
    detour = finite_float(row["detour_ratio"])
    evidence_values_valid = all(
        value is not None for value in (current, proposed, uplift, detour)
    )
    model_evidence_score = (
        score_prediction(row)[0] if evidence_values_valid else None
    )

    if city_pair is not None:
        company_origin_il_id, added_stop_il_id = city_pair
        if (
            company_origin_il_id is not None
            and added_stop_il_id is not None
            and company_origin_il_id == added_stop_il_id
        ):
            return (
                "APPROVE",
                "Added stop is in the company's origin city.",
                10.0,
                model_evidence_score,
                "ORIGIN_CITY_HARD_APPROVE",
            )

    proposed_ok = str(row["prediction_status"]).strip().upper() == "SUCCESS"
    current_ok = (
        str(row["current_route_prediction_status"]).strip().upper() == "SUCCESS"
    )
    if not proposed_ok or not current_ok:
        return (
            "REVIEW",
            "Current-route or proposed-route prediction failed.",
            0.0,
            model_evidence_score,
            None,
        )

    if not evidence_values_valid:
        return (
            "REVIEW",
            "Required prediction evidence is missing or invalid.",
            0.0,
            None,
            None,
        )

    assert current is not None and proposed is not None
    assert uplift is not None and detour is not None
    assert model_evidence_score is not None
    if current < MIN_CURRENT_ROUTE_PREDICTION:
        return (
            "REJECT",
            "Current-route prediction is below the minimum demand.",
            -10.0,
            model_evidence_score,
            None,
        )
    if proposed < MIN_PROPOSED_ROUTE_PREDICTION:
        return (
            "REJECT",
            "Proposed-route prediction is below the minimum demand.",
            -10.0,
            model_evidence_score,
            None,
        )
    if detour >= CLEARLY_UNREASONABLE_DETOUR_RATIO:
        return (
            "REJECT",
            "Detour is clearly unreasonable.",
            -10.0,
            model_evidence_score,
            None,
        )
    if detour >= LARGE_DETOUR_RATIO:
        return (
            "REJECT",
            "Large detour outweighs the available demand evidence.",
            -8.0,
            model_evidence_score,
            None,
        )
    if uplift <= STRONG_NEGATIVE_UPLIFT:
        return (
            "REJECT",
            "Strong negative predicted uplift.",
            -8.0,
            model_evidence_score,
            None,
        )

    score, evidence = score_prediction(row)
    scenario = str(row["training_scenario"]).strip().upper()
    has_support = any(
        parse_bool(row[column])
        for column in (
            "has_same_company_exact_proposed_history",
            "has_any_exact_proposed_history",
            "has_similar_route_history",
        )
    )

    if (
        score >= APPROVE_SCORE
        and uplift >= POSITIVE_UPLIFT
        and detour <= ACCEPTABLE_DETOUR_RATIO
        and scenario != "COLD_START"
        and has_support
    ):
        decision = "APPROVE"
    elif score <= REJECT_SCORE or uplift <= MATERIAL_NEGATIVE_UPLIFT:
        decision = "REJECT"
    else:
        decision = "REVIEW"

    return (
        decision,
        "; ".join(evidence) + f" (score {score:.1f}).",
        score,
        model_evidence_score,
        None,
    )


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    database_path = resolve_path(args.database)

    frame = pd.read_csv(input_path, dtype="string")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    city_pairs = load_city_pairs(frame, database_path)
    decisions: list[tuple[str, str, float, float | None, str | None]] = []
    city_values: list[tuple[int | None, int | None]] = []
    warnings: list[str | None] = []
    for _, row in frame.iterrows():
        company_id = finite_float(row["FIRMA_ID"])
        stop_id = finite_float(row["CANDIDATE_STOP_UETDS_YER_ID"])
        key = (
            None
            if company_id is None or stop_id is None
            else (int(company_id), int(stop_id))
        )
        city_pair = None if key is None else city_pairs.get(key)
        city_pair = city_pair or (None, None)
        decisions.append(decide(row, city_pair))
        city_values.append(city_pair)
        row_warnings = decision_warnings(row)
        warnings.append("|".join(row_warnings) if row_warnings else None)

    frame["business_decision"] = [item[0] for item in decisions]
    frame["decision_reason"] = [item[1] for item in decisions]
    frame["decision_score"] = [item[2] for item in decisions]
    frame["decision_override"] = [item[4] for item in decisions]
    frame["model_evidence_score"] = [item[3] for item in decisions]
    frame["decision_warnings"] = warnings
    frame["is_company_origin_city"] = [
        origin_id is not None
        and added_id is not None
        and origin_id == added_id
        for origin_id, added_id in city_values
    ]
    frame["added_stop_il_id"] = [item[1] for item in city_values]
    frame["company_origin_il_id"] = [item[0] for item in city_values]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)

    counts = frame["business_decision"].value_counts()
    print(f"Output: {output_path}")
    for decision in ("APPROVE", "REVIEW", "REJECT"):
        print(f"{decision}: {int(counts.get(decision, 0))}")


if __name__ == "__main__":
    main()
