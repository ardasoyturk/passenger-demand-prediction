"""CLI for one mixed v4.2/v4.4 proposed-trip prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from inference.engine import predict_proposals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict passenger demand for one proposed trip.")
    parser.add_argument("--firma-id", required=True, type=int, help="Company identifier")
    parser.add_argument("--guzergah-kodu", required=True, type=int, help="Company-specific route code")
    parser.add_argument("--date", required=True, help="Proposal date in YYYY-MM-DD format")
    parser.add_argument("--time", required=True, help="Departure time in HH:MM format")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print normalized lookup keys, history matches, and baseline source",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proposed = pd.DataFrame([{
        "FIRMA_ID": args.firma_id,
        "GUZERGAH_KODU": args.guzergah_kodu,
        "SEFER_TARIHI": args.date,
        "SEFER_SAATI": args.time,
    }])
    row = predict_proposals(proposed, debug=args.debug).iloc[0]

    print("\nProposed trip")
    print(f"Company: {row.FIRMA_ID}")
    print(f"Route: {row.GUZERGAH_KODU}")
    print(f"Date: {row.SEFER_TARIHI:%Y-%m-%d}")
    print(f"Departure time: {row.SEFER_SAATI}")
    print(f"\nExpected passengers: {row.v4_2_hybrid_prediction:.2f}")
    print(f"Weekday baseline: {row.weekday_baseline_prediction:.2f}")
    print(f"v4.2 hybrid weight: {row.v4_2_hybrid_weight:.2f}")
    if args.debug:
        print("\nBaseline debug")
        for column in (
            "normalized_FIRMA_ID",
            "normalized_GUZERGAH_KODU",
            "normalized_SEFER_SAATI",
            "canonical_mapping_source",
            "departure_30min_bucket",
            "weekday",
            "company_route_time_weekday_count",
            "company_route_time_weekday_mean",
            "company_route_time_count",
            "company_route_time_mean",
            "company_route_count",
            "company_route_mean",
            "canonical_fallback_count",
            "canonical_fallback_mean",
            "previous_weekday_baseline_prediction",
            "previous_weekday_baseline_source",
            "weekday_baseline_source",
        ):
            print(f"{column}: {row[column]}")
    for threshold in (10, 20, 30, 43):
        print(
            f"P(demand >= {threshold}): {row[f'probability_ge_{threshold}']:.6f} "
            f"(frozen cutoff {row[f'cutoff_ge_{threshold}']:.2f}; "
            f"class weights {row[f'classifier_variant_ge_{threshold}']})"
        )
    print("\nMixed decisions:")
    for threshold in (10, 20, 30, 43):
        print(f">={threshold}: {int(row[f'mixed_decision_ge_{threshold}'])}")
    print(f"\nFinal demand label: {row.mixed_demand_label}")
    print(f"Prediction reliability: {row.prediction_reliability}")
    print(f"Reliability reason: {row.reliability_reason}")
    if row.classifier_monotonicity_violation:
        print("Raw classifier monotonicity violation: yes (mixed decisions corrected)")


if __name__ == "__main__":
    main()