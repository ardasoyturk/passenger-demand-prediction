from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from inference.api.routes.stop_addition import (
    GeneralStopAdditionRequest,
    predict_general_stop_addition,
)
from inference.stop_addition.general_baseline import demand_label, select_baseline

COUNT_MEAN_COLUMNS = (
    (
        "proposed_same_company_prior_trip_count",
        "proposed_same_company_prior_mean_demand",
    ),
    (
        "proposed_all_company_prior_trip_count",
        "proposed_all_company_prior_mean_demand",
    ),
    (
        "similar_same_company_prior_trip_count",
        "similar_same_company_weighted_mean_demand",
    ),
    (
        "similar_all_company_prior_trip_count",
        "similar_all_company_weighted_mean_demand",
    ),
    ("base_same_company_prior_trip_count", "base_same_company_prior_mean_demand"),
    ("base_all_company_prior_trip_count", "base_all_company_prior_mean_demand"),
    ("company_prior_trip_count", "company_prior_mean_demand"),
)


def evidence_row(selected_index: int) -> dict[str, float | int]:
    row: dict[str, float | int] = {}
    for index, (count_column, mean_column) in enumerate(COUNT_MEAN_COLUMNS):
        row[count_column] = 5 if index == selected_index else 0
        row[mean_column] = float(10 + index) if index == selected_index else np.nan
    return row


class GeneralBaselineTests(unittest.TestCase):
    def test_fallback_order_is_frozen(self) -> None:
        frame = pd.DataFrame([evidence_row(index) for index in range(7)])
        predictions, sources = select_baseline(frame)
        self.assertEqual(predictions.tolist(), [10, 11, 12, 13, 14, 15, 16])
        self.assertEqual(
            sources.tolist(),
            [
                "SAME_COMPANY_EXACT",
                "ALL_COMPANY_EXACT",
                "SIMILAR_ROUTE",
                "SIMILAR_ROUTE",
                "CURRENT_ROUTE",
                "CURRENT_ROUTE",
                "COMPANY",
            ],
        )

    def test_inference_has_no_global_average_fallback(self) -> None:
        frame = pd.DataFrame([evidence_row(-1)])
        predictions, sources = select_baseline(frame)
        self.assertTrue(np.isnan(predictions[0]))
        self.assertEqual(sources.iloc[0], "NO_PRIOR_EVIDENCE")

    def test_fixed_demand_thresholds(self) -> None:
        self.assertEqual(demand_label(9.999), "CLEAR_FAILURE")
        self.assertEqual(demand_label(10), "WEAK_DEMAND")
        self.assertEqual(demand_label(20), "MODERATE_DEMAND")
        self.assertEqual(demand_label(30), "STRONG_DEMAND")
        self.assertEqual(demand_label(43), "CAPACITY_PRESSURE")

    @patch(
        "inference.api.routes.stop_addition.add_current_route_predictions",
        side_effect=AssertionError("general inference called current-route model"),
    )
    @patch(
        "inference.api.routes.stop_addition.load_stop_addition_contract",
        side_effect=AssertionError("general inference loaded CatBoost"),
    )
    @patch("inference.api.routes.stop_addition.business_rules.decision_warnings")
    @patch("inference.api.routes.stop_addition.business_rules.decide")
    @patch("inference.api.routes.stop_addition.build_general_baseline")
    @patch("inference.api.routes.stop_addition.validate_and_build_requests")
    @patch("inference.api.routes.stop_addition.load_reference_data")
    def test_general_endpoint_does_not_call_models_or_classifiers(
        self,
        load_reference_data,
        validate_requests,
        build_baseline,
        decide,
        decision_warnings,
        _load_contract,
        _current_route_model,
    ) -> None:
        load_reference_data.return_value = (
            {},
            {},
            {},
            {44: {"il_id": 6}},
            {49: 34},
        )
        targets = pd.DataFrame([{"training_row_id": 1}])
        similarities = pd.DataFrame()
        validate_requests.return_value = (
            targets,
            pd.DataFrame(),
            similarities,
            [
                {
                    "_input_row_number": 1,
                    "prediction_error": None,
                    "detour_ratio": 0.05,
                    "added_haversine_km": 8.0,
                    "selected_insertion_index": 2,
                }
            ],
        )
        build_baseline.return_value = pd.DataFrame(
            [
                {
                    **evidence_row(0),
                    "general_baseline_prediction": 25.0,
                    "general_baseline_source": "SAME_COMPANY_EXACT",
                    "general_demand_label": "MODERATE_DEMAND",
                    "general_current_route_prediction": 20.0,
                    "general_current_route_source": "CURRENT_ROUTE_SAME_COMPANY",
                    "similar_routes_with_prior_history_count": 2,
                    "best_similarity_with_prior_history": 0.8,
                    "training_scenario": "SAME_COMPANY_EXACT",
                    "historical_evidence_level": "HIGH",
                }
            ]
        )
        decide.return_value = ("REVIEW", "test", 0.0, 1.0, None)
        decision_warnings.return_value = []

        result = predict_general_stop_addition(
            GeneralStopAdditionRequest(
                firma_id=49,
                current_guzergah_kodu=5866,
                candidate_stop_uetds_yer_id=44,
            ),
            object(),
        )

        self.assertEqual(result["prediction_mode"], "GENERAL_SQL_BASELINE")
        self.assertEqual(result["proposed_route_prediction"], 25.0)
        self.assertEqual(result["demand_label"], "MODERATE_DEMAND")
        self.assertEqual(result["predicted_uplift"], 5.0)


if __name__ == "__main__":
    unittest.main()
