"""Adapter from stop-addition evaluation to the shared demand engine."""

from typing import Any

import math
import pandas as pd

from inference.engine import FrozenArtifacts, predict_proposals


def add_current_route_predictions(
    output_rows: list[dict[str, Any]],
    positions: list[int],
    artifacts: FrozenArtifacts,
) -> None:
    """Populate current-route demand while preserving row-level failures."""

    def predict_subset(subset_positions: list[int]) -> None:
        proposals = pd.DataFrame(
            [
                {
                    "FIRMA_ID": output_rows[position]["FIRMA_ID"],
                    "GUZERGAH_KODU": output_rows[position]["CURRENT_GUZERGAH_KODU"],
                    "SEFER_TARIHI": output_rows[position]["REQUESTED_DATE"],
                    "SEFER_SAATI": output_rows[position]["REQUESTED_TIME"],
                }
                for position in subset_positions
            ]
        )
        try:
            predictions = predict_proposals(
                proposals,
                artifacts=artifacts,
                debug=True,
            )
            if len(predictions) != len(subset_positions):
                raise RuntimeError(
                    "Production current-route inference changed the batch row count."
                )
            prediction_rows = predictions.to_dict("records")
            for position, prediction in zip(
                subset_positions, prediction_rows, strict=True
            ):
                value = prediction["v4_2_hybrid_prediction"]
                if not math.isfinite(float(value)):
                    raise RuntimeError(
                        "Production current-route inference returned a non-finite value."
                    )
                output_rows[position].update(
                    {
                        "current_route_prediction": float(value),
                        "current_route_prediction_status": "SUCCESS",
                        "current_route_prediction_error": None,
                        "current_route_reliability": str(
                            prediction["prediction_reliability"]
                        ),
                        "current_route_reliability_reason": str(
                            prediction["reliability_reason"]
                        ),
                        "current_route_baseline_source": str(
                            prediction["weekday_baseline_source"]
                        ),
                        "current_route_history_exact_time_weekday_count": int(
                            prediction["company_route_time_weekday_count"]
                        ),
                        "current_route_history_exact_time_count": int(
                            prediction["company_route_time_count"]
                        ),
                        "current_route_history_company_route_count": int(
                            prediction["company_route_count"]
                        ),
                        "current_route_history_canonical_time_weekday_count": int(
                            prediction["canonical_route_time_weekday_count"]
                        ),
                        "current_route_history_canonical_route_count": int(
                            prediction["canonical_route_count"]
                        ),
                    }
                )
        except Exception as exc:
            if len(subset_positions) > 1:
                midpoint = len(subset_positions) // 2
                predict_subset(subset_positions[:midpoint])
                predict_subset(subset_positions[midpoint:])
                return
            position = subset_positions[0]
            output_rows[position]["current_route_prediction_status"] = "ERROR"
            output_rows[position]["current_route_prediction_error"] = str(exc)

    if positions:
        predict_subset(positions)
