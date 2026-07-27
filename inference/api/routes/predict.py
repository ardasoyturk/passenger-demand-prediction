"""Demand-prediction endpoint."""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from inference.api.depends import Artifacts, Database
from inference.api.schemas import (
    DetailedPrediction,
    FrequentDepartureTime,
    PredictionRequest,
    ReliabilityEvidence,
    SimplifiedPrediction,
    ThresholdProbabilities,
)
from inference.engine import predict_proposals


router = APIRouter(tags=["prediction"])


@router.post(
    "/predict",
    response_model=SimplifiedPrediction | DetailedPrediction,
    summary="Predict demand for a proposed trip",
)
def predict(
    proposal: PredictionRequest,
    artifacts: Artifacts,
    db: Database,
    detail: bool = Query(default=False),
) -> SimplifiedPrediction | DetailedPrediction:
    frame = pd.DataFrame(
        [
            {
                "FIRMA_ID": proposal.firma_id,
                "GUZERGAH_KODU": proposal.guzergah_kodu,
                "SEFER_TARIHI": proposal.sefer_tarihi.isoformat(),
                "SEFER_SAATI": proposal.sefer_saati.isoformat(),
            }
        ]
    )
    try:
        row = predict_proposals(frame, artifacts=artifacts, debug=not detail).iloc[0].to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if detail:
        return DetailedPrediction.model_validate(row)
    frequent_departures: list[FrequentDepartureTime] = []
    exact_time_count = int(row["company_route_time_count"])
    company_route_count = int(row["company_route_count"])
    if exact_time_count < 10 and company_route_count >= 50:
        frequent_departures = _frequent_departure_times(
            db,
            firma_id=proposal.firma_id,
            guzergah_kodu=proposal.guzergah_kodu,
            before_date=proposal.sefer_tarihi.isoformat(),
        )
    return SimplifiedPrediction(
        expected_demand=float(row["v4_2_hybrid_prediction"]),
        baseline_demand=float(row["weekday_baseline_prediction"]),
        demand_label=str(row["mixed_demand_label"]),
        reliability=str(row["prediction_reliability"]),
        reliability_reason=str(row["reliability_reason"]),
        probabilities=ThresholdProbabilities(
            ge_10=float(row["probability_ge_10"]),
            ge_20=float(row["probability_ge_20"]),
            ge_30=float(row["probability_ge_30"]),
            ge_43=float(row["probability_ge_43"]),
        ),
        reliability_evidence=ReliabilityEvidence(
            exact_time_weekday_count=int(row["company_route_time_weekday_count"]),
            exact_time_count=exact_time_count,
            company_route_count=company_route_count,
            canonical_time_weekday_count=int(row["canonical_route_time_weekday_count"]),
            canonical_route_count=int(row["canonical_route_count"]),
            baseline_source=str(row["weekday_baseline_source"]),
            frequent_departure_times=frequent_departures,
        ),
    )


def _frequent_departure_times(
    db: Database,
    *,
    firma_id: int,
    guzergah_kodu: int,
    before_date: str,
) -> list[FrequentDepartureTime]:
    rows = db.execute(
        """
        SELECT
            EXTRACT(HOUR FROM SEFER_SAATI)::INTEGER AS departure_hour,
            EXTRACT(MINUTE FROM SEFER_SAATI)::INTEGER AS departure_minute,
            COUNT(*)::BIGINT AS trip_count,
            COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER () AS route_share
        FROM model_data_base
        WHERE FIRMA_ID = ?
          AND GUZERGAH_KODU = ?
          AND SEFER_TARIHI >= DATE '2023-01-01'
          AND SEFER_TARIHI < CAST(? AS DATE)
        GROUP BY 1, 2
        ORDER BY trip_count DESC, departure_hour, departure_minute
        LIMIT 5
        """,
        [firma_id, guzergah_kodu, before_date],
    ).fetchall()
    return [
        FrequentDepartureTime(
            departure_time=f"{row[0]:02d}:{row[1]:02d}",
            trip_count=int(row[2]),
            route_share=float(row[3]),
        )
        for row in rows
    ]
