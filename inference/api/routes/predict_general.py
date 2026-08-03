"""General (time-independent) demand-prediction endpoint.

The prediction comes from the historical-average baseline only: the v4.2
hybrid regressor and the v4.4 classifiers are not run for this mode, and the
demand label is assigned algorithmically from the baseline value.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from inference.api.depends import Database
from inference.api.schemas import GeneralPrediction, GeneralPredictionRequest
from inference.general import predict_general


router = APIRouter(tags=["prediction"])


@router.post(
    "/predict-general",
    response_model=GeneralPrediction,
    summary="Predict general route demand from company and route only",
)
def predict_general_route(
    proposal: GeneralPredictionRequest,
    db: Database,
) -> GeneralPrediction:
    try:
        row = predict_general(db, proposal.firma_id, proposal.guzergah_kodu)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return GeneralPrediction.model_validate(row)
