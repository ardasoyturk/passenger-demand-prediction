"""Curated MCP tools for passenger-demand predictions."""

from __future__ import annotations

from datetime import date, time

import duckdb
from fastapi import HTTPException
from fastmcp import FastMCP

from inference.api.routes.predict import predict
from inference.api.routes.predict_general import predict_general_route
from inference.api.routes.stop_addition import (
    GeneralStopAdditionRequest,
    StopAdditionRequest,
    available_routes,
    predict_general_stop_addition,
    predict_stop_addition,
)
from inference.api.schemas import GeneralPredictionRequest, PredictionRequest
from inference.engine import DB_PATH, FrozenArtifacts

mcp = FastMCP(
    "Passenger Demand Predictions",
    instructions=(
        "Use these tools to obtain passenger-demand predictions. Never invent "
        "prediction values; call the tool that matches the user's request."
    ),
)

_artifacts: FrozenArtifacts | None = None


def set_artifacts(artifacts: FrozenArtifacts | None) -> None:
    """Share the model artifacts owned by the FastAPI application lifespan."""
    global _artifacts
    _artifacts = artifacts


def _loaded_artifacts() -> FrozenArtifacts:
    if _artifacts is None:
        raise RuntimeError("Prediction artifacts are not loaded")
    return _artifacts


def _tool_error(exc: HTTPException) -> ValueError:
    return ValueError(str(exc.detail))


@mcp.tool
def predict_trip_demand(
    firma_id: int,
    guzergah_kodu: int,
    sefer_tarihi: date,
    sefer_saati: time,
) -> dict:
    """Predict demand for a specific future bus trip.

    Use this when the user provides a company, company-specific route, date,
    and departure time. The result includes expected and baseline demand,
    demand label, threshold probabilities, and reliability evidence.
    """
    request = PredictionRequest(
        firma_id=firma_id,
        guzergah_kodu=guzergah_kodu,
        sefer_tarihi=sefer_tarihi,
        sefer_saati=sefer_saati,
    )
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        try:
            result = predict(request, _loaded_artifacts(), db, detail=False)
        except HTTPException as exc:
            raise _tool_error(exc) from exc
    return result.model_dump(mode="json")


@mcp.tool
def predict_general_route_demand(firma_id: int, guzergah_kodu: int) -> dict:
    """Estimate a route's general demand without a date or departure time.

    Use this only when the user asks about general/historical route demand and
    has not specified a particular trip. This is a historical baseline, not a
    live date-specific model prediction.
    """
    request = GeneralPredictionRequest(firma_id=firma_id, guzergah_kodu=guzergah_kodu)
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        try:
            result = predict_general_route(request, db)
        except HTTPException as exc:
            raise _tool_error(exc) from exc
    return result.model_dump(mode="json")


@mcp.tool
def find_stop_addition_options(
    firma_id: int | None = None,
    search: str | None = None,
) -> list[dict]:
    """Find valid routes and candidate stops for a stop-addition analysis.

    Use this before a stop-addition prediction when the user gives a stop name
    instead of its ID, or has not provided a candidate stop. Returns company,
    current route, and candidate stop identifiers and names.
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        try:
            options = available_routes(
                db,
                firma_id=firma_id,
                search=search,
                has_trip_history_both=False,
                limit=50,
                offset=0,
            )
        except HTTPException as exc:
            raise _tool_error(exc) from exc
    return [option.model_dump(mode="json") for option in options]


@mcp.tool
def predict_stop_addition_demand(
    firma_id: int,
    current_guzergah_kodu: int,
    candidate_stop_uetds_yer_id: int,
    requested_date: date,
    requested_time: time,
) -> dict:
    """Predict the impact of adding one stop to a specific scheduled trip.

    Returns the existing and proposed route demand, expected uplift, added
    Haversine distance, historical evidence, and the business recommendation.
    Use for a date-specific stop-addition question.
    """
    request = StopAdditionRequest(
        firma_id=firma_id,
        current_guzergah_kodu=current_guzergah_kodu,
        candidate_stop_uetds_yer_id=candidate_stop_uetds_yer_id,
        requested_date=requested_date,
        requested_time=requested_time,
    )
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        try:
            result = predict_stop_addition(request, _loaded_artifacts(), db)
        except HTTPException as exc:
            raise _tool_error(exc) from exc
    return result


@mcp.tool
def predict_general_stop_addition_demand(
    firma_id: int,
    current_guzergah_kodu: int,
    candidate_stop_uetds_yer_id: int,
) -> dict:
    """Estimate the general impact of adding one stop without a schedule.

    This uses the historical SQL baseline and returns proposed/current demand,
    estimated uplift, added distance, evidence, and business recommendation.
    """
    request = GeneralStopAdditionRequest(
        firma_id=firma_id,
        current_guzergah_kodu=current_guzergah_kodu,
        candidate_stop_uetds_yer_id=candidate_stop_uetds_yer_id,
    )
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        try:
            result = predict_general_stop_addition(request, db)
        except HTTPException as exc:
            raise _tool_error(exc) from exc
    return result


# The AI SDK's HTTP transport establishes a GET-based inbound SSE stream, so
# this must remain session-based (the default), not stateless.
mcp_app = mcp.http_app(path="/", stateless_http=False, json_response=True)
