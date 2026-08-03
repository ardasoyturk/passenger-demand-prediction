"""Curated MCP tools for passenger-demand predictions."""

from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path

import duckdb
from fastapi import HTTPException
from fastmcp import FastMCP

from inference.api.routes.durak import get_durak
from inference.api.routes.predict import predict
from inference.api.routes.predict_general import predict_general_route
from inference.api.routes.route import get_route, list_company_routes
from inference.api.routes.stop_addition import (
    GeneralStopAdditionRequest,
    StopAdditionRequest,
    available_routes,
    predict_general_stop_addition,
    predict_stop_addition,
)
from inference.api.schemas import GeneralPredictionRequest, PredictionRequest
from inference.engine import DB_PATH, FrozenArtifacts

_CITY_DATA_PATH = Path(__file__).with_name("data") / "cities_of_turkey.json"
with _CITY_DATA_PATH.open(encoding="utf-8") as city_data_file:
    _TURKEY_CITIES = json.load(city_data_file)
CITY_NAMES_BY_ID = {
    int(city["id"]): str(city["name"])
    for city in _TURKEY_CITIES
    if isinstance(city, dict) and "id" in city and "name" in city
}
if set(CITY_NAMES_BY_ID) != set(range(1, 82)):
    raise RuntimeError("Turkey city data must contain exactly the 81 province IDs")

mcp = FastMCP(
    "Passenger Demand and Route Intelligence",
    instructions=(
        "Use these tools to obtain passenger-demand predictions and factual "
        "route or stop data. Never invent values; call the matching tool."
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


def _city_name(il_id: int | None) -> str | None:
    return CITY_NAMES_BY_ID.get(il_id) if il_id is not None else None


@mcp.tool
def list_routes_for_company(firma_id: int) -> dict:
    """List every known route code for a company with origin/destination names.

    Use this when a user knows the company but not the route code, or asks what
    routes a company operates.
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        result = list_company_routes(firma_id, db)
    return result.model_dump(mode="json")


@mcp.tool
def get_company_route_details(firma_id: int, guzergah_kodu: int) -> dict:
    """Get an ordered route-stop list suitable for describing or mapping a route.

    Returns company and canonical route identifiers plus every stop's order,
    name, administrative IDs, latitude, and longitude. Use these coordinates
    when the user asks to visualize, map, or explain a company route.
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        try:
            result = get_route(firma_id, guzergah_kodu, db)
        except HTTPException as exc:
            raise _tool_error(exc) from exc
    return result.model_dump(mode="json")


@mcp.tool
def get_canonical_route_details(canonical_guzergah_id: int) -> dict:
    """Get route stops and company-specific aliases for a canonical route ID.

    Use this when the user supplies a canonical physical-route ID instead of a
    company and route code. The ordered stops include map coordinates.
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        aliases = db.execute(
            """
            SELECT
                route.firma_id::INTEGER,
                route.guzergah_kodu::INTEGER,
                company.unvan
            FROM guzergah_canonical AS route
            LEFT JOIN ats_firma AS company ON company.firma_id = route.firma_id
            WHERE route.canonical_guzergah_id = ?
            ORDER BY route.firma_id, route.guzergah_kodu
            """,
            [canonical_guzergah_id],
        ).fetchall()
        if not aliases:
            raise ValueError("Canonical route not found")
        try:
            detail = get_route(int(aliases[0][0]), int(aliases[0][1]), db)
        except HTTPException as exc:
            raise _tool_error(exc) from exc
    return {
        "canonical_guzergah_id": canonical_guzergah_id,
        "company_routes": [
            {
                "firma_id": int(row[0]),
                "guzergah_kodu": int(row[1]),
                "firma_unvan": row[2],
            }
            for row in aliases
        ],
        "duraklar": [stop.model_dump(mode="json") for stop in detail.duraklar],
    }


@mcp.tool
def get_stop_details(durak_id: int) -> dict:
    """Get authoritative stop information by internal stop ID.

    Returns UETDS code/name, type, province/district IDs, country, and map
    coordinates. Use this for questions about one specific stop.
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        try:
            result = get_durak(durak_id, db)
        except HTTPException as exc:
            raise _tool_error(exc) from exc
    return result.model_dump(mode="json")


@mcp.tool
def search_stops(query: str, limit: int = 20) -> list[dict]:
    """Search stops by name, short name, UETDS code, or numeric stop ID.

    Use this to resolve a user's natural-language stop reference before calling
    a route or stop-addition tool. Returns at most 50 matching stops.
    """
    normalized = query.strip()
    if not normalized:
        raise ValueError("Stop search query cannot be empty")
    bounded_limit = max(1, min(limit, 50))
    pattern = f"%{normalized.casefold()}%"
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        rows = db.execute(
            """
            SELECT
                id, uetds_kodu, turu, uetds_adi, il_id, ilce_id, kisa_adi,
                ulke_id, ulke_adi, enlem, boylam
            FROM ats_yer
            WHERE CAST(id AS VARCHAR) = ?
               OR contains(lower(coalesce(uetds_kodu, '')), ?)
               OR contains(lower(coalesce(uetds_adi, '')), ?)
               OR contains(lower(coalesce(kisa_adi, '')), ?)
            ORDER BY
                CASE WHEN CAST(id AS VARCHAR) = ? THEN 0 ELSE 1 END,
                uetds_adi NULLS LAST,
                id
            LIMIT ?
            """,
            [normalized, pattern[1:-1], pattern[1:-1], pattern[1:-1], normalized, bounded_limit],
        ).fetchall()
    columns = (
        "id",
        "uetds_kodu",
        "turu",
        "uetds_adi",
        "il_id",
        "ilce_id",
        "kisa_adi",
        "ulke_id",
        "ulke_adi",
        "enlem",
        "boylam",
    )
    return [dict(zip(columns, row, strict=True)) for row in rows]


@mcp.tool
def get_company_details(firma_id: int) -> dict:
    """Get company identity, operating location, route count, and trip history.

    Use this for factual questions about a company ID. Demand statistics are
    descriptive historical values, not future predictions.
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        row = db.execute(
            """
            SELECT
                company.firma_id::INTEGER,
                company.unvan,
                company.faaliyet_il_id,
                (SELECT COUNT(DISTINCT route.guzergah_kodu)
                 FROM guzergah AS route
                 WHERE route.firma_id = company.firma_id)::INTEGER AS route_count,
                (SELECT COUNT(*)
                 FROM model_data_base AS trip
                 WHERE trip.FIRMA_ID = company.firma_id)::BIGINT AS trip_count,
                (SELECT MIN(trip.SEFER_TARIHI)
                 FROM model_data_base AS trip
                 WHERE trip.FIRMA_ID = company.firma_id) AS first_trip_date,
                (SELECT MAX(trip.SEFER_TARIHI)
                 FROM model_data_base AS trip
                 WHERE trip.FIRMA_ID = company.firma_id) AS last_trip_date,
                (SELECT AVG(trip.target)
                 FROM model_data_base AS trip
                 WHERE trip.FIRMA_ID = company.firma_id) AS historical_mean_demand
            FROM ats_firma AS company
            WHERE company.firma_id = ?
            """,
            [firma_id],
        ).fetchone()
    if row is None:
        raise ValueError("Company not found")
    columns = (
        "firma_id",
        "unvan",
        "faaliyet_il_id",
        "route_count",
        "trip_count",
        "first_trip_date",
        "last_trip_date",
        "historical_mean_demand",
    )
    result = dict(zip(columns, row, strict=True))
    result["faaliyet_il_adi"] = _city_name(result["faaliyet_il_id"])
    return result


@mcp.tool
def search_companies(query: str, limit: int = 20) -> list[dict]:
    """Search companies by title or exact numeric company ID."""
    normalized = query.strip()
    if not normalized:
        raise ValueError("Company search query cannot be empty")
    bounded_limit = max(1, min(limit, 50))
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        rows = db.execute(
            """
            SELECT firma_id::INTEGER, unvan, faaliyet_il_id
            FROM ats_firma
            WHERE CAST(firma_id AS VARCHAR) = ?
               OR contains(lower(coalesce(unvan, '')), lower(?))
            ORDER BY
                CASE WHEN CAST(firma_id AS VARCHAR) = ? THEN 0 ELSE 1 END,
                unvan NULLS LAST,
                firma_id
            LIMIT ?
            """,
            [normalized, normalized, normalized, bounded_limit],
        ).fetchall()
    columns = ("firma_id", "unvan", "faaliyet_il_id")
    results = [dict(zip(columns, row, strict=True)) for row in rows]
    for result in results:
        result["faaliyet_il_adi"] = _city_name(result["faaliyet_il_id"])
    return results


@mcp.tool
def get_trip_details(sefer_id: int) -> dict:
    """Get one historical trip by trip ID, including observed passenger demand.

    This returns a past observation, not a prediction. The company-specific and
    canonical route identifiers can be used with the route tools.
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        row = db.execute(
            """
            SELECT
                trip.SEFER_ID,
                trip.SEFER_TARIHI,
                trip.SEFER_SAATI,
                trip.FIRMA_ID::INTEGER,
                company.unvan,
                trip.GUZERGAH_KODU::INTEGER,
                trip.canonical_guzergah_id,
                trip.target AS observed_demand
            FROM model_data_base AS trip
            LEFT JOIN ats_firma AS company ON company.firma_id = trip.FIRMA_ID
            WHERE trip.SEFER_ID = ?
            """,
            [sefer_id],
        ).fetchone()
    if row is None:
        raise ValueError("Trip not found")
    columns = (
        "sefer_id",
        "sefer_tarihi",
        "sefer_saati",
        "firma_id",
        "firma_unvan",
        "guzergah_kodu",
        "canonical_guzergah_id",
        "observed_demand",
    )
    return dict(zip(columns, row, strict=True))


@mcp.tool
def get_company_route_history_summary(firma_id: int, guzergah_kodu: int) -> dict:
    """Summarize observed historical demand for one company route.

    Returns trip count, coverage dates, mean/median/min/max observed demand, and
    the number of distinct departure times. This is descriptive, not a forecast.
    """
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        row = db.execute(
            """
            SELECT
                FIRMA_ID::INTEGER,
                GUZERGAH_KODU::INTEGER,
                any_value(canonical_guzergah_id) AS canonical_guzergah_id,
                COUNT(*)::BIGINT AS trip_count,
                MIN(SEFER_TARIHI) AS first_trip_date,
                MAX(SEFER_TARIHI) AS last_trip_date,
                AVG(target) AS mean_observed_demand,
                MEDIAN(target) AS median_observed_demand,
                MIN(target)::BIGINT AS min_observed_demand,
                MAX(target)::BIGINT AS max_observed_demand,
                COUNT(DISTINCT SEFER_SAATI)::INTEGER AS distinct_departure_times
            FROM model_data_base
            WHERE FIRMA_ID = ? AND GUZERGAH_KODU = ?
            GROUP BY FIRMA_ID, GUZERGAH_KODU
            """,
            [firma_id, guzergah_kodu],
        ).fetchone()
    if row is None:
        raise ValueError("No historical trips found for company route")
    columns = (
        "firma_id",
        "guzergah_kodu",
        "canonical_guzergah_id",
        "trip_count",
        "first_trip_date",
        "last_trip_date",
        "mean_observed_demand",
        "median_observed_demand",
        "min_observed_demand",
        "max_observed_demand",
        "distinct_departure_times",
    )
    return dict(zip(columns, row, strict=True))


@mcp.tool
def list_routes_serving_stop(durak_id: int, limit: int = 50) -> list[dict]:
    """List company routes whose ordered canonical stop list includes a stop."""
    bounded_limit = max(1, min(limit, 200))
    with duckdb.connect(str(DB_PATH), read_only=True) as db:
        rows = db.execute(
            """
            SELECT
                route.firma_id::INTEGER,
                company.unvan,
                route.guzergah_kodu::INTEGER,
                route.canonical_guzergah_id,
                list_position(route.duraklar, ?)::INTEGER AS stop_order,
                len(route.duraklar)::INTEGER AS route_stop_count
            FROM guzergah_canonical AS route
            LEFT JOIN ats_firma AS company ON company.firma_id = route.firma_id
            WHERE list_contains(route.duraklar, ?)
            ORDER BY route.firma_id, route.guzergah_kodu
            LIMIT ?
            """,
            [durak_id, durak_id, bounded_limit],
        ).fetchall()
    columns = (
        "firma_id",
        "firma_unvan",
        "guzergah_kodu",
        "canonical_guzergah_id",
        "stop_order",
        "route_stop_count",
    )
    return [dict(zip(columns, row, strict=True)) for row in rows]


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
