"""Company route and ordered route-stop endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from inference.api.depends import Database
from inference.api.schemas import (
    CompanyRoutesResponse,
    RouteDetailResponse,
    RouteDurak,
    RouteSummary,
)


router = APIRouter(prefix="/route", tags=["route"])


@router.get("/{firma_id}", response_model=CompanyRoutesResponse)
def list_company_routes(firma_id: int, db: Database) -> CompanyRoutesResponse:
    rows = db.execute(
        """
        SELECT
            route.guzergah_kodu,
            departure.uetds_adi AS kalkis_durak_adi,
            arrival.uetds_adi AS varis_durak_adi
        FROM guzergah AS route
        LEFT JOIN ats_yer AS departure ON departure.id = route.kalkis_uetds_yer_id
        LEFT JOIN ats_yer AS arrival ON arrival.id = route.varis_uetds_yer_id
        WHERE route.firma_id = ?
        ORDER BY route.guzergah_kodu
        """,
        [firma_id],
    ).fetchall()
    return CompanyRoutesResponse(
        firma_id=firma_id,
        routes=[
            RouteSummary(
                guzergah_kodu=row[0],
                kalkis_durak_adi=row[1],
                varis_durak_adi=row[2],
            )
            for row in rows
        ],
    )


@router.get("/{firma_id}/{guzergah_kodu}", response_model=RouteDetailResponse)
def get_route(firma_id: int, guzergah_kodu: int, db: Database) -> RouteDetailResponse:
    route = db.execute(
        """
        SELECT canonical_guzergah_id
        FROM guzergah_canonical
        WHERE firma_id = ? AND guzergah_kodu = ?
        """,
        [firma_id, guzergah_kodu],
    ).fetchone()
    if route is None:
        raise HTTPException(status_code=404, detail="Route not found")

    stops = db.execute(
        """
        WITH ordered_stops AS (
            SELECT
                stop.ordinality AS original_sira,
                stop.durak_id
            FROM guzergah_canonical AS route,
            UNNEST(route.duraklar) WITH ORDINALITY AS stop(durak_id, ordinality)
            WHERE route.firma_id = ? AND route.guzergah_kodu = ?
        ),
        unique_stops AS (
            SELECT original_sira, durak_id
            FROM ordered_stops
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY durak_id
                ORDER BY original_sira
            ) = 1
        )
        SELECT
            ROW_NUMBER() OVER (ORDER BY stop.original_sira)::INTEGER AS sira,
            stop.durak_id::INTEGER AS durak_id,
            place.uetds_adi AS durak_adi,
            place.kisa_adi,
            place.il_id,
            place.ilce_id,
            place.enlem,
            place.boylam
        FROM unique_stops AS stop
        LEFT JOIN ats_yer AS place ON place.id = stop.durak_id
        ORDER BY stop.original_sira
        """,
        [firma_id, guzergah_kodu],
    ).fetchall()
    return RouteDetailResponse(
        firma_id=firma_id,
        guzergah_kodu=guzergah_kodu,
        canonical_guzergah_id=int(route[0]),
        duraklar=[
            RouteDurak(
                sira=row[0],
                durak_id=row[1],
                durak_adi=row[2],
                kisa_adi=row[3],
                il_id=row[4],
                ilce_id=row[5],
                enlem=row[6],
                boylam=row[7],
            )
            for row in stops
        ],
    )
