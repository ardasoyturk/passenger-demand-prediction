"""Stop lookup endpoints backed by ``ats_yer``."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from inference.api.depends import Database
from inference.api.schemas import Durak, PaginatedDurakResponse


router = APIRouter(prefix="/durak", tags=["durak"])

_DURAK_COLUMNS = """
    id, uetds_kodu, turu, uetds_adi, il_id, ilce_id, kisa_adi,
    ulke_id, ulke_adi, enlem, boylam
"""


@router.get("", response_model=PaginatedDurakResponse)
def list_durak(
    db: Database,
    il_id: int | None = Query(default=None),
    ilce_id: int | None = Query(default=None),
    turu: str | None = Query(default=None, min_length=1),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
) -> PaginatedDurakResponse:
    clauses: list[str] = []
    parameters: list[object] = []
    for column, value in (("il_id", il_id), ("ilce_id", ilce_id), ("turu", turu)):
        if value is not None:
            clauses.append(f"{column} = ?")
            parameters.append(value)
    where_sql = " WHERE " + " AND ".join(clauses) if clauses else ""

    total = db.execute(
        f"SELECT COUNT(*) FROM ats_yer{where_sql}", parameters
    ).fetchone()[0]
    rows = db.execute(
        f"""
        SELECT {_DURAK_COLUMNS}
        FROM ats_yer
        {where_sql}
        ORDER BY uetds_adi NULLS LAST, id
        LIMIT ? OFFSET ?
        """,
        [*parameters, page_size, (page - 1) * page_size],
    ).fetchdf().to_dict("records")
    return PaginatedDurakResponse(
        items=[Durak.model_validate(row) for row in rows],
        total=int(total),
        page=page,
        page_size=page_size,
    )


@router.get("/{durak_id}", response_model=Durak)
def get_durak(durak_id: int, db: Database) -> Durak:
    row = db.execute(
        f"SELECT {_DURAK_COLUMNS} FROM ats_yer WHERE id = ?",
        [durak_id],
    ).fetchdf()
    if row.empty:
        raise HTTPException(status_code=404, detail="Durak not found")
    return Durak.model_validate(row.iloc[0].to_dict())

