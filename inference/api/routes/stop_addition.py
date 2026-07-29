"""Stop-addition prediction and business-decision endpoint."""

from __future__ import annotations

from datetime import date, time
from functools import lru_cache
from pathlib import Path
from typing import Any

import math
import pandas as pd
from catboost import CatBoostRegressor
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from inference.api.depends import Artifacts, Database
from inference.engine import PROJECT_DIR
from inference.stop_addition import business_rules
from inference.stop_addition.contracts import MODEL_PATH, MODEL_SUMMARY_PATH
from inference.stop_addition.current_route_adapter import add_current_route_predictions
from inference.stop_addition.proposed_route import (
    build_feature_rows,
    load_contract,
    load_reference_data,
    normalise_request_columns,
    validate_and_build_requests,
)

router = APIRouter(tags=["stop-addition"])
AVAILABLE_ROUTES_CSV = (
    PROJECT_DIR / "results" / "stop_addition" / "one_stop_route_pairs.csv"
)

NOT_FOUND_ERRORS = {
    "COMPANY_NOT_FOUND",
    "CURRENT_ROUTE_NOT_FOUND_FOR_COMPANY",
    "CANDIDATE_STOP_NOT_FOUND",
}
INVALID_REQUEST_ERRORS = {
    "CURRENT_ROUTE_TOO_SHORT",
    "CANDIDATE_STOP_ALREADY_IN_ROUTE",
    "ROUTE_OR_CANDIDATE_COORDINATES_UNUSABLE",
    "NO_USABLE_INTERMEDIATE_INSERTION",
}


class StopAdditionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    firma_id: int
    current_guzergah_kodu: int
    candidate_stop_uetds_yer_id: int
    requested_date: date
    requested_time: time


class AvailableRoute(BaseModel):
    firma_id: int
    current_guzergah_kodu: int
    origin_stop_id: int
    origin_name: str | None
    destination_stop_id: int
    destination_name: str | None
    candidate_stop_id: int
    candidate_stop_name: str | None


@router.get(
    "/stop-addition/available-routes",
    response_model=list[AvailableRoute],
    summary="List historically observed stop-addition options",
)
def available_routes(
    db: Database,
    firma_id: int | None = Query(default=None, ge=0),
    search: str | None = Query(default=None, max_length=100),
    has_trip_history_both: bool = Query(
        default=False,
        description=(
            "Only return options with historical trips for both the current "
            "route and the stop-added route."
        ),
    ),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[AvailableRoute]:
    """Return a filtered, paginated list of useful route/candidate choices."""

    csv_path = Path(AVAILABLE_ROUTES_CSV)
    if not csv_path.is_file():
        raise HTTPException(
            status_code=503,
            detail=(
                "Stop-addition route options are unavailable: "
                "results/stop_addition/one_stop_route_pairs.csv was not found."
            ),
        )

    normalized_search = search.strip().casefold() if search else None
    rows = db.execute(
        """
        WITH pairs AS (
            SELECT DISTINCT
                TRY_CAST(
                    base_stop_addition_canonical_route_id AS UBIGINT
                ) AS base_canonical_id,
                TRY_CAST(
                    variant_stop_addition_canonical_route_id AS UBIGINT
                ) AS variant_canonical_id,
                TRY_CAST(added_stop_uetds_yer_id AS BIGINT) AS added_stop_id,
                TRY_CAST(base_historical_trip_rows AS BIGINT)
                    AS base_historical_trip_rows,
                TRY_CAST(variant_historical_trip_rows AS BIGINT)
                    AS variant_historical_trip_rows
            FROM read_csv_auto(
                ?,
                header = true,
                types = {
                    'base_stop_addition_canonical_route_id': 'VARCHAR',
                    'variant_stop_addition_canonical_route_id': 'VARCHAR',
                    'added_stop_uetds_yer_id': 'VARCHAR'
                }
            )
        ),
        base_routes AS (
            SELECT DISTINCT
                c.stop_addition_canonical_route_id,
                c.firma_id,
                c.guzergah_kodu,
                d.duraklar
            FROM guzergah_canonical_stop_addition c
            JOIN guzergah_durak_listesi_stop_addition d
              ON d.firma_id = c.firma_id
             AND d.guzergah_kodu = c.guzergah_kodu
        ),
        options AS (
            SELECT DISTINCT
                b.firma_id,
                b.guzergah_kodu AS current_guzergah_kodu,
                b.duraklar[1] AS origin_stop_id,
                origin.uetds_adi AS origin_name,
                b.duraklar[len(b.duraklar)] AS destination_stop_id,
                destination.uetds_adi AS destination_name,
                p.added_stop_id AS candidate_stop_id,
                candidate.uetds_adi AS candidate_stop_name,
                p.base_canonical_id,
                p.variant_canonical_id,
                p.base_historical_trip_rows,
                p.variant_historical_trip_rows
            FROM pairs p
            JOIN base_routes b
              ON b.stop_addition_canonical_route_id = p.base_canonical_id
            LEFT JOIN ats_yer origin ON origin.id = b.duraklar[1]
            LEFT JOIN ats_yer destination
              ON destination.id = b.duraklar[len(b.duraklar)]
            LEFT JOIN ats_yer candidate ON candidate.id = p.added_stop_id
        )
        SELECT
            firma_id,
            current_guzergah_kodu,
            origin_stop_id,
            origin_name,
            destination_stop_id,
            destination_name,
            candidate_stop_id,
            candidate_stop_name
        FROM options
        WHERE origin_stop_id IN (
                  SELECT id
                  FROM ats_yer
                  WHERE ulke_id = 8 AND enlem IS NOT NULL AND boylam IS NOT NULL
              )
          AND destination_stop_id IN (
                  SELECT id
                  FROM ats_yer
                  WHERE ulke_id = 8 AND enlem IS NOT NULL AND boylam IS NOT NULL
              )
          AND candidate_stop_id IN (
                  SELECT id
                  FROM ats_yer
                  WHERE ulke_id = 8 AND enlem IS NOT NULL AND boylam IS NOT NULL
              )
          AND (
              ? = false
              OR (
                  coalesce(base_historical_trip_rows, 0) > 0
                  AND coalesce(variant_historical_trip_rows, 0) > 0
              )
          )
          AND (? IS NULL OR firma_id = ?)
          AND (
              ? IS NULL
              OR contains(lower(coalesce(origin_name, '')), ?)
              OR contains(lower(coalesce(destination_name, '')), ?)
              OR contains(lower(coalesce(candidate_stop_name, '')), ?)
              OR contains(CAST(firma_id AS VARCHAR), ?)
              OR contains(CAST(current_guzergah_kodu AS VARCHAR), ?)
          )
        ORDER BY
            firma_id,
            origin_name,
            destination_name,
            candidate_stop_name
        LIMIT ? OFFSET ?
        """,
        [
            str(csv_path),
            has_trip_history_both,
            firma_id,
            firma_id,
            normalized_search,
            normalized_search,
            normalized_search,
            normalized_search,
            normalized_search,
            normalized_search,
            limit,
            offset,
        ],
    ).fetchall()
    return [
        AvailableRoute(
            firma_id=row[0],
            current_guzergah_kodu=row[1],
            origin_stop_id=row[2],
            origin_name=row[3],
            destination_stop_id=row[4],
            destination_name=row[5],
            candidate_stop_id=row[6],
            candidate_stop_name=row[7],
        )
        for row in rows
    ]


@lru_cache(maxsize=1)
def load_stop_addition_contract(
) -> tuple[CatBoostRegressor, list[str], list[str], str]:
    """Load and validate the immutable stop-addition model contract once."""

    return load_contract(MODEL_PATH, MODEL_SUMMARY_PATH)


def _finite_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


@router.post(
    "/predict-stop-addition",
    summary="Evaluate a proposed stop addition",
)
def predict_stop_addition(
    proposal: StopAdditionRequest,
    artifacts: Artifacts,
    db: Database,
) -> dict[str, Any]:
    """Run proposed/current demand, uplift, and frozen business rules."""

    frame = normalise_request_columns(
        pd.DataFrame(
            [
                {
                    "FIRMA_ID": proposal.firma_id,
                    "CURRENT_GUZERGAH_KODU": proposal.current_guzergah_kodu,
                    "CANDIDATE_STOP_UETDS_YER_ID": (
                        proposal.candidate_stop_uetds_yer_id
                    ),
                    "REQUESTED_DATE": proposal.requested_date.isoformat(),
                    "REQUESTED_TIME": proposal.requested_time.isoformat(),
                }
            ]
        )
    )

    try:
        model, features, categorical, _target = load_stop_addition_contract()
        routes, canonical_routes, coordinates, place_info, companies = (
            load_reference_data(db)
        )
        targets, pairs, similarities, output_rows = validate_and_build_requests(
            frame,
            db,
            routes,
            canonical_routes,
            coordinates,
            place_info,
            companies,
        )
        output = output_rows[0]
        validation_error = output.get("prediction_error")
        if validation_error:
            status_code = 404 if validation_error in NOT_FOUND_ERRORS else 422
            if validation_error not in NOT_FOUND_ERRORS | INVALID_REQUEST_ERRORS:
                status_code = 500
            raise HTTPException(status_code=status_code, detail=validation_error)

        feature_rows = build_feature_rows(db, targets, pairs, similarities)
        if len(feature_rows) != 1:
            raise RuntimeError(
                f"Expected one feature row, created {len(feature_rows)}."
            )
        feature_row = feature_rows.iloc[0]
        missing = [name for name in features if name not in feature_rows.columns]
        if missing:
            raise RuntimeError(
                f"Inference failed to create model features: {missing}"
            )
        matrix = feature_rows.loc[:, features].copy()
        for column in categorical:
            matrix[column] = (
                matrix[column].astype("string").fillna("__MISSING__").astype(str)
            )
        prediction = float(model.predict(matrix)[0])
        if not math.isfinite(prediction):
            raise RuntimeError("Stop-addition model returned a non-finite prediction.")

        output.update(
            {
                "training_scenario": str(feature_row["training_scenario"]),
                "has_same_company_exact_proposed_history": bool(
                    feature_row["inference_has_same_company_exact"]
                ),
                "has_any_exact_proposed_history": bool(
                    feature_row["inference_has_any_exact"]
                ),
                "has_similar_route_history": bool(
                    feature_row["inference_has_similar"]
                ),
                "current_route_expected_demand_proxy": _finite_float(
                    feature_row["current_route_expected_demand_proxy"]
                ),
                "proposed_route_hierarchical_baseline": _finite_float(
                    feature_row["inference_hierarchical_baseline"]
                ),
                "proposed_route_baseline_source": str(
                    feature_row["proposed_route_baseline_source"]
                ),
                "proposed_route_history_same_company_time_count": int(
                    feature_row["proposed_same_company_time_prior_trip_count"]
                ),
                "proposed_route_history_same_company_route_count": int(
                    feature_row["proposed_same_company_prior_trip_count"]
                ),
                "proposed_route_history_all_company_time_count": int(
                    feature_row["proposed_all_company_time_prior_trip_count"]
                ),
                "proposed_route_history_all_company_route_count": int(
                    feature_row["proposed_all_company_prior_trip_count"]
                ),
                "proposed_route_history_similar_route_count": int(
                    feature_row["similar_routes_with_prior_history_count"]
                ),
                "proposed_route_history_similar_trip_count": int(
                    feature_row["similar_all_company_prior_trip_count"]
                ),
                "proposed_route_prediction": prediction,
                "current_route_prediction": None,
                "predicted_uplift": None,
                "current_route_prediction_status": "PENDING",
                "current_route_prediction_error": None,
                "current_route_reliability": None,
                "current_route_reliability_reason": None,
                "current_route_baseline_source": None,
                "current_route_history_exact_time_weekday_count": None,
                "current_route_history_exact_time_count": None,
                "current_route_history_company_route_count": None,
                "current_route_history_canonical_time_weekday_count": None,
                "current_route_history_canonical_route_count": None,
                "prediction_status": "SUCCESS",
                "prediction_error": None,
            }
        )
        add_current_route_predictions(output_rows, [0], artifacts)
        if output["current_route_prediction_status"] == "SUCCESS":
            output["predicted_uplift"] = (
                prediction - float(output["current_route_prediction"])
            )

        city_pair = (
            companies.get(proposal.firma_id),
            place_info[proposal.candidate_stop_uetds_yer_id]["il_id"],
        )
        decision = business_rules.decide(pd.Series(output), city_pair)
        warnings = business_rules.decision_warnings(pd.Series(output))
        output.update(
            {
                "business_decision": decision[0],
                "decision_reason": decision[1],
                "decision_score": decision[2],
                "model_evidence_score": decision[3],
                "decision_override": decision[4],
                "decision_warnings": "|".join(warnings) if warnings else None,
                "is_company_origin_city": (
                    city_pair[0] is not None
                    and city_pair[1] is not None
                    and city_pair[0] == city_pair[1]
                ),
                "company_origin_il_id": city_pair[0],
                "added_stop_il_id": city_pair[1],
            }
        )
        output.pop("_input_row_number", None)
        return output
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Stop-addition inference failed: {exc}",
        ) from exc
