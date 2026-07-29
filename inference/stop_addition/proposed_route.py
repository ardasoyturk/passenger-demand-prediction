"""Stop-addition request validation, route construction, and demand inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from catboost import CatBoostRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
from inference.engine import (
    FrozenArtifacts,
    load_frozen_artifacts,
)
from inference.stop_addition import feature_builder as training
from inference.stop_addition.contracts import MODEL_PATH, MODEL_SUMMARY_PATH
from inference.stop_addition.current_route_adapter import (
    add_current_route_predictions,
)
from inference.stop_addition.geography import (
    count_missing_coordinates,
    corridor_similarity,
    final_similarity_score,
    find_best_insertion,
    jaccard_similarity,
    ordered_sequence_similarity,
    route_haversine_km,
    route_length_similarity,
)

REQUIRED_REQUEST_COLUMNS = {
    "FIRMA_ID",
    "CURRENT_GUZERGAH_KODU",
    "CANDIDATE_STOP_UETDS_YER_ID",
    "REQUESTED_DATE",
    "REQUESTED_TIME",
}
ALIASES = {
    "GUZERGAH_KODU": "CURRENT_GUZERGAH_KODU",
    "current_GUZERGAH_KODU": "CURRENT_GUZERGAH_KODU",
    "added_stop_uetds_yer_id": "CANDIDATE_STOP_UETDS_YER_ID",
    "target_date": "REQUESTED_DATE",
    "departure_time": "REQUESTED_TIME",
}
PROHIBITED_MODEL_COLUMNS = {
    "canonical_guzergah_id",
    "history_mask_type",
    "combined_weight",
    "scenario_weight",
    "support_weight",
}
TOP_K_SIMILAR = 10
MIN_SIMILARITY = 0.60
MAX_SIMILAR_CANDIDATES = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict demand for proposed intercity stop additions."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="CSV or Parquet request file.",
    )
    parser.add_argument(
        "--output",
        default="stop_addition_predictions.csv",
        help="CSV or Parquet batch output.",
    )
    parser.add_argument("--database", default="analysis.duckdb")
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
    )
    parser.add_argument(
        "--model-summary",
        default=str(MODEL_SUMMARY_PATH),
    )
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype="string")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format {suffix!r}; use CSV or Parquet.")


def write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format {suffix!r}; use CSV or Parquet.")


def normalise_request_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.rename(
        columns={source: target for source, target in ALIASES.items() if target not in frame}
    ).copy()
    missing = REQUIRED_REQUEST_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(
            f"Input is missing required columns: {sorted(missing)}. "
            f"Accepted canonical names: {sorted(REQUIRED_REQUEST_COLUMNS)}"
        )
    if "REQUEST_ID" not in frame:
        frame.insert(0, "REQUEST_ID", range(1, len(frame) + 1))
    frame.insert(0, "_input_row_number", range(1, len(frame) + 1))
    return frame


def parse_integer(value: Any, label: str) -> int:
    if value is None or pd.isna(value) or not str(value).strip():
        raise ValueError(f"{label} is required")
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or not float(numeric).is_integer():
        raise ValueError(f"{label} must be an integer")
    return int(numeric)


def parse_date(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError("REQUESTED_DATE must be a valid date")
    return pd.Timestamp(parsed).normalize()


def parse_time(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    parsed = pd.to_datetime(text, errors="coerce")
    if not text or pd.isna(parsed):
        raise ValueError("REQUESTED_TIME must be a valid time")
    return pd.Timestamp(parsed).strftime("%H:%M:%S")


def stop_list_json(stops: tuple[int, ...] | None) -> str | None:
    return None if stops is None else json.dumps(list(stops), ensure_ascii=False)


def usable_coordinate(
    coordinate: tuple[float | None, float | None] | None,
) -> bool:
    if coordinate is None or any(value is None for value in coordinate):
        return False
    lat, lon = float(coordinate[0]), float(coordinate[1])
    return (
        math.isfinite(lat)
        and math.isfinite(lon)
        and -90.0 <= lat <= 90.0
        and -180.0 <= lon <= 180.0
    )


def load_reference_data(
    con: duckdb.DuckDBPyConnection,
) -> tuple[
    dict[tuple[int, int], dict[str, Any]],
    dict[int, tuple[int, ...]],
    dict[int, tuple[float | None, float | None]],
    dict[int, dict[str, Any]],
    dict[int, int | None],
]:
    training.require_schema(con, training.TRIP_TABLE_DEFAULT)
    training.validate_route_mapping_uniqueness(con)
    mappings = con.execute(
        """
        SELECT
            d.guzergah_kodu,
            d.firma_id,
            d.duraklar,
            c.stop_addition_canonical_route_id
        FROM guzergah_durak_listesi_stop_addition d
        JOIN guzergah_canonical_stop_addition c
          ON c.guzergah_kodu = d.guzergah_kodu
         AND c.firma_id = d.firma_id
        """
    ).fetchdf()
    routes: dict[tuple[int, int], dict[str, Any]] = {}
    canonical_routes: dict[int, tuple[int, ...]] = {}
    for row in mappings.itertuples(index=False):
        stops = tuple(int(value) for value in row.duraklar)
        canonical_id = int(row.stop_addition_canonical_route_id)
        routes[(int(row.firma_id), int(row.guzergah_kodu))] = {
            "canonical_id": canonical_id,
            "stops": stops,
        }
        canonical_routes.setdefault(canonical_id, stops)

    places = con.execute(
        "SELECT id, il_id, uetds_adi, enlem, boylam FROM ats_yer"
    ).fetchdf()
    coordinates = {
        int(row.id): (
            None if pd.isna(row.enlem) else float(row.enlem),
            None if pd.isna(row.boylam) else float(row.boylam),
        )
        for row in places.itertuples(index=False)
    }
    place_info = {
        int(row.id): {
            "il_id": None if pd.isna(row.il_id) else int(row.il_id),
            "name": None if pd.isna(row.uetds_adi) else str(row.uetds_adi),
        }
        for row in places.itertuples(index=False)
    }
    companies = {
        int(row[0]): None if row[1] is None else int(row[1])
        for row in con.execute(
            "SELECT firma_id, faaliyet_il_id FROM ats_firma"
        ).fetchall()
        if row[0] is not None
    }
    return routes, canonical_routes, coordinates, place_info, companies


def canonical_route_id(
    con: duckdb.DuckDBPyConnection, stops: tuple[int, ...]
) -> int:
    return int(con.execute("SELECT hash(?::INTEGER[])", [list(stops)]).fetchone()[0])


def similarity_rows(
    pair_id: int,
    proposed_route_id: int,
    proposed_stops: tuple[int, ...],
    proposed_km: float,
    canonical_routes: dict[int, tuple[int, ...]],
    coordinates: dict[int, tuple[float | None, float | None]],
) -> list[dict[str, Any]]:
    candidates: list[tuple[int, tuple[int, ...]]] = [
        (route_id, stops)
        for route_id, stops in canonical_routes.items()
        if route_id != proposed_route_id
        and stops[0] == proposed_stops[0]
        and stops[-1] == proposed_stops[-1]
    ]
    candidates.sort(key=lambda item: (abs(len(item[1]) - len(proposed_stops)), item[0]))
    scored: list[dict[str, Any]] = []
    for route_id, stops in candidates[:MAX_SIMILAR_CANDIDATES]:
        candidate_km = route_haversine_km(stops, coordinates)
        ordered = ordered_sequence_similarity(proposed_stops, stops)
        jaccard = jaccard_similarity(proposed_stops, stops)
        stop_difference = len(stops) - len(proposed_stops)
        length = route_length_similarity(proposed_km, candidate_km)
        corridor_km, corridor = corridor_similarity(
            proposed_stops, stops, coordinates
        )
        score = final_similarity_score(
            ordered_score=ordered,
            stop_set_score=jaccard,
            length_score=length,
            corridor_score=corridor,
            stop_count_difference=stop_difference,
        )
        if score >= MIN_SIMILARITY:
            scored.append(
                {
                    "pair_id": pair_id,
                    "similar_stop_addition_canonical_route_id": str(route_id),
                    "ordered_sequence_similarity": ordered,
                    "stop_set_jaccard": jaccard,
                    "stop_count_difference": stop_difference,
                    "route_length_similarity": length,
                    "corridor_mean_nearest_stop_km": corridor_km,
                    "corridor_similarity": corridor,
                    "similarity_score": score,
                }
            )
    scored.sort(
        key=lambda row: (
            -float(row["similarity_score"]),
            int(row["similar_stop_addition_canonical_route_id"]),
        )
    )
    for rank, row in enumerate(scored[:TOP_K_SIMILAR], start=1):
        row["similarity_rank"] = rank
    return scored[:TOP_K_SIMILAR]


def validate_and_build_requests(
    frame: pd.DataFrame,
    con: duckdb.DuckDBPyConnection,
    routes: dict[tuple[int, int], dict[str, Any]],
    canonical_routes: dict[int, tuple[int, ...]],
    coordinates: dict[int, tuple[float | None, float | None]],
    place_info: dict[int, dict[str, Any]],
    companies: dict[int, int | None],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    similarities: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []

    for raw in frame.to_dict("records"):
        output = {
            **raw,
            "prediction_status": "ERROR",
            "prediction_error": None,
            "current_route_prediction": None,
            "current_route_prediction_status": "NOT_ATTEMPTED",
            "current_route_prediction_error": None,
            "predicted_uplift": None,
        }
        try:
            firma_id = parse_integer(raw["FIRMA_ID"], "FIRMA_ID")
            route_code = parse_integer(
                raw["CURRENT_GUZERGAH_KODU"], "CURRENT_GUZERGAH_KODU"
            )
            candidate = parse_integer(
                raw["CANDIDATE_STOP_UETDS_YER_ID"],
                "CANDIDATE_STOP_UETDS_YER_ID",
            )
            target_date = parse_date(raw["REQUESTED_DATE"])
            target_time = parse_time(raw["REQUESTED_TIME"])
            if firma_id not in companies:
                raise ValueError("COMPANY_NOT_FOUND")
            route = routes.get((firma_id, route_code))
            if route is None:
                raise ValueError("CURRENT_ROUTE_NOT_FOUND_FOR_COMPANY")
            if candidate not in place_info:
                raise ValueError("CANDIDATE_STOP_NOT_FOUND")

            base_stops = route["stops"]
            if len(base_stops) < 2:
                raise ValueError("CURRENT_ROUTE_TOO_SHORT")
            if candidate in base_stops:
                raise ValueError("CANDIDATE_STOP_ALREADY_IN_ROUTE")
            base_missing = count_missing_coordinates(base_stops, coordinates)
            candidate_coord = coordinates.get(candidate)
            if (
                base_missing
                or any(not usable_coordinate(coordinates.get(stop)) for stop in base_stops)
                or not usable_coordinate(candidate_coord)
            ):
                raise ValueError("ROUTE_OR_CANDIDATE_COORDINATES_UNUSABLE")

            insertion_index, minimum_added = find_best_insertion(
                base_stops, candidate, coordinates
            )
            if insertion_index is None or minimum_added is None:
                raise ValueError("NO_USABLE_INTERMEDIATE_INSERTION")
            proposed_stops = (
                base_stops[:insertion_index]
                + (candidate,)
                + base_stops[insertion_index:]
            )
            base_km = route_haversine_km(base_stops, coordinates)
            proposed_km = route_haversine_km(proposed_stops, coordinates)
            if (
                base_km is None
                or proposed_km is None
                or not math.isfinite(base_km)
                or not math.isfinite(proposed_km)
                or base_km <= 0
            ):
                raise ValueError("ROUTE_OR_CANDIDATE_COORDINATES_UNUSABLE")
            added_km = proposed_km - base_km
            variant_id = canonical_route_id(con, proposed_stops)
            # A repeated hash must resolve to the same physical route if it exists.
            existing_stops = canonical_routes.get(variant_id)
            if existing_stops is not None and existing_stops != proposed_stops:
                raise RuntimeError("CANONICAL_ROUTE_HASH_COLLISION")

            pair_id = int(raw["_input_row_number"])
            place = place_info[candidate]
            company_origin_il_id = companies[firma_id]
            pair = {
                "pair_id": pair_id,
                "base_stop_addition_canonical_route_id": str(route["canonical_id"]),
                "variant_stop_addition_canonical_route_id": str(variant_id),
                "added_stop_uetds_yer_id": str(candidate),
                "added_stop_il_id": place["il_id"],
                "origin_stop_id": base_stops[0],
                "destination_stop_id": base_stops[-1],
                "base_stop_count": len(base_stops),
                "variant_stop_count": len(proposed_stops),
                "observed_added_stop_index": insertion_index,
                "best_haversine_insertion_index": insertion_index,
                "observed_index_is_min_haversine_index": True,
                "base_route_haversine_km": base_km,
                "variant_route_haversine_km": proposed_km,
                "added_haversine_km": added_km,
                "detour_ratio": added_km / base_km,
                "base_missing_coordinate_stop_count": base_missing,
                "variant_missing_coordinate_stop_count": count_missing_coordinates(
                    proposed_stops, coordinates
                ),
            }
            target_ts = pd.Timestamp(
                f"{target_date.date()} {target_time}"
            )
            valid.append(
                {
                    "training_row_id": pair_id,
                    "pair_id": pair_id,
                    "SEFER_ID": None,
                    "target_date": target_date.date(),
                    "target_time": target_ts.time(),
                    "target_variant_guzergah_kodu": None,
                    "FIRMA_ID": firma_id,
                    **{k: v for k, v in pair.items() if k != "pair_id"},
                    "company_origin_il_id": company_origin_il_id,
                    "is_company_origin_city": (
                        company_origin_il_id is not None
                        and company_origin_il_id == place["il_id"]
                    ),
                    "iso_weekday": target_date.isoweekday(),
                    "half_hour_bucket": target_ts.hour * 2 + target_ts.minute // 30,
                    "year": target_date.year,
                    "month": target_date.month,
                    "day_of_month": target_date.day,
                    "departure_hour": target_ts.hour,
                    "departure_minute": target_ts.minute,
                    "is_weekend": target_date.isoweekday() in (6, 7),
                    "data_split": "INFERENCE",
                    "target_passenger_count": None,
                }
            )
            pairs.append(pair)
            similarities.extend(
                similarity_rows(
                    pair_id,
                    variant_id,
                    proposed_stops,
                    proposed_km,
                    canonical_routes,
                    coordinates,
                )
            )
            output.update(
                {
                    "FIRMA_ID": firma_id,
                    "CURRENT_GUZERGAH_KODU": route_code,
                    "CANDIDATE_STOP_UETDS_YER_ID": candidate,
                    "added_stop_uetds_yer_id": candidate,
                    "REQUESTED_DATE": str(target_date.date()),
                    "REQUESTED_TIME": target_time,
                    "base_stop_list": stop_list_json(base_stops),
                    "proposed_stop_list": stop_list_json(proposed_stops),
                    "selected_insertion_index": insertion_index,
                    "base_route_haversine_km": base_km,
                    "variant_route_haversine_km": proposed_km,
                    "added_haversine_km": added_km,
                    "detour_ratio": added_km / base_km,
                    "variant_stop_addition_canonical_route_id": str(variant_id),
                }
            )
        except Exception as exc:  # keep independent rows alive
            output["prediction_error"] = str(exc)
            output["current_route_prediction_error"] = (
                f"PROPOSED_REQUEST_VALIDATION_FAILED: {exc}"
            )
        outputs.append(output)

    return (
        pd.DataFrame(valid),
        pd.DataFrame(pairs, columns=training.SAFE_PAIR_COLUMNS),
        pd.DataFrame(similarities, columns=training.SAFE_SIMILARITY_COLUMNS),
        outputs,
    )


def build_feature_rows(
    con: duckdb.DuckDBPyConnection,
    targets: pd.DataFrame,
    pairs: pd.DataFrame,
    similarities: pd.DataFrame,
) -> pd.DataFrame:
    training.register_static_inputs(con, pairs, similarities)
    con.register("inference_target_rows_df", targets)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE target_rows AS
        SELECT * FROM inference_target_rows_df
        """
    )
    training.create_mapped_history(con, training.TRIP_TABLE_DEFAULT)
    training.create_relevant_routes(con)
    training.create_daily_aggregates(con)
    training.create_all_exact_and_company_features(con)
    training.create_similar_route_features(con)
    training.create_final_training_table(con)
    # Match the scenario builder's inference-safe recalculation exactly.
    return con.execute(
        """
        SELECT
            *,
            CASE
                WHEN proposed_same_company_time_prior_trip_count > 0
                    THEN proposed_same_company_time_prior_mean_demand
                WHEN proposed_all_company_time_prior_trip_count > 0
                    THEN proposed_all_company_time_prior_mean_demand
                WHEN proposed_same_company_prior_trip_count > 0
                    THEN proposed_same_company_prior_mean_demand
                WHEN proposed_all_company_prior_trip_count > 0
                    THEN proposed_all_company_prior_mean_demand
                WHEN similar_same_company_prior_trip_count > 0
                    THEN similar_same_company_weighted_mean_demand
                WHEN similar_all_company_prior_trip_count > 0
                    THEN similar_all_company_weighted_mean_demand
                WHEN base_same_company_prior_trip_count > 0
                    THEN base_same_company_prior_mean_demand
                WHEN base_all_company_prior_trip_count > 0
                    THEN base_all_company_prior_mean_demand
                WHEN company_prior_trip_count > 0 THEN company_prior_mean_demand
            END AS inference_hierarchical_baseline,
            proposed_same_company_prior_trip_count > 0
                AS inference_has_same_company_exact,
            proposed_all_company_prior_trip_count > 0
                AS inference_has_any_exact,
            COALESCE(similar_routes_with_prior_history_count, 0) > 0
                AS inference_has_similar,
            CASE
                WHEN proposed_same_company_prior_trip_count > 0
                    THEN 'SAME_COMPANY_EXACT'
                WHEN proposed_all_company_prior_trip_count > 0
                    THEN 'OTHER_COMPANY_EXACT'
                WHEN COALESCE(similar_routes_with_prior_history_count, 0) > 0
                    THEN 'SIMILAR_ROUTE_ONLY'
                ELSE 'COLD_START'
            END AS training_scenario
        FROM stop_addition_training_data
        ORDER BY training_row_id
        """
    ).fetchdf()


def load_contract(
    model_path: Path, summary_path: Path
) -> tuple[CatBoostRegressor, list[str], list[str], str]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    features = [str(value) for value in payload["feature_names"]]
    categorical = [str(value) for value in payload["categorical_feature_names"]]
    target = str(payload["target"])
    if len(features) != 131 or len(set(features)) != 131:
        raise RuntimeError(
            f"Expected exactly 131 unique summary features, found {len(features)}."
        )
    prohibited = PROHIBITED_MODEL_COLUMNS & set(features)
    if prohibited:
        raise RuntimeError(f"Prohibited columns in model contract: {sorted(prohibited)}")
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    if list(model.feature_names_) != features:
        raise RuntimeError(
            "Model feature_names_ differ from model-summary feature_names/order."
        )
    if not set(categorical).issubset(features):
        raise RuntimeError("Categorical contract contains non-feature columns.")
    return model, features, categorical, target


def predict_current_routes(
    output_rows: list[dict[str, Any]],
    positions: list[int],
) -> None:
    """Call the production current-route engine and isolate row-level failures."""

    if not positions:
        return
    try:
        artifacts = load_frozen_artifacts()
    except Exception as exc:
        error = f"PRODUCTION_ARTIFACT_LOAD_FAILED: {exc}"
        for position in positions:
            output_rows[position]["current_route_prediction_status"] = "ERROR"
            output_rows[position]["current_route_prediction_error"] = error
        return

    add_current_route_predictions(output_rows, positions, artifacts)


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    output_path = resolve_path(args.output)
    database_path = resolve_path(args.database)
    model_path = resolve_path(args.model)
    summary_path = resolve_path(args.model_summary)
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be non-negative")

    requests = normalise_request_columns(read_frame(input_path))
    if args.limit is not None:
        requests = requests.head(args.limit).copy()
    input_count = len(requests)
    model, features, categorical, _target = load_contract(model_path, summary_path)

    con = duckdb.connect(str(database_path), read_only=True)
    try:
        (
            routes,
            canonical_routes,
            coordinates,
            place_info,
            companies,
        ) = load_reference_data(con)
        targets, pairs, similarities, output_rows = validate_and_build_requests(
            requests,
            con,
            routes,
            canonical_routes,
            coordinates,
            place_info,
            companies,
        )
        feature_rows = (
            pd.DataFrame()
            if targets.empty
            else build_feature_rows(con, targets, pairs, similarities)
        )
    finally:
        con.close()

    features_by_id = {
        int(row["training_row_id"]): row
        for row in feature_rows.to_dict("records")
    }
    successful_positions: list[int] = []
    successful_feature_rows: list[dict[str, Any]] = []
    for position, output in enumerate(output_rows):
        if output["prediction_error"] is not None:
            continue
        row = features_by_id.get(int(output["_input_row_number"]))
        if row is None:
            output["prediction_error"] = "FEATURE_ROW_NOT_CREATED"
            continue
        row["proposed_route_hierarchical_baseline"] = row[
            "inference_hierarchical_baseline"
        ]
        row["has_same_company_exact_proposed_history"] = row[
            "inference_has_same_company_exact"
        ]
        row["has_any_exact_proposed_history"] = row["inference_has_any_exact"]
        row["has_similar_route_history"] = row["inference_has_similar"]
        row["is_exact_proposed_route_cold_start"] = not row[
            "inference_has_any_exact"
        ]
        successful_positions.append(position)
        successful_feature_rows.append(row)

    if successful_feature_rows:
        matrix = pd.DataFrame(successful_feature_rows)
        missing = [name for name in features if name not in matrix]
        if missing:
            raise RuntimeError(f"Inference failed to create model features: {missing}")
        matrix = matrix.loc[:, features].copy()
        for column in categorical:
            matrix[column] = (
                matrix[column].astype("string").fillna("__MISSING__").astype(str)
            )
        if list(matrix.columns) != features or matrix.shape[1] != 131:
            raise RuntimeError("Inference feature order/count violates model contract.")
        predictions = model.predict(matrix)
        for position, row, prediction in zip(
            successful_positions, successful_feature_rows, predictions
        ):
            if not math.isfinite(float(prediction)):
                output_rows[position]["prediction_error"] = "NON_FINITE_PREDICTION"
                continue
            proxy = row["current_route_expected_demand_proxy"]
            proxy_uplift = (
                None
                if proxy is None or pd.isna(proxy)
                else float(prediction) - float(proxy)
            )
            output_rows[position].update(
                {
                    "training_scenario": row["training_scenario"],
                    "has_same_company_exact_proposed_history": bool(
                        row["inference_has_same_company_exact"]
                    ),
                    "has_any_exact_proposed_history": bool(
                        row["inference_has_any_exact"]
                    ),
                    "has_similar_route_history": bool(row["inference_has_similar"]),
                    "current_route_expected_demand_proxy": proxy,
                    "proposed_route_hierarchical_baseline": row[
                        "inference_hierarchical_baseline"
                    ],
                    "proposed_route_prediction": float(prediction),
                    "proxy_based_predicted_uplift": proxy_uplift,
                    "current_route_prediction": None,
                    "predicted_uplift": None,
                    "current_route_prediction_status": "PENDING",
                    "current_route_prediction_error": None,
                    "prediction_status": "SUCCESS",
                    "prediction_error": None,
                }
            )

    predict_current_routes(output_rows, successful_positions)
    for position in successful_positions:
        output = output_rows[position]
        if output.get("current_route_prediction_status") == "SUCCESS":
            output["predicted_uplift"] = (
                float(output["proposed_route_prediction"])
                - float(output["current_route_prediction"])
            )

    output = pd.DataFrame(output_rows).sort_values("_input_row_number")
    if len(output) != input_count:
        raise RuntimeError(
            f"Output row count {len(output)} differs from input row count {input_count}."
        )
    output = output.drop(columns=["_input_row_number"])
    write_frame(output, output_path)

    successful = output["prediction_status"].eq("SUCCESS")
    scenario_counts = (
        output.loc[successful, "training_scenario"].value_counts().to_dict()
        if successful.any()
        else {}
    )
    mean_prediction = (
        float(output.loc[successful, "proposed_route_prediction"].mean())
        if successful.any()
        else None
    )
    uplift_values = pd.to_numeric(
        output.loc[successful, "proxy_based_predicted_uplift"], errors="coerce"
    )
    mean_uplift = float(uplift_values.mean()) if uplift_values.notna().any() else None
    current_successful = output["current_route_prediction_status"].eq("SUCCESS")
    real_uplift_values = pd.to_numeric(
        output.loc[current_successful, "predicted_uplift"], errors="coerce"
    )
    mean_real_uplift = (
        float(real_uplift_values.mean())
        if real_uplift_values.notna().any()
        else None
    )
    print(f"Input rows: {input_count}")
    print(f"Successful predictions: {int(successful.sum())}")
    print(f"Failed rows: {int((~successful).sum())}")
    print(
        "Successful current-route predictions: "
        f"{int(current_successful.sum())}"
    )
    print(
        "Failed current-route predictions: "
        f"{int((successful & ~current_successful).sum())}"
    )
    print(f"Scenario distribution: {json.dumps(scenario_counts, sort_keys=True)}")
    print(f"Mean proposed-route prediction: {mean_prediction}")
    print(f"Mean production-model uplift: {mean_real_uplift}")
    print(f"Mean proxy-based uplift: {mean_uplift}")
    print(f"Output path: {output_path}")


if __name__ == "__main__":
    main()
