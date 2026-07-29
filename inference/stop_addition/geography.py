"""Geographic primitives and legacy analysis helpers for stop-addition inference."""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "duckdb is required. Run this script inside the project's uv environment."
    ) from exc

EARTH_RADIUS_KM = 6371.0088
ROUTE_TABLE = "guzergah_canonical_stop_addition"
TRIP_TABLE_DEFAULT = "seferler_model"
PLACE_TABLE = "ats_yer"

PAIR_ID_COLUMNS = [
    "base_stop_addition_canonical_route_id",
    "variant_stop_addition_canonical_route_id",
    "added_stop_uetds_yer_id",
]

REQUIRED_PAIR_COLUMNS = set(PAIR_ID_COLUMNS) | {"base_stop_list", "variant_stop_list"}


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    rows = con.execute(f"DESCRIBE {quote_ident(table)}").fetchall()
    return {str(row[0]) for row in rows}


def require_schema(con: duckdb.DuckDBPyConnection, trip_table: str) -> None:
    expected = {
        ROUTE_TABLE: {
            "guzergah_kodu",
            "firma_id",
            "duraklar",
            "stop_addition_canonical_route_id",
        },
        trip_table: {
            "SEFER_TARIHI",
            "SEFER_SAATI",
            "GUZERGAH_KODU",
            "FIRMA_ID",
            "SEFER_SAYISI",
        },
        PLACE_TABLE: {"id", "uetds_adi", "il_id", "enlem", "boylam"},
    }
    for table, required in expected.items():
        actual = table_columns(con, table)
        missing = required - actual
        if missing:
            raise RuntimeError(
                f"{table} is missing required columns: {sorted(missing)}. "
                f"Available columns: {sorted(actual)}"
            )


def parse_stop_list(value: Any) -> tuple[int, ...]:
    if value is None or pd.isna(value):
        return tuple()
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value)
    text = str(value).strip()
    if not text:
        return tuple()
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError) as exc:
        raise ValueError(f"Could not parse stop list: {text[:200]}") from exc
    return tuple(int(v) for v in parsed)


def id_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def load_pairs(path: Path) -> pd.DataFrame:
    pairs = pd.read_csv(
        path,
        dtype={
            "base_stop_addition_canonical_route_id": "string",
            "variant_stop_addition_canonical_route_id": "string",
            "added_stop_uetds_yer_id": "string",
        },
    )
    missing = REQUIRED_PAIR_COLUMNS - set(pairs.columns)
    if missing:
        raise RuntimeError(
            f"Pair CSV is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(pairs.columns)}"
        )

    pairs = pairs.copy()
    for column in PAIR_ID_COLUMNS:
        pairs[column] = pairs[column].map(id_text)

    pairs["base_stops_tuple"] = pairs["base_stop_list"].map(parse_stop_list)
    pairs["variant_stops_tuple"] = pairs["variant_stop_list"].map(parse_stop_list)
    pairs["added_stop_int"] = pairs["added_stop_uetds_yer_id"].astype("int64")

    # Keep the same deterministic pair_id order used by analyse_training_support_v2.py.
    pairs = pairs.drop_duplicates(PAIR_ID_COLUMNS).reset_index(drop=True)
    if "pair_id" not in pairs.columns:
        pairs.insert(0, "pair_id", range(1, len(pairs) + 1))
    else:
        pairs["pair_id"] = pd.to_numeric(pairs["pair_id"], errors="raise").astype("int64")

    return pairs


def haversine_km(
    lat1: float | None,
    lon1: float | None,
    lat2: float | None,
    lon2: float | None,
) -> float | None:
    values = (lat1, lon1, lat2, lon2)
    if any(value is None or pd.isna(value) for value in values):
        return None

    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    dphi = math.radians(float(lat2) - float(lat1))
    dlambda = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return EARTH_RADIUS_KM * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def route_haversine_km(
    stops: Iterable[int], coordinates: dict[int, tuple[float | None, float | None]]
) -> float | None:
    stop_list = list(stops)
    if len(stop_list) < 2:
        return None

    total = 0.0
    for left, right in zip(stop_list, stop_list[1:]):
        left_coord = coordinates.get(int(left))
        right_coord = coordinates.get(int(right))
        if left_coord is None or right_coord is None:
            return None
        segment = haversine_km(*left_coord, *right_coord)
        if segment is None:
            return None
        total += segment
    return total


def count_missing_coordinates(
    stops: Iterable[int], coordinates: dict[int, tuple[float | None, float | None]]
) -> int:
    missing = 0
    for stop in stops:
        coord = coordinates.get(int(stop))
        if coord is None or any(value is None or pd.isna(value) for value in coord):
            missing += 1
    return missing


def insertion_added_distance(
    base_stops: tuple[int, ...],
    added_stop: int,
    insertion_index: int,
    coordinates: dict[int, tuple[float | None, float | None]],
) -> float | None:
    if insertion_index <= 0 or insertion_index >= len(base_stops):
        return None
    left = base_stops[insertion_index - 1]
    right = base_stops[insertion_index]
    left_coord = coordinates.get(left)
    added_coord = coordinates.get(added_stop)
    right_coord = coordinates.get(right)
    if left_coord is None or added_coord is None or right_coord is None:
        return None
    left_added = haversine_km(*left_coord, *added_coord)
    added_right = haversine_km(*added_coord, *right_coord)
    left_right = haversine_km(*left_coord, *right_coord)
    if None in (left_added, added_right, left_right):
        return None
    return float(left_added + added_right - left_right)


def find_best_insertion(
    base_stops: tuple[int, ...],
    added_stop: int,
    coordinates: dict[int, tuple[float | None, float | None]],
) -> tuple[int | None, float | None]:
    best_index: int | None = None
    best_added: float | None = None
    for index in range(1, len(base_stops)):
        added_km = insertion_added_distance(base_stops, added_stop, index, coordinates)
        if added_km is None:
            continue
        if best_added is None or added_km < best_added:
            best_added = added_km
            best_index = index
    return best_index, best_added


def observed_added_index(base_stops: tuple[int, ...], variant_stops: tuple[int, ...]) -> int | None:
    if len(variant_stops) != len(base_stops) + 1:
        return None
    left = 0
    while left < len(base_stops) and base_stops[left] == variant_stops[left]:
        left += 1
    if variant_stops[:left] + variant_stops[left + 1 :] == base_stops:
        return left
    return None


def lcs_length(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for index, right_value in enumerate(right, start=1):
            if left_value == right_value:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def ordered_sequence_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    return lcs_length(left, right) / denominator


def jaccard_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def route_length_similarity(left_km: float | None, right_km: float | None) -> float | None:
    if left_km is None or right_km is None:
        return None
    denominator = max(left_km, right_km)
    if denominator <= 0:
        return 1.0
    return max(0.0, 1.0 - abs(left_km - right_km) / denominator)


def mean_nearest_stop_distance(
    source: tuple[int, ...],
    target: tuple[int, ...],
    coordinates: dict[int, tuple[float | None, float | None]],
) -> float | None:
    target_coords = [coordinates.get(stop) for stop in target]
    target_coords = [coord for coord in target_coords if coord is not None]
    if not target_coords:
        return None

    distances: list[float] = []
    for stop in source:
        source_coord = coordinates.get(stop)
        if source_coord is None:
            continue
        values = [haversine_km(*source_coord, *target_coord) for target_coord in target_coords]
        values = [value for value in values if value is not None]
        if values:
            distances.append(min(values))
    if not distances:
        return None
    return sum(distances) / len(distances)


def corridor_similarity(
    left: tuple[int, ...],
    right: tuple[int, ...],
    coordinates: dict[int, tuple[float | None, float | None]],
) -> tuple[float | None, float | None]:
    left_to_right = mean_nearest_stop_distance(left, right, coordinates)
    right_to_left = mean_nearest_stop_distance(right, left, coordinates)
    if left_to_right is None or right_to_left is None:
        return None, None
    mean_distance = (left_to_right + right_to_left) / 2.0
    score = 1.0 / (1.0 + mean_distance / 50.0)
    return mean_distance, score


def final_similarity_score(
    ordered_score: float,
    stop_set_score: float,
    length_score: float | None,
    corridor_score: float | None,
    stop_count_difference: int,
) -> float:
    length_value = 0.0 if length_score is None else length_score
    corridor_value = 0.0 if corridor_score is None else corridor_score
    stop_count_score = 1.0 / (1.0 + abs(stop_count_difference))
    return (
        0.40 * ordered_score
        + 0.25 * stop_set_score
        + 0.15 * length_value
        + 0.15 * corridor_value
        + 0.05 * stop_count_score
    )


def route_names(stops: Iterable[int], place_info: dict[int, dict[str, Any]]) -> str:
    return " -> ".join(
        place_info.get(int(stop), {}).get("name") or str(int(stop)) for stop in stops
    )


def json_stop_list(stops: Iterable[int]) -> str:
    return json.dumps([int(stop) for stop in stops], ensure_ascii=False)


def load_database_reference(
    con: duckdb.DuckDBPyConnection, trip_table: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, tuple[float | None, float | None]], dict[int, dict[str, Any]]]:
    routes = con.execute(
        f"""
        SELECT
            guzergah_kodu,
            firma_id,
            duraklar,
            stop_addition_canonical_route_id
        FROM {quote_ident(ROUTE_TABLE)}
        """
    ).fetchdf()

    history = con.execute(
        f"""
        WITH mapped_trips AS (
            SELECT
                gc.stop_addition_canonical_route_id,
                s.FIRMA_ID AS firma_id,
                s.SEFER_SAYISI,
                s.SEFER_TARIHI
            FROM {quote_ident(trip_table)} s
            JOIN {quote_ident(ROUTE_TABLE)} gc
              ON gc.guzergah_kodu = s.GUZERGAH_KODU
             AND gc.firma_id = s.FIRMA_ID
        )
        SELECT
            stop_addition_canonical_route_id,
            COUNT(*) AS exact_route_trip_count,
            COUNT(DISTINCT firma_id) AS exact_route_company_count,
            AVG(SEFER_SAYISI) AS exact_route_average_demand,
            MEDIAN(SEFER_SAYISI) AS exact_route_median_demand,
            SUM(SEFER_SAYISI) AS exact_route_total_demand,
            MIN(SEFER_TARIHI) AS exact_route_first_trip_date,
            MAX(SEFER_TARIHI) AS exact_route_last_trip_date
        FROM mapped_trips
        GROUP BY stop_addition_canonical_route_id
        """
    ).fetchdf()

    places = con.execute(
        f"""
        SELECT id, uetds_adi, il_id, enlem, boylam
        FROM {quote_ident(PLACE_TABLE)}
        """
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
            "name": None if pd.isna(row.uetds_adi) else str(row.uetds_adi),
            "il_id": None if pd.isna(row.il_id) else int(row.il_id),
            "lat": None if pd.isna(row.enlem) else float(row.enlem),
            "lon": None if pd.isna(row.boylam) else float(row.boylam),
        }
        for row in places.itertuples(index=False)
    }
    return routes, history, coordinates, place_info


def build_unique_routes(
    route_mappings: pd.DataFrame,
    coordinates: dict[int, tuple[float | None, float | None]],
) -> dict[str, dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for row in route_mappings.itertuples(index=False):
        canonical_id = id_text(row.stop_addition_canonical_route_id)
        if canonical_id is None or canonical_id in unique:
            continue
        stops = tuple(int(stop) for stop in row.duraklar)
        if len(stops) < 2:
            continue
        unique[canonical_id] = {
            "canonical_id": canonical_id,
            "stops": stops,
            "origin": stops[0],
            "destination": stops[-1],
            "stop_count": len(stops),
            "haversine_km": route_haversine_km(stops, coordinates),
            "missing_coordinate_stop_count": count_missing_coordinates(stops, coordinates),
        }
    return unique


def make_history_lookup(history: pd.DataFrame) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in history.itertuples(index=False):
        data = row._asdict()
        key = id_text(data.pop("stop_addition_canonical_route_id"))
        if key is not None:
            lookup[key] = data
    return lookup


def load_optional_support(output_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    support_path = output_dir / "stop_addition_pair_company_support.csv"
    bucket_path = output_dir / "stop_addition_comparable_buckets.csv"
    coverage_path = output_dir / "stop_addition_pair_coverage.csv"

    support = pd.read_csv(support_path) if support_path.exists() else None
    buckets = pd.read_csv(bucket_path) if bucket_path.exists() else None
    coverage = pd.read_csv(coverage_path) if coverage_path.exists() else None

    for df in (support, buckets, coverage):
        if df is not None:
            for column in PAIR_ID_COLUMNS:
                if column in df.columns:
                    df[column] = df[column].map(id_text)
    return support, buckets, coverage


def summarise_support_for_pair(
    pair_id: int,
    support: pd.DataFrame | None,
    coverage: pd.DataFrame | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "same_company_rows_with_both_sides": 0,
        "same_company_with_date_overlap_count": 0,
        "strong_support_company_count": 0,
        "moderate_support_company_count": 0,
        "weak_support_company_count": 0,
        "best_support_level": None,
        "max_comparable_bucket_count": 0,
        "total_comparable_base_trip_count": 0,
        "total_comparable_variant_trip_count": 0,
        "best_weighted_average_uplift": None,
        "median_weighted_average_uplift": None,
        "pair_coverage_status": None,
    }

    if coverage is not None and "pair_id" in coverage.columns:
        coverage_match = coverage[coverage["pair_id"] == pair_id]
        if not coverage_match.empty:
            c = coverage_match.iloc[0]
            row["same_company_with_date_overlap_count"] = int(
                0 if pd.isna(c.get("same_company_with_date_overlap_count")) else c.get("same_company_with_date_overlap_count")
            )
            row["pair_coverage_status"] = c.get("pair_coverage_status")

    if support is None or "pair_id" not in support.columns:
        return row

    subset = support[support["pair_id"] == pair_id]
    if subset.empty:
        return row

    row["same_company_rows_with_both_sides"] = int(len(subset))
    levels = subset["support_level"].fillna("UNKNOWN").astype(str)
    row["strong_support_company_count"] = int((levels == "STRONG_SUPPORT").sum())
    row["moderate_support_company_count"] = int((levels == "MODERATE_SUPPORT").sum())
    row["weak_support_company_count"] = int((levels == "WEAK_SUPPORT").sum())
    priority = ["STRONG_SUPPORT", "MODERATE_SUPPORT", "WEAK_SUPPORT", "NO_COMPARABLE_BUCKETS", "NO_DATE_OVERLAP"]
    for level in priority:
        if (levels == level).any():
            row["best_support_level"] = level
            break

    if "comparable_bucket_count" in subset.columns:
        row["max_comparable_bucket_count"] = int(subset["comparable_bucket_count"].fillna(0).max())
    if "comparable_base_trip_count" in subset.columns:
        row["total_comparable_base_trip_count"] = int(subset["comparable_base_trip_count"].fillna(0).sum())
    if "comparable_variant_trip_count" in subset.columns:
        row["total_comparable_variant_trip_count"] = int(subset["comparable_variant_trip_count"].fillna(0).sum())
    if "weighted_average_uplift" in subset.columns:
        uplift = subset["weighted_average_uplift"].dropna()
        if not uplift.empty:
            row["best_weighted_average_uplift"] = float(
                subset.sort_values(
                    ["support_level", "comparable_bucket_count"],
                    ascending=[True, False],
                )["weighted_average_uplift"].dropna().iloc[0]
            )
            row["median_weighted_average_uplift"] = float(uplift.median())
    return row


def score_similar_routes_for_pair(
    pair_id: int,
    variant_route: dict[str, Any],
    base_id: str,
    variant_id: str,
    routes: dict[str, dict[str, Any]],
    history_lookup: dict[str, dict[str, Any]],
    coordinates: dict[int, tuple[float | None, float | None]],
    top_k: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates = [
        route
        for route in routes.values()
        if route["canonical_id"] not in {base_id, variant_id}
        and route["origin"] == variant_route["origin"]
        and route["destination"] == variant_route["destination"]
    ]
    candidates.sort(key=lambda route: abs(route["stop_count"] - variant_route["stop_count"]))
    candidates = candidates[:max_candidates]

    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        ordered_score = ordered_sequence_similarity(variant_route["stops"], candidate["stops"])
        stop_set_score = jaccard_similarity(variant_route["stops"], candidate["stops"])
        stop_count_difference = int(candidate["stop_count"] - variant_route["stop_count"])
        length_score = route_length_similarity(variant_route["haversine_km"], candidate["haversine_km"])
        corridor_distance, corridor_score = corridor_similarity(variant_route["stops"], candidate["stops"], coordinates)
        score = final_similarity_score(
            ordered_score=ordered_score,
            stop_set_score=stop_set_score,
            length_score=length_score,
            corridor_score=corridor_score,
            stop_count_difference=stop_count_difference,
        )
        history = history_lookup.get(candidate["canonical_id"], {})
        scored.append(
            {
                "pair_id": pair_id,
                "base_stop_addition_canonical_route_id": base_id,
                "variant_stop_addition_canonical_route_id": variant_id,
                "similar_stop_addition_canonical_route_id": candidate["canonical_id"],
                "similar_stop_list": json_stop_list(candidate["stops"]),
                "similar_origin_stop_id": candidate["origin"],
                "similar_destination_stop_id": candidate["destination"],
                "similar_stop_count": candidate["stop_count"],
                "similar_route_haversine_km": candidate["haversine_km"],
                "ordered_sequence_similarity": ordered_score,
                "stop_set_jaccard": stop_set_score,
                "stop_count_difference": stop_count_difference,
                "route_length_similarity": length_score,
                "corridor_mean_nearest_stop_km": corridor_distance,
                "corridor_similarity": corridor_score,
                "similarity_score": score,
                "similar_route_trip_count": history.get("exact_route_trip_count", 0),
                "similar_route_company_count": history.get("exact_route_company_count", 0),
                "similar_route_average_demand": history.get("exact_route_average_demand"),
                "similar_route_median_demand": history.get("exact_route_median_demand"),
                "similar_route_first_trip_date": history.get("exact_route_first_trip_date"),
                "similar_route_last_trip_date": history.get("exact_route_last_trip_date"),
            }
        )

    scored.sort(
        key=lambda row: (
            row["similarity_score"],
            row["similar_route_trip_count"],
            row["similar_route_company_count"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(scored[:top_k], start=1):
        row["similarity_rank"] = rank
    return scored[:top_k]


def build_geographic_analysis(
    pairs: pd.DataFrame,
    routes: dict[str, dict[str, Any]],
    history_lookup: dict[str, dict[str, Any]],
    coordinates: dict[int, tuple[float | None, float | None]],
    place_info: dict[int, dict[str, Any]],
    support: pd.DataFrame | None,
    coverage: pd.DataFrame | None,
    top_k_similar: int,
    max_candidates: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feature_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []

    for pair in pairs.itertuples(index=False):
        pair_id = int(pair.pair_id)
        base_id = id_text(pair.base_stop_addition_canonical_route_id)
        variant_id = id_text(pair.variant_stop_addition_canonical_route_id)
        added_stop = int(pair.added_stop_int)
        base_stops = tuple(pair.base_stops_tuple)
        variant_stops = tuple(pair.variant_stops_tuple)

        observed_index = observed_added_index(base_stops, variant_stops)
        best_index, best_added_km = find_best_insertion(base_stops, added_stop, coordinates)
        observed_added_km = (
            insertion_added_distance(base_stops, added_stop, observed_index, coordinates)
            if observed_index is not None
            else None
        )
        base_km = route_haversine_km(base_stops, coordinates)
        variant_km = route_haversine_km(variant_stops, coordinates)
        added_km = None if base_km is None or variant_km is None else variant_km - base_km
        detour_ratio = None if base_km is None or base_km <= 0 or added_km is None else added_km / base_km

        added_info = place_info.get(added_stop, {})
        base_history = history_lookup.get(base_id or "", {})
        variant_history = history_lookup.get(variant_id or "", {})
        support_summary = summarise_support_for_pair(pair_id, support, coverage)

        variant_route = routes.get(variant_id or "")
        similar = []
        if variant_route is not None and base_id is not None and variant_id is not None:
            similar = score_similar_routes_for_pair(
                pair_id=pair_id,
                variant_route=variant_route,
                base_id=base_id,
                variant_id=variant_id,
                routes=routes,
                history_lookup=history_lookup,
                coordinates=coordinates,
                top_k=top_k_similar,
                max_candidates=max_candidates,
            )
            evidence_rows.extend(similar)

        best_similarity_score = similar[0]["similarity_score"] if similar else None
        best_similar_trip_count = similar[0]["similar_route_trip_count"] if similar else 0
        history_weighted_similarity = None
        history_candidates = [row for row in similar if row.get("similar_route_trip_count", 0) > 0]
        if history_candidates:
            total_weight = sum(
                float(row["similarity_score"]) * math.log1p(float(row["similar_route_trip_count"]))
                for row in history_candidates
            )
            if total_weight > 0:
                history_weighted_similarity = sum(
                    float(row["similarity_score"])
                    * math.log1p(float(row["similar_route_trip_count"]))
                    * float(row["similar_route_average_demand"])
                    for row in history_candidates
                    if row.get("similar_route_average_demand") is not None
                ) / total_weight

        row = {
            "pair_id": pair_id,
            "base_stop_addition_canonical_route_id": base_id,
            "variant_stop_addition_canonical_route_id": variant_id,
            "added_stop_uetds_yer_id": added_stop,
            "added_stop_name": added_info.get("name") or getattr(pair, "added_stop_name", None),
            "added_stop_il_id": added_info.get("il_id"),
            "added_stop_lat": added_info.get("lat"),
            "added_stop_lon": added_info.get("lon"),
            "origin_stop_id": base_stops[0] if base_stops else None,
            "destination_stop_id": base_stops[-1] if base_stops else None,
            "origin_stop_name": place_info.get(base_stops[0], {}).get("name") if base_stops else None,
            "destination_stop_name": place_info.get(base_stops[-1], {}).get("name") if base_stops else None,
            "base_stop_count": len(base_stops),
            "variant_stop_count": len(variant_stops),
            "observed_added_stop_index": observed_index,
            "best_haversine_insertion_index": best_index,
            "observed_index_is_min_haversine_index": observed_index == best_index if observed_index is not None and best_index is not None else None,
            "observed_added_segment_haversine_km": observed_added_km,
            "minimum_added_segment_haversine_km": best_added_km,
            "observed_minus_minimum_added_km": None if observed_added_km is None or best_added_km is None else observed_added_km - best_added_km,
            "base_route_haversine_km": base_km,
            "variant_route_haversine_km": variant_km,
            "added_haversine_km": added_km,
            "detour_ratio": detour_ratio,
            "base_missing_coordinate_stop_count": count_missing_coordinates(base_stops, coordinates),
            "variant_missing_coordinate_stop_count": count_missing_coordinates(variant_stops, coordinates),
            "base_stop_list": json_stop_list(base_stops),
            "variant_stop_list": json_stop_list(variant_stops),
            "base_route_names": route_names(base_stops, place_info),
            "variant_route_names": route_names(variant_stops, place_info),
            "base_exact_route_trip_count": base_history.get("exact_route_trip_count", 0),
            "base_exact_route_company_count": base_history.get("exact_route_company_count", 0),
            "base_exact_route_average_demand": base_history.get("exact_route_average_demand"),
            "base_exact_route_median_demand": base_history.get("exact_route_median_demand"),
            "base_exact_route_first_trip_date": base_history.get("exact_route_first_trip_date"),
            "base_exact_route_last_trip_date": base_history.get("exact_route_last_trip_date"),
            "variant_exact_route_trip_count": variant_history.get("exact_route_trip_count", 0),
            "variant_exact_route_company_count": variant_history.get("exact_route_company_count", 0),
            "variant_exact_route_average_demand": variant_history.get("exact_route_average_demand"),
            "variant_exact_route_median_demand": variant_history.get("exact_route_median_demand"),
            "variant_exact_route_first_trip_date": variant_history.get("exact_route_first_trip_date"),
            "variant_exact_route_last_trip_date": variant_history.get("exact_route_last_trip_date"),
            "all_history_average_uplift": None
            if base_history.get("exact_route_average_demand") is None or variant_history.get("exact_route_average_demand") is None
            else float(variant_history.get("exact_route_average_demand")) - float(base_history.get("exact_route_average_demand")),
            "similar_route_count_scored": len(similar),
            "best_similarity_score": best_similarity_score,
            "best_similar_route_trip_count": best_similar_trip_count,
            "similar_history_weighted_average_demand": history_weighted_similarity,
        }
        row.update(support_summary)
        feature_rows.append(row)

    features = pd.DataFrame(feature_rows)
    evidence = pd.DataFrame(evidence_rows)
    summary = build_summary(features, evidence)
    return features, evidence, summary


def numeric_summary(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"mean": None, "median": None, "p10": None, "p90": None}
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p10": float(values.quantile(0.10)),
        "p90": float(values.quantile(0.90)),
    }


def build_summary(features: pd.DataFrame, evidence: pd.DataFrame) -> dict[str, Any]:
    if features.empty:
        return {"input_pairs": 0}

    valid_geo = features["added_haversine_km"].notna()
    exact_both = (features["base_exact_route_trip_count"].fillna(0) > 0) & (
        features["variant_exact_route_trip_count"].fillna(0) > 0
    )
    support_counts = (
        features["best_support_level"].fillna("NO_SAME_COMPANY_SUPPORT").value_counts().to_dict()
        if "best_support_level" in features.columns
        else {}
    )
    coverage_counts = (
        features["pair_coverage_status"].fillna("UNKNOWN").value_counts().to_dict()
        if "pair_coverage_status" in features.columns
        else {}
    )

    return {
        "input_pairs": int(len(features)),
        "pairs_with_complete_geography": int(valid_geo.sum()),
        "pairs_missing_some_coordinates": int((~valid_geo).sum()),
        "pairs_where_observed_insertion_is_min_haversine": int(
            features["observed_index_is_min_haversine_index"].fillna(False).sum()
        ),
        "pairs_with_exact_history_on_both_routes": int(exact_both.sum()),
        "pairs_with_same_company_rows": int((features["same_company_rows_with_both_sides"].fillna(0) > 0).sum()),
        "pairs_with_strong_or_moderate_support": int(
            (
                features["best_support_level"].isin(["STRONG_SUPPORT", "MODERATE_SUPPORT"])
                if "best_support_level" in features.columns
                else pd.Series(False, index=features.index)
            ).sum()
        ),
        "pairs_with_similar_route_candidates": int((features["similar_route_count_scored"].fillna(0) > 0).sum()),
        "similar_route_evidence_rows": int(len(evidence)),
        "support_level_counts": {str(k): int(v) for k, v in support_counts.items()},
        "pair_coverage_status_counts": {str(k): int(v) for k, v in coverage_counts.items()},
        "added_haversine_km_summary": numeric_summary(features["added_haversine_km"]),
        "detour_ratio_summary": numeric_summary(features["detour_ratio"]),
        "all_history_average_uplift_summary": numeric_summary(features["all_history_average_uplift"]),
        "same_company_weighted_uplift_summary": numeric_summary(features["median_weighted_average_uplift"]),
        "best_similarity_score_summary": numeric_summary(features["best_similarity_score"]),
        "methodology": {
            "route_source": ROUTE_TABLE,
            "trip_mapping": "seferler_model joined to guzergah_canonical_stop_addition by GUZERGAH_KODU + FIRMA_ID",
            "canonical_id_used": "stop_addition_canonical_route_id",
            "old_canonical_guzergah_id_used": False,
            "geography": "Segment-sum Haversine distance using ats_yer.enlem and ats_yer.boylam",
            "similarity_scope": "same origin and destination, excluding the base and variant route themselves",
            "warning": "Haversine approximates geographic distance, not road distance or actual operating cost.",
        },
    }


def write_outputs(output_dir: Path, features: pd.DataFrame, evidence: pd.DataFrame, summary: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    features_path = output_dir / "one_stop_route_pair_geographic_features.csv"
    evidence_path = output_dir / "route_pair_similarity_evidence.csv"
    summary_path = output_dir / "route_pair_geographic_summary.json"

    features.to_csv(features_path, index=False)
    evidence.to_csv(evidence_path, index=False)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    pairs_path = Path(args.pairs)
    output_dir = Path(args.output_dir)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pair CSV not found: {pairs_path}")

    pairs = load_pairs(pairs_path)
    support, buckets, coverage = load_optional_support(output_dir)
    del buckets  # The pair-level support file already aggregates comparable buckets.

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        require_schema(con, args.trip_table)
        route_mappings, route_history, coordinates, place_info = load_database_reference(con, args.trip_table)
    finally:
        con.close()

    routes = build_unique_routes(route_mappings, coordinates)
    history_lookup = make_history_lookup(route_history)

    features, evidence, summary = build_geographic_analysis(
        pairs=pairs,
        routes=routes,
        history_lookup=history_lookup,
        coordinates=coordinates,
        place_info=place_info,
        support=support,
        coverage=coverage,
        top_k_similar=args.top_k_similar,
        max_candidates=args.max_candidates,
    )
    write_outputs(output_dir, features, evidence, summary)

    print("\nRoute-pair geography analysis")
    print("================================")
    print(f"Input pairs: {summary['input_pairs']}")
    print(f"Pairs with complete geography: {summary['pairs_with_complete_geography']}")
    print(
        "Observed insertion is minimum-Haversine insertion: "
        f"{summary['pairs_where_observed_insertion_is_min_haversine']}"
    )
    print(
        "Pairs with strong/moderate same-company support: "
        f"{summary['pairs_with_strong_or_moderate_support']}"
    )
    print(f"Pairs with similar-route candidates: {summary['pairs_with_similar_route_candidates']}")
    print(f"Similarity evidence rows: {summary['similar_route_evidence_rows']}")
    print(f"\nOutputs written to: {output_dir.resolve()}")

