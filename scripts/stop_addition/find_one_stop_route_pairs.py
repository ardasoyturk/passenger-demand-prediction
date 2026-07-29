from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from scripts.shared.duckdb_utils import quote_ident
from scripts.stop_addition.common import ROUTE_TABLE

TRIP_TABLE_CANDIDATES = ("seferler_model", "seferler_canonical", "seferler_clean", "seferler")


def table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    return bool(
        con.execute(
            """
            SELECT COUNT(*) > 0
            FROM information_schema.tables
            WHERE lower(table_name) = lower(?)
            """,
            [table_name],
        ).fetchone()[0]
    )


def table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> dict[str, str]:
    rows = con.execute(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE lower(table_name) = lower(?)
        ORDER BY ordinal_position
        """,
        [table_name],
    ).fetchall()
    return {str(name): str(dtype) for name, dtype in rows}


def resolve_column(columns: dict[str, str], candidates: tuple[str, ...], label: str) -> str:
    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise RuntimeError(
        f"Could not find {label}. Tried {candidates}. Available columns: {list(columns)}"
    )


def normalise_stop_list(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            text = text.strip("[]")
            value = [] if not text else [part.strip() for part in text.split(",")]

    try:
        stops = tuple(int(stop) for stop in value if stop is not None)
    except (TypeError, ValueError):
        return None

    return stops if len(stops) >= 2 else None


def detect_trip_table(con: duckdb.DuckDBPyConnection, requested: str | None) -> str:
    if requested:
        if not table_exists(con, requested):
            raise RuntimeError(f"Requested trip table does not exist: {requested}")
        return requested

    for candidate in TRIP_TABLE_CANDIDATES:
        if table_exists(con, candidate):
            return candidate
    raise RuntimeError(
        "No historical trip table found. Tried: " + ", ".join(TRIP_TABLE_CANDIDATES)
    )


def load_routes(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict[str, str]]:
    if not table_exists(con, ROUTE_TABLE):
        raise RuntimeError(f"Required table does not exist: {ROUTE_TABLE}")

    columns = table_columns(con, ROUTE_TABLE)
    canonical_col = resolve_column(
        columns,
        ("stop_addition_canonical_route_id",),
        "clean stop-addition canonical route ID",
    )
    stops_col = resolve_column(columns, ("duraklar", "stop_list", "durak_listesi"), "stop list")

    optional = {}
    lower_map = {column.lower(): column for column in columns}
    for logical, candidates in {
        "guzergah_kodu": ("guzergah_kodu",),
        "firma_id": ("firma_id",),
    }.items():
        for candidate in candidates:
            if candidate in lower_map:
                optional[logical] = lower_map[candidate]
                break

    selected = [
        f"{quote_ident(canonical_col)} AS stop_addition_canonical_route_id",
        f"{quote_ident(stops_col)} AS stop_list",
    ]
    for logical, actual in optional.items():
        selected.append(f"{quote_ident(actual)} AS {logical}")

    routes = con.execute(
        f"SELECT {', '.join(selected)} FROM {quote_ident(ROUTE_TABLE)}"
    ).fetchdf()
    return routes, optional


def build_canonical_catalog(routes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    routes = routes.copy()
    routes["normalised_stop_list"] = routes["stop_list"].map(normalise_stop_list)

    invalid = routes[routes["normalised_stop_list"].isna()].copy()
    valid = routes[routes["normalised_stop_list"].notna()].copy()

    valid["stop_count"] = valid["normalised_stop_list"].map(len)
    valid["has_repeated_stop"] = valid["normalised_stop_list"].map(
        lambda stops: len(stops) != len(set(stops))
    )

    # One row per physical ordered stop list. Keep all route/company mappings separately.
    canonical_catalog = (
        valid.sort_values("stop_addition_canonical_route_id")
        .drop_duplicates(subset=["normalised_stop_list"])
        [["stop_addition_canonical_route_id", "normalised_stop_list", "stop_count", "has_repeated_stop"]]
        .reset_index(drop=True)
    )
    return canonical_catalog, invalid


def find_pairs(catalog: pd.DataFrame) -> pd.DataFrame:
    route_by_stops = {
        row.normalised_stop_list: row.stop_addition_canonical_route_id
        for row in catalog.itertuples(index=False)
    }

    records: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, int, int]] = set()

    for row in catalog.itertuples(index=False):
        variant_stops = row.normalised_stop_list
        if len(variant_stops) < 3:
            continue

        # Only intermediate positions are removable. Origins/destinations remain identical.
        for added_index in range(1, len(variant_stops) - 1):
            base_stops = variant_stops[:added_index] + variant_stops[added_index + 1 :]
            base_id = route_by_stops.get(base_stops)
            if base_id is None:
                continue

            key = (base_id, row.stop_addition_canonical_route_id, added_index, variant_stops[added_index])
            if key in seen:
                continue
            seen.add(key)

            records.append(
                {
                    "base_stop_addition_canonical_route_id": base_id,
                    "variant_stop_addition_canonical_route_id": row.stop_addition_canonical_route_id,
                    "added_stop_uetds_yer_id": variant_stops[added_index],
                    "added_stop_index": added_index,
                    "base_stop_count": len(base_stops),
                    "variant_stop_count": len(variant_stops),
                    "base_stop_list": list(base_stops),
                    "variant_stop_list": list(variant_stops),
                }
            )

    return pd.DataFrame.from_records(records)


def attach_route_mappings(routes: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping_cols = ["stop_addition_canonical_route_id"]
    for col in ("guzergah_kodu", "firma_id"):
        if col in routes.columns:
            mapping_cols.append(col)

    mappings = routes[mapping_cols].drop_duplicates().copy()
    represented_ids = pd.unique(
        pd.concat(
            [
                pairs["base_stop_addition_canonical_route_id"],
                pairs["variant_stop_addition_canonical_route_id"],
            ],
            ignore_index=True,
        )
    ) if not pairs.empty else []

    represented = mappings[mappings["stop_addition_canonical_route_id"].isin(represented_ids)].copy()
    return mappings, represented


def build_trip_support(
    con: duckdb.DuckDBPyConnection,
    trip_table: str,
    mappings: pd.DataFrame,
    pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    trip_cols = table_columns(con, trip_table)
    trip_route_col = resolve_column(trip_cols, ("GUZERGAH_KODU", "guzergah_kodu"), "trip route code")
    trip_company_col = resolve_column(trip_cols, ("FIRMA_ID", "firma_id"), "trip company ID")
    date_col = resolve_column(trip_cols, ("SEFER_TARIHI", "sefer_tarihi"), "trip date")

    demand_col = None
    for candidate in ("SEFER_SAYISI", "sefer_sayisi", "COUNT(*)", "demand"):
        if candidate in trip_cols:
            demand_col = candidate
            break

    if "guzergah_kodu" not in mappings.columns or "firma_id" not in mappings.columns:
        raise RuntimeError(
            f"{ROUTE_TABLE} must contain guzergah_kodu and firma_id to calculate historical trip support."
        )

    con.register("stop_addition_route_mapping_df", mappings)

    demand_select = (
        f"SUM(TRY_CAST(t.{quote_ident(demand_col)} AS DOUBLE)) AS total_passenger_count"
        if demand_col
        else "NULL::DOUBLE AS total_passenger_count"
    )

    trip_support = con.execute(
        f"""
        SELECT
            m.stop_addition_canonical_route_id,
            COUNT(*) AS historical_trip_rows,
            COUNT(DISTINCT t.{quote_ident(trip_company_col)}) AS historical_company_count,
            MIN(t.{quote_ident(date_col)}) AS first_trip_date,
            MAX(t.{quote_ident(date_col)}) AS last_trip_date,
            {demand_select}
        FROM {quote_ident(trip_table)} t
        JOIN stop_addition_route_mapping_df m
          ON TRY_CAST(t.{quote_ident(trip_route_col)} AS BIGINT) = TRY_CAST(m.guzergah_kodu AS BIGINT)
         AND TRY_CAST(t.{quote_ident(trip_company_col)} AS BIGINT) = TRY_CAST(m.firma_id AS BIGINT)
        GROUP BY m.stop_addition_canonical_route_id
        """
    ).fetchdf()

    if pairs.empty:
        summary = {
            "historical_trip_rows_for_matched_routes": 0,
            "companies_represented_in_history": 0,
        }
        return trip_support, summary

    matched_ids = set(pairs["base_stop_addition_canonical_route_id"]) | set(
        pairs["variant_stop_addition_canonical_route_id"]
    )
    support_for_pairs = trip_support[
        trip_support["stop_addition_canonical_route_id"].isin(matched_ids)
    ]

    # Count distinct historical rows once, even when one route participates in many pairs.
    con.register(
        "matched_stop_addition_ids_df",
        pd.DataFrame({"stop_addition_canonical_route_id": list(matched_ids)}),
    )
    matched_history = con.execute(
        f"""
        SELECT
            COUNT(*) AS historical_trip_rows_for_matched_routes,
            COUNT(DISTINCT t.{quote_ident(trip_company_col)}) AS companies_represented_in_history
        FROM {quote_ident(trip_table)} t
        JOIN stop_addition_route_mapping_df m
          ON TRY_CAST(t.{quote_ident(trip_route_col)} AS BIGINT) = TRY_CAST(m.guzergah_kodu AS BIGINT)
         AND TRY_CAST(t.{quote_ident(trip_company_col)} AS BIGINT) = TRY_CAST(m.firma_id AS BIGINT)
        JOIN matched_stop_addition_ids_df x
          ON x.stop_addition_canonical_route_id = m.stop_addition_canonical_route_id
        """
    ).fetchone()

    summary = {
        "historical_trip_rows_for_matched_routes": int(matched_history[0]),
        "companies_represented_in_history": int(matched_history[1]),
        "matched_routes_with_any_history": int((support_for_pairs["historical_trip_rows"] > 0).sum()),
    }
    return trip_support, summary


def add_names(con: duckdb.DuckDBPyConnection, pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty or not table_exists(con, "ats_yer"):
        return pairs

    yer_cols = table_columns(con, "ats_yer")
    try:
        id_col = resolve_column(yer_cols, ("id",), "ats_yer ID")
        name_col = resolve_column(yer_cols, ("uetds_adi",), "ats_yer name")
    except RuntimeError:
        return pairs

    names = con.execute(
        f"SELECT {quote_ident(id_col)} AS stop_id, {quote_ident(name_col)} AS stop_name FROM ats_yer"
    ).fetchdf()
    name_map = dict(zip(names["stop_id"], names["stop_name"]))

    result = pairs.copy()
    result["added_stop_name"] = result["added_stop_uetds_yer_id"].map(name_map)
    result["base_route_names"] = result["base_stop_list"].map(
        lambda stops: " -> ".join(str(name_map.get(stop, stop)) for stop in stops)
    )
    result["variant_route_names"] = result["variant_stop_list"].map(
        lambda stops: " -> ".join(str(name_map.get(stop, stop)) for stop in stops)
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find clean canonical route pairs differing by exactly one intermediate stop."
    )
    parser.add_argument("--db", default="analysis.duckdb", help="Path to DuckDB database")
    parser.add_argument("--trip-table", default=None, help="Historical trip table override")
    parser.add_argument(
        "--output-dir",
        default="results/stop_addition",
        help="Directory for CSV and JSON outputs",
    )
    parser.add_argument("--examples", type=int, default=20, help="Number of example pairs to print")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(args.db, read_only=True)
    try:
        trip_table = detect_trip_table(con, args.trip_table)
        routes, _ = load_routes(con)
        catalog, invalid_routes = build_canonical_catalog(routes)
        pairs = find_pairs(catalog)
        mappings, represented_mappings = attach_route_mappings(routes, pairs)
        trip_support, history_summary = build_trip_support(
            con, trip_table, mappings, pairs
        )
        pairs = add_names(con, pairs)

        if not pairs.empty:
            base_support = trip_support.rename(
                columns={
                    "stop_addition_canonical_route_id": "base_stop_addition_canonical_route_id",
                    "historical_trip_rows": "base_historical_trip_rows",
                    "historical_company_count": "base_historical_company_count",
                    "first_trip_date": "base_first_trip_date",
                    "last_trip_date": "base_last_trip_date",
                    "total_passenger_count": "base_total_passenger_count",
                }
            )
            variant_support = trip_support.rename(
                columns={
                    "stop_addition_canonical_route_id": "variant_stop_addition_canonical_route_id",
                    "historical_trip_rows": "variant_historical_trip_rows",
                    "historical_company_count": "variant_historical_company_count",
                    "first_trip_date": "variant_first_trip_date",
                    "last_trip_date": "variant_last_trip_date",
                    "total_passenger_count": "variant_total_passenger_count",
                }
            )
            pairs = pairs.merge(base_support, on="base_stop_addition_canonical_route_id", how="left")
            pairs = pairs.merge(variant_support, on="variant_stop_addition_canonical_route_id", how="left")

        repeated_stop_routes = int(catalog["has_repeated_stop"].sum())
        duplicate_physical_mappings = int(
            routes.assign(normalised_stop_list=routes["stop_list"].map(normalise_stop_list))
            .dropna(subset=["normalised_stop_list"])
            .duplicated(subset=["normalised_stop_list"], keep=False)
            .sum()
        )

        summary = {
            "route_table": ROUTE_TABLE,
            "trip_table": trip_table,
            "valid_route_pairs": int(len(pairs)),
            "unique_base_routes": int(pairs["base_stop_addition_canonical_route_id"].nunique()) if not pairs.empty else 0,
            "unique_variant_routes": int(pairs["variant_stop_addition_canonical_route_id"].nunique()) if not pairs.empty else 0,
            "companies_represented_by_route_mappings": int(represented_mappings["firma_id"].nunique())
            if "firma_id" in represented_mappings.columns
            else None,
            **history_summary,
            "canonical_physical_routes_scanned": int(len(catalog)),
            "invalid_or_too_short_route_rows": int(len(invalid_routes)),
            "clean_routes_still_containing_repeated_stops": repeated_stop_routes,
            "route_mapping_rows_part_of_duplicate_physical_stop_lists": duplicate_physical_mappings,
        }

        pairs.to_csv(output_dir / "one_stop_route_pairs.csv", index=False)
        trip_support.to_csv(output_dir / "one_stop_route_trip_support.csv", index=False)
        invalid_routes.to_csv(output_dir / "invalid_stop_addition_routes.csv", index=False)
        with (output_dir / "one_stop_route_pair_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)

        print("\nOne-stop route-pair analysis")
        print("=" * 36)
        for key, value in summary.items():
            print(f"{key}: {value}")

        print("\nExample matched pairs")
        print("=" * 36)
        if pairs.empty:
            print("No valid pairs found.")
        else:
            example_columns = [
                col
                for col in (
                    "base_stop_addition_canonical_route_id",
                    "variant_stop_addition_canonical_route_id",
                    "added_stop_uetds_yer_id",
                    "added_stop_name",
                    "base_route_names",
                    "variant_route_names",
                    "base_historical_trip_rows",
                    "variant_historical_trip_rows",
                )
                if col in pairs.columns
            ]
            print(pairs[example_columns].head(args.examples).to_string(index=False))

        print(f"\nOutputs written to: {output_dir.resolve()}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
