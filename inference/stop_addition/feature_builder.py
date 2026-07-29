"""DuckDB feature construction used by stop-addition inference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    import duckdb
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "duckdb is required. Run this script inside the project's uv environment."
    ) from exc

ROUTE_TABLE = "guzergah_canonical_stop_addition"
TRIP_TABLE_DEFAULT = "seferler_model"
FIRMA_TABLE = "ats_firma"
PLACE_TABLE = "ats_yer"

PAIR_ID_COLUMNS = [
    "base_stop_addition_canonical_route_id",
    "variant_stop_addition_canonical_route_id",
]

SAFE_PAIR_COLUMNS = [
    "pair_id",
    "base_stop_addition_canonical_route_id",
    "variant_stop_addition_canonical_route_id",
    "added_stop_uetds_yer_id",
    "added_stop_il_id",
    "origin_stop_id",
    "destination_stop_id",
    "base_stop_count",
    "variant_stop_count",
    "observed_added_stop_index",
    "best_haversine_insertion_index",
    "observed_index_is_min_haversine_index",
    "base_route_haversine_km",
    "variant_route_haversine_km",
    "added_haversine_km",
    "detour_ratio",
    "base_missing_coordinate_stop_count",
    "variant_missing_coordinate_stop_count",
]

SAFE_SIMILARITY_COLUMNS = [
    "pair_id",
    "similar_stop_addition_canonical_route_id",
    "ordered_sequence_similarity",
    "stop_set_jaccard",
    "stop_count_difference",
    "route_length_similarity",
    "corridor_mean_nearest_stop_km",
    "corridor_similarity",
    "similarity_score",
    "similarity_rank",
]

RECENT_WINDOWS = (30, 90, 180)


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def table_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    return {str(row[0]) for row in con.execute(f"DESCRIBE {quote_ident(table)}").fetchall()}


def require_schema(con: duckdb.DuckDBPyConnection, trip_table: str) -> None:
    expected = {
        ROUTE_TABLE: {
            "guzergah_kodu",
            "firma_id",
            "duraklar",
            "stop_addition_canonical_route_id",
        },
        trip_table: {
            "SEFER_ID",
            "SEFER_TARIHI",
            "SEFER_SAATI",
            "GUZERGAH_KODU",
            "FIRMA_ID",
            "SEFER_SAYISI",
        },
        FIRMA_TABLE: {"firma_id", "faaliyet_il_id"},
        PLACE_TABLE: {"id", "il_id"},
    }
    for table, required in expected.items():
        actual = table_columns(con, table)
        missing = required - actual
        if missing:
            raise RuntimeError(
                f"{table} is missing required columns: {sorted(missing)}. "
                f"Available columns: {sorted(actual)}"
            )


def id_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def load_pair_features(path: Path) -> pd.DataFrame:
    dtype_map = {
        "base_stop_addition_canonical_route_id": "string",
        "variant_stop_addition_canonical_route_id": "string",
        "added_stop_uetds_yer_id": "string",
    }
    frame = pd.read_csv(path, dtype=dtype_map)
    missing = set(SAFE_PAIR_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Pair geography CSV is missing columns: {sorted(missing)}. "
            f"Available columns: {sorted(frame.columns)}"
        )

    frame = frame[SAFE_PAIR_COLUMNS].copy()
    for column in [
        "base_stop_addition_canonical_route_id",
        "variant_stop_addition_canonical_route_id",
        "added_stop_uetds_yer_id",
    ]:
        frame[column] = frame[column].map(id_text)

    frame["pair_id"] = pd.to_numeric(frame["pair_id"], errors="raise").astype("int64")
    if frame["pair_id"].duplicated().any():
        duplicated = frame.loc[frame["pair_id"].duplicated(False), "pair_id"].tolist()[:10]
        raise RuntimeError(f"Duplicate pair_id values in pair geography CSV: {duplicated}")
    return frame


def load_similarity(path: Path, min_similarity: float, top_k: int) -> pd.DataFrame:
    dtype_map = {"similar_stop_addition_canonical_route_id": "string"}
    frame = pd.read_csv(path, dtype=dtype_map)
    missing = set(SAFE_SIMILARITY_COLUMNS) - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"Similarity CSV is missing columns: {sorted(missing)}. "
            f"Available columns: {sorted(frame.columns)}"
        )

    frame = frame[SAFE_SIMILARITY_COLUMNS].copy()
    frame["similar_stop_addition_canonical_route_id"] = frame[
        "similar_stop_addition_canonical_route_id"
    ].map(id_text)
    frame["pair_id"] = pd.to_numeric(frame["pair_id"], errors="raise").astype("int64")
    frame["similarity_score"] = pd.to_numeric(
        frame["similarity_score"], errors="coerce"
    )
    frame["similarity_rank"] = pd.to_numeric(
        frame["similarity_rank"], errors="coerce"
    )
    frame = frame[
        frame["similarity_score"].ge(min_similarity)
        & frame["similarity_rank"].le(top_k)
        & frame["similar_stop_addition_canonical_route_id"].notna()
    ].copy()
    frame = frame.sort_values(["pair_id", "similarity_rank", "similarity_score"])
    frame = frame.drop_duplicates(
        ["pair_id", "similar_stop_addition_canonical_route_id"], keep="first"
    )
    return frame


def validate_route_mapping_uniqueness(
    con: duckdb.DuckDBPyConnection,
) -> None:
    duplicate_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT guzergah_kodu, firma_id
            FROM {quote_ident(ROUTE_TABLE)}
            GROUP BY guzergah_kodu, firma_id
            HAVING COUNT(*) > 1
        ) d
        """
    ).fetchone()[0]
    if duplicate_count:
        raise RuntimeError(
            f"{ROUTE_TABLE} has {duplicate_count} duplicate "
            "(guzergah_kodu, firma_id) mappings. Resolve these before training-data generation."
        )


def register_static_inputs(
    con: duckdb.DuckDBPyConnection,
    pairs: pd.DataFrame,
    similarity: pd.DataFrame,
) -> None:
    con.register("pair_features_df", pairs)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE pair_features AS
        SELECT
            CAST(pair_id AS BIGINT) AS pair_id,
            CAST(base_stop_addition_canonical_route_id AS UBIGINT)
                AS base_stop_addition_canonical_route_id,
            CAST(variant_stop_addition_canonical_route_id AS UBIGINT)
                AS variant_stop_addition_canonical_route_id,
            CAST(added_stop_uetds_yer_id AS BIGINT) AS added_stop_uetds_yer_id,
            TRY_CAST(added_stop_il_id AS BIGINT) AS added_stop_il_id,
            TRY_CAST(origin_stop_id AS BIGINT) AS origin_stop_id,
            TRY_CAST(destination_stop_id AS BIGINT) AS destination_stop_id,
            TRY_CAST(base_stop_count AS INTEGER) AS base_stop_count,
            TRY_CAST(variant_stop_count AS INTEGER) AS variant_stop_count,
            TRY_CAST(observed_added_stop_index AS INTEGER) AS observed_added_stop_index,
            TRY_CAST(best_haversine_insertion_index AS INTEGER)
                AS best_haversine_insertion_index,
            TRY_CAST(observed_index_is_min_haversine_index AS BOOLEAN)
                AS observed_index_is_min_haversine_index,
            TRY_CAST(base_route_haversine_km AS DOUBLE) AS base_route_haversine_km,
            TRY_CAST(variant_route_haversine_km AS DOUBLE) AS variant_route_haversine_km,
            TRY_CAST(added_haversine_km AS DOUBLE) AS added_haversine_km,
            TRY_CAST(detour_ratio AS DOUBLE) AS detour_ratio,
            TRY_CAST(base_missing_coordinate_stop_count AS INTEGER)
                AS base_missing_coordinate_stop_count,
            TRY_CAST(variant_missing_coordinate_stop_count AS INTEGER)
                AS variant_missing_coordinate_stop_count
        FROM pair_features_df
        """
    )

    con.register("similarity_df", similarity)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE similarity_static AS
        SELECT
            CAST(pair_id AS BIGINT) AS pair_id,
            CAST(similar_stop_addition_canonical_route_id AS UBIGINT)
                AS similar_stop_addition_canonical_route_id,
            TRY_CAST(ordered_sequence_similarity AS DOUBLE)
                AS ordered_sequence_similarity,
            TRY_CAST(stop_set_jaccard AS DOUBLE) AS stop_set_jaccard,
            TRY_CAST(stop_count_difference AS INTEGER) AS stop_count_difference,
            TRY_CAST(route_length_similarity AS DOUBLE) AS route_length_similarity,
            TRY_CAST(corridor_mean_nearest_stop_km AS DOUBLE)
                AS corridor_mean_nearest_stop_km,
            TRY_CAST(corridor_similarity AS DOUBLE) AS corridor_similarity,
            TRY_CAST(similarity_score AS DOUBLE) AS similarity_score,
            TRY_CAST(similarity_rank AS INTEGER) AS similarity_rank
        FROM similarity_df
        """
    )


def create_mapped_history(
    con: duckdb.DuckDBPyConnection,
    trip_table: str,
) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE mapped_history AS
        SELECT
            s.SEFER_ID,
            s.SEFER_TARIHI,
            s.SEFER_SAATI,
            s.GUZERGAH_KODU,
            s.FIRMA_ID,
            s.SEFER_SAYISI,
            r.stop_addition_canonical_route_id,
            CAST(EXTRACT(ISODOW FROM s.SEFER_TARIHI) AS INTEGER) AS iso_weekday,
            CAST(EXTRACT(HOUR FROM s.SEFER_SAATI) AS INTEGER) * 2
                + CAST(FLOOR(EXTRACT(MINUTE FROM s.SEFER_SAATI) / 30) AS INTEGER)
                AS half_hour_bucket
        FROM {quote_ident(trip_table)} s
        JOIN {quote_ident(ROUTE_TABLE)} r
          ON r.guzergah_kodu = s.GUZERGAH_KODU
         AND r.firma_id = s.FIRMA_ID
        WHERE s.SEFER_TARIHI IS NOT NULL
          AND s.SEFER_SAATI IS NOT NULL
          AND s.SEFER_SAYISI IS NOT NULL
        """
    )


def create_target_rows(
    con: duckdb.DuckDBPyConnection,
    train_end: str,
    validation_end: str,
    test_end: str,
) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE target_rows AS
        SELECT
            ROW_NUMBER() OVER (
                ORDER BY h.SEFER_TARIHI, h.SEFER_SAATI, h.SEFER_ID, p.pair_id
            ) AS training_row_id,
            p.pair_id,
            h.SEFER_ID,
            h.SEFER_TARIHI AS target_date,
            h.SEFER_SAATI AS target_time,
            h.GUZERGAH_KODU AS target_variant_guzergah_kodu,
            h.FIRMA_ID,
            p.base_stop_addition_canonical_route_id,
            p.variant_stop_addition_canonical_route_id,
            p.added_stop_uetds_yer_id,
            COALESCE(p.added_stop_il_id, y.il_id) AS added_stop_il_id,
            p.origin_stop_id,
            p.destination_stop_id,
            p.base_stop_count,
            p.variant_stop_count,
            p.observed_added_stop_index,
            p.best_haversine_insertion_index,
            p.observed_index_is_min_haversine_index,
            p.base_route_haversine_km,
            p.variant_route_haversine_km,
            p.added_haversine_km,
            p.detour_ratio,
            p.base_missing_coordinate_stop_count,
            p.variant_missing_coordinate_stop_count,
            f.faaliyet_il_id AS company_origin_il_id,
            CASE
                WHEN COALESCE(p.added_stop_il_id, y.il_id) IS NOT NULL
                 AND f.faaliyet_il_id = COALESCE(p.added_stop_il_id, y.il_id)
                THEN TRUE ELSE FALSE
            END AS is_company_origin_city,
            h.iso_weekday,
            h.half_hour_bucket,
            CAST(EXTRACT(YEAR FROM h.SEFER_TARIHI) AS INTEGER) AS year,
            CAST(EXTRACT(MONTH FROM h.SEFER_TARIHI) AS INTEGER) AS month,
            CAST(EXTRACT(DAY FROM h.SEFER_TARIHI) AS INTEGER) AS day_of_month,
            CAST(EXTRACT(HOUR FROM h.SEFER_SAATI) AS INTEGER) AS departure_hour,
            CAST(EXTRACT(MINUTE FROM h.SEFER_SAATI) AS INTEGER) AS departure_minute,
            CASE WHEN h.iso_weekday IN (6, 7) THEN TRUE ELSE FALSE END AS is_weekend,
            CASE
                WHEN h.SEFER_TARIHI <= DATE {sql_string(train_end)} THEN 'TRAIN'
                WHEN h.SEFER_TARIHI <= DATE {sql_string(validation_end)} THEN 'VALIDATION'
                WHEN h.SEFER_TARIHI <= DATE {sql_string(test_end)} THEN 'TEST'
                ELSE 'FINAL_EVAL'
            END AS data_split,
            h.SEFER_SAYISI AS target_passenger_count
        FROM mapped_history h
        JOIN pair_features p
          ON p.variant_stop_addition_canonical_route_id
           = h.stop_addition_canonical_route_id
        LEFT JOIN {quote_ident(FIRMA_TABLE)} f
          ON f.firma_id = h.FIRMA_ID
        LEFT JOIN {quote_ident(PLACE_TABLE)} y
          ON y.id = p.added_stop_uetds_yer_id
        """
    )


def create_relevant_routes(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE relevant_routes AS
        SELECT base_stop_addition_canonical_route_id AS route_id FROM pair_features
        UNION
        SELECT variant_stop_addition_canonical_route_id AS route_id FROM pair_features
        UNION
        SELECT similar_stop_addition_canonical_route_id AS route_id FROM similarity_static
        """
    )


def create_daily_aggregates(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE route_company_time_daily AS
        SELECT
            h.stop_addition_canonical_route_id AS route_id,
            h.FIRMA_ID AS firma_id,
            h.iso_weekday,
            h.half_hour_bucket,
            h.SEFER_TARIHI AS history_date,
            COUNT(*)::BIGINT AS trip_count,
            SUM(h.SEFER_SAYISI)::DOUBLE AS demand_sum,
            SUM(h.SEFER_SAYISI * h.SEFER_SAYISI)::DOUBLE AS demand_sum_sq
        FROM mapped_history h
        JOIN relevant_routes r
          ON r.route_id = h.stop_addition_canonical_route_id
        GROUP BY 1, 2, 3, 4, 5
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE route_time_daily AS
        SELECT
            route_id,
            iso_weekday,
            half_hour_bucket,
            history_date,
            SUM(trip_count)::BIGINT AS trip_count,
            SUM(demand_sum)::DOUBLE AS demand_sum,
            SUM(demand_sum_sq)::DOUBLE AS demand_sum_sq
        FROM route_company_time_daily
        GROUP BY 1, 2, 3, 4
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE route_company_daily AS
        SELECT
            route_id,
            firma_id,
            history_date,
            SUM(trip_count)::BIGINT AS trip_count,
            SUM(demand_sum)::DOUBLE AS demand_sum,
            SUM(demand_sum_sq)::DOUBLE AS demand_sum_sq
        FROM route_company_time_daily
        GROUP BY 1, 2, 3
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE route_daily AS
        SELECT
            route_id,
            history_date,
            SUM(trip_count)::BIGINT AS trip_count,
            SUM(demand_sum)::DOUBLE AS demand_sum,
            SUM(demand_sum_sq)::DOUBLE AS demand_sum_sq
        FROM route_company_time_daily
        GROUP BY 1, 2
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE company_time_daily AS
        SELECT
            FIRMA_ID AS firma_id,
            iso_weekday,
            half_hour_bucket,
            SEFER_TARIHI AS history_date,
            COUNT(*)::BIGINT AS trip_count,
            SUM(SEFER_SAYISI)::DOUBLE AS demand_sum,
            SUM(SEFER_SAYISI * SEFER_SAYISI)::DOUBLE AS demand_sum_sq
        FROM mapped_history
        GROUP BY 1, 2, 3, 4
        """
    )
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE company_daily AS
        SELECT
            firma_id,
            history_date,
            SUM(trip_count)::BIGINT AS trip_count,
            SUM(demand_sum)::DOUBLE AS demand_sum,
            SUM(demand_sum_sq)::DOUBLE AS demand_sum_sq
        FROM company_time_daily
        GROUP BY 1, 2
        """
    )


def _partition_columns(alias_names: Iterable[str]) -> str:
    return ", ".join(alias_names)


def create_prior_feature_table(
    con: duckdb.DuckDBPyConnection,
    *,
    output_table: str,
    target_table: str,
    target_id_column: str,
    history_table: str,
    key_pairs: list[tuple[str, str]],
    prefix: str,
    recent_windows: tuple[int, ...] = RECENT_WINDOWS,
) -> None:
    key_aliases = [f"k{i}" for i in range(len(key_pairs))]
    history_keys = ",\n                ".join(
        f"h.{quote_ident(history_column)} AS {alias}"
        for alias, (_, history_column) in zip(key_aliases, key_pairs)
    )
    target_keys = ",\n                ".join(
        f"t.{quote_ident(target_column)} AS {alias}"
        for alias, (target_column, _) in zip(key_aliases, key_pairs)
    )
    partition = _partition_columns(key_aliases)

    recent_expressions = []
    recent_output = []
    for days in recent_windows:
        recent_expressions.extend(
            [
                f"SUM(trip_count) OVER (PARTITION BY {partition} ORDER BY event_ts "
                f"RANGE BETWEEN INTERVAL '{days} days' PRECEDING "
                "AND CURRENT ROW) "
                f"AS recent_{days}_count",
                f"SUM(demand_sum) OVER (PARTITION BY {partition} ORDER BY event_ts "
                f"RANGE BETWEEN INTERVAL '{days} days' PRECEDING "
                "AND CURRENT ROW) "
                f"AS recent_{days}_sum",
            ]
        )
        recent_output.extend(
            [
                f"COALESCE(recent_{days}_count, 0)::BIGINT "
                f"AS {prefix}_recent_{days}_trip_count",
                f"CASE WHEN recent_{days}_count > 0 "
                f"THEN recent_{days}_sum / recent_{days}_count END "
                f"AS {prefix}_recent_{days}_mean_demand",
            ]
        )

    recent_sql = ",\n                ".join(recent_expressions)
    if recent_sql:
        recent_sql = ",\n                " + recent_sql

    output_sql = ",\n            ".join(recent_output)
    if output_sql:
        output_sql = ",\n            " + output_sql

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE {quote_ident(output_table)} AS
        WITH events AS (
            SELECT
                {history_keys},
                CAST(h.history_date AS TIMESTAMP) + INTERVAL '12 hours' AS event_ts,
                NULL::BIGINT AS target_id,
                h.trip_count,
                h.demand_sum,
                h.demand_sum_sq
            FROM {quote_ident(history_table)} h

            UNION ALL

            SELECT
                {target_keys},
                CAST(t.target_date AS TIMESTAMP) AS event_ts,
                t.{quote_ident(target_id_column)} AS target_id,
                0::BIGINT AS trip_count,
                0::DOUBLE AS demand_sum,
                0::DOUBLE AS demand_sum_sq
            FROM {quote_ident(target_table)} t
        ),
        windowed AS (
            SELECT
                target_id,
                SUM(trip_count) OVER (
                    PARTITION BY {partition}
                    ORDER BY event_ts
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS prior_count,
                SUM(demand_sum) OVER (
                    PARTITION BY {partition}
                    ORDER BY event_ts
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS prior_sum,
                SUM(demand_sum_sq) OVER (
                    PARTITION BY {partition}
                    ORDER BY event_ts
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS prior_sum_sq
                {recent_sql}
            FROM events
        )
        SELECT
            target_id AS {quote_ident(target_id_column)},
            COALESCE(prior_count, 0)::BIGINT AS {prefix}_prior_trip_count,
            CASE WHEN prior_count > 0 THEN prior_sum / prior_count END
                AS {prefix}_prior_mean_demand,
            CASE
                WHEN prior_count > 1 THEN SQRT(GREATEST(
                    prior_sum_sq / prior_count
                    - (prior_sum / prior_count) * (prior_sum / prior_count),
                    0
                ))
            END AS {prefix}_prior_std_demand
            {output_sql}
        FROM windowed
        WHERE target_id IS NOT NULL
        """
    )


def create_all_exact_and_company_features(con: duckdb.DuckDBPyConnection) -> None:
    definitions = [
        (
            "feat_base_company_time",
            "route_company_time_daily",
            [
                ("base_stop_addition_canonical_route_id", "route_id"),
                ("FIRMA_ID", "firma_id"),
                ("iso_weekday", "iso_weekday"),
                ("half_hour_bucket", "half_hour_bucket"),
            ],
            "base_same_company_time",
        ),
        (
            "feat_base_route_time",
            "route_time_daily",
            [
                ("base_stop_addition_canonical_route_id", "route_id"),
                ("iso_weekday", "iso_weekday"),
                ("half_hour_bucket", "half_hour_bucket"),
            ],
            "base_all_company_time",
        ),
        (
            "feat_base_company",
            "route_company_daily",
            [
                ("base_stop_addition_canonical_route_id", "route_id"),
                ("FIRMA_ID", "firma_id"),
            ],
            "base_same_company",
        ),
        (
            "feat_base_route",
            "route_daily",
            [("base_stop_addition_canonical_route_id", "route_id")],
            "base_all_company",
        ),
        (
            "feat_variant_company_time",
            "route_company_time_daily",
            [
                ("variant_stop_addition_canonical_route_id", "route_id"),
                ("FIRMA_ID", "firma_id"),
                ("iso_weekday", "iso_weekday"),
                ("half_hour_bucket", "half_hour_bucket"),
            ],
            "proposed_same_company_time",
        ),
        (
            "feat_variant_route_time",
            "route_time_daily",
            [
                ("variant_stop_addition_canonical_route_id", "route_id"),
                ("iso_weekday", "iso_weekday"),
                ("half_hour_bucket", "half_hour_bucket"),
            ],
            "proposed_all_company_time",
        ),
        (
            "feat_variant_company",
            "route_company_daily",
            [
                ("variant_stop_addition_canonical_route_id", "route_id"),
                ("FIRMA_ID", "firma_id"),
            ],
            "proposed_same_company",
        ),
        (
            "feat_variant_route",
            "route_daily",
            [("variant_stop_addition_canonical_route_id", "route_id")],
            "proposed_all_company",
        ),
        (
            "feat_company_time",
            "company_time_daily",
            [
                ("FIRMA_ID", "firma_id"),
                ("iso_weekday", "iso_weekday"),
                ("half_hour_bucket", "half_hour_bucket"),
            ],
            "company_time",
        ),
        (
            "feat_company",
            "company_daily",
            [("FIRMA_ID", "firma_id")],
            "company",
        ),
    ]

    for output_table, history_table, key_pairs, prefix in definitions:
        create_prior_feature_table(
            con,
            output_table=output_table,
            target_table="target_rows",
            target_id_column="training_row_id",
            history_table=history_table,
            key_pairs=key_pairs,
            prefix=prefix,
        )


def create_similar_route_features(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE target_similarity AS
        SELECT
            ROW_NUMBER() OVER (
                ORDER BY t.training_row_id, s.similarity_rank,
                         s.similar_stop_addition_canonical_route_id
            ) AS similar_target_id,
            t.training_row_id,
            t.target_date,
            t.FIRMA_ID,
            t.iso_weekday,
            t.half_hour_bucket,
            s.similar_stop_addition_canonical_route_id,
            s.ordered_sequence_similarity,
            s.stop_set_jaccard,
            s.stop_count_difference,
            s.route_length_similarity,
            s.corridor_mean_nearest_stop_km,
            s.corridor_similarity,
            s.similarity_score,
            s.similarity_rank
        FROM target_rows t
        JOIN similarity_static s USING (pair_id)
        """
    )

    create_prior_feature_table(
        con,
        output_table="feat_similar_route_time_candidate",
        target_table="target_similarity",
        target_id_column="similar_target_id",
        history_table="route_time_daily",
        key_pairs=[
            ("similar_stop_addition_canonical_route_id", "route_id"),
            ("iso_weekday", "iso_weekday"),
            ("half_hour_bucket", "half_hour_bucket"),
        ],
        prefix="candidate_all_company_time",
    )
    create_prior_feature_table(
        con,
        output_table="feat_similar_company_time_candidate",
        target_table="target_similarity",
        target_id_column="similar_target_id",
        history_table="route_company_time_daily",
        key_pairs=[
            ("similar_stop_addition_canonical_route_id", "route_id"),
            ("FIRMA_ID", "firma_id"),
            ("iso_weekday", "iso_weekday"),
            ("half_hour_bucket", "half_hour_bucket"),
        ],
        prefix="candidate_same_company_time",
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE similar_features AS
        WITH candidates AS (
            SELECT
                ts.*,
                ar.candidate_all_company_time_prior_trip_count,
                ar.candidate_all_company_time_prior_mean_demand,
                ar.candidate_all_company_time_recent_90_trip_count,
                ar.candidate_all_company_time_recent_90_mean_demand,
                cr.candidate_same_company_time_prior_trip_count,
                cr.candidate_same_company_time_prior_mean_demand,
                cr.candidate_same_company_time_recent_90_trip_count,
                cr.candidate_same_company_time_recent_90_mean_demand,
                ts.similarity_score
                    * LN(1 + ar.candidate_all_company_time_prior_trip_count)
                    AS all_company_weight,
                ts.similarity_score
                    * LN(1 + cr.candidate_same_company_time_prior_trip_count)
                    AS same_company_weight,
                ts.similarity_score
                    * LN(1 + ar.candidate_all_company_time_recent_90_trip_count)
                    AS all_company_recent_90_weight,
                ts.similarity_score
                    * LN(1 + cr.candidate_same_company_time_recent_90_trip_count)
                    AS same_company_recent_90_weight
            FROM target_similarity ts
            LEFT JOIN feat_similar_route_time_candidate ar USING (similar_target_id)
            LEFT JOIN feat_similar_company_time_candidate cr USING (similar_target_id)
        )
        SELECT
            training_row_id,
            COUNT(*)::INTEGER AS similar_candidate_count,
            (COUNT(*) FILTER (
                WHERE candidate_all_company_time_prior_trip_count > 0
            ))::INTEGER AS similar_routes_with_prior_history_count,
            (COUNT(*) FILTER (
                WHERE candidate_same_company_time_prior_trip_count > 0
            ))::INTEGER AS same_company_similar_routes_with_prior_history_count,
            COALESCE(SUM(candidate_all_company_time_prior_trip_count), 0)::BIGINT
                AS similar_all_company_prior_trip_count,
            COALESCE(SUM(candidate_same_company_time_prior_trip_count), 0)::BIGINT
                AS similar_same_company_prior_trip_count,
            MAX(similarity_score) FILTER (
                WHERE candidate_all_company_time_prior_trip_count > 0
            ) AS best_similarity_with_prior_history,
            SUM(
                candidate_all_company_time_prior_mean_demand * all_company_weight
            ) FILTER (
                WHERE candidate_all_company_time_prior_trip_count > 0
            ) / NULLIF(SUM(all_company_weight) FILTER (
                WHERE candidate_all_company_time_prior_trip_count > 0
            ), 0) AS similar_all_company_weighted_mean_demand,
            SUM(
                candidate_same_company_time_prior_mean_demand * same_company_weight
            ) FILTER (
                WHERE candidate_same_company_time_prior_trip_count > 0
            ) / NULLIF(SUM(same_company_weight) FILTER (
                WHERE candidate_same_company_time_prior_trip_count > 0
            ), 0) AS similar_same_company_weighted_mean_demand,
            SUM(
                candidate_all_company_time_recent_90_mean_demand
                * all_company_recent_90_weight
            ) FILTER (
                WHERE candidate_all_company_time_recent_90_trip_count > 0
            ) / NULLIF(SUM(all_company_recent_90_weight) FILTER (
                WHERE candidate_all_company_time_recent_90_trip_count > 0
            ), 0) AS similar_all_company_recent_90_weighted_mean_demand,
            SUM(
                candidate_same_company_time_recent_90_mean_demand
                * same_company_recent_90_weight
            ) FILTER (
                WHERE candidate_same_company_time_recent_90_trip_count > 0
            ) / NULLIF(SUM(same_company_recent_90_weight) FILTER (
                WHERE candidate_same_company_time_recent_90_trip_count > 0
            ), 0) AS similar_same_company_recent_90_weighted_mean_demand
        FROM candidates
        GROUP BY training_row_id
        """
    )


def create_final_training_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE stop_addition_training_data AS
        SELECT
            t.*,
            f_base_ct.* EXCLUDE (training_row_id),
            f_base_rt.* EXCLUDE (training_row_id),
            f_base_c.* EXCLUDE (training_row_id),
            f_base_r.* EXCLUDE (training_row_id),
            f_variant_ct.* EXCLUDE (training_row_id),
            f_variant_rt.* EXCLUDE (training_row_id),
            f_variant_c.* EXCLUDE (training_row_id),
            f_variant_r.* EXCLUDE (training_row_id),
            f_company_t.* EXCLUDE (training_row_id),
            f_company.* EXCLUDE (training_row_id),
            sf.* EXCLUDE (training_row_id),

            CASE
                WHEN f_base_ct.base_same_company_time_prior_trip_count >= 10
                    THEN f_base_ct.base_same_company_time_prior_mean_demand
                WHEN f_base_rt.base_all_company_time_prior_trip_count >= 10
                    THEN f_base_rt.base_all_company_time_prior_mean_demand
                WHEN f_base_c.base_same_company_prior_trip_count >= 10
                    THEN f_base_c.base_same_company_prior_mean_demand
                WHEN f_base_r.base_all_company_prior_trip_count > 0
                    THEN f_base_r.base_all_company_prior_mean_demand
                ELSE f_company.company_prior_mean_demand
            END AS current_route_expected_demand_proxy,
            CASE
                WHEN f_base_ct.base_same_company_time_prior_trip_count >= 10
                    THEN 'BASE_SAME_COMPANY_WEEKDAY_TIME'
                WHEN f_base_rt.base_all_company_time_prior_trip_count >= 10
                    THEN 'BASE_ALL_COMPANY_WEEKDAY_TIME'
                WHEN f_base_c.base_same_company_prior_trip_count >= 10
                    THEN 'BASE_SAME_COMPANY'
                WHEN f_base_r.base_all_company_prior_trip_count > 0
                    THEN 'BASE_ALL_COMPANY'
                WHEN f_company.company_prior_trip_count > 0
                    THEN 'COMPANY_GLOBAL'
                ELSE 'NO_PRIOR_EVIDENCE'
            END AS current_route_proxy_source,

            CASE
                WHEN f_variant_ct.proposed_same_company_time_prior_trip_count >= 10
                    THEN f_variant_ct.proposed_same_company_time_prior_mean_demand
                WHEN f_variant_rt.proposed_all_company_time_prior_trip_count >= 10
                    THEN f_variant_rt.proposed_all_company_time_prior_mean_demand
                WHEN sf.similar_same_company_prior_trip_count >= 10
                    THEN sf.similar_same_company_weighted_mean_demand
                WHEN sf.similar_all_company_prior_trip_count > 0
                    THEN sf.similar_all_company_weighted_mean_demand
                WHEN f_base_ct.base_same_company_time_prior_trip_count >= 10
                    THEN f_base_ct.base_same_company_time_prior_mean_demand
                WHEN f_base_rt.base_all_company_time_prior_trip_count > 0
                    THEN f_base_rt.base_all_company_time_prior_mean_demand
                ELSE f_company.company_prior_mean_demand
            END AS proposed_route_hierarchical_baseline,
            CASE
                WHEN f_variant_ct.proposed_same_company_time_prior_trip_count >= 10
                    THEN 'PROPOSED_SAME_COMPANY_EXACT_WEEKDAY_TIME'
                WHEN f_variant_rt.proposed_all_company_time_prior_trip_count >= 10
                    THEN 'PROPOSED_ALL_COMPANY_EXACT_WEEKDAY_TIME'
                WHEN sf.similar_same_company_prior_trip_count >= 10
                    THEN 'SAME_COMPANY_SIMILAR_ROUTES'
                WHEN sf.similar_all_company_prior_trip_count > 0
                    THEN 'ALL_COMPANY_SIMILAR_ROUTES'
                WHEN f_base_ct.base_same_company_time_prior_trip_count >= 10
                    THEN 'CURRENT_ROUTE_SAME_COMPANY'
                WHEN f_base_rt.base_all_company_time_prior_trip_count > 0
                    THEN 'CURRENT_ROUTE_ALL_COMPANY'
                WHEN f_company.company_prior_trip_count > 0
                    THEN 'COMPANY_GLOBAL'
                ELSE 'NO_PRIOR_EVIDENCE'
            END AS proposed_route_baseline_source,
            CASE
                WHEN f_variant_ct.proposed_same_company_time_prior_trip_count >= 10
                    THEN 'HIGH'
                WHEN f_variant_rt.proposed_all_company_time_prior_trip_count >= 30
                    THEN 'MEDIUM_HIGH'
                WHEN sf.similar_same_company_prior_trip_count >= 10
                    THEN 'MEDIUM'
                WHEN sf.similar_all_company_prior_trip_count >= 30
                    THEN 'MEDIUM_LOW'
                ELSE 'LOW'
            END AS historical_evidence_level,
            f_base_ct.base_same_company_time_prior_trip_count > 0
                AS has_same_company_current_route_history,
            f_variant_ct.proposed_same_company_time_prior_trip_count > 0
                AS has_same_company_exact_proposed_history,
            f_variant_rt.proposed_all_company_time_prior_trip_count > 0
                AS has_any_exact_proposed_history,
            COALESCE(sf.similar_routes_with_prior_history_count, 0) > 0
                AS has_similar_route_history,
            f_variant_rt.proposed_all_company_time_prior_trip_count = 0
                AS is_exact_proposed_route_cold_start
        FROM target_rows t
        LEFT JOIN feat_base_company_time f_base_ct USING (training_row_id)
        LEFT JOIN feat_base_route_time f_base_rt USING (training_row_id)
        LEFT JOIN feat_base_company f_base_c USING (training_row_id)
        LEFT JOIN feat_base_route f_base_r USING (training_row_id)
        LEFT JOIN feat_variant_company_time f_variant_ct USING (training_row_id)
        LEFT JOIN feat_variant_route_time f_variant_rt USING (training_row_id)
        LEFT JOIN feat_variant_company f_variant_c USING (training_row_id)
        LEFT JOIN feat_variant_route f_variant_r USING (training_row_id)
        LEFT JOIN feat_company_time f_company_t USING (training_row_id)
        LEFT JOIN feat_company f_company USING (training_row_id)
        LEFT JOIN similar_features sf USING (training_row_id)
        """
    )


def recommended_feature_metadata(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    columns = [str(row[0]) for row in con.execute("DESCRIBE stop_addition_training_data").fetchall()]

    excluded = {
        "training_row_id",
        "pair_id",
        "SEFER_ID",
        "target_date",
        "target_time",
        "target_variant_guzergah_kodu",
        "base_stop_addition_canonical_route_id",
        "variant_stop_addition_canonical_route_id",
        "data_split",
        "target_passenger_count",
        "current_route_proxy_source",
        "proposed_route_baseline_source",
        "historical_evidence_level",
        # The hard rule is applied outside the model. Keep the flag for reporting,
        # but do not let it influence the demand model by default.
        "is_company_origin_city",
        "company_origin_il_id",
        "observed_index_is_min_haversine_index",
    }
    recommended = [column for column in columns if column not in excluded]
    categorical_candidates = {
        "FIRMA_ID",
        "added_stop_uetds_yer_id",
        "added_stop_il_id",
        "origin_stop_id",
        "destination_stop_id",
        "iso_weekday",
        "month",
        "half_hour_bucket",
    }
    categorical = [column for column in recommended if column in categorical_candidates]

    return {
        "target_column": "target_passenger_count",
        "split_column": "data_split",
        "recommended_feature_columns": recommended,
        "recommended_categorical_columns": categorical,
        "metadata_columns_not_for_model": sorted(excluded - {"target_passenger_count", "data_split"}),
        "notes": [
            "Historical features use trips from dates strictly before target_date.",
            "Old canonical_guzergah_id is not used.",
            "Route IDs are retained as metadata but excluded from recommended model features to reduce memorisation and improve unseen-route generalisation.",
            "is_company_origin_city is retained for the external hard-approval rule and excluded from the demand model by default.",
            "Retrospective analysis columns such as all-history uplift and support_level are intentionally not imported.",
        ],
    }


def build_summary(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    result = con.execute(
        """
        SELECT
            COUNT(*) AS training_rows,
            COUNT(DISTINCT pair_id) AS unique_pairs,
            COUNT(DISTINCT variant_stop_addition_canonical_route_id) AS unique_variant_routes,
            COUNT(DISTINCT FIRMA_ID) AS unique_companies,
            MIN(target_date) AS min_target_date,
            MAX(target_date) AS max_target_date,
            COUNT(*) FILTER (WHERE is_company_origin_city) AS hard_approval_rows,
            COUNT(*) FILTER (
                WHERE base_same_company_time_prior_trip_count > 0
            ) AS rows_with_same_company_current_route_history,
            COUNT(*) FILTER (
                WHERE proposed_same_company_time_prior_trip_count > 0
            ) AS rows_with_same_company_exact_proposed_history,
            COUNT(*) FILTER (
                WHERE proposed_all_company_time_prior_trip_count > 0
            ) AS rows_with_any_exact_proposed_history,
            COUNT(*) FILTER (
                WHERE similar_routes_with_prior_history_count > 0
            ) AS rows_with_similar_route_history,
            AVG(target_passenger_count) AS target_mean,
            MEDIAN(target_passenger_count) AS target_median
        FROM stop_addition_training_data
        """
    )
    scalar = result.fetchone()
    names = [desc[0] for desc in result.description]
    summary = dict(zip(names, scalar))

    summary["split_counts"] = {
        str(split): int(count)
        for split, count in con.execute(
            """
            SELECT data_split, COUNT(*)
            FROM stop_addition_training_data
            GROUP BY data_split
            ORDER BY data_split
            """
        ).fetchall()
    }
    summary["baseline_source_counts"] = {
        str(source): int(count)
        for source, count in con.execute(
            """
            SELECT proposed_route_baseline_source, COUNT(*)
            FROM stop_addition_training_data
            GROUP BY proposed_route_baseline_source
            ORDER BY COUNT(*) DESC
            """
        ).fetchall()
    }
    summary["evidence_level_counts"] = {
        str(level): int(count)
        for level, count in con.execute(
            """
            SELECT historical_evidence_level, COUNT(*)
            FROM stop_addition_training_data
            GROUP BY historical_evidence_level
            ORDER BY COUNT(*) DESC
            """
        ).fetchall()
    }
    summary["methodology"] = {
        "target": "SEFER_SAYISI of a real historical trip on a one-stop variant route",
        "trip_mapping": (
            "seferler_model joined to guzergah_canonical_stop_addition by "
            "GUZERGAH_KODU + FIRMA_ID"
        ),
        "canonical_id": "stop_addition_canonical_route_id",
        "old_canonical_guzergah_id_used": False,
        "leakage_rule": "Only history_date < target_date contributes to historical features.",
        "similar_route_weight": "similarity_score * ln(1 + prior_trip_count)",
        "hard_rule": (
            "is_company_origin_city is exported for unconditional approval but is "
            "not a recommended demand-model feature"
        ),
    }
    return summary


def json_default(value: object) -> object:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def export_outputs(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
    sample_rows: int,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "stop_addition_training_data.parquet"
    sample_path = output_dir / "stop_addition_training_data_sample.csv"
    summary_path = output_dir / "stop_addition_training_data_summary.json"
    metadata_path = output_dir / "stop_addition_training_features.json"

    con.execute(
        f"""
        COPY (
            SELECT *
            FROM stop_addition_training_data
            ORDER BY target_date, target_time, SEFER_ID, pair_id
        ) TO {sql_string(parquet_path.resolve())}
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT *
            FROM stop_addition_training_data
            ORDER BY target_date, target_time, SEFER_ID, pair_id
            LIMIT {max(0, int(sample_rows))}
        ) TO {sql_string(sample_path.resolve())}
        (HEADER, DELIMITER ',')
        """
    )

    summary = build_summary(con)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    metadata = recommended_feature_metadata(con)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )
    return parquet_path, sample_path, summary_path, metadata_path


def main() -> None:
    args = parse_args()
    db_path = Path(args.db).resolve()
    pair_path = Path(args.pair_features).resolve()
    similarity_path = Path(args.similarity_evidence).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not db_path.exists():
        raise SystemExit(f"DuckDB database not found: {db_path}")
    if not pair_path.exists():
        raise SystemExit(f"Pair geography CSV not found: {pair_path}")
    if not similarity_path.exists():
        raise SystemExit(f"Similarity evidence CSV not found: {similarity_path}")
    if not 0 <= args.min_similarity <= 1:
        raise SystemExit("--min-similarity must be between 0 and 1.")
    if args.top_k_similar < 1:
        raise SystemExit("--top-k-similar must be at least 1.")

    pairs = load_pair_features(pair_path)
    similarity = load_similarity(
        similarity_path,
        min_similarity=args.min_similarity,
        top_k=args.top_k_similar,
    )
    unknown_pair_ids = sorted(set(similarity["pair_id"]) - set(pairs["pair_id"]))
    if unknown_pair_ids:
        raise SystemExit(
            "Similarity evidence contains pair_id values absent from pair geography: "
            f"{unknown_pair_ids[:10]}"
        )

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        require_schema(con, args.trip_table)
        validate_route_mapping_uniqueness(con)
        register_static_inputs(con, pairs, similarity)

        print("Building mapped stop-addition trip history...")
        create_mapped_history(con, args.trip_table)

        print("Creating target variant-route trips...")
        create_target_rows(
            con,
            train_end=args.train_end,
            validation_end=args.validation_end,
            test_end=args.test_end,
        )
        target_count = con.execute("SELECT COUNT(*) FROM target_rows").fetchone()[0]
        if target_count == 0:
            raise RuntimeError(
                "No target rows were created. Check that variant canonical route IDs "
                "from the pair CSV exist in the clean route mapping and trip history."
            )

        print("Aggregating prior route and company history...")
        create_relevant_routes(con)
        create_daily_aggregates(con)
        create_all_exact_and_company_features(con)

        print("Building leakage-safe similar-route features...")
        create_similar_route_features(con)

        print("Assembling final training table...")
        create_final_training_table(con)

        parquet_path, sample_path, summary_path, metadata_path = export_outputs(
            con,
            output_dir,
            sample_rows=args.sample_rows,
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        print("\nStop-addition training-data build")
        print("=" * 39)
        print(f"Training rows: {summary['training_rows']:,}")
        print(f"Unique route pairs: {summary['unique_pairs']:,}")
        print(f"Unique variant routes: {summary['unique_variant_routes']:,}")
        print(f"Unique companies: {summary['unique_companies']:,}")
        print(
            "Rows with same-company current-route history: "
            f"{summary['rows_with_same_company_current_route_history']:,}"
        )
        print(
            "Rows with exact proposed-route history: "
            f"{summary['rows_with_any_exact_proposed_history']:,}"
        )
        print(
            "Rows with similar-route history: "
            f"{summary['rows_with_similar_route_history']:,}"
        )
        print(f"\nParquet: {parquet_path}")
        print(f"Sample CSV: {sample_path}")
        print(f"Summary: {summary_path}")
        print(f"Feature metadata: {metadata_path}")
    finally:
        con.close()

