"""Batch-only DuckDB feature generation for proposed-trip inference."""

from __future__ import annotations

from datetime import date
from time import perf_counter
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from inference import features as v43


HISTORY_START = "2023-01-01"
INPUT_COLUMNS = ["FIRMA_ID", "GUZERGAH_KODU", "SEFER_TARIHI", "SEFER_SAATI"]


def _normalize(proposals: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in INPUT_COLUMNS if column not in proposals]
    if missing:
        raise ValueError(f"Missing required proposal column(s): {missing}")
    if proposals.empty:
        raise ValueError("No proposed trips were supplied")
    frame = proposals.copy()
    frame["_proposal_order"] = np.arange(len(frame), dtype=np.int64)
    for column in ("FIRMA_ID", "GUZERGAH_KODU"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or np.any(numeric % 1 != 0):
            raise ValueError(f"{column} must contain integer identifiers")
        frame[column] = numeric.astype("int64")
    parsed_dates = pd.to_datetime(frame["SEFER_TARIHI"], format="%Y-%m-%d", errors="coerce")
    if parsed_dates.isna().any():
        bad = frame.loc[parsed_dates.isna(), "SEFER_TARIHI"].astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(f"SEFER_TARIHI must use YYYY-MM-DD; invalid value(s): {bad}")
    frame["SEFER_TARIHI"] = parsed_dates.dt.normalize()
    if frame["SEFER_TARIHI"].dt.date.min() <= date.fromisoformat(HISTORY_START):
        raise ValueError(f"Proposal dates must be later than {HISTORY_START}")
    time_text = frame["SEFER_SAATI"].astype(str).str.strip()
    parsed_times = pd.to_datetime(time_text, format="%H:%M", errors="coerce")
    has_seconds = time_text.str.fullmatch(r"\d{2}:\d{2}:\d{2}")
    if has_seconds.any():
        parsed_times.loc[has_seconds] = pd.to_datetime(
            time_text.loc[has_seconds], format="%H:%M:%S", errors="coerce"
        )
    if parsed_times.isna().any():
        bad = time_text[parsed_times.isna()].drop_duplicates().head(10).tolist()
        raise ValueError(f"SEFER_SAATI must use HH:MM or HH:MM:SS; invalid value(s): {bad}")
    frame["SEFER_SAATI"] = parsed_times.dt.strftime("%H:%M:%S")
    return frame


def _register_and_map(conn: duckdb.DuckDBPyConnection, frame: pd.DataFrame) -> None:
    """Register the complete proposal DataFrame exactly once."""

    conn.register("proposal_batch_input", frame)
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE proposal_route_mapping AS
        WITH proposal_route_keys AS (
            SELECT DISTINCT FIRMA_ID, GUZERGAH_KODU
            FROM proposal_batch_input
        ),
        historical_candidates AS (
            SELECT
                history.FIRMA_ID,
                history.GUZERGAH_KODU,
                history.canonical_guzergah_id,
                COUNT(*) AS historical_row_count,
                MAX(history.SEFER_TARIHI) AS latest_history_date
            FROM model_data_base AS history
            JOIN proposal_route_keys AS keys
              ON keys.FIRMA_ID = history.FIRMA_ID
             AND keys.GUZERGAH_KODU = history.GUZERGAH_KODU
            WHERE history.canonical_guzergah_id IS NOT NULL
            GROUP BY history.FIRMA_ID, history.GUZERGAH_KODU, history.canonical_guzergah_id
        ),
        historical_mapping AS (
            SELECT * EXCLUDE (historical_row_count, latest_history_date)
            FROM historical_candidates
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY FIRMA_ID, GUZERGAH_KODU
                ORDER BY historical_row_count DESC, latest_history_date DESC
            ) = 1
        ),
        current_mapping AS (
            SELECT firma_id, guzergah_kodu, canonical_guzergah_id
            FROM guzergah_canonical
        )
        SELECT
            keys.FIRMA_ID,
            keys.GUZERGAH_KODU,
            COALESCE(historical.canonical_guzergah_id, current.canonical_guzergah_id)::UBIGINT
                AS canonical_guzergah_id,
            CASE
                WHEN historical.canonical_guzergah_id IS NOT NULL THEN 'model_data_base_history'
                WHEN current.canonical_guzergah_id IS NOT NULL THEN 'guzergah_canonical'
            END AS canonical_mapping_source
        FROM proposal_route_keys AS keys
        LEFT JOIN historical_mapping AS historical USING (FIRMA_ID, GUZERGAH_KODU)
        LEFT JOIN current_mapping AS current
          ON current.firma_id = keys.FIRMA_ID
         AND current.guzergah_kodu = keys.GUZERGAH_KODU
    """)
    missing_routes = conn.execute("""
        SELECT FIRMA_ID, GUZERGAH_KODU
        FROM proposal_route_mapping
        WHERE canonical_guzergah_id IS NULL
    """).fetchdf()
    if not missing_routes.empty:
        raise ValueError(
            "Unknown company-route combination(s) or missing canonical route: "
            f"{missing_routes.to_dict('records')}"
        )
    duplicate_mappings = conn.execute("""
        SELECT FIRMA_ID, GUZERGAH_KODU, COUNT(*) AS mapping_count
        FROM proposal_route_mapping
        GROUP BY FIRMA_ID, GUZERGAH_KODU
        HAVING COUNT(*) <> 1
    """).fetchdf()
    if not duplicate_mappings.empty:
        raise ValueError(
            "Company-route mapping is not many-to-one: "
            f"{duplicate_mappings.to_dict('records')}"
        )
    earliest = frame["SEFER_TARIHI"].min().date().isoformat()
    history_count = conn.execute(
        "SELECT COUNT(*) FROM model_data_base WHERE SEFER_TARIHI >= ? AND SEFER_TARIHI < ?",
        [HISTORY_START, earliest],
    ).fetchone()[0]
    if not history_count:
        raise ValueError(f"No historical feature rows exist before {earliest}")
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE v4_3_target AS
        SELECT
            -(proposal._proposal_order + 1)::BIGINT AS SEFER_ID,
            CAST(proposal.SEFER_TARIHI AS DATE) AS SEFER_TARIHI,
            proposal.FIRMA_ID,
            proposal.GUZERGAH_KODU,
            mapping.canonical_guzergah_id,
            0::BIGINT AS target,
            MONTH(proposal.SEFER_TARIHI)::BIGINT AS month,
            WEEKOFYEAR(proposal.SEFER_TARIHI)::BIGINT AS week_of_year,
            ISODOW(proposal.SEFER_TARIHI)::BIGINT AS day_of_week,
            (HOUR(CAST(proposal.SEFER_SAATI AS TIME)) * 60
                + MINUTE(CAST(proposal.SEFER_SAATI AS TIME)))::BIGINT AS departure_minute,
            (HOUR(CAST(proposal.SEFER_SAATI AS TIME)) * 2
                + CASE WHEN MINUTE(CAST(proposal.SEFER_SAATI AS TIME)) >= 30
                    THEN 1 ELSE 0 END)::BIGINT AS departure_30min_bucket,
            proposal._proposal_order,
            proposal.SEFER_SAATI,
            mapping.canonical_mapping_source
        FROM proposal_batch_input AS proposal
        JOIN proposal_route_mapping AS mapping USING (FIRMA_ID, GUZERGAH_KODU)
    """)


def _global_statistics(
    conn: duckdb.DuckDBPyConnection,
    log: Any = None,
) -> None:
    _log = log or (lambda _msg: None)
    _log("Global statistics: building reference-date aggregates...")
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE v4_3_global_stats AS
        WITH reference_dates AS (
            SELECT DISTINCT SEFER_TARIHI AS reference_date FROM v4_3_target
        )
        SELECT
            reference.reference_date,
            AVG(history.target)::DOUBLE AS global_average,
            MEDIAN(history.target)::DOUBLE AS global_median,
            STDDEV_SAMP(history.target)::DOUBLE AS global_std,
            MAX(history.target)::DOUBLE AS global_maximum,
            QUANTILE_CONT(history.target, 0.10)::DOUBLE AS global_p10,
            QUANTILE_CONT(history.target, 0.25)::DOUBLE AS global_p25,
            QUANTILE_CONT(history.target, 0.75)::DOUBLE AS global_p75,
            QUANTILE_CONT(history.target, 0.90)::DOUBLE AS global_p90,
            AVG(CASE WHEN history.target < 10 THEN 1.0 ELSE 0.0 END)::DOUBLE AS global_below_10_rate,
            AVG(CASE WHEN history.target > 10 THEN 1.0 ELSE 0.0 END)::DOUBLE AS global_above_10_rate,
            AVG(CASE WHEN history.target > 20 THEN 1.0 ELSE 0.0 END)::DOUBLE AS global_above_20_rate,
            AVG(CASE WHEN history.target > 30 THEN 1.0 ELSE 0.0 END)::DOUBLE AS global_above_30_rate,
            AVG(CASE WHEN history.target > 40 THEN 1.0 ELSE 0.0 END)::DOUBLE AS global_above_40_rate,
            AVG(CASE WHEN history.target > 60 THEN 1.0 ELSE 0.0 END)::DOUBLE AS global_above_60_rate,
            AVG(CASE WHEN history.target > 100 THEN 1.0 ELSE 0.0 END)::DOUBLE AS global_above_100_rate
        FROM reference_dates AS reference
        LEFT JOIN model_data_base AS history
          ON history.SEFER_TARIHI >= DATE '{HISTORY_START}'
         AND history.SEFER_TARIHI < reference.reference_date
        GROUP BY reference.reference_date
    """)
    _log("Global statistics complete")


def _long_term_statistics(
    conn: duckdb.DuckDBPyConnection,
    log: Any = None,
) -> None:
    _log = log or (lambda _msg: None)
    _global_statistics(conn, log=_log)
    total = len(v43.HISTORICAL_GROUP_DEFINITIONS)
    for index, (prefix, group_columns) in enumerate(v43.HISTORICAL_GROUP_DEFINITIONS, start=1):
        _log(f"Long-term aggregation {index}/{total}: {prefix}...")
        reference_columns = ", ".join(f"target.{column}" for column in group_columns)
        grouped_columns = ", ".join(f"reference.{column}" for column in group_columns)
        join_sql = v43._join_condition("reference", "history", group_columns)
        extra_sql = ""
        if prefix == v43.DISTRIBUTION_PREFIX:
            extra_sql = f""",
                QUANTILE_CONT(history.target, 0.10)::DOUBLE AS {prefix}_p10,
                QUANTILE_CONT(history.target, 0.25)::DOUBLE AS {prefix}_p25,
                QUANTILE_CONT(history.target, 0.75)::DOUBLE AS {prefix}_p75,
                AVG(CASE WHEN history.target < 10 THEN 1.0 ELSE 0.0 END)::DOUBLE AS {prefix}_below_10_rate,
                AVG(CASE WHEN history.target > 10 THEN 1.0 ELSE 0.0 END)::DOUBLE AS {prefix}_above_10_rate,
                AVG(CASE WHEN history.target > 20 THEN 1.0 ELSE 0.0 END)::DOUBLE AS {prefix}_above_20_rate,
                AVG(CASE WHEN history.target > 30 THEN 1.0 ELSE 0.0 END)::DOUBLE AS {prefix}_above_30_rate,
                AVG(CASE WHEN history.target > 40 THEN 1.0 ELSE 0.0 END)::DOUBLE AS {prefix}_above_40_rate
            """
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE v4_3_{prefix}_stats AS
            WITH reference_keys AS (
                SELECT DISTINCT {reference_columns}, target.SEFER_TARIHI AS reference_date
                FROM v4_3_target AS target
            )
            SELECT
                {grouped_columns}, reference.reference_date,
                AVG(history.target)::DOUBLE AS {prefix}_average,
                MEDIAN(history.target)::DOUBLE AS {prefix}_median,
                STDDEV_SAMP(history.target)::DOUBLE AS {prefix}_std,
                MAX(history.target)::DOUBLE AS {prefix}_maximum,
                QUANTILE_CONT(history.target, 0.90)::DOUBLE AS {prefix}_p90,
                AVG(CASE WHEN history.target > 60 THEN 1.0 ELSE 0.0 END)::DOUBLE AS {prefix}_above_60_rate,
                AVG(CASE WHEN history.target > 100 THEN 1.0 ELSE 0.0 END)::DOUBLE AS {prefix}_above_100_rate,
                COUNT(history.target)::BIGINT AS {prefix}_count
                {extra_sql}
            FROM reference_keys AS reference
            LEFT JOIN model_data_base AS history
              ON {join_sql}
             AND history.SEFER_TARIHI >= DATE '{HISTORY_START}'
             AND history.SEFER_TARIHI < reference.reference_date
            GROUP BY {grouped_columns}, reference.reference_date
        """)


def _recent_statistics(
    conn: duckdb.DuckDBPyConnection,
    log: Any = None,
) -> None:
    _log = log or (lambda _msg: None)
    total = len(v43.RECENT_GROUP_DEFINITIONS)
    for index, (prefix, group_columns) in enumerate(v43.RECENT_GROUP_DEFINITIONS, start=1):
        _log(f"Recent aggregation {index}/{total}: {prefix}...")
        reference_columns = ", ".join(f"target.{column}" for column in group_columns)
        grouped_columns = ", ".join(f"reference.{column}" for column in group_columns)
        join_sql = v43._join_condition("reference", "history", group_columns)
        expressions: list[str] = []
        for window in v43.RECENT_WINDOWS:
            condition = (
                f"history.SEFER_TARIHI >= reference.reference_date "
                f"- INTERVAL '{window} days'"
            )
            expressions.extend([
                f"AVG(history.target) FILTER (WHERE {condition})::DOUBLE AS {prefix}_recent_{window}d_average",
                f"COUNT(history.target) FILTER (WHERE {condition})::BIGINT AS {prefix}_recent_{window}d_count",
            ])
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE v4_3_{prefix}_recent_stats AS
            WITH reference_keys AS (
                SELECT DISTINCT {reference_columns}, target.SEFER_TARIHI AS reference_date
                FROM v4_3_target AS target
            )
            SELECT
                {grouped_columns}, reference.reference_date, {', '.join(expressions)}
            FROM reference_keys AS reference
            LEFT JOIN model_data_base AS history
              ON {join_sql}
             AND history.SEFER_TARIHI >= reference.reference_date - INTERVAL '180 days'
             AND history.SEFER_TARIHI < reference.reference_date
            GROUP BY {grouped_columns}, reference.reference_date
        """)


def _feature_table(conn: duckdb.DuckDBPyConnection) -> None:
    select = [
        "base.SEFER_ID", "base.SEFER_TARIHI", "base.FIRMA_ID", "base.GUZERGAH_KODU",
        "base.canonical_guzergah_id", "base.target", "base.month", "base.week_of_year",
        "base.day_of_week", "base.departure_minute", "base.departure_30min_bucket",
        "base._proposal_order", "base.SEFER_SAATI", "base.canonical_mapping_source",
    ]
    joins: list[str] = []
    aliases: dict[str, str] = {}
    for index, (prefix, group_columns) in enumerate(v43.HISTORICAL_GROUP_DEFINITIONS):
        alias = f"long_term_{index}"
        aliases[prefix] = alias
        condition = v43._join_condition("base", alias, group_columns)
        condition += f" AND base.SEFER_TARIHI = {alias}.reference_date"
        joins.append(f"LEFT JOIN v4_3_{prefix}_stats AS {alias} ON {condition}")
        select.extend(v43._long_term_expressions(alias, prefix))
    for index, (prefix, group_columns) in enumerate(v43.RECENT_GROUP_DEFINITIONS):
        alias = f"recent_{index}"
        condition = v43._join_condition("base", alias, group_columns)
        condition += f" AND base.SEFER_TARIHI = {alias}.reference_date"
        joins.append(f"LEFT JOIN v4_3_{prefix}_recent_stats AS {alias} ON {condition}")
        fallback = f"COALESCE({aliases[prefix]}.{prefix}_average, global_stats.global_average)"
        for window in v43.RECENT_WINDOWS:
            select.extend([
                f"COALESCE({alias}.{prefix}_recent_{window}d_average, {fallback}) AS {prefix}_recent_{window}d_average",
                f"COALESCE({alias}.{prefix}_recent_{window}d_count, 0)::BIGINT AS {prefix}_recent_{window}d_count",
            ])
    specific = aliases["company_route_time_weekday"]
    canonical_time = aliases["canonical_route_time_weekday"]
    canonical = aliases["canonical_route"]
    select.extend([
        f"COALESCE({specific}.company_route_time_weekday_average, "
        f"{canonical_time}.canonical_route_time_weekday_average, "
        f"{canonical}.canonical_route_average, global_stats.global_average) AS baseline_prediction",
        f"CASE WHEN {specific}.company_route_time_weekday_average IS NOT NULL THEN 'company_route_time_weekday' "
        f"WHEN {canonical_time}.canonical_route_time_weekday_average IS NOT NULL THEN 'canonical_route_time_weekday' "
        f"WHEN {canonical}.canonical_route_average IS NOT NULL THEN 'canonical_route' "
        "ELSE 'overall_average' END AS baseline_source",
    ])
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE v4_3_features AS
        SELECT {', '.join(select)}
        FROM v4_3_target AS base
        JOIN v4_3_global_stats AS global_stats
          ON global_stats.reference_date = base.SEFER_TARIHI
        {' '.join(joins)}
    """)


def _direct_baseline(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE proposal_baseline_direct_stats AS
        SELECT
            proposal.SEFER_ID,
            COUNT(history.target)::BIGINT AS company_route_count,
            AVG(history.target)::DOUBLE AS company_route_mean,
            COUNT(history.target) FILTER (
                WHERE history.SEFER_SAATI = CAST(proposal.SEFER_SAATI AS TIME)
            )::BIGINT AS company_route_time_count,
            AVG(history.target) FILTER (
                WHERE history.SEFER_SAATI = CAST(proposal.SEFER_SAATI AS TIME)
            )::DOUBLE AS company_route_time_mean,
            COUNT(history.target) FILTER (
                WHERE history.SEFER_SAATI = CAST(proposal.SEFER_SAATI AS TIME)
                  AND history.day_of_week = proposal.day_of_week
            )::BIGINT AS company_route_time_weekday_count,
            AVG(history.target) FILTER (
                WHERE history.SEFER_SAATI = CAST(proposal.SEFER_SAATI AS TIME)
                  AND history.day_of_week = proposal.day_of_week
            )::DOUBLE AS company_route_time_weekday_mean
        FROM v4_3_target AS proposal
        LEFT JOIN model_data_base AS history
          ON history.FIRMA_ID = proposal.FIRMA_ID
         AND history.GUZERGAH_KODU = proposal.GUZERGAH_KODU
         AND history.SEFER_TARIHI >= DATE '{HISTORY_START}'
         AND history.SEFER_TARIHI < proposal.SEFER_TARIHI
        GROUP BY proposal.SEFER_ID
    """)


def _fetch(conn: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    columns = list(dict.fromkeys([
        "SEFER_ID", "SEFER_TARIHI", "FIRMA_ID", "GUZERGAH_KODU",
        *v43.FEATURE_COLUMNS, "baseline_prediction", "baseline_source",
    ]))
    frame = conn.execute(f"""
        SELECT
            {', '.join(f'features.{column}' for column in columns)},
            features._proposal_order, features.SEFER_SAATI, features.canonical_mapping_source,
            direct.company_route_time_weekday_count AS debug_company_route_time_weekday_count,
            direct.company_route_time_weekday_mean AS debug_company_route_time_weekday_mean,
            direct.company_route_time_count AS debug_company_route_time_count,
            direct.company_route_time_mean AS debug_company_route_time_mean,
            direct.company_route_count AS debug_company_route_count,
            direct.company_route_mean AS debug_company_route_mean,
            features.canonical_route_time_weekday_count AS debug_canonical_route_time_weekday_count,
            CASE WHEN features.canonical_route_time_weekday_count > 0
                THEN features.canonical_route_time_weekday_average END AS debug_canonical_route_time_weekday_mean,
            features.canonical_route_count AS debug_canonical_route_count,
            CASE WHEN features.canonical_route_count > 0
                THEN features.canonical_route_average END AS debug_canonical_route_mean,
            CASE WHEN features.canonical_route_time_weekday_count > 0
                THEN features.canonical_route_time_weekday_count ELSE features.canonical_route_count END::BIGINT
                AS canonical_fallback_count,
            CASE WHEN features.canonical_route_time_weekday_count > 0
                THEN features.canonical_route_time_weekday_average
                WHEN features.canonical_route_count > 0 THEN features.canonical_route_average END::DOUBLE
                AS canonical_fallback_mean,
            COALESCE(
                direct.company_route_time_weekday_mean, direct.company_route_time_mean,
                direct.company_route_mean,
                CASE WHEN features.canonical_route_time_weekday_count > 0
                    THEN features.canonical_route_time_weekday_average END,
                CASE WHEN features.canonical_route_count > 0 THEN features.canonical_route_average END,
                global_stats.global_average
            )::DOUBLE AS corrected_baseline_prediction,
            CASE
                WHEN direct.company_route_time_weekday_count > 0 THEN 'company_route_time_weekday'
                WHEN direct.company_route_time_count > 0 THEN 'company_route_time'
                WHEN direct.company_route_count > 0 THEN 'company_route'
                WHEN features.canonical_route_time_weekday_count > 0 THEN 'canonical_route_time_weekday'
                WHEN features.canonical_route_count > 0 THEN 'canonical_route'
                ELSE 'global'
            END AS corrected_baseline_source
        FROM v4_3_features AS features
        JOIN proposal_baseline_direct_stats AS direct USING (SEFER_ID)
        JOIN v4_3_global_stats AS global_stats
          ON global_stats.reference_date = features.SEFER_TARIHI
        ORDER BY features._proposal_order
    """).fetchdf()
    frame["previous_weekday_baseline_prediction"] = frame["baseline_prediction"]
    frame["previous_weekday_baseline_source"] = frame["baseline_source"].replace(
        {"overall_average": "global"}
    )
    frame["baseline_prediction"] = frame.pop("corrected_baseline_prediction")
    frame["baseline_source"] = frame.pop("corrected_baseline_source")
    if ((frame["debug_company_route_time_count"] > 0) & (frame["baseline_source"] == "global")).any():
        raise AssertionError("Exact company-route-time history exists but baseline source is global")
    v43.validate_feature_frame(frame)
    return frame


def build_feature_matrix(
    conn: duckdb.DuckDBPyConnection,
    proposals: pd.DataFrame,
    timings: dict[str, float],
    log: Any = None,
) -> pd.DataFrame:
    """Build the complete matrix with no per-proposal or per-date SQL loops."""

    _log = log or (lambda _msg: None)

    started = perf_counter()
    _log(f"Normalizing {proposals.shape[0]:,} proposals...")
    frame = _normalize(proposals)
    _log("Registering proposals and resolving canonical route mapping...")
    _register_and_map(conn, frame)
    timings["canonical_mapping"] = perf_counter() - started
    _log(f"Canonical mapping: {len(frame):,} proposals mapped ({timings['canonical_mapping']:.3f}s)")

    started = perf_counter()
    _long_term_statistics(conn, log=_log)
    timings["long_term_aggregations"] = perf_counter() - started
    _log(f"Long-term aggregations complete ({timings['long_term_aggregations']:.3f}s)")

    started = perf_counter()
    _recent_statistics(conn, log=_log)
    timings["recent_aggregations"] = perf_counter() - started
    _log(f"Recent aggregations complete ({timings['recent_aggregations']:.3f}s)")

    started = perf_counter()
    _log("Building feature table and joining historical statistics...")
    _feature_table(conn)
    _log("Computing direct company-route baselines...")
    _direct_baseline(conn)
    _log("Fetching final feature matrix...")
    result = _fetch(conn)
    timings["feature_joins"] = perf_counter() - started
    _log(f"Feature joins complete ({timings['feature_joins']:.3f}s)")
    return result