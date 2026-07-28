"""Generic DuckDB feature construction pipeline for CatBoost v4.x training.

All versions share the same structural steps:

1. Create source (history + target) temp tables
2. Build long-term aggregate statistics
3. Build rolling recent-window statistics
4. Assemble the final feature table via LEFT JOINs

Version-specific extensions (distribution features, holiday calendars, etc.)
are injected through ``extra_*`` SQL fragment parameters so each version's
``common.py`` stays self-contained.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import duckdb
import pandas as pd

from scripts.shared.constants import (
    CATEGORICAL_FEATURES,
    HISTORICAL_GROUP_DEFINITIONS,
    HISTORICAL_STATISTIC_SUFFIXES,
    RECENT_GROUP_DEFINITIONS,
    RECENT_WINDOWS,
    SOURCE_COLUMNS,
)
from scripts.shared.paths import DB_PATH, DUCKDB_THREADS, TEMP_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sql_identifier_list(columns: Sequence[str]) -> str:
    return ", ".join(columns)


def join_condition(
    base_alias: str,
    aggregate_alias: str,
    columns: Sequence[str],
) -> str:
    return " AND ".join(
        f"{base_alias}.{column} = {aggregate_alias}.{column}"
        for column in columns
    )


def long_term_expressions(alias: str, prefix: str) -> list[str]:
    """Return the standard 8 COALESCE expressions for *prefix*."""
    return [
        f"COALESCE({alias}.{prefix}_average, global_stats.global_average) AS {prefix}_average",
        f"COALESCE({alias}.{prefix}_median, global_stats.global_median) AS {prefix}_median",
        f"COALESCE({alias}.{prefix}_std, global_stats.global_std) AS {prefix}_std",
        f"COALESCE({alias}.{prefix}_maximum, global_stats.global_maximum) AS {prefix}_maximum",
        f"COALESCE({alias}.{prefix}_p90, global_stats.global_p90) AS {prefix}_p90",
        f"COALESCE({alias}.{prefix}_above_60_rate, global_stats.global_above_60_rate) AS {prefix}_above_60_rate",
        f"COALESCE({alias}.{prefix}_above_100_rate, global_stats.global_above_100_rate) AS {prefix}_above_100_rate",
        f"COALESCE({alias}.{prefix}_count, 0)::BIGINT AS {prefix}_count",
    ]


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect(*, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect(str(DB_PATH), read_only=read_only)
    temp_path = TEMP_DIR.as_posix().replace("'", "''")
    connection.execute(f"SET threads = {DUCKDB_THREADS}")
    connection.execute(f"SET temp_directory = '{temp_path}'")
    return connection


# ---------------------------------------------------------------------------
# Environment validation
# ---------------------------------------------------------------------------

def ensure_environment(
    *,
    require_model: bool = False,
    model_path: Path | None = None,
    extra_paths: Sequence[tuple[Path, str]] = (),
) -> None:
    """Validate that required filesystem artefacts exist.

    *extra_paths* is a sequence of ``(Path, description)`` tuples that are
    checked in addition to the database and (optionally) the model.
    """
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database not found: {DB_PATH}")
    if require_model and model_path is not None and not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    for path, description in extra_paths:
        if not path.exists():
            raise FileNotFoundError(f"{description}: {path}")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Source tables
# ---------------------------------------------------------------------------

def create_source_tables(
    conn: duckdb.DuckDBPyConnection,
    config: Any,
    *,
    table_prefix: str,
) -> None:
    """Create ``{prefix}_history`` and ``{prefix}_target`` temp tables.

    Always validates that the history period ends strictly before the target
    period starts (leakage guard).
    """
    columns = sql_identifier_list(SOURCE_COLUMNS)
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table_prefix}_history AS
        SELECT {columns}
        FROM model_data_base
        WHERE SEFER_TARIHI >= DATE '{config.history_start}'
          AND SEFER_TARIHI < DATE '{config.history_end_exclusive}'
    """)
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table_prefix}_target AS
        SELECT {columns}
        FROM model_data_base
        WHERE SEFER_TARIHI >= DATE '{config.target_start}'
          AND SEFER_TARIHI < DATE '{config.target_end_exclusive}'
    """)

    history_count = conn.execute(
        f"SELECT COUNT(*) FROM {table_prefix}_history"
    ).fetchone()[0]
    target_count = conn.execute(
        f"SELECT COUNT(*) FROM {table_prefix}_target"
    ).fetchone()[0]
    if not history_count or not target_count:
        raise RuntimeError("History or target period is empty.")

    latest_history = conn.execute(
        f"SELECT MAX(SEFER_TARIHI) FROM {table_prefix}_history"
    ).fetchone()[0]
    earliest_target = conn.execute(
        f"SELECT MIN(SEFER_TARIHI) FROM {table_prefix}_target"
    ).fetchone()[0]
    if latest_history >= earliest_target:
        raise RuntimeError(
            f"Leakage guard failed: history ends {latest_history}, "
            f"target starts {earliest_target}."
        )
    print(f"History rows: {history_count:,}; target rows: {target_count:,}")
    print(f"Leakage cutoff verified: {latest_history} < {earliest_target}")


# ---------------------------------------------------------------------------
# Long-term statistics
# ---------------------------------------------------------------------------

def create_long_term_statistics(
    conn: duckdb.DuckDBPyConnection,
    *,
    table_prefix: str,
    history_where_sql: str = "TRUE",
    extra_global_sql: str = "",
    extra_group_sqls: dict[str, str] | None = None,
) -> None:
    """Build global + per-group long-term aggregate tables.

    *extra_global_sql* is appended to the ``SELECT`` of the global stats
    query (e.g. additional quantiles for v4.3 distribution features).

    *extra_group_sqls* maps ``prefix → extra SQL`` that is appended to
    the per-group aggregation for that prefix only.
    """
    if extra_group_sqls is None:
        extra_group_sqls = {}

    global_extra = f",\n{extra_global_sql}" if extra_global_sql else ""
    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table_prefix}_global_stats AS
        SELECT
            AVG(target)::DOUBLE AS global_average,
            MEDIAN(target)::DOUBLE AS global_median,
            STDDEV_SAMP(target)::DOUBLE AS global_std,
            MAX(target)::DOUBLE AS global_maximum,
            QUANTILE_CONT(target, 0.90)::DOUBLE AS global_p90,
            AVG(CASE WHEN target > 60 THEN 1.0 ELSE 0.0 END)::DOUBLE
                AS global_above_60_rate,
            AVG(CASE WHEN target > 100 THEN 1.0 ELSE 0.0 END)::DOUBLE
                AS global_above_100_rate
            {global_extra}
        FROM {table_prefix}_history
        WHERE {history_where_sql}
    """)

    for prefix, group_columns in HISTORICAL_GROUP_DEFINITIONS:
        group_sql = sql_identifier_list(group_columns)
        extra = extra_group_sqls.get(prefix, "")
        extra_sql = f",\n{extra}" if extra else ""
        print(f"  Long-term aggregation: {prefix}")
        conn.execute(f"""
            CREATE OR REPLACE TEMP TABLE {table_prefix}_{prefix}_stats AS
            SELECT
                {group_sql},
                AVG(target)::DOUBLE AS {prefix}_average,
                MEDIAN(target)::DOUBLE AS {prefix}_median,
                STDDEV_SAMP(target)::DOUBLE AS {prefix}_std,
                MAX(target)::DOUBLE AS {prefix}_maximum,
                QUANTILE_CONT(target, 0.90)::DOUBLE AS {prefix}_p90,
                AVG(CASE WHEN target > 60 THEN 1.0 ELSE 0.0 END)::DOUBLE
                    AS {prefix}_above_60_rate,
                AVG(CASE WHEN target > 100 THEN 1.0 ELSE 0.0 END)::DOUBLE
                    AS {prefix}_above_100_rate,
                COUNT(*)::BIGINT AS {prefix}_count
                {extra_sql}
            FROM {table_prefix}_history
            WHERE {history_where_sql}
            GROUP BY {group_sql}
        """)


# ---------------------------------------------------------------------------
# Recent-window statistics
# ---------------------------------------------------------------------------

def create_recent_statistics(
    conn: duckdb.DuckDBPyConnection,
    config: Any,
    *,
    table_prefix: str,
    history_table: str,
) -> None:
    """Build rolling-window features using only strictly-earlier dates."""
    for prefix, group_columns in RECENT_GROUP_DEFINITIONS:
        reference_columns = ", ".join(
            f"source.{column}" for column in group_columns
        )
        grouped_columns = ", ".join(
            f"reference.{column}" for column in group_columns
        )
        join_sql = join_condition("reference", "history", group_columns)
        for window in RECENT_WINDOWS:
            print(f"  Recent aggregation: {prefix}, {window}d")
            conn.execute(f"""
                CREATE OR REPLACE TEMP TABLE
                    {table_prefix}_{prefix}_recent_{window}d_stats AS
                WITH reference_dates AS (
                    SELECT DISTINCT
                        {reference_columns},
                        source.SEFER_TARIHI AS reference_date
                    FROM {table_prefix}_target AS source
                )
                SELECT
                    {grouped_columns},
                    reference.reference_date,
                    AVG(history.target)::DOUBLE
                        AS {prefix}_recent_{window}d_average,
                    COUNT(history.target)::BIGINT
                        AS {prefix}_recent_{window}d_count
                FROM reference_dates AS reference
                LEFT JOIN {history_table} AS history
                    ON {join_sql}
                    AND history.SEFER_TARIHI
                        >= reference.reference_date - INTERVAL '{window} days'
                    AND history.SEFER_TARIHI
                        < reference.reference_date
                GROUP BY
                    {grouped_columns},
                    reference.reference_date
            """)


# ---------------------------------------------------------------------------
# Feature table assembly
# ---------------------------------------------------------------------------

def create_feature_table(
    conn: duckdb.DuckDBPyConnection,
    *,
    table_prefix: str,
    feature_columns: Sequence[str],
    extra_selects: Sequence[str] = (),
    extra_joins: Sequence[str] = (),
) -> None:
    """Assemble the final feature table by LEFT JOINing all aggregate tables.

    *extra_selects* are additional SQL expressions appended after the
    standard features (e.g. distribution COALESCEs, holiday columns).

    *extra_joins* are additional JOIN clauses appended after the standard
    aggregate joins (e.g. holiday calendar join).
    """
    select_expressions = [
        "base.SEFER_ID",
        "base.SEFER_TARIHI",
        "base.FIRMA_ID",
        "base.GUZERGAH_KODU",
        "base.canonical_guzergah_id",
        "base.target",
        "base.month",
        "base.week_of_year",
        "base.day_of_week",
        "base.departure_minute",
        "base.departure_30min_bucket",
    ]
    joins: list[str] = []
    aliases: dict[str, str] = {}

    # Long-term aggregate joins
    for index, (prefix, group_columns) in enumerate(
        HISTORICAL_GROUP_DEFINITIONS
    ):
        alias = f"long_term_{index}"
        aliases[prefix] = alias
        joins.append(
            f"LEFT JOIN {table_prefix}_{prefix}_stats AS {alias} "
            f"ON {join_condition('base', alias, group_columns)}"
        )
        select_expressions.extend(long_term_expressions(alias, prefix))

    # Recent-window joins
    recent_index = len(HISTORICAL_GROUP_DEFINITIONS)
    for prefix, group_columns in RECENT_GROUP_DEFINITIONS:
        fallback = (
            f"COALESCE({aliases[prefix]}.{prefix}_average, "
            "global_stats.global_average)"
        )
        for window in RECENT_WINDOWS:
            alias = f"recent_{recent_index}"
            join = join_condition("base", alias, group_columns)
            join += f" AND base.SEFER_TARIHI = {alias}.reference_date"
            joins.append(
                f"LEFT JOIN {table_prefix}_{prefix}_recent_{window}d_stats "
                f"AS {alias} ON {join}"
            )
            select_expressions.extend(
                [
                    f"COALESCE({alias}.{prefix}_recent_{window}d_average, "
                    f"{fallback}) AS {prefix}_recent_{window}d_average",
                    f"COALESCE({alias}.{prefix}_recent_{window}d_count, "
                    f"0)::BIGINT AS {prefix}_recent_{window}d_count",
                ]
            )
            recent_index += 1

    # Baseline prediction (weekday-aware cascade)
    specific = aliases["company_route_time_weekday"]
    canonical_time = aliases["canonical_route_time_weekday"]
    canonical = aliases["canonical_route"]
    select_expressions.extend(
        [
            f"COALESCE({specific}.company_route_time_weekday_average, "
            f"{canonical_time}.canonical_route_time_weekday_average, "
            f"{canonical}.canonical_route_average, "
            "global_stats.global_average) AS baseline_prediction",
            f"CASE WHEN {specific}.company_route_time_weekday_average "
            "IS NOT NULL THEN 'company_route_time_weekday' "
            f"WHEN {canonical_time}.canonical_route_time_weekday_average "
            "IS NOT NULL THEN 'canonical_route_time_weekday' "
            f"WHEN {canonical}.canonical_route_average IS NOT NULL "
            "THEN 'canonical_route' ELSE 'overall_average' "
            "END AS baseline_source",
        ]
    )

    # Version-specific extensions
    select_expressions.extend(extra_selects)
    joins.extend(extra_joins)

    select_sql = ",\n            ".join(select_expressions)
    joins_sql = "\n".join(joins)

    conn.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table_prefix}_features AS
        SELECT
            {select_sql}
        FROM {table_prefix}_target AS base
        CROSS JOIN {table_prefix}_global_stats AS global_stats
        {joins_sql}
    """)

    source_count = conn.execute(
        f"SELECT COUNT(*) FROM {table_prefix}_target"
    ).fetchone()[0]
    feature_count = conn.execute(
        f"SELECT COUNT(*) FROM {table_prefix}_features"
    ).fetchone()[0]
    if source_count != feature_count:
        raise RuntimeError(
            f"Feature joins changed row count: "
            f"{source_count:,} -> {feature_count:,}"
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_feature_frame(
    frame: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str] | None = None,
    rate_suffix: str = "_rate",
) -> None:
    """Validate a feature DataFrame for missing values and type constraints."""
    if categorical_columns is None:
        categorical_columns = CATEGORICAL_FEATURES

    expected = list(feature_columns)
    if len(expected) != len(set(expected)):
        raise RuntimeError("Feature columns contain duplicates.")
    missing_columns = [col for col in expected if col not in frame.columns]
    if missing_columns:
        raise RuntimeError(f"Missing feature columns: {missing_columns}")

    for column in categorical_columns:
        if column in frame.columns:
            frame[column] = frame[column].astype("string").fillna("missing")

    missing = frame[expected].isna().sum()
    if int(missing.sum()) != 0:
        raise RuntimeError(
            "Feature matrix contains missing values:\n"
            + missing[missing > 0].to_string()
        )

    rate_columns = [
        col for col in expected if col.endswith(rate_suffix)
    ]
    invalid_rates = {
        col: int((~frame[col].between(0.0, 1.0)).sum())
        for col in rate_columns
        if not frame[col].between(0.0, 1.0).all()
    }
    if invalid_rates:
        raise RuntimeError(
            f"Historical rates outside [0, 1]: {invalid_rates}"
        )
