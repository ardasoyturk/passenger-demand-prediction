"""Generic DuckDB SQL helpers."""

from __future__ import annotations

from pathlib import Path

import duckdb


def quote_ident(name: str) -> str:
    """Safely quote a SQL identifier."""
    return '"' + name.replace('"', '""') + '"'


def sql_string(value: str | Path) -> str:
    """Safely quote a SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def table_columns(con: duckdb.DuckDBPyConnection, table: str) -> set[str]:
    """Return column names of a table via DESCRIBE."""
    return {
        str(row[0])
        for row in con.execute(f"DESCRIBE {quote_ident(table)}").fetchall()
    }


def parquet_schema(
    con: duckdb.DuckDBPyConnection, input_path: Path
) -> list[tuple[str, str]]:
    """Return [(column_name, sql_type)] for a parquet file."""
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet({sql_string(input_path)})"
    ).fetchall()
    return [(str(row[0]), str(row[1]).upper()) for row in rows]


def parquet_columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    """Return column names of a parquet file."""
    return {name for name, _ in parquet_schema(con, path)}


def select_expression(columns: list[str]) -> str:
    """Build a SQL-safe comma-separated column list."""
    return ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
