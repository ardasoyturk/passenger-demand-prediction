"""Shared FastAPI dependencies for immutable models and read-only DuckDB."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Annotated

import duckdb
from fastapi import Depends, Request

from inference.engine import DB_PATH, FrozenArtifacts


def get_db() -> Iterator[duckdb.DuckDBPyConnection]:
    connection = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        yield connection
    finally:
        connection.close()


def get_artifacts(request: Request) -> FrozenArtifacts:
    return request.app.state.artifacts


Database = Annotated[duckdb.DuckDBPyConnection, Depends(get_db)]
Artifacts = Annotated[FrozenArtifacts, Depends(get_artifacts)]

