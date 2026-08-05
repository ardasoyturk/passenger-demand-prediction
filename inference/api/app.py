"""FastAPI application for the internal passenger-demand demo."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastmcp.utilities.lifespan import combine_lifespans

from inference.api.depends import Artifacts, Database
from inference.api.mcp import mcp_app, set_artifacts
from inference.api.routes import (
    durak,
    gateway,
    openai_compatible,
    predict,
    predict_general,
    route,
    stop_addition,
)
from inference.api.routes.stop_addition import load_stop_addition_contract
from inference.api.schemas import HealthResponse
from inference.engine import load_frozen_artifacts


def _cors_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS", "*")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return origins or ["*"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifacts = load_frozen_artifacts()
    set_artifacts(app.state.artifacts)
    load_stop_addition_contract()
    yield
    set_artifacts(None)
    del app.state.artifacts


app = FastAPI(
    title="Passenger Demand Demo API",
    version="1.0.0",
    lifespan=combine_lifespans(lifespan, mcp_app.lifespan),
)

origins = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials="*" not in origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)
app.include_router(predict_general.router)
app.include_router(durak.router)
app.include_router(route.router)
app.include_router(stop_addition.router)
app.include_router(gateway.router)
app.include_router(openai_compatible.router)
app.mount("/mcp", mcp_app)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(db: Database, _artifacts: Artifacts) -> HealthResponse:
    db.execute("SELECT 1").fetchone()
    return HealthResponse(status="ok", database="ok", artifacts="loaded")
