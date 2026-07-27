"""FastAPI application for the internal passenger-demand demo."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from inference.api.depends import Artifacts, Database
from inference.api.routes import durak, predict, route
from inference.api.schemas import HealthResponse
from inference.engine import load_frozen_artifacts


def _cors_origins() -> list[str]:
    configured = os.getenv("CORS_ORIGINS", "*")
    origins = [origin.strip() for origin in configured.split(",") if origin.strip()]
    return origins or ["*"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifacts = load_frozen_artifacts()
    yield
    del app.state.artifacts


app = FastAPI(
    title="Passenger Demand Demo API",
    version="1.0.0",
    lifespan=lifespan,
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
app.include_router(durak.router)
app.include_router(route.router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health(db: Database, _artifacts: Artifacts) -> HealthResponse:
    db.execute("SELECT 1").fetchone()
    return HealthResponse(status="ok", database="ok", artifacts="loaded")
