"""HTTP route modules for the inference API."""

from inference.api.routes import (
    durak,
    gateway,
    openai_compatible,
    predict,
    predict_general,
    route,
    stop_addition,
)

__all__ = [
    "durak",
    "gateway",
    "openai_compatible",
    "predict",
    "predict_general",
    "route",
    "stop_addition",
]
