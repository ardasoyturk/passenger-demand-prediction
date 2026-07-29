"""JSON serialization helpers for pandas/numpy types."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def json_default(value: Any) -> Any:
    """Default serializer for json.dumps handling pandas datetime and NaN."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, data: Any, **kwargs: Any) -> None:
    """Write JSON with standard settings."""
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=json_default, **kwargs),
        encoding="utf-8",
    )
