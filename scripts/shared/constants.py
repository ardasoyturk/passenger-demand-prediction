from __future__ import annotations

from scripts.shared.paths import RECENT_WINDOWS

CATEGORICAL_FEATURES = [
    "FIRMA_ID",
    "GUZERGAH_KODU",
    "canonical_guzergah_id",
    "month",
    "day_of_week",
    "departure_30min_bucket",
]

CALENDAR_NUMERIC_FEATURES = ["week_of_year", "departure_minute"]

HISTORICAL_GROUP_DEFINITIONS = [
    (
        "company_route_time_weekday",
        [
            "FIRMA_ID",
            "canonical_guzergah_id",
            "departure_30min_bucket",
            "day_of_week",
        ],
    ),
    (
        "company_route_time",
        ["FIRMA_ID", "canonical_guzergah_id", "departure_30min_bucket"],
    ),
    (
        "canonical_route_time_weekday",
        ["canonical_guzergah_id", "departure_30min_bucket", "day_of_week"],
    ),
    ("canonical_route", ["canonical_guzergah_id"]),
    ("company", ["FIRMA_ID"]),
]

RECENT_GROUP_DEFINITIONS = [
    HISTORICAL_GROUP_DEFINITIONS[0],
    HISTORICAL_GROUP_DEFINITIONS[2],
]

HISTORICAL_STATISTIC_SUFFIXES = [
    "average",
    "median",
    "std",
    "maximum",
    "p90",
    "above_60_rate",
    "above_100_rate",
    "count",
]

SOURCE_COLUMNS = [
    "SEFER_ID",
    "SEFER_TARIHI",
    "FIRMA_ID",
    "GUZERGAH_KODU",
    "canonical_guzergah_id",
    "target",
    "month",
    "week_of_year",
    "day_of_week",
    "departure_minute",
    "departure_30min_bucket",
]

HISTORICAL_FEATURES = [
    f"{prefix}_{suffix}"
    for prefix, _ in HISTORICAL_GROUP_DEFINITIONS
    for suffix in HISTORICAL_STATISTIC_SUFFIXES
]

RECENT_FEATURES = [
    f"{prefix}_recent_{window}d_{suffix}"
    for prefix, _ in RECENT_GROUP_DEFINITIONS
    for window in RECENT_WINDOWS
    for suffix in ("average", "count")
]

V4_1_FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + CALENDAR_NUMERIC_FEATURES
    + HISTORICAL_FEATURES
    + RECENT_FEATURES
)
