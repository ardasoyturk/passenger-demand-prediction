"""Shared constants and metadata for stop-addition pipeline."""

from __future__ import annotations

ROUTE_TABLE = "guzergah_canonical_stop_addition"
TRIP_TABLE_DEFAULT = "seferler_model"
FIRMA_TABLE = "ats_firma"
PLACE_TABLE = "ats_yer"

RECENT_WINDOWS = (30, 90, 180)

FEATURE_METADATA_EXCLUDED = {
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

CATEGORICAL_CANDIDATES = {
    "FIRMA_ID",
    "added_stop_uetds_yer_id",
    "added_stop_il_id",
    "origin_stop_id",
    "destination_stop_id",
    "iso_weekday",
    "month",
    "half_hour_bucket",
}
