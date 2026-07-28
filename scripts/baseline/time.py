from scripts.shared.baseline_runner import run_baseline

run_baseline(
    grouping_levels=[
        (["FIRMA_ID", "canonical_guzergah_id", "departure_30min_bucket"],
         "company_route_time_prediction"),
        (["canonical_guzergah_id", "departure_30min_bucket"],
         "canonical_time_prediction"),
        (["canonical_guzergah_id"],
         "canonical_route_prediction"),
    ],
    output_label="Baseline (company+route+time)",
)
