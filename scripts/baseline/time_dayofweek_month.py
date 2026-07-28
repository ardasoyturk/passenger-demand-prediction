from scripts.shared.baseline_runner import run_baseline

run_baseline(
    grouping_levels=[
        (["FIRMA_ID", "canonical_guzergah_id", "departure_30min_bucket", "day_of_week", "month"],
         "company_route_time_weekday_month_prediction"),
        (["FIRMA_ID", "canonical_guzergah_id", "departure_30min_bucket", "day_of_week"],
         "company_route_time_weekday_prediction"),
        (["canonical_guzergah_id", "departure_30min_bucket", "day_of_week", "month"],
         "canonical_route_time_weekday_month_prediction"),
        (["canonical_guzergah_id", "departure_30min_bucket", "day_of_week"],
         "canonical_route_time_weekday_prediction"),
        (["canonical_guzergah_id"],
         "canonical_route_prediction"),
    ],
    output_label="Month-aware baseline",
)
