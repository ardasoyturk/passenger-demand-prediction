import duckdb
from pathlib import Path

conn = duckdb.connect(str(Path("analysis.duckdb").resolve()), read_only=False)
conn.execute("SET threads = 8")
conn.execute("SET temp_directory = 'duckdb_temp'")

conn.execute("""
    CREATE OR REPLACE TEMP TABLE history_until_2025 AS
    SELECT SEFER_TARIHI, FIRMA_ID, GUZERGAH_KODU, canonical_guzergah_id, target,
           month, week_of_year, day_of_week, departure_minute, departure_30min_bucket
    FROM model_data_base
    WHERE SEFER_TARIHI >= DATE '2023-01-01'
      AND SEFER_TARIHI < DATE '2025-01-01'
""")

conn.execute("""
    CREATE OR REPLACE TEMP TABLE global_stats_for_train AS
    SELECT AVG(target)::DOUBLE AS global_average FROM history_until_2025
    WHERE SEFER_TARIHI >= DATE '2023-01-01' AND SEFER_TARIHI < DATE '2024-01-01'
""")

conn.execute("""
    CREATE OR REPLACE TEMP TABLE company_route_time_weekday_for_train AS
    SELECT
        FIRMA_ID,
        canonical_guzergah_id,
        departure_30min_bucket,
        day_of_week,
        AVG(target)::DOUBLE AS company_route_time_weekday_average
    FROM history_until_2025
    WHERE SEFER_TARIHI >= DATE '2023-01-01' AND SEFER_TARIHI < DATE '2024-01-01'
    GROUP BY FIRMA_ID, canonical_guzergah_id, departure_30min_bucket, day_of_week
""")

conn.execute("""
    CREATE OR REPLACE TEMP TABLE company_route_time_weekday_recent_30d_for_train AS
    SELECT
        reference.FIRMA_ID,
        reference.canonical_guzergah_id,
        reference.departure_30min_bucket,
        reference.day_of_week,
        reference.SEFER_TARIHI AS reference_date,
        AVG(history.target)::DOUBLE AS company_route_time_weekday_recent_30d_average
    FROM history_until_2025 AS reference
    INNER JOIN history_until_2025 AS history
        ON reference.FIRMA_ID = history.FIRMA_ID
        AND reference.canonical_guzergah_id = history.canonical_guzergah_id
        AND reference.departure_30min_bucket = history.departure_30min_bucket
        AND reference.day_of_week = history.day_of_week
        AND history.SEFER_TARIHI >= reference.SEFER_TARIHI - INTERVAL '30 days'
        AND history.SEFER_TARIHI < reference.SEFER_TARIHI
    WHERE reference.SEFER_TARIHI >= DATE '2024-01-01'
      AND reference.SEFER_TARIHI < DATE '2025-01-01'
    GROUP BY
        reference.FIRMA_ID,
        reference.canonical_guzergah_id,
        reference.departure_30min_bucket,
        reference.day_of_week,
        reference.SEFER_TARIHI
""")

# Test 1: reference alias directly
conn.execute("""
    CREATE OR REPLACE TEMP TABLE test_alias_direct AS
    SELECT
        base.*,
        COALESCE(v3.company_route_time_weekday_average, global_stats.global_average) AS company_route_time_weekday_average,
        COALESCE(agg.company_route_time_weekday_recent_30d_average, company_route_time_weekday_average) AS company_route_time_weekday_recent_30d_average
    FROM (
        SELECT *
        FROM history_until_2025
        WHERE SEFER_TARIHI >= DATE '2024-01-01' AND SEFER_TARIHI < DATE '2025-01-01'
    ) AS base
    CROSS JOIN global_stats_for_train AS global_stats
    LEFT JOIN company_route_time_weekday_for_train AS v3
        ON base.FIRMA_ID = v3.FIRMA_ID
        AND base.canonical_guzergah_id = v3.canonical_guzergah_id
        AND base.departure_30min_bucket = v3.departure_30min_bucket
        AND base.day_of_week = v3.day_of_week
    LEFT JOIN company_route_time_weekday_recent_30d_for_train AS agg
        ON base.FIRMA_ID = agg.FIRMA_ID
        AND base.canonical_guzergah_id = agg.canonical_guzergah_id
        AND base.departure_30min_bucket = agg.departure_30min_bucket
        AND base.day_of_week = agg.day_of_week
        AND base.SEFER_TARIHI = agg.reference_date
""")

# Test 2: use a distinct alias name
conn.execute("""
    CREATE OR REPLACE TEMP TABLE test_distinct_alias AS
    SELECT
        base.*,
        COALESCE(v3.company_route_time_weekday_average, global_stats.global_average) AS company_route_time_weekday_avg,
        COALESCE(agg.company_route_time_weekday_recent_30d_average, company_route_time_weekday_avg) AS company_route_time_weekday_recent_30d_average
    FROM (
        SELECT *
        FROM history_until_2025
        WHERE SEFER_TARIHI >= DATE '2024-01-01' AND SEFER_TARIHI < DATE '2025-01-01'
    ) AS base
    CROSS JOIN global_stats_for_train AS global_stats
    LEFT JOIN company_route_time_weekday_for_train AS v3
        ON base.FIRMA_ID = v3.FIRMA_ID
        AND base.canonical_guzergah_id = v3.canonical_guzergah_id
        AND base.departure_30min_bucket = v3.departure_30min_bucket
        AND base.day_of_week = v3.day_of_week
    LEFT JOIN company_route_time_weekday_recent_30d_for_train AS agg
        ON base.FIRMA_ID = agg.FIRMA_ID
        AND base.canonical_guzergah_id = agg.canonical_guzergah_id
        AND base.departure_30min_bucket = agg.departure_30min_bucket
        AND base.day_of_week = agg.day_of_week
        AND base.SEFER_TARIHI = agg.reference_date
""")

print("Direct alias reference:")
print(conn.execute("SELECT COUNT(*) - COUNT(company_route_time_weekday_recent_30d_average) AS missing FROM test_alias_direct").fetchdf().to_string())

print("\nDistinct alias reference:")
print(conn.execute("SELECT COUNT(*) - COUNT(company_route_time_weekday_recent_30d_average) AS missing FROM test_distinct_alias").fetchdf().to_string())

conn.close()
