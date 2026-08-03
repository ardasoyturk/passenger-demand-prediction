"""General (time-independent) demand prediction from company + route only.

This mode answers "what demand should this company expect on this route in
general?" without a departure date or time. It deliberately does **not** run
the frozen v4.2 hybrid regressor or the v4.4 threshold classifiers: the
prediction is the historical-average baseline queried without time grouping,
and the demand label is assigned algorithmically from that baseline using the
same 10/20/30/43 thresholds.

Baseline fallback order:
    1. company + route average (FIRMA_ID + GUZERGAH_KODU)
    2. canonical route average (shared physical route, any company)
    3. global historical average
"""

from __future__ import annotations

from typing import Any

import duckdb


HISTORY_START = "2023-01-01"
THRESHOLDS = (10, 20, 30, 43)

BASELINE_SOURCES = ("company_route", "canonical_route", "global")


def _resolve_canonical_route(
    conn: duckdb.DuckDBPyConnection,
    firma_id: int,
    guzergah_kodu: int,
) -> tuple[int, str]:
    """Map one company route to its canonical route (history first)."""

    row = conn.execute(
        """
        WITH historical_candidates AS (
            SELECT
                history.canonical_guzergah_id,
                COUNT(*) AS historical_row_count,
                MAX(history.SEFER_TARIHI) AS latest_history_date
            FROM model_data_base AS history
            WHERE history.FIRMA_ID = ?
              AND history.GUZERGAH_KODU = ?
              AND history.canonical_guzergah_id IS NOT NULL
            GROUP BY history.canonical_guzergah_id
        ),
        historical_mapping AS (
            SELECT canonical_guzergah_id
            FROM historical_candidates
            QUALIFY ROW_NUMBER() OVER (
                ORDER BY historical_row_count DESC, latest_history_date DESC
            ) = 1
        )
        SELECT
            COALESCE(
                historical.canonical_guzergah_id,
                current_mapping.canonical_guzergah_id
            )::UBIGINT AS canonical_guzergah_id,
            CASE
                WHEN historical.canonical_guzergah_id IS NOT NULL
                    THEN 'model_data_base_history'
                WHEN current_mapping.canonical_guzergah_id IS NOT NULL
                    THEN 'guzergah_canonical'
            END AS canonical_mapping_source
        FROM (SELECT 1) AS dummy
        LEFT JOIN historical_mapping AS historical ON TRUE
        LEFT JOIN guzergah_canonical AS current_mapping
          ON current_mapping.firma_id = ?
         AND current_mapping.guzergah_kodu = ?
        """,
        [firma_id, guzergah_kodu, firma_id, guzergah_kodu],
    ).fetchone()
    if row is None or row[0] is None:
        raise ValueError(
            "Unknown company-route combination or missing canonical route: "
            f"FIRMA_ID={firma_id}, GUZERGAH_KODU={guzergah_kodu}"
        )
    return int(row[0]), str(row[1])


def _baseline_statistics(
    conn: duckdb.DuckDBPyConnection,
    firma_id: int,
    guzergah_kodu: int,
    canonical_guzergah_id: int,
) -> dict[str, Any]:
    """Single-scan company-route, canonical-route, and global aggregates."""

    row = conn.execute(
        f"""
        SELECT
            COUNT(target) FILTER (
                WHERE FIRMA_ID = ? AND GUZERGAH_KODU = ?
            )::BIGINT AS company_route_count,
            AVG(target) FILTER (
                WHERE FIRMA_ID = ? AND GUZERGAH_KODU = ?
            )::DOUBLE AS company_route_mean,
            COUNT(target) FILTER (
                WHERE canonical_guzergah_id = ?
            )::BIGINT AS canonical_route_count,
            AVG(target) FILTER (
                WHERE canonical_guzergah_id = ?
            )::DOUBLE AS canonical_route_mean,
            COUNT(target)::BIGINT AS global_count,
            AVG(target)::DOUBLE AS global_mean
        FROM model_data_base
        WHERE SEFER_TARIHI >= DATE '{HISTORY_START}'
        """,
        [
            firma_id,
            guzergah_kodu,
            firma_id,
            guzergah_kodu,
            canonical_guzergah_id,
            canonical_guzergah_id,
        ],
    ).fetchone()
    if row is None:
        raise RuntimeError("Baseline statistics query returned no row")
    return {
        "company_route_count": int(row[0]),
        "company_route_mean": float(row[1]) if row[1] is not None else None,
        "canonical_route_count": int(row[2]),
        "canonical_route_mean": float(row[3]) if row[3] is not None else None,
        "global_count": int(row[4]),
        "global_mean": float(row[5]),
    }


def _demand_label(demand: float) -> str:
    """Classify the baseline value algorithmically on the frozen thresholds."""

    if demand >= 43:
        return "CAPACITY_PRESSURE"
    if demand >= 30:
        return "STRONG_DEMAND"
    if demand >= 20:
        return "MODERATE_DEMAND"
    if demand >= 10:
        return "WEAK_DEMAND"
    return "CLEAR_FAILURE"


def _reliability(source: str, count: int) -> tuple[str, str]:
    """Confidence from the amount and specificity of the averaged history."""

    if source == "global":
        return (
            "NO_HISTORY",
            "Bu firma-güzergâh ve fiziksel rota için geçmiş sefer bulunamadı; "
            "genel tarihsel ortalama kullanıldı",
        )
    if count < 10:
        level = "UNSAFE"
    elif count < 30:
        level = "MEDIUM" if source == "company_route" else "LOW"
    else:
        level = "HIGH" if source == "company_route" else "MEDIUM"
    if source == "company_route":
        reason = f"Aynı firma ve güzergâhta {count} adet geçmiş sefer var"
    else:
        reason = (
            "Firma-güzergâh geçmişi olmadığı için aynı fiziksel rotadaki "
            f"{count} adet geçmiş sefer kullanıldı"
        )
    return level, reason


def predict_general(
    conn: duckdb.DuckDBPyConnection,
    firma_id: int,
    guzergah_kodu: int,
) -> dict[str, Any]:
    """Predict general route demand from company + route history only."""

    canonical_guzergah_id, _ = _resolve_canonical_route(
        conn, firma_id, guzergah_kodu
    )
    stats = _baseline_statistics(conn, firma_id, guzergah_kodu, canonical_guzergah_id)

    if stats["company_route_count"] > 0:
        source = "company_route"
        expected = stats["company_route_mean"]
        count = stats["company_route_count"]
    elif stats["canonical_route_count"] > 0:
        source = "canonical_route"
        expected = stats["canonical_route_mean"]
        count = stats["canonical_route_count"]
    else:
        source = "global"
        expected = stats["global_mean"]
        count = stats["global_count"]

    reliability, reason = _reliability(source, count)
    return {
        "FIRMA_ID": firma_id,
        "GUZERGAH_KODU": guzergah_kodu,
        "canonical_route_id": canonical_guzergah_id,
        "expected_demand": float(expected),
        "baseline_source": source,
        "baseline_trip_count": int(count),
        "company_route_count": stats["company_route_count"],
        "company_route_mean": stats["company_route_mean"],
        "canonical_route_count": stats["canonical_route_count"],
        "canonical_route_mean": stats["canonical_route_mean"],
        "demand_label": _demand_label(float(expected)),
        "prediction_reliability": reliability,
        "reliability_reason": reason,
    }
