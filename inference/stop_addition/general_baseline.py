"""Time-independent SQL baseline for general stop-addition demand."""

from __future__ import annotations

from typing import Any

import duckdb
import numpy as np
import pandas as pd

DEMAND_BANDS = ("<10", "10-19", "20-29", "30-42", "43+")

# This ordering is the evaluated SQL baseline contract. Sub-levels only resolve
# same-company versus all-company evidence within the named business fallback.
BASELINE_LEVELS = (
    (
        "proposed_same_company_prior_trip_count",
        "proposed_same_company_prior_mean_demand",
        "SAME_COMPANY_EXACT",
    ),
    (
        "proposed_all_company_prior_trip_count",
        "proposed_all_company_prior_mean_demand",
        "ALL_COMPANY_EXACT",
    ),
    (
        "similar_same_company_prior_trip_count",
        "similar_same_company_weighted_mean_demand",
        "SIMILAR_ROUTE",
    ),
    (
        "similar_all_company_prior_trip_count",
        "similar_all_company_weighted_mean_demand",
        "SIMILAR_ROUTE",
    ),
    (
        "base_same_company_prior_trip_count",
        "base_same_company_prior_mean_demand",
        "CURRENT_ROUTE",
    ),
    (
        "base_all_company_prior_trip_count",
        "base_all_company_prior_mean_demand",
        "CURRENT_ROUTE",
    ),
    ("company_prior_trip_count", "company_prior_mean_demand", "COMPANY"),
)


def select_baseline(
    frame: pd.DataFrame,
    *,
    global_fallback: float | None = None,
) -> tuple[np.ndarray, pd.Series]:
    """Select the strongest available non-temporal evidence for every row."""

    prediction = pd.Series(np.nan, index=frame.index, dtype="float64")
    source = pd.Series("NO_PRIOR_EVIDENCE", index=frame.index, dtype="string")
    for count_column, mean_column, label in BASELINE_LEVELS:
        eligible = (
            prediction.isna()
            & frame[count_column].fillna(0).gt(0)
            & frame[mean_column].notna()
        )
        prediction.loc[eligible] = frame.loc[eligible, mean_column].astype(float)
        source.loc[eligible] = label
    if global_fallback is not None:
        prediction = prediction.fillna(global_fallback)
        source.loc[source.eq("NO_PRIOR_EVIDENCE")] = "GLOBAL_MEDIAN_FALLBACK"
    return prediction.to_numpy(dtype=np.float64), source


def demand_label(demand: float) -> str:
    """Convert a numeric baseline directly to the frozen demand bands."""

    if demand >= 43:
        return "CAPACITY_PRESSURE"
    if demand >= 30:
        return "STRONG_DEMAND"
    if demand >= 20:
        return "MODERATE_DEMAND"
    if demand >= 10:
        return "WEAK_DEMAND"
    return "CLEAR_FAILURE"


def demand_band(values: pd.Series) -> pd.Categorical:
    """Vectorized form used by grouped training evaluation."""

    return pd.cut(
        values,
        bins=[-np.inf, 9, 19, 29, 42, np.inf],
        labels=DEMAND_BANDS,
        ordered=True,
    )


def build_statistics(
    con: duckdb.DuckDBPyConnection,
    targets: pd.DataFrame,
    similarities: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the general baseline aggregates from all available history.

    Exact, current-route, and company aggregates use the same COUNT/AVG
    definitions as the leakage-safe training builder. Similar routes use its
    similarity_score * LN(1 + trip_count) weighted-average definition, but do
    not group by a requested weekday or departure time.
    """

    con.register("general_stop_addition_targets_df", targets)
    con.register("general_stop_addition_similarities_df", similarities)
    return con.execute(
        """
        WITH targets AS (
            SELECT
                CAST(training_row_id AS BIGINT) AS training_row_id,
                CAST(FIRMA_ID AS BIGINT) AS firma_id,
                CAST(base_stop_addition_canonical_route_id AS UBIGINT)
                    AS base_route_id,
                CAST(variant_stop_addition_canonical_route_id AS UBIGINT)
                    AS proposed_route_id
            FROM general_stop_addition_targets_df
        ),
        history AS (
            SELECT
                CAST(s.FIRMA_ID AS BIGINT) AS firma_id,
                c.stop_addition_canonical_route_id AS route_id,
                CAST(s.SEFER_SAYISI AS DOUBLE) AS demand
            FROM seferler_model s
            JOIN guzergah_canonical_stop_addition c
              ON c.guzergah_kodu = s.GUZERGAH_KODU
             AND c.firma_id = s.FIRMA_ID
            WHERE s.SEFER_SAYISI IS NOT NULL
        ),
        route_company AS (
            SELECT
                route_id,
                firma_id,
                COUNT(*)::BIGINT AS trip_count,
                AVG(demand)::DOUBLE AS mean_demand
            FROM history
            GROUP BY route_id, firma_id
        ),
        route_all AS (
            SELECT
                route_id,
                COUNT(*)::BIGINT AS trip_count,
                AVG(demand)::DOUBLE AS mean_demand
            FROM history
            GROUP BY route_id
        ),
        company AS (
            SELECT
                firma_id,
                COUNT(*)::BIGINT AS trip_count,
                AVG(demand)::DOUBLE AS mean_demand
            FROM history
            GROUP BY firma_id
        ),
        similar_candidates AS (
            SELECT
                t.training_row_id,
                CAST(s.similar_stop_addition_canonical_route_id AS UBIGINT)
                    AS similar_route_id,
                CAST(s.similarity_score AS DOUBLE) AS similarity_score,
                COALESCE(rc.trip_count, 0)::BIGINT AS same_company_trip_count,
                rc.mean_demand AS same_company_mean_demand,
                COALESCE(ra.trip_count, 0)::BIGINT AS all_company_trip_count,
                ra.mean_demand AS all_company_mean_demand,
                CAST(s.similarity_score AS DOUBLE)
                    * LN(1 + COALESCE(rc.trip_count, 0))
                    AS same_company_weight,
                CAST(s.similarity_score AS DOUBLE)
                    * LN(1 + COALESCE(ra.trip_count, 0))
                    AS all_company_weight
            FROM targets t
            JOIN general_stop_addition_similarities_df s
              ON CAST(s.pair_id AS BIGINT) = t.training_row_id
            LEFT JOIN route_company rc
              ON rc.route_id = CAST(
                    s.similar_stop_addition_canonical_route_id AS UBIGINT
                 )
             AND rc.firma_id = t.firma_id
            LEFT JOIN route_all ra
              ON ra.route_id = CAST(
                    s.similar_stop_addition_canonical_route_id AS UBIGINT
                 )
        ),
        similar_aggregates AS (
            SELECT
                training_row_id,
                COUNT(*) FILTER (
                    WHERE all_company_trip_count > 0
                )::INTEGER AS similar_routes_with_prior_history_count,
                COUNT(*) FILTER (
                    WHERE same_company_trip_count > 0
                )::INTEGER
                    AS same_company_similar_routes_with_prior_history_count,
                COALESCE(SUM(same_company_trip_count), 0)::BIGINT
                    AS similar_same_company_prior_trip_count,
                SUM(same_company_mean_demand * same_company_weight) FILTER (
                    WHERE same_company_trip_count > 0
                ) / NULLIF(SUM(same_company_weight) FILTER (
                    WHERE same_company_trip_count > 0
                ), 0) AS similar_same_company_weighted_mean_demand,
                COALESCE(SUM(all_company_trip_count), 0)::BIGINT
                    AS similar_all_company_prior_trip_count,
                SUM(all_company_mean_demand * all_company_weight) FILTER (
                    WHERE all_company_trip_count > 0
                ) / NULLIF(SUM(all_company_weight) FILTER (
                    WHERE all_company_trip_count > 0
                ), 0) AS similar_all_company_weighted_mean_demand,
                MAX(similarity_score) FILTER (
                    WHERE all_company_trip_count > 0
                ) AS best_similarity_with_prior_history
            FROM similar_candidates
            GROUP BY training_row_id
        )
        SELECT
            t.training_row_id,
            COALESCE(proposed_company.trip_count, 0)::BIGINT
                AS proposed_same_company_prior_trip_count,
            proposed_company.mean_demand
                AS proposed_same_company_prior_mean_demand,
            COALESCE(proposed_all.trip_count, 0)::BIGINT
                AS proposed_all_company_prior_trip_count,
            proposed_all.mean_demand AS proposed_all_company_prior_mean_demand,
            COALESCE(
                similar_aggregates.similar_routes_with_prior_history_count, 0
            )::INTEGER
                AS similar_routes_with_prior_history_count,
            COALESCE(
                similar_aggregates.same_company_similar_routes_with_prior_history_count,
                0
            )::INTEGER AS same_company_similar_routes_with_prior_history_count,
            COALESCE(
                similar_aggregates.similar_same_company_prior_trip_count, 0
            )::BIGINT
                AS similar_same_company_prior_trip_count,
            similar_aggregates.similar_same_company_weighted_mean_demand,
            COALESCE(
                similar_aggregates.similar_all_company_prior_trip_count, 0
            )::BIGINT
                AS similar_all_company_prior_trip_count,
            similar_aggregates.similar_all_company_weighted_mean_demand,
            similar_aggregates.best_similarity_with_prior_history,
            COALESCE(base_company.trip_count, 0)::BIGINT
                AS base_same_company_prior_trip_count,
            base_company.mean_demand AS base_same_company_prior_mean_demand,
            COALESCE(base_all.trip_count, 0)::BIGINT
                AS base_all_company_prior_trip_count,
            base_all.mean_demand AS base_all_company_prior_mean_demand,
            COALESCE(company.trip_count, 0)::BIGINT AS company_prior_trip_count,
            company.mean_demand AS company_prior_mean_demand
        FROM targets t
        LEFT JOIN route_company proposed_company
          ON proposed_company.route_id = t.proposed_route_id
         AND proposed_company.firma_id = t.firma_id
        LEFT JOIN route_all proposed_all
          ON proposed_all.route_id = t.proposed_route_id
        LEFT JOIN similar_aggregates USING (training_row_id)
        LEFT JOIN route_company base_company
          ON base_company.route_id = t.base_route_id
         AND base_company.firma_id = t.firma_id
        LEFT JOIN route_all base_all ON base_all.route_id = t.base_route_id
        LEFT JOIN company ON company.firma_id = t.firma_id
        ORDER BY t.training_row_id
        """
    ).fetchdf()


def evaluate_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach baseline choice, current-route baseline, scenario, and quality."""

    result = frame.copy()
    prediction, source = select_baseline(result)
    result["general_baseline_prediction"] = prediction
    result["general_baseline_source"] = source
    result["general_demand_label"] = [
        None if not np.isfinite(value) else demand_label(float(value))
        for value in prediction
    ]
    result["general_current_route_prediction"] = np.select(
        [
            result["base_same_company_prior_trip_count"].gt(0),
            result["base_all_company_prior_trip_count"].gt(0),
            result["company_prior_trip_count"].gt(0),
        ],
        [
            result["base_same_company_prior_mean_demand"],
            result["base_all_company_prior_mean_demand"],
            result["company_prior_mean_demand"],
        ],
        default=np.nan,
    )
    result["general_current_route_source"] = np.select(
        [
            result["base_same_company_prior_trip_count"].gt(0),
            result["base_all_company_prior_trip_count"].gt(0),
            result["company_prior_trip_count"].gt(0),
        ],
        ["CURRENT_ROUTE_SAME_COMPANY", "CURRENT_ROUTE_ALL_COMPANY", "COMPANY"],
        default="NO_PRIOR_EVIDENCE",
    )
    result["training_scenario"] = np.select(
        [
            result["proposed_same_company_prior_trip_count"].gt(0),
            result["proposed_all_company_prior_trip_count"].gt(0),
            result["similar_routes_with_prior_history_count"].gt(0),
        ],
        ["SAME_COMPANY_EXACT", "OTHER_COMPANY_EXACT", "SIMILAR_ROUTE_ONLY"],
        default="COLD_START",
    )
    result["historical_evidence_level"] = np.select(
        [
            result["proposed_same_company_prior_trip_count"].gt(0),
            result["proposed_all_company_prior_trip_count"].gt(0),
            result["similar_same_company_prior_trip_count"].gt(0),
            result["similar_all_company_prior_trip_count"].gt(0),
        ],
        ["HIGH", "MEDIUM_HIGH", "MEDIUM", "MEDIUM_LOW"],
        default="LOW",
    )
    return result


def build_general_baseline(
    con: duckdb.DuckDBPyConnection,
    targets: pd.DataFrame,
    similarities: pd.DataFrame,
) -> pd.DataFrame:
    """Build and select the production general stop-addition SQL baseline."""

    return evaluate_statistics(build_statistics(con, targets, similarities))


def finite_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    converted = float(value)
    return converted if np.isfinite(converted) else None
