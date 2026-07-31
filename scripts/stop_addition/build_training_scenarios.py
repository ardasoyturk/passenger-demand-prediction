from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import duckdb

from scripts.shared.duckdb_utils import parquet_schema, quote_ident, sql_string
from scripts.shared.serialization import write_json
from scripts.stop_addition.common import CATEGORICAL_CANDIDATES, FEATURE_METADATA_EXCLUDED


SCENARIOS = {
    "SAME_COMPANY_EXACT",
    "OTHER_COMPANY_EXACT",
    "SIMILAR_ROUTE_ONLY",
    "COLD_START",
}
MASK_TYPES = {
    "NONE",
    "MASK_SAME_COMPANY_EXACT",
    "MASK_ALL_EXACT",
    "MASK_ALL_ROUTE_EVIDENCE",
}
EVALUATION_SPLITS = {"VALIDATION", "TEST", "FINAL_EVAL"}
REQUIRED_COLUMNS = {
    "training_row_id",
    "target_passenger_count",
    "data_split",
    "target_date",
    "target_time",
    "FIRMA_ID",
    "proposed_same_company_time_prior_trip_count",
    "proposed_same_company_time_prior_mean_demand",
    "proposed_all_company_time_prior_trip_count",
    "proposed_all_company_time_prior_mean_demand",
    "proposed_same_company_prior_trip_count",
    "proposed_same_company_prior_mean_demand",
    "proposed_all_company_prior_trip_count",
    "proposed_all_company_prior_mean_demand",
    "same_company_similar_routes_with_prior_history_count",
    "similar_routes_with_prior_history_count",
    "similar_same_company_prior_trip_count",
    "similar_same_company_weighted_mean_demand",
    "similar_all_company_prior_trip_count",
    "similar_all_company_weighted_mean_demand",
    "base_same_company_prior_trip_count",
    "base_same_company_prior_mean_demand",
    "base_all_company_prior_trip_count",
    "base_all_company_prior_mean_demand",
    "company_prior_trip_count",
    "company_prior_mean_demand",
    "proposed_route_hierarchical_baseline",
    "proposed_route_baseline_source",
    "historical_evidence_level",
    "has_same_company_exact_proposed_history",
    "has_any_exact_proposed_history",
    "is_exact_proposed_route_cold_start",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create evidence-masked stop-addition demand training scenarios."
    )
    parser.add_argument(
        "--input",
        default="results/stop_addition/stop_addition_training_data.parquet",
    )
    parser.add_argument("--output-dir", default="results/stop_addition")
    parser.add_argument("--same-company-mask-rate", type=float, default=0.20)
    parser.add_argument("--all-exact-mask-rate", type=float, default=0.20)
    parser.add_argument("--all-route-mask-rate", type=float, default=0.10)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--mask-evaluation-splits", action="store_true")
    parser.add_argument("--sample-rows", type=int, default=2000)
    return parser.parse_args()


def validate_rates(args: argparse.Namespace) -> None:
    for name in (
        "same_company_mask_rate",
        "all_exact_mask_rate",
        "all_route_mask_rate",
    ):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1.")
    if args.sample_rows < 0:
        raise SystemExit("--sample-rows must be non-negative.")


def require_columns(schema: list[tuple[str, str]]) -> None:
    columns = {name for name, _ in schema}
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise RuntimeError(
            f"Input Parquet is missing required columns: {sorted(missing)}. "
            f"Available columns: {sorted(columns)}"
        )
    if "canonical_guzergah_id" in columns:
        raise RuntimeError(
            "Input contains forbidden old column canonical_guzergah_id; use the clean "
            "stop_addition_canonical_route_id route identifiers."
        )
    if "scenario_training_row_id" in columns:
        raise RuntimeError(
            "Input already contains scenario_training_row_id; pass the original "
            "stop-addition training dataset."
        )


def mask_value(column: str, sql_type: str) -> str:
    if "count" in column.lower() or sql_type == "BOOLEAN":
        return f"CAST(0 AS {sql_type}) AS {quote_ident(column)}"
    return f"CAST(NULL AS {sql_type}) AS {quote_ident(column)}"


def masked_projection(
    schema: list[tuple[str, str]], prefixes: tuple[str, ...], mask_best_similarity: bool
) -> str:
    expressions: list[str] = []
    for column, sql_type in schema:
        should_mask = column.startswith(prefixes) or (
            mask_best_similarity and column == "best_similarity_with_prior_history"
        )
        expressions.append(
            mask_value(column, sql_type) if should_mask else quote_ident(column)
        )
    return ",\n                ".join(expressions)


def sampling_predicate(rate: float, seed: int, label: str) -> str:
    threshold = int(round(rate * 1_000_000))
    return (
        f"hash(training_row_id, {seed}, {sql_string(label)}) % 1000000 "
        f"< {threshold}"
    )


def create_scenarios(
    con: duckdb.DuckDBPyConnection,
    input_path: Path,
    schema: list[tuple[str, str]],
    args: argparse.Namespace,
) -> None:
    con.execute(
        f"CREATE VIEW source_rows AS SELECT * FROM read_parquet({sql_string(input_path)})"
    )
    eligible_split = (
        "TRUE" if args.mask_evaluation_splits else "data_split = 'TRAIN'"
    )
    same_projection = masked_projection(
        schema, ("proposed_same_company_",), False
    )
    exact_projection = masked_projection(
        schema, ("proposed_same_company_", "proposed_all_company_"), False
    )
    route_projection = masked_projection(
        schema,
        ("proposed_same_company_", "proposed_all_company_", "similar_"),
        True,
    )
    same_sample = sampling_predicate(
        args.same_company_mask_rate, args.random_seed, "MASK_SAME_COMPANY_EXACT"
    )
    exact_sample = sampling_predicate(
        args.all_exact_mask_rate, args.random_seed, "MASK_ALL_EXACT"
    )
    route_sample = sampling_predicate(
        args.all_route_mask_rate, args.random_seed, "MASK_ALL_ROUTE_EVIDENCE"
    )

    con.execute(
        f"""
        CREATE TABLE scenario_raw AS
        SELECT *, 'NONE'::VARCHAR AS history_mask_type FROM source_rows
        UNION ALL BY NAME
        SELECT {same_projection},
               'MASK_SAME_COMPANY_EXACT'::VARCHAR AS history_mask_type
        FROM source_rows
        WHERE {eligible_split}
          AND proposed_same_company_prior_trip_count > 0
          AND {same_sample}
        UNION ALL BY NAME
        SELECT {exact_projection},
               'MASK_ALL_EXACT'::VARCHAR AS history_mask_type
        FROM source_rows
        WHERE {eligible_split}
          AND proposed_all_company_prior_trip_count > 0
          AND {exact_sample}
        UNION ALL BY NAME
        SELECT {route_projection},
               'MASK_ALL_ROUTE_EVIDENCE'::VARCHAR AS history_mask_type
        FROM source_rows
        WHERE {eligible_split}
          AND (
              proposed_all_company_prior_trip_count > 0
              OR similar_routes_with_prior_history_count > 0
          )
          AND {route_sample}
        """
    )

    con.execute(
        """
        CREATE TABLE scenario_recalculated AS
        SELECT
            * EXCLUDE (
                proposed_route_hierarchical_baseline,
                proposed_route_baseline_source,
                historical_evidence_level,
                has_same_company_exact_proposed_history,
                has_any_exact_proposed_history,
                has_similar_route_history,
                is_exact_proposed_route_cold_start
            ),
            CASE
                WHEN proposed_same_company_time_prior_trip_count > 0
                    THEN proposed_same_company_time_prior_mean_demand
                WHEN proposed_all_company_time_prior_trip_count > 0
                    THEN proposed_all_company_time_prior_mean_demand
                WHEN proposed_same_company_prior_trip_count > 0
                    THEN proposed_same_company_prior_mean_demand
                WHEN proposed_all_company_prior_trip_count > 0
                    THEN proposed_all_company_prior_mean_demand
                WHEN similar_same_company_prior_trip_count > 0
                    THEN similar_same_company_weighted_mean_demand
                WHEN similar_all_company_prior_trip_count > 0
                    THEN similar_all_company_weighted_mean_demand
                WHEN base_same_company_prior_trip_count > 0
                    THEN base_same_company_prior_mean_demand
                WHEN base_all_company_prior_trip_count > 0
                    THEN base_all_company_prior_mean_demand
                WHEN company_prior_trip_count > 0
                    THEN company_prior_mean_demand
                ELSE NULL
            END AS proposed_route_hierarchical_baseline,
            CASE
                WHEN proposed_same_company_time_prior_trip_count > 0
                    THEN 'PROPOSED_SAME_COMPANY_EXACT_WEEKDAY_TIME'
                WHEN proposed_all_company_time_prior_trip_count > 0
                    THEN 'PROPOSED_ALL_COMPANY_EXACT_WEEKDAY_TIME'
                WHEN proposed_same_company_prior_trip_count > 0
                    THEN 'PROPOSED_SAME_COMPANY_EXACT_ROUTE'
                WHEN proposed_all_company_prior_trip_count > 0
                    THEN 'PROPOSED_ALL_COMPANY_EXACT_ROUTE'
                WHEN similar_same_company_prior_trip_count > 0
                    THEN 'SAME_COMPANY_SIMILAR_ROUTES'
                WHEN similar_all_company_prior_trip_count > 0
                    THEN 'ALL_COMPANY_SIMILAR_ROUTES'
                WHEN base_same_company_prior_trip_count > 0
                    THEN 'CURRENT_ROUTE_SAME_COMPANY'
                WHEN base_all_company_prior_trip_count > 0
                    THEN 'CURRENT_ROUTE_ALL_COMPANY'
                WHEN company_prior_trip_count > 0 THEN 'COMPANY_GLOBAL'
                ELSE 'NO_PRIOR_EVIDENCE'
            END AS proposed_route_baseline_source,
            CASE
                WHEN proposed_same_company_prior_trip_count > 0 THEN 'HIGH'
                WHEN proposed_all_company_prior_trip_count > 0 THEN 'MEDIUM_HIGH'
                WHEN similar_same_company_prior_trip_count > 0 THEN 'MEDIUM'
                WHEN similar_all_company_prior_trip_count > 0 THEN 'MEDIUM_LOW'
                ELSE 'LOW'
            END AS historical_evidence_level,
            proposed_same_company_prior_trip_count > 0
                AS has_same_company_exact_proposed_history,
            proposed_all_company_prior_trip_count > 0
                AS has_any_exact_proposed_history,
            COALESCE(similar_routes_with_prior_history_count, 0) > 0
                AS has_similar_route_history,
            proposed_all_company_prior_trip_count = 0
                AS is_exact_proposed_route_cold_start
        FROM scenario_raw
        """
    )
    con.execute(
        """
        CREATE TABLE stop_addition_training_scenarios AS
        SELECT
            row_number() OVER (
                ORDER BY training_row_id,
                    CASE history_mask_type
                        WHEN 'NONE' THEN 0
                        WHEN 'MASK_SAME_COMPANY_EXACT' THEN 1
                        WHEN 'MASK_ALL_EXACT' THEN 2
                        ELSE 3
                    END
            )::BIGINT AS scenario_training_row_id,
            *,
            CASE
                WHEN proposed_same_company_prior_trip_count > 0
                    THEN 'SAME_COMPANY_EXACT'
                WHEN proposed_all_company_prior_trip_count > 0
                    THEN 'OTHER_COMPANY_EXACT'
                WHEN COALESCE(similar_routes_with_prior_history_count, 0) > 0
                    THEN 'SIMILAR_ROUTE_ONLY'
                ELSE 'COLD_START'
            END AS training_scenario
        FROM scenario_recalculated
        """
    )


def scalar(con: duckdb.DuckDBPyConnection, query: str) -> int:
    return int(con.execute(query).fetchone()[0])


def validate_output(
    con: duckdb.DuckDBPyConnection,
    schema: list[tuple[str, str]],
    mask_evaluation_splits: bool,
) -> None:
    failures: list[str] = []
    target_mismatches = scalar(
        con,
        """
        SELECT COUNT(*)
        FROM stop_addition_training_scenarios s
        JOIN source_rows o USING (training_row_id)
        WHERE s.target_passenger_count IS DISTINCT FROM o.target_passenger_count
        """,
    )
    if target_mismatches:
        failures.append(f"{target_mismatches} masked targets changed")

    families = {
        "MASK_SAME_COMPANY_EXACT": ("proposed_same_company_",),
        "MASK_ALL_EXACT": ("proposed_same_company_", "proposed_all_company_"),
        "MASK_ALL_ROUTE_EVIDENCE": (
            "proposed_same_company_",
            "proposed_all_company_",
            "similar_",
        ),
    }
    type_by_column = dict(schema)
    for mask_type, prefixes in families.items():
        columns = [
            name
            for name, _ in schema
            if name.startswith(prefixes)
            or (
                mask_type == "MASK_ALL_ROUTE_EVIDENCE"
                and name == "best_similarity_with_prior_history"
            )
        ]
        count_columns = [name for name in columns if "count" in name.lower()]
        statistical_columns = [
            name
            for name in columns
            if name not in count_columns and type_by_column[name] != "BOOLEAN"
        ]
        if count_columns:
            condition = " OR ".join(
                f"COALESCE({quote_ident(name)}, -1) <> 0" for name in count_columns
            )
            bad = scalar(
                con,
                f"""
                SELECT COUNT(*) FROM stop_addition_training_scenarios
                WHERE history_mask_type = {sql_string(mask_type)} AND ({condition})
                """,
            )
            if bad:
                failures.append(f"{bad} {mask_type} rows have nonzero masked counts")
        if statistical_columns:
            condition = " OR ".join(
                f"{quote_ident(name)} IS NOT NULL" for name in statistical_columns
            )
            bad = scalar(
                con,
                f"""
                SELECT COUNT(*) FROM stop_addition_training_scenarios
                WHERE history_mask_type = {sql_string(mask_type)} AND ({condition})
                """,
            )
            if bad:
                failures.append(
                    f"{bad} {mask_type} rows have non-null masked statistics"
                )

    if not mask_evaluation_splits:
        split_list = ", ".join(sql_string(value) for value in EVALUATION_SPLITS)
        bad = scalar(
            con,
            f"""
            SELECT COUNT(*) FROM stop_addition_training_scenarios
            WHERE data_split IN ({split_list}) AND history_mask_type <> 'NONE'
            """,
        )
        if bad:
            failures.append(f"{bad} evaluation rows were masked")

    duplicate_ids = scalar(
        con,
        """
        SELECT COUNT(*) - COUNT(DISTINCT scenario_training_row_id)
        FROM stop_addition_training_scenarios
        """,
    )
    if duplicate_ids:
        failures.append(f"{duplicate_ids} duplicate scenario_training_row_id values")
    invalid_scenarios = scalar(
        con,
        f"""
        SELECT COUNT(*) FROM stop_addition_training_scenarios
        WHERE training_scenario NOT IN ({", ".join(map(sql_string, sorted(SCENARIOS)))})
           OR training_scenario IS NULL
        """,
    )
    if invalid_scenarios:
        failures.append(f"{invalid_scenarios} invalid training_scenario values")
    invalid_masks = scalar(
        con,
        f"""
        SELECT COUNT(*) FROM stop_addition_training_scenarios
        WHERE history_mask_type NOT IN ({", ".join(map(sql_string, sorted(MASK_TYPES)))})
           OR history_mask_type IS NULL
        """,
    )
    if invalid_masks:
        failures.append(f"{invalid_masks} invalid history_mask_type values")
    output_columns = {
        str(row[0])
        for row in con.execute(
            "DESCRIBE stop_addition_training_scenarios"
        ).fetchall()
    }
    if "canonical_guzergah_id" in output_columns:
        failures.append("forbidden canonical_guzergah_id was introduced")
    if failures:
        raise RuntimeError("Scenario validation failed:\n- " + "\n- ".join(failures))


def grouped_counts(
    con: duckdb.DuckDBPyConnection, columns: list[str]
) -> dict[str, Any]:
    select = ", ".join(map(quote_ident, columns))
    rows = con.execute(
        f"""
        SELECT {select}, COUNT(*) AS row_count
        FROM stop_addition_training_scenarios
        GROUP BY {select}
        ORDER BY {select}
        """
    ).fetchall()
    if len(columns) == 1:
        return {str(row[0]): int(row[1]) for row in rows}
    nested: dict[str, Any] = {}
    for row in rows:
        cursor = nested
        for value in row[:-2]:
            cursor = cursor.setdefault(str(value), {})
        cursor[str(row[-2])] = int(row[-1])
    return nested


def grouped_stat(
    con: duckdb.DuckDBPyConnection, aggregate: str
) -> dict[str, float | None]:
    return {
        str(name): (None if value is None else float(value))
        for name, value in con.execute(
            f"""
            SELECT training_scenario, {aggregate}(target_passenger_count)
            FROM stop_addition_training_scenarios
            GROUP BY training_scenario
            ORDER BY training_scenario
            """
        ).fetchall()
    }


def build_summary(
    con: duckdb.DuckDBPyConnection, args: argparse.Namespace
) -> dict[str, Any]:
    original = scalar(con, "SELECT COUNT(*) FROM source_rows")
    final = scalar(con, "SELECT COUNT(*) FROM stop_addition_training_scenarios")
    return {
        "original_row_count": original,
        "generated_masked_row_count": final - original,
        "final_row_count": final,
        "rows_by_data_split": grouped_counts(con, ["data_split"]),
        "rows_by_training_scenario": grouped_counts(con, ["training_scenario"]),
        "rows_by_history_mask_type": grouped_counts(con, ["history_mask_type"]),
        "scenario_counts_by_split": grouped_counts(
            con, ["data_split", "training_scenario"]
        ),
        "mask_counts_by_split": grouped_counts(
            con, ["data_split", "history_mask_type"]
        ),
        "target_mean_by_scenario": grouped_stat(con, "AVG"),
        "target_median_by_scenario": grouped_stat(con, "MEDIAN"),
        "baseline_source_counts_by_mask": grouped_counts(
            con, ["history_mask_type", "proposed_route_baseline_source"]
        ),
        "evidence_level_counts_by_mask": grouped_counts(
            con, ["history_mask_type", "historical_evidence_level"]
        ),
        "duplicate_training_row_id_count": scalar(
            con,
            """
            SELECT COUNT(*)
            FROM (
                SELECT training_row_id
                FROM stop_addition_training_scenarios
                GROUP BY training_row_id
                HAVING COUNT(*) > 1
            )
            """,
        ),
        "parameters": {
            "same_company_mask_rate": args.same_company_mask_rate,
            "all_exact_mask_rate": args.all_exact_mask_rate,
            "all_route_mask_rate": args.all_route_mask_rate,
            "random_seed": args.random_seed,
            "mask_evaluation_splits": args.mask_evaluation_splits,
        },
    }


def feature_metadata(
    con: duckdb.DuckDBPyConnection, input_path: Path
) -> dict[str, Any]:
    columns = [
        str(row[0])
        for row in con.execute(
            "DESCRIBE stop_addition_training_scenarios"
        ).fetchall()
    ]
    excluded = FEATURE_METADATA_EXCLUDED | {"scenario_training_row_id"}
    recommended = [name for name in columns if name not in excluded]
    categorical = sorted(CATEGORICAL_CANDIDATES | {"training_scenario", "history_mask_type"})
    return {
        "target_column": "target_passenger_count",
        "split_column": "data_split",
        "source_file": str(input_path),
        "recommended_feature_columns": recommended,
        "recommended_categorical_columns": [
            name for name in recommended if name in categorical
        ],
        "metadata_columns_not_for_model": sorted(
            excluded - {"target_passenger_count", "data_split"}
        ),
        "notes": [
            "Masked copies preserve the leakage-safe cutoff of the source dataset.",
            "history_mask_type and training_scenario explicitly describe available production evidence.",
            "is_company_origin_city is an external hard-approval rule and is excluded.",
            "Old canonical_guzergah_id and retrospective/future-derived support are not used.",
        ],
    }


def export_outputs(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
    input_path: Path,
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "stop_addition_training_scenarios.parquet"
    sample_path = output_dir / "stop_addition_training_scenarios_sample.csv"
    summary_path = output_dir / "stop_addition_training_scenarios_summary.json"
    features_path = output_dir / "stop_addition_training_scenario_features.json"
    con.execute(
        f"""
        COPY (
            SELECT * FROM stop_addition_training_scenarios
            ORDER BY scenario_training_row_id
        ) TO {sql_string(parquet_path)} (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT * FROM stop_addition_training_scenarios
            ORDER BY scenario_training_row_id
            LIMIT {int(args.sample_rows)}
        ) TO {sql_string(sample_path)} (HEADER, DELIMITER ',')
        """
    )
    summary = build_summary(con, args)
    write_json(summary_path, summary)
    write_json(features_path, feature_metadata(con, input_path))
    return parquet_path, sample_path, summary_path, features_path, summary


def main() -> None:
    args = parse_args()
    validate_rates(args)
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input Parquet not found: {input_path}")

    con = duckdb.connect()
    try:
        schema = parquet_schema(con, input_path)
        require_columns(schema)
        create_scenarios(con, input_path, schema, args)
        validate_output(con, schema, args.mask_evaluation_splits)
        paths = export_outputs(con, output_dir, input_path, args)
    finally:
        con.close()

    summary = paths[-1]
    print(
        "Created stop-addition training scenarios: "
        f"{summary['original_row_count']:,} original + "
        f"{summary['generated_masked_row_count']:,} masked = "
        f"{summary['final_row_count']:,} rows."
    )
    print(
        "Scenario distribution: "
        + ", ".join(
            f"{name}={count:,}"
            for name, count in summary["rows_by_training_scenario"].items()
        )
    )
    print(f"Parquet: {paths[0]}")


if __name__ == "__main__":
    main()
