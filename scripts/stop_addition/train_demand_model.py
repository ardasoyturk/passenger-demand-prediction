from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import catboost
import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from scripts.shared.duckdb_utils import parquet_columns, select_expression, sql_string


TARGET = "target_passenger_count"
WEIGHT_COLUMN = "combined_weight"
SCENARIO_COLUMN = "training_scenario"
FORBIDDEN_FEATURES = {
    "history_mask_type",
    "scenario_training_row_id",
    "training_row_id",
    "scenario_weight",
    "source_row_weight",
    "combined_weight",
    "data_split",
}
DATASETS = {
    "untouched_validation": ("main", "VALIDATION"),
    "diagnostic_validation": ("diagnostic_validation", None),
    "untouched_test": ("main", "TEST"),
    "diagnostic_test": ("diagnostic_test", None),
}
DEMAND_BANDS = ("<10", "10-19", "20-29", "30-42", "43+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare unweighted and combined-weight CatBoost regressors "
            "on the frozen stop-addition scenario dataset."
        )
    )
    parser.add_argument(
        "--input",
        default="results/stop_addition/stop_addition_training_scenarios_weighted.parquet",
    )
    parser.add_argument(
        "--features",
        default="results/stop_addition/stop_addition_training_scenario_features.json",
    )
    parser.add_argument(
        "--diagnostic-validation",
        default=(
            "results/stop_addition/"
            "stop_addition_training_scenarios_diagnostic_validation.parquet"
        ),
    )
    parser.add_argument(
        "--diagnostic-test",
        default=(
            "results/stop_addition/"
            "stop_addition_training_scenarios_diagnostic_test.parquet"
        ),
    )
    parser.add_argument("--output-dir", default="results/stop_addition")
    parser.add_argument("--model-dir", default="models/stop_addition")
    parser.add_argument("--iterations", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--task-type",
        choices=("GPU", "CPU"),
        default="GPU",
    )
    parser.add_argument("--devices", default="0")
    parser.add_argument("--thread-count", type=int, default=-1)
    parser.add_argument(
        "--max-exact-mae-degradation-rate",
        type=float,
        default=0.02,
        help=(
            "Maximum allowed weighted-model MAE degradation on untouched validation "
            "SAME_COMPANY_EXACT rows."
        ),
    )
    return parser.parse_args()


def require_files(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.is_file()]
    if missing:
        raise SystemExit("Required frozen input files are missing:\n- " + "\n- ".join(missing))


def load_feature_contract(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    recommended = [str(value) for value in payload["recommended_feature_columns"]]
    categorical = [
        str(value) for value in payload["recommended_categorical_columns"]
    ]
    features = [name for name in recommended if name not in FORBIDDEN_FEATURES]
    if SCENARIO_COLUMN not in features:
        features.append(SCENARIO_COLUMN)
    categorical = [
        name
        for name in categorical
        if name in features and name not in FORBIDDEN_FEATURES
    ]
    if SCENARIO_COLUMN not in categorical:
        categorical.append(SCENARIO_COLUMN)
    forbidden_present = FORBIDDEN_FEATURES.intersection(features)
    if forbidden_present:
        raise RuntimeError(
            f"Forbidden model features survived filtering: {sorted(forbidden_present)}"
        )
    if len(features) != len(set(features)):
        raise RuntimeError("Feature contract contains duplicate feature names.")
    return features, categorical


def validate_schemas(
    con: duckdb.DuckDBPyConnection,
    paths: dict[str, Path],
    features: list[str],
) -> None:
    required_common = set(features) | {TARGET, SCENARIO_COLUMN, "data_split"}
    for name in ("input", "diagnostic_validation", "diagnostic_test"):
        columns = parquet_columns(con, paths[name])
        required = required_common - ({"data_split"} if name.startswith("diagnostic") else set())
        missing = required - columns
        if missing:
            raise RuntimeError(
                f"{name} is missing required columns: {sorted(missing)}"
            )
        if "canonical_guzergah_id" in columns:
            raise RuntimeError(f"{name} contains forbidden canonical_guzergah_id.")
    input_columns = parquet_columns(con, paths["input"])
    if WEIGHT_COLUMN not in input_columns:
        raise RuntimeError(f"Training input is missing {WEIGHT_COLUMN}.")


def load_frame(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    features: list[str],
    *,
    where: str | None = None,
    include_weight: bool = False,
    include_ids: bool = False,
) -> pd.DataFrame:
    columns = list(features) + [TARGET, SCENARIO_COLUMN, "data_split", "target_date"]
    if include_weight:
        columns.append(WEIGHT_COLUMN)
    if include_ids:
        columns.extend(["scenario_training_row_id", "training_row_id", "history_mask_type"])
    columns = list(dict.fromkeys(columns))
    predicate = f"WHERE {where}" if where else ""
    return con.execute(
        f"""
        SELECT {select_expression(columns)}
        FROM read_parquet({sql_string(path)})
        {predicate}
        ORDER BY target_date, scenario_training_row_id
        """
    ).fetch_df()


def prepare_features(
    frame: pd.DataFrame,
    features: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    matrix = frame[features].copy()
    for column in categorical:
        matrix[column] = matrix[column].astype("string").fillna("__MISSING__").astype(str)
    return matrix


def model_parameters(args: argparse.Namespace) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_seed": args.random_seed,
        "task_type": args.task_type,
        "thread_count": args.thread_count,
        "allow_writing_files": False,
        "verbose": 100,
    }
    if args.task_type == "GPU":
        parameters["devices"] = args.devices
    return parameters


def train_model(
    name: str,
    parameters: dict[str, Any],
    train_matrix: pd.DataFrame,
    train_target: np.ndarray,
    validation_matrix: pd.DataFrame,
    validation_target: np.ndarray,
    categorical: list[str],
    early_stopping_rounds: int,
    weights: np.ndarray | None,
) -> CatBoostRegressor:
    print(f"\nTraining {name} CatBoost model...")
    train_pool = Pool(
        train_matrix,
        label=train_target,
        weight=weights,
        cat_features=categorical,
    )
    validation_pool = Pool(
        validation_matrix,
        label=validation_target,
        cat_features=categorical,
    )
    model = CatBoostRegressor(**parameters)
    model.fit(
        train_pool,
        eval_set=validation_pool,
        early_stopping_rounds=early_stopping_rounds,
        use_best_model=True,
    )
    return model


def demand_band(target: pd.Series) -> pd.Categorical:
    return pd.cut(
        target,
        bins=[-np.inf, 9, 19, 29, 42, np.inf],
        labels=DEMAND_BANDS,
        ordered=True,
    )


def metric_record(
    dataset: str,
    model_name: str,
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    breakdown_type: str,
    breakdown_value: str,
) -> dict[str, Any]:
    error = prediction - target
    return {
        "dataset": dataset,
        "model": model_name,
        "breakdown_type": breakdown_type,
        "breakdown_value": breakdown_value,
        "row_count": int(len(target)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "target_mean": float(np.mean(target)),
        "prediction_mean": float(np.mean(prediction)),
    }


def evaluate_predictions(
    dataset: str,
    frame: pd.DataFrame,
    predictions: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    target = frame[TARGET].to_numpy(dtype=np.float64)
    records: list[dict[str, Any]] = []
    bands = demand_band(frame[TARGET])
    for model_name, prediction in predictions.items():
        records.append(
            metric_record(
                dataset,
                model_name,
                target,
                prediction,
                breakdown_type="OVERALL",
                breakdown_value="ALL",
            )
        )
        for scenario in sorted(frame[SCENARIO_COLUMN].dropna().unique()):
            mask = frame[SCENARIO_COLUMN].eq(scenario).to_numpy()
            records.append(
                metric_record(
                    dataset,
                    model_name,
                    target[mask],
                    prediction[mask],
                    breakdown_type="TRAINING_SCENARIO",
                    breakdown_value=str(scenario),
                )
            )
        for band in DEMAND_BANDS:
            mask = np.asarray(bands == band)
            if mask.any():
                records.append(
                    metric_record(
                        dataset,
                        model_name,
                        target[mask],
                        prediction[mask],
                        breakdown_type="DEMAND_GROUP",
                        breakdown_value=band,
                    )
                )
        for scenario in sorted(frame[SCENARIO_COLUMN].dropna().unique()):
            scenario_mask = frame[SCENARIO_COLUMN].eq(scenario).to_numpy()
            for band in DEMAND_BANDS:
                mask = scenario_mask & np.asarray(bands == band)
                if mask.any():
                    records.append(
                        metric_record(
                            dataset,
                            model_name,
                            target[mask],
                            prediction[mask],
                            breakdown_type="SCENARIO_X_DEMAND_GROUP",
                            breakdown_value=f"{scenario}|{band}",
                        )
                    )
    return records


def comparison_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "breakdown_type", "breakdown_value", "row_count"]
    values = ["mae", "rmse", "bias", "target_mean", "prediction_mean"]
    wide = metrics.pivot(index=keys, columns="model", values=values).reset_index()
    wide.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in wide.columns
    ]
    wide["weighted_minus_unweighted_mae"] = (
        wide["mae_weighted"] - wide["mae_unweighted"]
    )
    wide["weighted_mae_improvement_rate"] = (
        -wide["weighted_minus_unweighted_mae"] / wide["mae_unweighted"]
    )
    wide["weighted_minus_unweighted_rmse"] = (
        wide["rmse_weighted"] - wide["rmse_unweighted"]
    )
    return wide.sort_values(["dataset", "breakdown_type", "breakdown_value"])


def select_model(
    comparisons: pd.DataFrame, max_exact_degradation_rate: float
) -> dict[str, Any]:
    def row(dataset: str, kind: str, value: str) -> pd.Series:
        matched = comparisons[
            comparisons["dataset"].eq(dataset)
            & comparisons["breakdown_type"].eq(kind)
            & comparisons["breakdown_value"].eq(value)
        ]
        if len(matched) != 1:
            raise RuntimeError(
                f"Expected one metric row for {dataset}/{kind}/{value}, got {len(matched)}."
            )
        return matched.iloc[0]

    diagnostic = row("diagnostic_validation", "OVERALL", "ALL")
    exact = row(
        "untouched_validation",
        "TRAINING_SCENARIO",
        "SAME_COMPANY_EXACT",
    )
    diagnostic_improved = float(diagnostic["weighted_minus_unweighted_mae"]) < 0
    exact_degradation_rate = float(
        (exact["mae_weighted"] - exact["mae_unweighted"])
        / exact["mae_unweighted"]
    )
    exact_guardrail_passed = exact_degradation_rate <= max_exact_degradation_rate
    selected = "weighted" if diagnostic_improved and exact_guardrail_passed else "unweighted"
    return {
        "selected_model": selected,
        "selection_dataset": "validation only",
        "diagnostic_validation_mae_unweighted": float(diagnostic["mae_unweighted"]),
        "diagnostic_validation_mae_weighted": float(diagnostic["mae_weighted"]),
        "diagnostic_validation_weighted_improvement_rate": float(
            diagnostic["weighted_mae_improvement_rate"]
        ),
        "untouched_exact_validation_mae_unweighted": float(exact["mae_unweighted"]),
        "untouched_exact_validation_mae_weighted": float(exact["mae_weighted"]),
        "untouched_exact_validation_degradation_rate": exact_degradation_rate,
        "max_allowed_exact_validation_degradation_rate": max_exact_degradation_rate,
        "diagnostic_improved": bool(diagnostic_improved),
        "exact_guardrail_passed": bool(exact_guardrail_passed),
        "rule": (
            "Select weighted only if diagnostic validation overall MAE improves and "
            "untouched SAME_COMPANY_EXACT validation MAE degradation is within the "
            "configured guardrail. Test metrics do not affect selection."
        ),
    }


def json_default(value: object) -> object:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.early_stopping_rounds < 1:
        raise SystemExit("--iterations and --early-stopping-rounds must be positive.")
    if args.max_exact_mae_degradation_rate < 0:
        raise SystemExit("--max-exact-mae-degradation-rate must be non-negative.")

    paths = {
        "input": Path(args.input).resolve(),
        "features": Path(args.features).resolve(),
        "diagnostic_validation": Path(args.diagnostic_validation).resolve(),
        "diagnostic_test": Path(args.diagnostic_test).resolve(),
    }
    require_files(paths)
    output_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    features, categorical = load_feature_contract(paths["features"])
    con = duckdb.connect()
    try:
        validate_schemas(con, paths, features)
        train = load_frame(
            con,
            paths["input"],
            features,
            where="data_split = 'TRAIN'",
            include_weight=True,
        )
        evaluation_frames = {
            "untouched_validation": load_frame(
                con,
                paths["input"],
                features,
                where="data_split = 'VALIDATION' AND history_mask_type = 'NONE'",
                include_ids=True,
            ),
            "diagnostic_validation": load_frame(
                con,
                paths["diagnostic_validation"],
                features,
                include_ids=True,
            ),
            "untouched_test": load_frame(
                con,
                paths["input"],
                features,
                where="data_split = 'TEST' AND history_mask_type = 'NONE'",
                include_ids=True,
            ),
            "diagnostic_test": load_frame(
                con,
                paths["diagnostic_test"],
                features,
                include_ids=True,
            ),
        }
    finally:
        con.close()

    train_matrix = prepare_features(train, features, categorical)
    validation_matrix = prepare_features(
        evaluation_frames["untouched_validation"], features, categorical
    )
    train_target = train[TARGET].to_numpy(dtype=np.float64)
    validation_target = evaluation_frames["untouched_validation"][TARGET].to_numpy(
        dtype=np.float64
    )
    parameters = model_parameters(args)

    unweighted = train_model(
        "unweighted",
        parameters,
        train_matrix,
        train_target,
        validation_matrix,
        validation_target,
        categorical,
        args.early_stopping_rounds,
        None,
    )
    weighted = train_model(
        "weighted",
        parameters,
        train_matrix,
        train_target,
        validation_matrix,
        validation_target,
        categorical,
        args.early_stopping_rounds,
        train[WEIGHT_COLUMN].to_numpy(dtype=np.float64),
    )
    if list(unweighted.feature_names_) != features or list(weighted.feature_names_) != features:
        raise RuntimeError("Trained CatBoost feature contract changed unexpectedly.")

    model_paths = {
        "unweighted": model_dir / "stop_addition_demand_unweighted.cbm",
        "weighted": model_dir / "stop_addition_demand_weighted.cbm",
    }
    unweighted.save_model(model_paths["unweighted"])
    weighted.save_model(model_paths["weighted"])

    all_metrics: list[dict[str, Any]] = []
    prediction_outputs: list[pd.DataFrame] = []
    for dataset_name, frame in evaluation_frames.items():
        matrix = prepare_features(frame, features, categorical)
        predictions = {
            "unweighted": unweighted.predict(matrix),
            "weighted": weighted.predict(matrix),
        }
        all_metrics.extend(evaluate_predictions(dataset_name, frame, predictions))
        output = frame[
            [
                "scenario_training_row_id",
                "training_row_id",
                "data_split",
                "target_date",
                SCENARIO_COLUMN,
                "history_mask_type",
                TARGET,
            ]
        ].copy()
        output.insert(0, "evaluation_dataset", dataset_name)
        output["unweighted_prediction"] = predictions["unweighted"]
        output["weighted_prediction"] = predictions["weighted"]
        prediction_outputs.append(output)

    metrics = pd.DataFrame(all_metrics)
    comparisons = comparison_rows(metrics)
    selection = select_model(comparisons, args.max_exact_mae_degradation_rate)
    selected_name = str(selection["selected_model"])
    selected_path = model_dir / "stop_addition_demand_selected.cbm"
    (weighted if selected_name == "weighted" else unweighted).save_model(selected_path)

    metrics_path = output_dir / "stop_addition_demand_model_metrics.csv"
    comparison_path = output_dir / "stop_addition_demand_model_comparison.csv"
    predictions_path = output_dir / "stop_addition_demand_model_predictions.parquet"
    feature_importance_path = (
        output_dir / "stop_addition_demand_model_feature_importance.csv"
    )
    summary_path = output_dir / "stop_addition_demand_model_summary.json"
    metrics.to_csv(metrics_path, index=False)
    comparisons.to_csv(comparison_path, index=False)
    prediction_frame = pd.concat(prediction_outputs, ignore_index=True)
    export_con = duckdb.connect()
    try:
        export_con.register("prediction_frame", prediction_frame)
        export_con.execute(
            f"""
            COPY prediction_frame TO {sql_string(predictions_path)}
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )
    finally:
        export_con.close()
    pd.DataFrame(
        {
            "feature": features,
            "unweighted_importance": unweighted.get_feature_importance(),
            "weighted_importance": weighted.get_feature_importance(),
        }
    ).sort_values("weighted_importance", ascending=False).to_csv(
        feature_importance_path, index=False
    )

    summary = {
        "schema_version": "stop_addition_demand_model_comparison_v1",
        "target": TARGET,
        "training_rows": int(len(train)),
        "feature_count": len(features),
        "feature_names": features,
        "categorical_feature_names": categorical,
        "forbidden_features": sorted(FORBIDDEN_FEATURES),
        "weight_column_for_weighted_model": WEIGHT_COLUMN,
        "catboost_version": catboost.__version__,
        "parameters": parameters,
        "best_iterations": {
            "unweighted": int(unweighted.get_best_iteration()),
            "weighted": int(weighted.get_best_iteration()),
        },
        "evaluation_row_counts": {
            name: int(len(frame)) for name, frame in evaluation_frames.items()
        },
        "selection": selection,
        "models": {name: str(path) for name, path in model_paths.items()},
        "selected_model_path": str(selected_path),
        "metrics_path": str(metrics_path),
        "comparison_path": str(comparison_path),
        "predictions_path": str(predictions_path),
        "feature_importance_path": str(feature_importance_path),
        "notes": [
            "combined_weight is used only by the weighted candidate.",
            "training_scenario is categorical; history_mask_type is excluded.",
            "Selection uses validation metrics only; test results are confirmatory.",
            "No dataset preparation artifacts are modified by this script.",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=json_default),
        encoding="utf-8",
    )

    print("\nStop-addition CatBoost comparison complete.")
    print(f"Selected model: {selected_name}")
    print(
        "Diagnostic validation MAE: "
        f"unweighted={selection['diagnostic_validation_mae_unweighted']:.6f}, "
        f"weighted={selection['diagnostic_validation_mae_weighted']:.6f}"
    )
    print(
        "Untouched exact validation MAE: "
        f"unweighted={selection['untouched_exact_validation_mae_unweighted']:.6f}, "
        f"weighted={selection['untouched_exact_validation_mae_weighted']:.6f}"
    )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
