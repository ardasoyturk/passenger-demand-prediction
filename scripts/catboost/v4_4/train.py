"""Train independent CatBoost v4.4 passenger-demand threshold classifiers.

This script must be launched manually.  Use --dry-run to validate the setup
without building feature matrices, CatBoost pools, models, or result files.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import catboost
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from scripts.catboost.v4_4 import common
from scripts.catboost.v4_3 import common as feature_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=6000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--early-stopping-rounds", type=int, default=300)
    parser.add_argument("--class-weights", choices=("none", "balanced"), default="balanced")
    parser.add_argument("--thresholds", type=common.parse_thresholds, default=list(common.DEFAULT_THRESHOLDS))
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be positive")
    if args.depth <= 0:
        raise ValueError("--depth must be positive")
    if args.early_stopping_rounds <= 0:
        raise ValueError("--early-stopping-rounds must be positive")


def model_parameters(args: argparse.Namespace) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "task_type": "GPU",
        "devices": "0",
        "random_seed": args.random_seed,
        "verbose": 100,
        "allow_writing_files": False,
    }
    if args.class_weights == "balanced":
        parameters["auto_class_weights"] = "Balanced"
    return parameters


def planned_paths(thresholds: list[int], class_weight_mode: str) -> list:
    paths = []
    for threshold in thresholds:
        paths.extend([
            common.model_path(threshold, class_weight_mode),
            common.metadata_path(threshold, class_weight_mode),
            common.cutoff_path(threshold, class_weight_mode),
        ])
    paths.extend(common.output_paths("validation", class_weight_mode).__dict__.values())
    return paths


def run_dry_run(args: argparse.Namespace) -> None:
    columns = common.validate_database_schema()
    splits, distributions = common.inspect_split_counts()
    print("v4.4 DRY RUN — no feature matrices or CatBoost pools will be created")
    print(f"Database: {common.DB_PATH}")
    print(f"model_data_base columns validated: {len(columns)}")
    print(f"Feature schema: {len(feature_pipeline.FEATURE_COLUMNS)} unique v4.3 features")
    print("Nominal training source: 2023-01-01 through 2024-12-31")
    print("Effective supervised rows: 2024-01-01 through 2024-12-31; 2023 is leakage-safe history")
    print("\nSplit row counts")
    print(splits.to_string(index=False))
    print("\n2024 supervised target class distributions")
    print(distributions.to_string(index=False))
    print("\nTraining parameters")
    print(json.dumps(model_parameters(args), indent=2))
    print(f"early_stopping_rounds: {args.early_stopping_rounds}")
    print(f"class_weight_mode: {args.class_weights}")
    print(f"thresholds: {args.thresholds}")
    print("\nPlanned output paths")
    for path in planned_paths(args.thresholds, args.class_weights):
        print(path)
    existing = [
        path for path in planned_paths(args.thresholds, args.class_weights)
        if path.exists()
    ]
    if existing:
        rendered = "\n".join(str(path) for path in existing)
        raise FileExistsError("Dry-run found output collisions:\n" + rendered)
    print("\nDry-run validation passed. No training was run and no outputs were created.")


def best_validation_auc(model: CatBoostClassifier) -> float:
    scores = model.get_best_score()
    for dataset_name in ("validation", "validation_0"):
        if dataset_name in scores and "AUC" in scores[dataset_name]:
            return float(scores[dataset_name]["AUC"])
    raise RuntimeError(f"Validation AUC missing from CatBoost best scores: {scores}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        run_dry_run(args)
        return

    common.validate_database_schema()
    common.ensure_directories()
    common.refuse_overwrite(planned_paths(args.thresholds, args.class_weights))
    parameters = model_parameters(args)
    print("v4.2 hybrid remains frozen; training separate v4.4 binary classifiers.")
    print("Building the shared 72-feature v4.3 matrices in DuckDB.")
    train = common.build_feature_matrix(common.TRAIN_CONFIG, training=True)
    validation = common.build_feature_matrix(common.PERIODS["validation"], training=False)
    validation = common.merge_comparison_predictions(validation, "validation")

    x_train = train[feature_pipeline.FEATURE_COLUMNS]
    x_validation = validation[feature_pipeline.FEATURE_COLUMNS]
    metric_frames = []
    prediction_frames = []
    calibration_frames = []
    summaries = []

    for threshold in args.thresholds:
        label_name = common.target_name(threshold)
        y_train = (train["target"].to_numpy() >= threshold).astype(np.int8)
        y_validation = (validation["target"].to_numpy() >= threshold).astype(np.int8)
        distribution = common.class_distribution(train["target"].to_numpy(), threshold)
        print(f"\n{label_name} training distribution")
        for key, value in distribution.items():
            print(f"{key}: {value}")

        model = CatBoostClassifier(**parameters)
        model.fit(
            x_train,
            y_train,
            cat_features=feature_pipeline.CATEGORICAL_FEATURES,
            eval_set=(x_validation, y_validation),
            early_stopping_rounds=args.early_stopping_rounds,
            use_best_model=True,
        )
        if list(model.feature_names_) != feature_pipeline.FEATURE_COLUMNS:
            raise RuntimeError("Trained classifier feature names/order changed unexpectedly")

        probability = model.predict_proba(x_validation)[:, 1]
        cutoffs = common.cutoff_candidates(y_validation, probability)
        selected = common.select_cutoff(cutoffs)
        frozen_cutoff = float(selected["probability_cutoff"])
        cutoffs.to_csv(common.cutoff_path(threshold, args.class_weights), index=False)

        metric_frames.append(
            common.metric_rows(validation, probability, frozen_cutoff, threshold, "validation")
        )
        prediction_frames.append(
            common.prediction_rows(validation, probability, frozen_cutoff, threshold)
        )
        calibration_frames.append(
            common.calibration_table(y_validation, probability, threshold=threshold, period="validation")
        )
        summaries.append(
            common.summary_row(y_validation, probability, frozen_cutoff, threshold, "validation")
        )

        metadata = {
            "schema_version": "catboost_v4_4_classifier_metadata_v1",
            "purpose": "passenger_demand_threshold_probability",
            "threshold": threshold,
            "target_name": label_name,
            "feature_count": len(feature_pipeline.FEATURE_COLUMNS),
            "feature_names": feature_pipeline.FEATURE_COLUMNS,
            "categorical_feature_names": feature_pipeline.CATEGORICAL_FEATURES,
            "nominal_training_date_range": {
                "start": "2023-01-01", "end_exclusive": "2025-01-01"
            },
            "effective_supervised_training_date_range": {
                "start": common.TRAIN_CONFIG.target_start,
                "end_exclusive": common.TRAIN_CONFIG.target_end_exclusive,
                "history_start": common.TRAIN_CONFIG.history_start,
                "long_term_history_end_exclusive": common.TRAIN_CONFIG.history_end_exclusive,
            },
            "validation_date_range": {
                "start": common.PERIODS["validation"].target_start,
                "end_exclusive": common.PERIODS["validation"].target_end_exclusive,
            },
            **distribution,
            "parameters": parameters,
            "early_stopping_rounds": args.early_stopping_rounds,
            "best_iteration": int(model.get_best_iteration()),
            "best_validation_auc": best_validation_auc(model),
            "catboost_version": catboost.__version__,
            "random_seed": args.random_seed,
            "class_weight_mode": args.class_weights,
            "cutoff_selection_period": "validation_2025_h1",
            "cutoff_selection_rule": "highest F1; then lower FNR; higher precision; closest to 0.50",
            "frozen_probability_cutoff": frozen_cutoff,
            "validation_metrics_at_frozen_cutoff": common.classification_metrics(
                y_validation, probability >= frozen_cutoff
            ),
            "validation_metrics_at_0_50": common.classification_metrics(
                y_validation, probability >= 0.5
            ),
            "validation_probability_metrics": common.probability_metrics(y_validation, probability),
        }
        model.save_model(str(common.model_path(threshold, args.class_weights)))
        common.metadata_path(threshold, args.class_weights).write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Frozen validation cutoff for {label_name}: {frozen_cutoff:.2f}")
        print(f"Saved model: {common.model_path(threshold, args.class_weights)}")

    paths = common.output_paths("validation", args.class_weights)
    pd.concat(metric_frames, ignore_index=True).to_csv(paths.metrics, index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(paths.predictions, index=False)
    pd.concat(calibration_frames, ignore_index=True).to_csv(paths.calibration, index=False)
    pd.DataFrame(summaries).to_csv(paths.summary, index=False)
    print(f"Saved validation outputs under {common.RESULTS_DIR}")
    print("Do not change cutoffs using 2025 H2 or 2026 results.")


if __name__ == "__main__":
    main()
