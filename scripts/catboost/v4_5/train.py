"""Train CatBoost v4.5 with predetermined religious-holiday features.

Training uses 2024 target rows, 2023 long-term history, and strictly earlier
dates for rolling features.  Model selection uses 2025 H1 validation only.
This script must be launched manually; importing it does not run training.
"""

from __future__ import annotations

import pandas as pd
from catboost import CatBoostRegressor

from scripts.catboost.v4_5 import common


ITERATIONS = 6000
EARLY_STOPPING_ROUNDS = 300
FEATURE_IMPORTANCE_PATH = (
    common.RESULTS_DIR / "catboost_v4_5_validation_feature_importance.csv"
)
TRAIN_CONFIG = common.PeriodConfig(
    "train_2024", "2024 training", "2023-01-01", "2024-01-01",
    "2024-01-01", "2025-01-01", 2_797_931, False,
)


def _refuse_overwrite() -> None:
    collisions = [
        path
        for path in (
            common.MODEL_PATH,
            FEATURE_IMPORTANCE_PATH,
            *common.output_paths(common.PERIODS["validation"]),
        )
        if path.exists()
    ]
    if collisions:
        rendered = "\n".join(f"  - {path}" for path in collisions)
        raise FileExistsError("Refusing to overwrite v4.5 artifacts:\n" + rendered)


def main() -> None:
    common.ensure_environment()
    _refuse_overwrite()
    print("Building the separate 73-feature v4.5 experiment; frozen systems are read only.")
    train = common.build_matrix(TRAIN_CONFIG, training=True)
    validation = common.build_matrix(common.PERIODS["validation"], training=False)
    x_train = train[common.FEATURE_COLUMNS]
    x_validation = validation[common.FEATURE_COLUMNS]
    model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=ITERATIONS,
        learning_rate=0.08,
        depth=8,
        task_type="GPU",
        devices="0",
        random_seed=42,
        verbose=100,
        metric_period=5,
        allow_writing_files=False,
    )
    model.fit(
        x_train,
        train["target"],
        cat_features=common.MODEL_CATEGORICAL_FEATURES,
        eval_set=(x_validation, validation["target"]),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )
    if list(model.feature_names_) != common.FEATURE_COLUMNS:
        raise RuntimeError("Trained model feature names/order changed unexpectedly.")
    validation["v4_5_prediction"] = model.predict(x_validation)
    comparison = common.merge_frozen_comparisons(
        validation, common.PERIODS["validation"]
    )
    importance = pd.DataFrame(
        {"feature": common.FEATURE_COLUMNS, "importance": model.get_feature_importance()}
    ).sort_values("importance", ascending=False)
    importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    model.save_model(str(common.MODEL_PATH))
    common.write_evaluation_outputs(comparison, common.PERIODS["validation"])
    print(f"Best iteration: {model.get_best_iteration()}")
    print(f"Saved trees: {model.tree_count_}")
    print(f"Saved v4.5 model: {common.MODEL_PATH}")


if __name__ == "__main__":
    main()

