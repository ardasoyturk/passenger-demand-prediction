"""Train the first CatBoost v4.3 business-distribution experiment.

Training rows are 2024, long-term features use 2023 only, and rolling
features use only dates strictly earlier than each 2024 row.  Model selection
and reporting use 2025 H1 validation only.  This script never reads test or
final-period targets.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from catboost import CatBoostRegressor
import pandas as pd

from scripts.catboost.v4_3 import common


ITERATIONS = 6000
EARLY_STOPPING_ROUNDS = 300
FEATURE_IMPORTANCE_PATH = (
    common.RESULTS_DIR / "catboost_v4_3_validation_feature_importance.csv"
)

TRAIN_CONFIG = common.PeriodConfig(
    "train_2024",
    "2024 training",
    "2023-01-01",
    "2024-01-01",
    "2024-01-01",
    "2025-01-01",
    2_797_931,
    False,
)


def _fetch_matrix(
    conn,
    config: common.PeriodConfig,
    *,
    training: bool,
) -> pd.DataFrame:
    common.create_source_tables(conn, config, training=training)
    if training:
        # Earlier 2024 rows are valid rolling history for later 2024 rows.
        # The strict reference-date predicate prevents same-day leakage.
        conn.execute("""
            CREATE OR REPLACE TEMP TABLE v4_3_recent_history AS
            SELECT * FROM v4_3_history
            UNION ALL
            SELECT * FROM v4_3_target
        """)
        recent_history_table = "v4_3_recent_history"
    else:
        recent_history_table = "v4_3_history"
    common.create_long_term_statistics(conn)
    common.create_recent_statistics(
        conn,
        config,
        history_table=recent_history_table,
    )
    common.create_feature_table(conn)
    selected = [
        "SEFER_ID",
        "SEFER_TARIHI",
        *common.FEATURE_COLUMNS,
        "target",
        "baseline_prediction",
        "baseline_source",
    ]
    frame = conn.execute(
        f"SELECT {common.sql_identifier_list(selected)} FROM v4_3_features"
    ).fetchdf()
    common.validate_feature_frame(frame)
    if len(frame) != config.expected_rows:
        raise RuntimeError(
            f"Expected {config.expected_rows:,} {config.key} rows; "
            f"received {len(frame):,}."
        )
    return frame


def _refuse_overwrite() -> None:
    expected_outputs = [
        common.MODEL_PATH,
        FEATURE_IMPORTANCE_PATH,
        common.RESULTS_DIR
        / "catboost_v4_3_validation_2025_h1_business_metrics.csv",
        common.RESULTS_DIR
        / "catboost_v4_3_validation_2025_h1_predictions.csv",
        common.RESULTS_DIR
        / "catboost_v4_3_validation_2025_h1_metadata.json",
    ]
    collisions = [path for path in expected_outputs if path.exists()]
    if collisions:
        rendered = "\n".join(f"  - {path}" for path in collisions)
        raise FileExistsError(
            "Refusing to overwrite a v4.3 experiment artifact:\n" + rendered
        )


def main() -> None:
    common.ensure_environment()
    _refuse_overwrite()
    print("v4.2 remains frozen; training a separate 72-feature v4.3 model.")
    print("Building 2024 training and 2025 H1 validation matrices in DuckDB.")

    conn = common.connect()
    try:
        train = _fetch_matrix(conn, TRAIN_CONFIG, training=True)
        validation = _fetch_matrix(
            conn,
            common.PERIODS["validation"],
            training=False,
        )
    finally:
        conn.close()

    x_train = train[common.FEATURE_COLUMNS]
    y_train = train["target"]
    x_validation = validation[common.FEATURE_COLUMNS]
    y_validation = validation["target"]
    print(f"Training matrix: {x_train.shape}")
    print(f"Validation matrix: {x_validation.shape}")

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
        y_train,
        cat_features=common.CATEGORICAL_FEATURES,
        eval_set=(x_validation, y_validation),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )
    if list(model.feature_names_) != common.FEATURE_COLUMNS:
        raise RuntimeError("Trained model feature names/order changed unexpectedly.")

    validation["v4_3_prediction"] = model.predict(x_validation)
    validation["v4_3_hybrid_prediction"] = (
        common.apply_frozen_hybrid_to_v4_3(validation)
    )
    comparison = common.merge_frozen_comparisons(
        validation,
        common.PERIODS["validation"],
    )

    feature_importance = pd.DataFrame(
        {
            "feature": common.FEATURE_COLUMNS,
            "importance": model.get_feature_importance(),
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    model.save_model(str(common.MODEL_PATH))
    common.write_comparison_outputs(
        comparison,
        common.PERIODS["validation"],
    )

    print(f"Best iteration: {model.get_best_iteration()}")
    print(f"Saved trees: {model.tree_count_}")
    print(f"Saved v4.3 model: {common.MODEL_PATH}")
    print(
        "Selection remains a business review: primary 10-40 MAE first, "
        "then FNR/FPR guardrails; overall MAE alone cannot promote v4.3."
    )


if __name__ == "__main__":
    main()
