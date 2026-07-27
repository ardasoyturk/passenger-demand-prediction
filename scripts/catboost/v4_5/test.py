"""Evaluate the frozen CatBoost v4.5 model on 2025 H2 exactly once."""

from __future__ import annotations

from scripts.catboost.v4_5 import common


def main() -> None:
    validation_outputs = common.output_paths(common.PERIODS["validation"])
    if not all(path.exists() for path in validation_outputs[:2]):
        raise RuntimeError("Run and review v4.5 validation before the 2025 H2 test.")
    config = common.PERIODS["test"]
    common.refuse_existing_outputs(config)
    frame = common.build_matrix(config, training=False)
    model = common.load_model()
    frame["v4_5_prediction"] = model.predict(frame[common.FEATURE_COLUMNS])
    comparison = common.merge_frozen_comparisons(frame, config)
    common.write_evaluation_outputs(comparison, config)


if __name__ == "__main__":
    main()
