"""Report the frozen CatBoost v4.5 model on the already-observed 2026 period."""

from __future__ import annotations

from scripts.catboost.v4_5 import common


def main() -> None:
    test_outputs = common.output_paths(common.PERIODS["test"])
    if not all(path.exists() for path in test_outputs[:2]):
        raise RuntimeError("Run the one-shot 2025 H2 test before final reporting.")
    config = common.PERIODS["final"]
    common.refuse_existing_outputs(config)
    frame = common.build_matrix(config, training=False)
    model = common.load_model()
    frame["v4_5_prediction"] = model.predict(frame[common.FEATURE_COLUMNS])
    comparison = common.merge_frozen_comparisons(frame, config)
    common.write_evaluation_outputs(comparison, config)


if __name__ == "__main__":
    main()
