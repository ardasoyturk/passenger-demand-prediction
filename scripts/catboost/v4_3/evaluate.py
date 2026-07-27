r"""One-period evaluation for the frozen CatBoost v4.3 experiment.

Usage:
    uv run python scripts/catboost/v4_3/evaluate.py test
    uv run python scripts/catboost/v4_3/evaluate.py final

The final period is reporting-only because it has already been observed.
Existing outputs are never overwritten, making the test run one-shot per
frozen model artifact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.catboost.v4_3 import common


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("period", choices=("test", "final"))
    return parser.parse_args()


def _require_workflow_order(period: str) -> None:
    validation_metrics = (
        common.RESULTS_DIR
        / "catboost_v4_3_validation_2025_h1_business_metrics.csv"
    )
    if not validation_metrics.exists():
        raise RuntimeError("Run and review v4.3 validation before later periods.")
    if period == "final":
        test_metrics = (
            common.RESULTS_DIR
            / "catboost_v4_3_test_2025_h2_business_metrics.csv"
        )
        if not test_metrics.exists():
            raise RuntimeError("Run the one-shot 2025 H2 test before final reporting.")


def _refuse_existing_period_outputs(period: str) -> None:
    config = common.PERIODS[period]
    stem = common.RESULTS_DIR / f"catboost_v4_3_{config.key}"
    collisions = [
        path
        for path in (
            Path(f"{stem}_business_metrics.csv"),
            Path(f"{stem}_predictions.csv"),
            Path(f"{stem}_metadata.json"),
        )
        if path.exists()
    ]
    if collisions:
        rendered = "\n".join(f"  - {path}" for path in collisions)
        raise FileExistsError(
            "Refusing to repeat or overwrite this frozen-period evaluation:\n"
            + rendered
        )


def main() -> None:
    args = _parse_args()
    _require_workflow_order(args.period)
    _refuse_existing_period_outputs(args.period)
    config = common.PERIODS[args.period]
    print(
        f"Evaluating frozen v4.3 on {config.label}. "
        f"Tuning allowed: {config.tuning_allowed}."
    )
    frame = common.build_evaluation_matrix(config)
    model = common.load_v4_3_model()
    frame["v4_3_prediction"] = model.predict(frame[common.FEATURE_COLUMNS])
    frame["v4_3_hybrid_prediction"] = (
        common.apply_frozen_hybrid_to_v4_3(frame)
    )
    comparison = common.merge_frozen_comparisons(frame, config)
    common.write_comparison_outputs(comparison, config)
    print("Evaluation artifacts written once; reruns refuse to overwrite them.")


if __name__ == "__main__":
    main()
