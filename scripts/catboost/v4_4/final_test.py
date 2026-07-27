"""Report trained v4.4 classifiers on 2026 data with validation-frozen cutoffs."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.catboost.v4_4.common import DEFAULT_THRESHOLDS, run_frozen_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-weights", choices=("none", "balanced"), default="balanced")
    args = parser.parse_args()
    run_frozen_evaluation(
        "final",
        list(DEFAULT_THRESHOLDS),
        class_weight_mode=args.class_weights,
    )


if __name__ == "__main__":
    main()
