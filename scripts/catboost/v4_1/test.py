import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.catboost.v4_1.common import (
    EvaluationConfig,
    run_evaluation,
)


CONFIG = EvaluationConfig(
    key="test_2025_h2",
    label="2025 H2 test",
    history_start="2023-01-01",
    history_end_exclusive="2025-07-01",
    target_start="2025-07-01",
    target_end_exclusive="2026-01-01",
    expected_rows=1_517_010,
    baseline_mae=10.244093697542297,
    baseline_rmse=16.999945314248993,
)


if __name__ == "__main__":
    run_evaluation(CONFIG)
