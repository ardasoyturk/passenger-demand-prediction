import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.catboost.v4_1.common import (
    EvaluationConfig,
    run_evaluation,
)


CONFIG = EvaluationConfig(
    key="final_2026",
    label="2026 official final evaluation",
    history_start="2023-01-01",
    history_end_exclusive="2026-01-01",
    target_start="2026-01-01",
    target_end_exclusive="2026-04-15",
    expected_rows=780_865,
    baseline_mae=9.86210681757341,
    baseline_rmse=14.361139756955968,
)


if __name__ == "__main__":
    run_evaluation(CONFIG)
