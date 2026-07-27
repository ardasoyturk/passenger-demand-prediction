import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.catboost.v4_2.common import PeriodConfig, run_frozen_evaluation


CONFIG = PeriodConfig(
    key="final_2026",
    label="2026 reporting-only final period",
    history_start="2023-01-01",
    history_end_exclusive="2026-01-01",
    target_start="2026-01-01",
    target_end_exclusive="2026-04-15",
    expected_rows=780_865,
)


if __name__ == "__main__":
    run_frozen_evaluation(CONFIG)
