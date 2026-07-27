import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.catboost.v4_2.common import PeriodConfig, run_validation


CONFIG = PeriodConfig(
    key="validation_2025_h1",
    label="2025 H1 validation",
    history_start="2023-01-01",
    history_end_exclusive="2025-01-01",
    target_start="2025-01-01",
    target_end_exclusive="2025-07-01",
    expected_rows=1_420_182,
)


if __name__ == "__main__":
    run_validation(CONFIG)
