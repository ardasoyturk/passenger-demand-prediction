from scripts.catboost.v4_2.common import PeriodConfig, run_frozen_evaluation


CONFIG = PeriodConfig(
    key="test_2025_h2",
    label="2025 H2 test",
    history_start="2023-01-01",
    history_end_exclusive="2025-07-01",
    target_start="2025-07-01",
    target_end_exclusive="2026-01-01",
    expected_rows=1_517_010,
)


if __name__ == "__main__":
    run_frozen_evaluation(CONFIG)
