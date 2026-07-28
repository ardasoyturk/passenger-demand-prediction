from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodConfig:
    key: str
    label: str
    history_start: str
    history_end_exclusive: str
    target_start: str
    target_end_exclusive: str
    expected_rows: int
    tuning_allowed: bool = False


STANDARD_PERIODS = {
    "validation": PeriodConfig(
        "validation_2025_h1",
        "2025 H1 validation",
        "2023-01-01",
        "2025-01-01",
        "2025-01-01",
        "2025-07-01",
        1_420_182,
        True,
    ),
    "test": PeriodConfig(
        "test_2025_h2",
        "2025 H2 test",
        "2023-01-01",
        "2025-07-01",
        "2025-07-01",
        "2026-01-01",
        1_517_010,
        False,
    ),
    "final": PeriodConfig(
        "final_2026",
        "2026 final (reporting only; already observed)",
        "2023-01-01",
        "2026-01-01",
        "2026-01-01",
        "2026-04-15",
        780_865,
        False,
    ),
}
