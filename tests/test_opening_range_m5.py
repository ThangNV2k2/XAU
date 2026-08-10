import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from opening_range_m5 import (
    VIETNAM_TZ,
    assess_opening_range_m5,
    build_m5_trade_plan,
    is_monitoring_time,
    seconds_until_next_m5_check,
)


def vn_time(hour: int, minute: int, second: int = 0) -> datetime:
    return datetime(2026, 8, 10, hour, minute, second, tzinfo=VIETNAM_TZ)


def m5_frame(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    local_index = pd.DatetimeIndex(
        [datetime.fromisoformat(value).replace(tzinfo=VIETNAM_TZ) for value, *_ in rows]
    )
    return pd.DataFrame(
        {
            "open": [row[1] for row in rows],
            "high": [row[2] for row in rows],
            "low": [row[3] for row in rows],
            "close": [row[4] for row in rows],
        },
        index=local_index.tz_convert("UTC"),
    )


class OpeningRangeM5Tests(unittest.TestCase):
    def setUp(self):
        self.reference = ("2026-08-10T20:30:00", 100.0, 105.0, 95.0, 101.0)

    def test_monitoring_window_is_exactly_2035_until_midnight_vietnam(self):
        self.assertFalse(is_monitoring_time(vn_time(20, 34, 59)))
        self.assertTrue(is_monitoring_time(vn_time(20, 35)))
        self.assertTrue(is_monitoring_time(vn_time(23, 59, 59)))
        midnight = datetime(2026, 8, 11, 0, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh"))
        self.assertFalse(is_monitoring_time(midnight))

    def test_scheduler_aligns_just_after_each_five_minute_close(self):
        self.assertEqual(seconds_until_next_m5_check(vn_time(20, 34, 58)), 4.0)
        self.assertEqual(seconds_until_next_m5_check(vn_time(20, 35, 1)), 1.0)
        self.assertEqual(seconds_until_next_m5_check(vn_time(20, 35, 2)), 300.0)
        self.assertEqual(seconds_until_next_m5_check(vn_time(20, 35, 3)), 299.0)

    def test_at_2035_uses_exact_2030_to_2035_candle(self):
        frame = m5_frame([self.reference])
        assessment = assess_opening_range_m5(frame, vn_time(20, 35))
        self.assertEqual(assessment.status, "WAIT_BREAKOUT")
        self.assertEqual(assessment.opening_high, 105.0)
        self.assertEqual(assessment.opening_low, 95.0)

    def test_breakout_candle_cannot_confirm_its_own_retest(self):
        frame = m5_frame(
            [
                self.reference,
                ("2026-08-10T20:35:00", 104.0, 107.0, 104.5, 106.5),
            ]
        )
        assessment = assess_opening_range_m5(frame, vn_time(20, 40))
        self.assertEqual(assessment.status, "WAIT_RETEST")
        self.assertEqual(assessment.breakout_side, "LONG")

    def test_long_signal_requires_closed_later_retest_candle(self):
        frame = m5_frame(
            [
                self.reference,
                ("2026-08-10T20:35:00", 104.0, 107.0, 103.8, 106.5),
                ("2026-08-10T20:40:00", 105.2, 107.0, 104.7, 106.7),
            ]
        )
        before_close = assess_opening_range_m5(frame, vn_time(20, 44, 59))
        self.assertEqual(before_close.status, "WAIT_RETEST")

        confirmed = assess_opening_range_m5(frame, vn_time(20, 45))
        self.assertEqual(confirmed.status, "SIGNAL")
        self.assertEqual(confirmed.breakout_side, "LONG")
        self.assertEqual(confirmed.confirmation_close, 106.7)
        self.assertAlmostEqual(confirmed.structural_stop, 104.5)

        plan = build_m5_trade_plan(confirmed, executable_price=106.8)
        self.assertAlmostEqual(plan.entry, 106.8)
        self.assertAlmostEqual(plan.stop_loss, 104.5)
        self.assertAlmostEqual(plan.take_profit_1, 110.25)
        self.assertAlmostEqual(plan.take_profit_2, 111.4)

    def test_short_signal_and_targets_are_symmetric(self):
        frame = m5_frame(
            [
                self.reference,
                ("2026-08-10T20:35:00", 96.0, 96.2, 93.5, 94.0),
                ("2026-08-10T20:40:00", 94.4, 95.3, 93.8, 94.0),
            ]
        )
        assessment = assess_opening_range_m5(frame, vn_time(20, 45))
        self.assertEqual(assessment.status, "SIGNAL")
        self.assertEqual(assessment.breakout_side, "SHORT")
        self.assertAlmostEqual(assessment.structural_stop, 95.5)

        plan = build_m5_trade_plan(assessment, executable_price=93.9)
        self.assertAlmostEqual(plan.risk, 1.6)
        self.assertAlmostEqual(plan.take_profit_1, 91.5)
        self.assertAlmostEqual(plan.take_profit_2, 90.7)

    def test_failed_long_breakout_can_reset_before_later_short_breakout(self):
        frame = m5_frame(
            [
                self.reference,
                ("2026-08-10T20:35:00", 104.0, 107.0, 103.8, 106.0),
                ("2026-08-10T20:40:00", 105.8, 106.0, 104.0, 104.8),
                ("2026-08-10T20:45:00", 96.0, 96.1, 93.5, 94.0),
                ("2026-08-10T20:50:00", 94.4, 95.3, 93.8, 94.0),
            ]
        )
        assessment = assess_opening_range_m5(frame, vn_time(20, 55))
        self.assertEqual(assessment.status, "SIGNAL")
        self.assertEqual(assessment.breakout_side, "SHORT")
        self.assertEqual(
            assessment.breakout_candle_start.astimezone(VIETNAM_TZ).strftime("%H:%M"),
            "20:45",
        )


if __name__ == "__main__":
    unittest.main()
