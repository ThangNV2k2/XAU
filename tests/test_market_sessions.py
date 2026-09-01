import unittest
from datetime import datetime, timezone

from market_sessions import timeframe_due, us_stock_session, xau_session


class MarketSessionTests(unittest.TestCase):
    def test_stock_hours_follow_new_york_dst(self):
        summer_open = us_stock_session(datetime(2026, 7, 6, 13, 40, 8, tzinfo=timezone.utc))
        winter_open = us_stock_session(datetime(2026, 1, 5, 14, 40, 8, tzinfo=timezone.utc))
        self.assertTrue(summer_open.is_open)
        self.assertTrue(winter_open.is_open)
        self.assertEqual(summer_open.local_time.hour, 9)
        self.assertEqual(winter_open.local_time.hour, 9)
        self.assertEqual(summer_open.phase, "OPENING_RANGE")
        self.assertFalse(summer_open.allow_new_entry)

    def test_stock_retest_midday_and_close_guards(self):
        retest = us_stock_session(datetime(2026, 7, 6, 14, 0, tzinfo=timezone.utc))
        midday = us_stock_session(datetime(2026, 7, 6, 16, 0, tzinfo=timezone.utc))
        closed = us_stock_session(datetime(2026, 7, 6, 19, 45, tzinfo=timezone.utc))
        self.assertEqual(retest.phase, "RETEST_WINDOW")
        self.assertTrue(retest.allow_new_entry)
        self.assertEqual(midday.phase, "MIDDAY")
        self.assertFalse(midday.allow_new_entry)
        self.assertFalse(closed.is_open)

    def test_xau_daily_maintenance_and_reopen(self):
        maintenance = xau_session(datetime(2026, 7, 6, 21, 0, tzinfo=timezone.utc))
        reopened = xau_session(datetime(2026, 7, 6, 22, 2, tzinfo=timezone.utc))
        self.assertFalse(maintenance.is_open)
        self.assertEqual(maintenance.phase, "MAINTENANCE")
        self.assertTrue(reopened.is_open)

    def test_timeframe_calls_are_bar_aligned(self):
        at_close = datetime(2026, 7, 6, 12, 15, 8, tzinfo=timezone.utc)
        self.assertTrue(timeframe_due("1min", at_close))
        self.assertTrue(timeframe_due("5min", at_close))
        self.assertTrue(timeframe_due("15min", at_close))
        self.assertFalse(timeframe_due("1h", at_close))
        self.assertFalse(timeframe_due("5min", at_close.replace(minute=16)))
        self.assertFalse(timeframe_due("1min", at_close.replace(second=2)))


if __name__ == "__main__":
    unittest.main()
