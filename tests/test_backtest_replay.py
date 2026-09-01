import unittest

import pandas as pd

from backtest.replay import build_timeframes, closed_frames_at, entry_is_allowed


class BacktestReplayTests(unittest.TestCase):
    def test_entry_fingerprint_and_cooldown_match_live_alert_guard(self):
        now = pd.Timestamp("2026-01-01 12:00", tz="UTC")
        self.assertFalse(entry_is_allowed(now - pd.Timedelta(minutes=61), "same", "same", now, 60))
        self.assertFalse(entry_is_allowed(now - pd.Timedelta(minutes=59), "old", "new", now, 60))
        self.assertTrue(entry_is_allowed(now - pd.Timedelta(minutes=60), "old", "new", now, 60))

    def test_forming_candles_are_never_visible(self):
        index = pd.date_range("2026-01-01 00:00", periods=20, freq="1min", tz="UTC")
        one_minute = pd.DataFrame(
            {
                "open": range(20),
                "high": range(20),
                "low": range(20),
                "close": range(20),
                "volume": [1] * 20,
            },
            index=index,
        )
        frames = build_timeframes(one_minute)
        visible = closed_frames_at(frames, pd.Timestamp("2026-01-01 00:10", tz="UTC"))
        self.assertEqual(visible["1min"].index[-1], pd.Timestamp("2026-01-01 00:09", tz="UTC"))
        self.assertEqual(visible["5min"].index[-1], pd.Timestamp("2026-01-01 00:05", tz="UTC"))
        self.assertNotIn(pd.Timestamp("2026-01-01 00:10", tz="UTC"), visible["5min"].index)


if __name__ == "__main__":
    unittest.main()
