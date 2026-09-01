import tempfile
import unittest
import zipfile
from pathlib import Path

from backtest.exness_tick_history import archive_url, ticks_to_one_minute


class ExnessTickHistoryTests(unittest.TestCase):
    def test_archive_urls(self):
        self.assertTrue(archive_url("XAUUSD", "2025").endswith("/2025/Exness_XAUUSD_2025.zip"))
        self.assertTrue(archive_url("XAUUSDm", "2026-08").endswith("/2026/08/Exness_XAUUSDm_2026_08.zip"))
        self.assertTrue(archive_url("XAUUSD", "2026-09-01").endswith("/2026/09/01/Exness_XAUUSD_2026_09_01.zip"))

    def test_chunk_boundary_keeps_minute_open_and_close(self):
        csv = """\"Exness\",\"Symbol\",\"Timestamp\",\"Bid\",\"Ask\"
\"exness\",\"XAUUSD\",\"2026-09-01 00:00:00.000Z\",100.0,100.2
\"exness\",\"XAUUSD\",\"2026-09-01 00:00:20.000Z\",101.0,101.2
\"exness\",\"XAUUSD\",\"2026-09-01 00:00:40.000Z\",99.0,99.2
\"exness\",\"XAUUSD\",\"2026-09-01 00:01:00.000Z\",102.0,102.3
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ticks.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("ticks.csv", csv)
            bars = ticks_to_one_minute([path], chunksize=2)
        first = bars.iloc[0]
        self.assertEqual(first["bid_open"], 100.0)
        self.assertEqual(first["bid_high"], 101.0)
        self.assertEqual(first["bid_low"], 99.0)
        self.assertEqual(first["bid_close"], 99.0)
        self.assertAlmostEqual(first["spread_max"], 0.2)
        self.assertEqual(first["volume"], 3)


if __name__ == "__main__":
    unittest.main()

