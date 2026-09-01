import unittest
from datetime import datetime, timezone

from market_scanner import ExnessMarketScanner


class NoCallClient:
    def __init__(self):
        self.calls = 0

    def get_rates(self, *_args, **_kwargs):
        self.calls += 1
        raise AssertionError("Không được gọi data ngoài giờ")


class ScannerTests(unittest.TestCase):
    def test_execution_mode_is_locked_to_signal_only(self):
        with self.assertRaisesRegex(ValueError, "signal_only"):
            ExnessMarketScanner(
                NoCallClient(),
                {"assets": [{"symbol": "AAPL", "type": "stock", "mode": "auto"}]},
            )

    def test_closed_stock_market_makes_zero_data_calls(self):
        client = NoCallClient()
        scanner = ExnessMarketScanner(
            client,
            {
                "assets": [{"symbol": "AAPL", "type": "stock", "scan_minutes": 5}],
                "scanner": {"timeframes": ["1min"], "bars_per_timeframe": 80},
            },
        )
        saturday = datetime(2026, 7, 4, 15, 0, 8, tzinfo=timezone.utc)
        outcome = scanner.scan_asset(scanner.assets[0], saturday, force=True)
        self.assertEqual(outcome.status, "CLOSED")
        self.assertEqual(client.calls, 0)

    def test_stock_scan_is_due_only_each_five_minutes(self):
        scanner = ExnessMarketScanner(
            NoCallClient(),
            {
                "assets": [{"symbol": "AAPL", "type": "stock", "scan_minutes": 5}],
                "scanner": {"timeframes": ["1min"], "bars_per_timeframe": 80, "close_settle_seconds": 8},
            },
        )
        asset = scanner.assets[0]
        self.assertTrue(scanner.is_asset_due(asset, datetime(2026, 7, 6, 14, 5, 8, tzinfo=timezone.utc)))
        self.assertFalse(scanner.is_asset_due(asset, datetime(2026, 7, 6, 14, 6, 8, tzinfo=timezone.utc)))


if __name__ == "__main__":
    unittest.main()
