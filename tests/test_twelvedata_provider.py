import unittest
from unittest.mock import patch

from data_provider.twelvedata_provider import TwelveDataProvider


class TwelveDataProviderTests(unittest.TestCase):
    def test_quote_is_normalized_for_world_spot_m5_alert(self):
        provider = TwelveDataProvider(symbol="XAU/USD", api_key="test-key")
        payload = {
            "symbol": "XAU/USD",
            "timestamp": 1_786_366_505,
            "close": "3402.15",
            "is_market_open": True,
        }
        with patch.object(provider, "_request", return_value=payload):
            quote = provider.get_quote()

        self.assertEqual(quote["symbol"], "XAU/USD")
        self.assertEqual(quote["last_quote_at"], 1_786_366_505)
        self.assertAlmostEqual(quote["close"], 3402.15)
        self.assertEqual(quote["source"], "Twelve Data XAU/USD spot")


if __name__ == "__main__":
    unittest.main()
