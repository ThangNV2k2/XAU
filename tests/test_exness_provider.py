import unittest
from types import SimpleNamespace

from data_provider.exness_mt5_provider import ExnessConnectionError, ExnessMT5Client


class FakeMT5:
    TIMEFRAME_M1 = 1
    TIMEFRAME_M5 = 5
    TIMEFRAME_M15 = 15
    TIMEFRAME_H1 = 60
    TIMEFRAME_H4 = 240
    TIMEFRAME_D1 = 1440

    def __init__(self, server="Exness-MT5Trial"):
        self.server = server
        self.selected = []
        self.shutdown_called = False

    def initialize(self, **_kwargs):
        return True

    def account_info(self):
        return SimpleNamespace(server=self.server, company="Exness Technologies", name="Demo")

    def shutdown(self):
        self.shutdown_called = True

    def last_error(self):
        return (1, "ok")

    def symbols_get(self):
        return (
            SimpleNamespace(name="XAUUSDm", visible=False, trade_mode=4),
            SimpleNamespace(name="AAPL", visible=True, trade_mode=4),
        )

    def symbol_select(self, symbol, enabled):
        self.selected.append((symbol, enabled))
        return True

    def copy_rates_from_pos(self, symbol, timeframe, position, count):
        self.last_rates_call = (symbol, timeframe, position, count)
        return [
            {
                "time": 1_700_000_000 + index * 60,
                "open": 2000 + index,
                "high": 2001 + index,
                "low": 1999 + index,
                "close": 2000.5 + index,
                "tick_volume": 100 + index,
                "spread": 20,
                "real_volume": 0,
            }
            for index in range(count)
        ]

    def symbol_info_tick(self, _symbol):
        return SimpleNamespace(bid=2399.8, ask=2400.2, last=0.0, time=1_700_000_000)

    def symbol_info(self, symbol):
        return SimpleNamespace(
            trade_mode=4,
            path=f"Exness\\Metals\\{symbol}",
            digits=2,
            point=0.01,
            volume_min=0.01,
            volume_step=0.01,
        )


class ExnessProviderTests(unittest.TestCase):
    def test_resolves_exness_suffix_and_normalizes_rates(self):
        fake = FakeMT5()
        client = ExnessMT5Client(mt5_module=fake)
        self.assertEqual(client.resolve_symbol("XAUUSD"), "XAUUSDm")
        frame = client.get_rates("XAUUSD", "1min", outputsize=80)
        self.assertEqual(len(frame), 81)
        self.assertEqual(list(frame.columns), ["open", "high", "low", "close", "volume", "spread", "real_volume"])
        self.assertEqual(str(frame.index.tz), "UTC")
        self.assertEqual(fake.last_rates_call, ("XAUUSDm", fake.TIMEFRAME_M1, 0, 81))

    def test_quote_uses_bid_ask_mid_and_exness_label(self):
        client = ExnessMT5Client(mt5_module=FakeMT5())
        quote = client.get_quote("XAUUSD")
        self.assertAlmostEqual(quote["close"], 2400.0)
        self.assertEqual(quote["symbol"], "XAUUSDm")
        self.assertIn("Exness MT5", quote["source"])

    def test_refuses_non_exness_terminal(self):
        fake = FakeMT5(server="OtherBroker-Live")
        fake.company = "Other Broker"

        def other_account():
            return SimpleNamespace(server="OtherBroker-Live", company="Other Broker", name="Demo")

        fake.account_info = other_account
        with self.assertRaises(ExnessConnectionError):
            ExnessMT5Client(mt5_module=fake)
        self.assertTrue(fake.shutdown_called)


if __name__ == "__main__":
    unittest.main()
