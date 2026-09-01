import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from lightweight_bot import journal_signal, _message_chunks, scan_job
from market_scanner import AssetSpec, ScanOutcome
from strategy_engine import MarketAnalysis, RetestSignal, TradePlan


def analysis(symbol, asset_type):
    plan = TradePlan(
        side="LONG",
        setup="EMA20_RETEST",
        entry_lower=99.9,
        entry_upper=100.1,
        preferred_entry=100.0,
        invalidation_level=99.7,
        stop_loss=99.0,
        take_profit_1=101.5,
        take_profit_2=102.5,
        risk=1.0,
        reward_risk_1=1.5,
        reward_risk_2=2.5,
    )
    return MarketAnalysis(
        symbol=symbol,
        asset_type=asset_type,
        checked_at=datetime(2026, 7, 6, 16, 0, tzinfo=timezone.utc),
        market_phase="POWER_HOUR",
        action="BUY",
        confidence=75,
        confidence_note="độ đầy đủ setup, không phải xác suất thắng",
        intraday_score=0.4,
        long_term_score=0.3,
        bias="LONG",
        bias_reason="1day trên EMA200 và EMA20>EMA50",
        horizon="4H–D1 nghiêng tăng",
        reason="Retest xác nhận",
        snapshots={},
        retest=RetestSignal(True, "EMA20_RETEST", 100.0, 99.8, 100.2, "Retest xác nhận"),
        plan=plan,
    )


class FakeScanner:
    def __init__(self, outcomes):
        self.outcomes = outcomes

    def scan_due(self, _now):
        return self.outcomes


class FakeState:
    def signal_is_new(self, _analysis, _cooldown):
        return True

    def error_is_due(self, _key, _now):
        return True


class FakeBot:
    def __init__(self):
        self.messages = []

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)


class TelegramAlertTests(unittest.IsolatedAsyncioTestCase):
    async def test_xau_and_stock_both_send_entry_4h_and_daily(self):
        xau = AssetSpec("XAUUSD", "Gold", "metal", 1)
        stock = AssetSpec("AAPL", "Apple", "stock", 5)
        outcomes = [
            ScanOutcome(xau, analysis("XAUUSD", "metal"), ("4h", "1day"), "OK", ""),
            ScanOutcome(stock, analysis("AAPL", "stock"), ("4h", "1day"), "OK", ""),
        ]
        bot = FakeBot()
        context = SimpleNamespace(
            bot=bot,
            bot_data={
                "scanner": FakeScanner(outcomes),
                "alert_state": FakeState(),
                "chat_id": 123,
                "config": {
                    "alerts": {
                        "telegram_enabled": True,
                        "entry_alerts": True,
                        "four_hour_summary": True,
                        "daily_summary": True,
                    }
                },
            },
        )
        await scan_job(context)
        texts = [item["text"] for item in bot.messages]
        self.assertEqual(len(texts), 4)
        self.assertTrue(any("XAU · TÍN HIỆU" in text and "XAUUSD" in text for text in texts))
        self.assertTrue(any("CỔ PHIẾU · CHỈ THÔNG BÁO" in text and "AAPL" in text for text in texts))
        daily = next(text for text in texts if "DỰ BÁO D1" in text)
        four_hour = next(text for text in texts if "DỰ BÁO 4H" in text)
        self.assertIn("XAUUSD", daily)
        self.assertIn("AAPL", daily)
        self.assertIn("XAUUSD", four_hour)
        self.assertIn("AAPL", four_hour)

    def test_grouped_messages_are_split_without_exceeding_telegram_limit(self):
        sections = [f"S{index}-" + "x" * 2000 for index in range(3)]
        chunks = _message_chunks("HEADER", sections, limit=4096)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(len(chunk) <= 4096 for chunk in chunks))
        self.assertEqual(sum(chunk.count("S") for chunk in chunks), 3)


if __name__ == "__main__":
    unittest.main()


class SignalJournalTests(unittest.TestCase):
    def test_each_signal_is_appended_with_the_fields_needed_to_score_it_later(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "journal.jsonl"
            journal_signal(analysis("XAUUSD", "metal"), str(path))
            journal_signal(analysis("AAPL", "stock"), str(path))
            lines = path.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        self.assertEqual(first["symbol"], "XAUUSD")
        self.assertEqual(first["side"], "LONG")
        self.assertEqual(first["bias"], "LONG")
        for field in ("entry", "stop_loss", "take_profit_1", "take_profit_2", "risk", "at"):
            self.assertIn(field, first)

    def test_a_wait_analysis_writes_nothing(self):
        waiting = analysis("XAUUSD", "metal")
        waiting = type(waiting)(**{**waiting.__dict__, "plan": None})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "journal.jsonl"
            journal_signal(waiting, str(path))
            self.assertFalse(path.exists())
