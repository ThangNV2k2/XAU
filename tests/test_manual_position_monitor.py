import unittest
from datetime import datetime, timedelta, timezone

from telegram_query_bot import (
    build_manual_position_state,
    manual_alert_is_due,
    manual_position_metrics,
    parse_manual_position_command,
    update_manual_position_event,
)


class ManualPositionMonitorTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 7, 30, 0, 0, tzinfo=timezone.utc)
        self.settings = {
            "price_tick": 0.01,
            "stop_atr_multiplier": 0.8,
            "minimum_stop_distance_pct": 0.0,
            "maximum_stop_distance": 7.0,
            "take_profit_1_r": 1.0,
            "take_profit_2_r": 2.0,
            "estimated_round_trip_fee_pct": 0.0,
            "estimated_slippage_bps": 0.0,
            "loss_warning_r": 0.5,
            "profit_reminder_r": 0.5,
            "close_repeat_seconds": 30,
        }

    def test_parser_accepts_requested_aliases_and_telegram_command(self):
        self.assertEqual(parse_manual_position_command("/long:100_5x"), ("LONG", 100.0, 5))
        self.assertEqual(parse_manual_position_command("/short 25,5 10x"), ("SHORT", 25.5, 10))
        self.assertEqual(parse_manual_position_command("sort:40_3x"), ("SHORT", 40.0, 3))
        self.assertIsNone(parse_manual_position_command("/long:abc_5x"))

    def test_long_position_uses_margin_times_leverage_and_atr_levels(self):
        position = build_manual_position_state(
            "LONG", 100.0, 5, 4000.0, 5.0, self.now, self.settings
        )
        self.assertAlmostEqual(position["notional_usdt"], 500.0)
        self.assertAlmostEqual(position["quantity_xau"], 0.125)
        self.assertAlmostEqual(position["stop_loss"], 3996.0)
        self.assertAlmostEqual(position["take_profit_1"], 4004.0)
        self.assertAlmostEqual(position["take_profit_2"], 4008.0)
        metrics = manual_position_metrics(position, 4004.0)
        self.assertAlmostEqual(metrics["gross_pnl_usdt"], 0.5)
        self.assertAlmostEqual(metrics["r_multiple"], 1.0)

    def test_short_position_pnl_is_directionally_correct(self):
        position = build_manual_position_state(
            "SHORT", 100.0, 5, 4000.0, 5.0, self.now, self.settings
        )
        self.assertAlmostEqual(position["stop_loss"], 4004.0)
        self.assertAlmostEqual(position["take_profit_1"], 3996.0)
        metrics = manual_position_metrics(position, 3996.0)
        self.assertAlmostEqual(metrics["gross_pnl_usdt"], 0.5)
        self.assertAlmostEqual(metrics["r_multiple"], 1.0)

    def test_stop_event_latches_until_dong_even_if_price_recovers(self):
        position = build_manual_position_state(
            "LONG", 100.0, 5, 4000.0, 5.0, self.now, self.settings
        )
        stopped = manual_position_metrics(position, 3995.0)
        event = update_manual_position_event(position, stopped, self.now, self.settings)
        self.assertEqual(event, "CLOSE_REQUIRED")
        self.assertEqual(position["close_required_reason"], "STOP_LOSS")
        recovered = manual_position_metrics(position, 4002.0)
        event = update_manual_position_event(
            position,
            recovered,
            self.now + timedelta(minutes=1),
            self.settings,
        )
        self.assertEqual(event, "CLOSE_REQUIRED")

    def test_close_warning_repeats_on_configured_interval(self):
        position = build_manual_position_state(
            "LONG", 100.0, 5, 4000.0, 5.0, self.now, self.settings
        )
        position["last_alert_event"] = "CLOSE_REQUIRED"
        position["last_alert_at"] = self.now.isoformat()
        self.assertFalse(
            manual_alert_is_due(
                position,
                "CLOSE_REQUIRED",
                self.now + timedelta(seconds=20),
                self.settings,
            )
        )
        self.assertTrue(
            manual_alert_is_due(
                position,
                "CLOSE_REQUIRED",
                self.now + timedelta(seconds=30),
                self.settings,
            )
        )


if __name__ == "__main__":
    unittest.main()
