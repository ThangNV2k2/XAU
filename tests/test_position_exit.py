import unittest
from datetime import datetime, timezone

import pandas as pd

from position_exit import PositionState, assess_closed_candle_exit
from strategy_engine import TechnicalSnapshot


def snap(interval="5min", score=-0.2, rsi=42.0, slope=-3.0):
    return TechnicalSnapshot(interval, 99.7, rsi, slope, 100.0, 100.2, 101.0, -0.1, 25.0, 1.0, score)


def position():
    return PositionState(
        symbol="XAUUSD",
        asset_type="metal",
        side="LONG",
        setup="BREAKOUT_RETEST",
        entry_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
        entry_price=100.0,
        initial_stop=98.8,
        current_stop=98.8,
        take_profit_1=101.8,
        take_profit_2=103.0,
        invalidation_level=99.8,
        initial_risk=1.2,
    )


class PositionExitTests(unittest.TestCase):
    def test_wick_through_level_does_not_exit(self):
        frame_5m = pd.DataFrame(
            {"open": [100, 100], "high": [101, 101], "low": [99.5, 99.4], "close": [100.1, 100.0]}
        )
        decision = assess_closed_candle_exit(position(), {"5min": frame_5m}, {"5min": snap()})
        self.assertEqual(decision.action, "HOLD")

    def test_two_5m_closes_and_adverse_momentum_exit(self):
        frame_5m = pd.DataFrame(
            {"open": [100, 99.8], "high": [100.1, 99.9], "low": [99.5, 99.3], "close": [99.7, 99.5]}
        )
        decision = assess_closed_candle_exit(position(), {"5min": frame_5m}, {"5min": snap()})
        self.assertEqual(decision.action, "FULL_EXIT")
        self.assertIn("STRUCTURE_INVALIDATED", decision.reason)

    def test_two_higher_timeframes_flip_exit(self):
        frame_5m = pd.DataFrame(
            {"open": [100, 100], "high": [101, 101], "low": [99.9, 99.9], "close": [100.2, 100.1]}
        )
        snapshots = {
            "5min": snap(score=0.1, rsi=51, slope=0),
            "15min": snap("15min", -0.2),
            "1h": snap("1h", -0.3),
            "4h": snap("4h", 0.1),
        }
        decision = assess_closed_candle_exit(position(), {"5min": frame_5m}, snapshots)
        self.assertEqual(decision.action, "FULL_EXIT")
        self.assertIn("REGIME_FLIP", decision.reason)

    def test_tp1_moves_remaining_stop_to_break_even(self):
        state = position()
        state.tp1_hit = True
        frame_5m = pd.DataFrame(
            {"open": [100, 100], "high": [101, 101], "low": [99.9, 99.9], "close": [100.2, 100.1]}
        )
        decision = assess_closed_candle_exit(
            state,
            {"5min": frame_5m},
            {"5min": snap(score=0.1, rsi=51, slope=0)},
        )
        self.assertEqual(decision.action, "MOVE_STOP")
        self.assertEqual(decision.new_stop, 100.0)


if __name__ == "__main__":
    unittest.main()

