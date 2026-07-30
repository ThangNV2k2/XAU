import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from backtest.peak_backtester import _wilson_interval
from indicators.signal_engine import compute_momentum_bias, compute_momentum_bias_series
from macro_risk import MacroRiskAssessment, assess_macro_risk, parse_fomc_calendar
from peak_analysis import (
    LiquidityTrapAssessment,
    PeakExecutionPlan,
    PeakLiquidityAssessment,
    PeakMap,
    PeakTradeGate,
    PeakZone,
    assess_peak_trade_gate,
    assess_setup_quality,
)


def zone(lower: float, upper: float, status: str) -> PeakZone:
    return PeakZone(
        lower=lower,
        upper=upper,
        center=(lower + upper) / 2,
        reliability="CAO",
        score=12,
        age_label="CŨ",
        newest_at=pd.Timestamp("2026-07-01", tz="UTC"),
        timeframes=("4h", "1h"),
        evidence_count=3,
        zigzag_count=1,
        reaction_atr=2.0,
        volume_spike=True,
        status=status,
        distance=0.0,
        support_confirmed=status == "ĐỈNH ĐÃ VƯỢT",
    )


class StrategyGuardTests(unittest.TestCase):
    def test_vectorized_momentum_matches_live_calculation(self):
        index = pd.date_range("2026-01-01", periods=80, freq="15min", tz="UTC")
        close = pd.Series([100 + i * 0.05 + ((i % 7) - 3) * 0.08 for i in range(80)], index=index)
        frame = pd.DataFrame(
            {
                "open": close.shift(1).fillna(close.iloc[0]),
                "high": close + 0.4,
                "low": close - 0.4,
                "close": close,
            },
            index=index,
        )
        weights = {"rsi": 1.0, "macd": 1.2, "ema_crossover": 1.0, "bollinger": 0.8}
        live = compute_momentum_bias(frame, weights)
        replay = compute_momentum_bias_series(frame, weights).iloc[-1]
        self.assertAlmostEqual(live.composite, float(replay.composite), places=10)

    def test_fomc_parser_uses_meeting_end_date(self):
        page = """
        <a id="x">2026 FOMC Meetings</a>
        <div class="row fomc-meeting">
          <div class="fomc-meeting__month"><strong>July</strong></div>
          <div class="fomc-meeting__date">28-29</div>
        </div>
        """
        events = parse_fomc_calendar(page, decision_hour_utc=18)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].scheduled_at.isoformat(), "2026-07-29T18:00:00+00:00")

    def test_fomc_parser_converts_new_york_daylight_saving(self):
        page = """
        <a id="x">2026 FOMC Meetings</a>
        <div class="row fomc-meeting">
          <div class="fomc-meeting__month"><strong>January</strong></div>
          <div class="fomc-meeting__date">27-28</div>
        </div>
        <div class="row fomc-meeting">
          <div class="fomc-meeting__month"><strong>July</strong></div>
          <div class="fomc-meeting__date">28-29</div>
        </div>
        """
        events = parse_fomc_calendar(page)
        self.assertEqual(events[0].scheduled_at.isoformat(), "2026-01-28T19:00:00+00:00")
        self.assertEqual(events[1].scheduled_at.isoformat(), "2026-07-29T18:00:00+00:00")

    @patch("macro_risk.fetch_fomc_events", return_value=([], "network down"))
    def test_macro_guard_fails_closed_when_official_calendar_is_missing(self, _fetch):
        result = assess_macro_risk(
            datetime(2026, 7, 20, tzinfo=timezone.utc),
            {
                "fetch_fomc_calendar": True,
                "fail_closed_if_calendar_unavailable": True,
                "manual_events": [
                    {"name": "Unrelated future event", "at_utc": "2027-01-01T00:00:00Z"}
                ],
            },
        )
        self.assertTrue(result.blocked)
        self.assertEqual(result.level, "UNKNOWN")

    def test_long_retest_keeps_converted_support_as_focus_zone(self):
        index = pd.date_range("2026-07-01", periods=8, freq="15min", tz="UTC")
        frame_15m = pd.DataFrame(
            {
                "open": [100.2, 100.3, 100.4, 100.5, 101.1, 101.4, 101.2, 101.1],
                "high": [100.8, 100.9, 100.8, 101.7, 101.6, 101.5, 101.5, 101.4],
                "low": [99.9, 100.0, 100.1, 100.3, 100.95, 101.0, 101.0, 101.0],
                "close": [100.4, 100.5, 100.6, 101.4, 101.2, 101.3, 101.2, 101.2],
                "volume": [100, 100, 100, 180, 130, 120, 110, 105],
            },
            index=index,
        )
        old_resistance = zone(100.0, 101.0, "ĐỈNH ĐÃ VƯỢT")
        overhead = zone(105.0, 106.0, "CẢN TRÊN")
        peak_map = PeakMap(
            current_price=101.2,
            resistance_zones=[overhead],
            converted_support_zones=[old_resistance],
            volume_available=True,
            scanned_peak_count=6,
        )
        frame_1h = pd.DataFrame(
            {
                "open": [100.5, 101.0],
                "high": [101.6, 101.8],
                "low": [100.2, 100.8],
                "close": [101.2, 101.4],
            },
            index=pd.date_range("2026-07-01", periods=2, freq="1h", tz="UTC"),
        )
        liquidity = PeakLiquidityAssessment(
            False,
            "TỐT",
            100.0,
            90.0,
            1.1,
            0.5,
            True,
            "volume/spread đạt",
        )
        neutral_trap = LiquidityTrapAssessment(
            False,
            False,
            False,
            False,
            1.0,
            0.8,
            0.5,
            "sạch",
        )
        gate = assess_peak_trade_gate(
            peak_map,
            frame_15m,
            {"15min": 0.3, "1h": 0.3, "4h": 0.3, "1day": 0.1},
            {
                "minimum_timeframe_score": 0.12,
                "require_1h_close_confirmation": True,
                "require_daily_context": True,
            },
            frame_1h=frame_1h,
            liquidity=liquidity,
            daily_pattern="HH/HL",
            trap=neutral_trap,
            macro_risk=MacroRiskAssessment(False, "NORMAL", "clear"),
        )
        self.assertEqual(gate.allowed_decision, "CANH LONG")
        self.assertIs(gate.resistance, old_resistance)

    def test_quality_score_is_blocked_during_macro_event(self):
        focus = zone(100.0, 101.0, "ĐANG TEST")
        gate = PeakTradeGate(
            "CANH SHORT",
            "confirmed",
            focus,
            None,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
        )
        plan = PeakExecutionPlan(
            "SHORT", 99.8, 100.0, 99.9, 101.2, 98.6, 97.0, 1.0, 2.23, "support"
        )
        liquidity = PeakLiquidityAssessment(
            False, "TỐT", 100.0, 90.0, 1.1, 0.5, True, "clear"
        )
        trap = LiquidityTrapAssessment(
            True, False, False, False, 1.2, 0.8, 0.5, "buy-side sweep"
        )
        macro = MacroRiskAssessment(
            True,
            "HIGH",
            "FOMC blackout",
            "FOMC",
            datetime(2026, 7, 29, 18, tzinfo=timezone.utc),
            30,
        )
        quality = assess_setup_quality(gate, plan, liquidity, trap, macro)
        self.assertGreaterEqual(quality.score, 70)
        self.assertFalse(quality.actionable)
        self.assertEqual(quality.recommended_risk_pct, 0.0)
        self.assertTrue(quality.paper_only)
        self.assertIn("FOMC blackout", quality.blockers)

    def test_wilson_interval_exposes_tiny_sample_uncertainty(self):
        interval = _wilson_interval(1, 3)
        self.assertIsNotNone(interval)
        lower, upper = interval
        self.assertLess(lower, 0.10)
        self.assertGreater(upper, 0.70)


if __name__ == "__main__":
    unittest.main()
