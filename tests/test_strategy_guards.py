import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from ai_analysis import AIPeakReview, enforce_peak_review_consistency
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
from telegram_query_bot import assess_proposed_order, format_peak_backtest_evidence


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

    def test_live_backtest_evidence_reports_rate_and_small_sample_warning(self):
        trades = [
            {"tier": "S · 90+", "net_r": 1.2},
            {"tier": "S · 90+", "net_r": -1.0},
            {"tier": "A · 80–89", "net_r": -0.5},
        ]
        with tempfile.TemporaryDirectory() as directory:
            trades_path = Path(directory) / "trades.json"
            trades_path.write_text(json.dumps(trades), encoding="utf-8")
            text = format_peak_backtest_evidence(
                {
                    "setup_quality": {
                        "backtest_summary_path": str(Path(directory) / "missing.json"),
                        "backtest_trades_path": str(trades_path),
                    },
                    "backtest": {"validation": {"minimum_trades": 100}},
                },
                "S · 90+",
            )
        self.assertIn("1/2 = 50.0%", text)
        self.assertIn("CI95%", text)
        self.assertIn("CHƯA ĐỦ MẪU 3/100", text)
        self.assertIn("không dùng như cam kết", text)

    def test_proposed_order_checks_entry_side_risk_and_paper_mode(self):
        quality = SimpleNamespace(
            actionable=True,
            blockers=(),
            score=96,
            tier="S · 90+",
            recommendation="PAPER · SETUP RẤT CHỌN LỌC",
            recommended_risk_pct=0.5,
            paper_only=True,
        )
        plan = PeakExecutionPlan(
            "LONG",
            3349.0,
            3351.0,
            3350.0,
            3345.0,
            3355.0,
            3360.0,
            1.0,
            2.0,
            "resistance",
        )
        opportunity = SimpleNamespace(
            execution_plan=plan,
            execution_reason="confirmed",
            quality=quality,
            realtime_quote=SimpleNamespace(price=3352.0),
        )
        settings = {
            "account_balance_usdt": 1000,
            "quantity_step": 0.001,
            "minimum_quantity": 0.001,
            "minimum_notional_usdt": 5,
            "max_leverage": 5,
            "max_margin_pct": 25,
            "estimated_round_trip_fee_pct": 0.1,
            "estimated_slippage_bps": 2,
        }
        assessment = assess_proposed_order(
            opportunity,
            "LONG",
            100.0,
            3350.0,
            5,
            settings,
        )
        self.assertTrue(assessment.setup_allowed)
        self.assertFalse(assessment.real_order_allowed)
        self.assertIn("CHỈ PAPER", assessment.verdict)
        self.assertAlmostEqual(assessment.actual_notional_usdt, 499.15)
        self.assertLess(assessment.estimated_risk_pct, 0.5)

        wrong_side = assess_proposed_order(
            opportunity,
            "SHORT",
            100.0,
            3350.0,
            5,
            settings,
        )
        self.assertFalse(wrong_side.setup_allowed)
        self.assertTrue(any("ngược với lệnh SHORT" in reason for reason in wrong_side.reasons))

    def test_ai_cannot_approve_a_rejected_user_order(self):
        review = AIPeakReview(
            decision="CANH LONG",
            review_vi="bullish",
            confirmation_vi="confirmed",
            invalidation_vi="below stop",
            risk_vi="risk",
            data_consistency=90,
        )
        snapshot = {
            "deterministic_gate": {"allowed_decision": "CANH LONG"},
            "code_execution_plan": {"side": "LONG"},
            "liquidity_guard": {"entries_allowed": True},
            "setup_quality_score": {"actionable": True, "paper_only": False},
            "macro_event_guard": {"blocked": False},
            "liquidity_sweep_fomo_guard": {
                "double_sweep": False,
                "fomo_extension": False,
            },
            "user_order_evaluation": {"setup_allowed": False},
        }
        enforced = enforce_peak_review_consistency(review, snapshot)
        self.assertEqual(enforced.decision, "CHỜ")


if __name__ == "__main__":
    unittest.main()
