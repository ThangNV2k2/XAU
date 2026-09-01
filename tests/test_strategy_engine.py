import unittest
from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from market_sessions import SessionState
from strategy_engine import (
    EMA200_WARMUP_BARS,
    EMA_SPREAD_ATR_DIVISOR,
    PRICE_DISTANCE_ATR_DIVISOR,
    TechnicalSnapshot,
    _rejection,
    analyze_market,
    higher_timeframe_bias,
    technical_frame,
    technical_snapshot,
)


def breakout_frame(side="LONG"):
    index = pd.date_range("2026-07-06 12:00", periods=100, freq="5min", tz="UTC")
    if side == "LONG":
        rows = [{"open": 99.5, "high": 100.0, "low": 99.0, "close": 99.5} for _ in index]
        rows[-2] = {"open": 99.7, "high": 101.2, "low": 99.6, "close": 101.0}
        rows[-1] = {"open": 100.7, "high": 101.0, "low": 99.95, "close": 100.85}
    else:
        rows = [{"open": 100.5, "high": 101.0, "low": 100.0, "close": 100.5} for _ in index]
        rows[-2] = {"open": 100.3, "high": 100.4, "low": 98.8, "close": 99.0}
        rows[-1] = {"open": 99.3, "high": 100.05, "low": 99.0, "close": 99.15}
    return pd.DataFrame(rows, index=index)


def snapshot(interval, score=0.5, side="LONG"):
    bullish = side == "LONG"
    return TechnicalSnapshot(
        interval=interval,
        close=100.85 if bullish else 99.15,
        rsi=55.0 if bullish else 45.0,
        rsi_slope=2.0 if bullish else -2.0,
        ema20=100.0,
        ema50=99.0 if bullish else 101.0,
        ema200=98.0 if bullish else 102.0,
        macd_histogram=0.2 if bullish else -0.2,
        adx=28.0,
        atr=1.0,
        score=score if bullish else -score,
    )


class StrategyEngineTests(unittest.TestCase):
    @staticmethod
    def _ramp_frame(bars):
        index = pd.date_range("2026-01-01", periods=bars, freq="1h", tz="UTC")
        close = pd.Series(
            [100 + step * 0.03 + ((step % 12) - 6) * 0.18 for step in range(bars)],
            index=index,
        )
        return pd.DataFrame(
            {
                "open": close.shift(1).fillna(close.iloc[0]),
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
            },
            index=index,
        )

    def test_standard_indicators_are_finite(self):
        result = technical_snapshot(self._ramp_frame(EMA200_WARMUP_BARS + 100), "1h")
        self.assertGreater(result.atr, 0)
        self.assertGreaterEqual(result.rsi, 0)
        self.assertLessEqual(result.rsi, 100)
        self.assertIsNotNone(result.ema200)
        self.assertGreaterEqual(result.score, -1)
        self.assertLessEqual(result.score, 1)

    def test_ema200_is_dropped_until_it_has_warmed_up(self):
        """A 260-bar EMA200 still carries ~7% of one arbitrary seed price."""
        short = technical_snapshot(self._ramp_frame(EMA200_WARMUP_BARS - 1), "1h")
        self.assertIsNone(short.ema200)
        warm = technical_snapshot(self._ramp_frame(EMA200_WARMUP_BARS), "1h")
        self.assertIsNotNone(warm.ema200)

    def test_live_slice_and_full_history_agree_on_the_last_closed_bar(self):
        """The scanner's tail() must not change the indicators it reports."""
        full = self._ramp_frame(3000)
        live = technical_frame(full.tail(EMA200_WARMUP_BARS + 200)).iloc[-1]
        replay = technical_frame(full).iloc[-1]
        for column in ("rsi", "ema20", "ema50", "ema200", "atr", "adx", "score"):
            expected = float(replay[column])
            # Relative, because EMA200 converges to a residual proportional to
            # price. At the old 260-bar window this gap was ~2e-3 on gold; the
            # bound below is roughly two orders of magnitude tighter than that.
            self.assertLessEqual(
                abs(float(live[column]) - expected) / max(abs(expected), 1e-9),
                1e-5,
                msg=f"{column} lệch giữa live slice và full history",
            )

    def test_atr_normalised_components_are_not_permanently_clipped(self):
        """Dividing by a bare 1x ATR pinned two of five components at +/-1."""
        frame = self._ramp_frame(3000)
        result = technical_frame(frame).dropna()
        spread = (result["ema20"] - result["ema50"]) / (
            EMA_SPREAD_ATR_DIVISOR * result["atr"]
        )
        distance = (result["close"] - result["ema20"]) / (
            PRICE_DISTANCE_ATR_DIVISOR * result["atr"]
        )
        for name, series in (("ema_component", spread), ("price_component", distance)):
            pinned = float((series.abs() >= 1.0).mean())
            self.assertLess(pinned, 0.35, f"{name} bị ghim ở +/-1 trên {pinned:.0%} số nến")

    def test_degenerate_atr_blocks_the_plan(self):
        """A frozen feed can still score high; the stop would be cents wide."""
        frames = {interval: breakout_frame("LONG") for interval in ("1min", "5min", "15min", "1h", "4h", "1day")}
        session = SessionState(
            True,
            "POWER_HOUR",
            "Cuối phiên Mỹ",
            datetime(2026, 7, 6, 14, 30, tzinfo=timezone.utc),
            True,
        )
        def frozen(_frame, interval):
            return replace(snapshot(interval), atr=0.0004)

        with patch("strategy_engine.technical_snapshot", side_effect=frozen):
            analysis = analyze_market(
                "AAPL",
                "stock",
                frames,
                {"digits": 2, "close": 100.0, "bid": 100.0, "ask": 100.0},
                session,
                {"minimum_atr_ratio": 0.0002},
                now=datetime(2026, 7, 6, 18, 30, tzinfo=timezone.utc),
            )
        self.assertEqual(analysis.action, "WAIT")
        self.assertIsNone(analysis.plan)
        self.assertIn("Biến động quá thấp", analysis.reason)

    def test_confidence_ignores_higher_timeframes_that_oppose_the_side(self):
        frames = {interval: breakout_frame("SHORT") for interval in ("1min", "5min", "15min", "1h", "4h", "1day")}
        session = SessionState(
            True,
            "POWER_HOUR",
            "Cuối phiên Mỹ",
            datetime(2026, 7, 6, 14, 30, tzinfo=timezone.utc),
            True,
        )
        # 15m/1h/4h all bullish while the entry frame is bearish: no side is
        # taken, so no higher timeframe may be counted as "aligned".
        def mixed(_frame, interval):
            return snapshot(interval, side="LONG" if interval in ("15min", "1h", "4h") else "SHORT")

        with patch("strategy_engine.technical_snapshot", side_effect=mixed):
            analysis = analyze_market(
                "AAPL",
                "stock",
                frames,
                {"digits": 2, "close": 99.15, "bid": 99.15, "ask": 99.17},
                session,
                now=datetime(2026, 7, 6, 18, 30, tzinfo=timezone.utc),
            )
        opposing = analysis.confidence
        with patch("strategy_engine.technical_snapshot", side_effect=lambda _f, i: snapshot(i, side="SHORT")):
            aligned = analyze_market(
                "AAPL",
                "stock",
                frames,
                {"digits": 2, "close": 99.15, "bid": 99.15, "ask": 99.17},
                session,
                now=datetime(2026, 7, 6, 18, 30, tzinfo=timezone.utc),
            ).confidence
        self.assertLess(opposing, aligned)

    def test_bias_blocks_counter_trend_entries(self):
        """150 lệnh SHORT trong sóng tăng 70% đã lỗ 34,75R vì thiếu bộ lọc này."""
        frames = {interval: breakout_frame("SHORT") for interval in ("1min", "5min", "15min", "1h", "4h", "1day")}
        session = SessionState(
            True, "POWER_HOUR", "Cuối phiên Mỹ",
            datetime(2026, 7, 6, 14, 30, tzinfo=timezone.utc), True,
        )

        def bearish_entry_bullish_daily(_frame, interval):
            if interval == "1day":
                return snapshot(interval, side="LONG")
            return snapshot(interval, side="SHORT")

        with patch("strategy_engine.technical_snapshot", side_effect=bearish_entry_bullish_daily):
            analysis = analyze_market(
                "XAUUSD", "metal", frames,
                {"digits": 2, "close": 99.15, "bid": 99.15, "ask": 99.17},
                session, {"entry_interval": "5min", "require_bias_alignment": True},
                now=datetime(2026, 7, 6, 18, 30, tzinfo=timezone.utc),
            )
        self.assertEqual(analysis.action, "WAIT")
        self.assertIsNone(analysis.plan)
        self.assertEqual(analysis.bias, "LONG")
        self.assertIn("ngược thiên hướng", analysis.reason)

    def test_bias_is_neutral_when_the_daily_frame_is_undecided(self):
        """NEUTRAL cũng chặn lệnh: chưa ngã ngũ thì không có gì chắc chắn."""
        undecided = replace(snapshot("1day"), close=99.5, ema20=100.5, ema50=100.0)
        bias, reason = higher_timeframe_bias({"1day": undecided}, {})
        self.assertEqual(bias, "NEUTRAL")
        self.assertIn("chưa ngã ngũ", reason)

    def test_bias_ignores_ema200_so_live_and_replay_use_one_rule(self):
        """Live có 1000 nến D1 (EMA200 sẵn), backtest chỉ ~390 (không có).

        Nếu luật rẽ nhánh theo EMA200 thì hai đường chạy hai luật khác nhau.
        """
        warm = replace(snapshot("1day"), ema200=200.0, close=101.0, ema20=100.5, ema50=100.0)
        cold = replace(warm, ema200=None)
        self.assertEqual(
            higher_timeframe_bias({"1day": warm}, {}),
            higher_timeframe_bias({"1day": cold}, {}),
        )
        self.assertEqual(higher_timeframe_bias({"1day": warm}, {})[0], "LONG")

    def test_zero_range_candle_is_never_a_rejection(self):
        flat = pd.Series({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0})
        self.assertFalse(_rejection(flat, "LONG", 0.60))
        self.assertFalse(_rejection(flat, "SHORT", 0.60))

    def test_long_requires_closed_breakout_retest_and_builds_structural_plan(self):
        frames = {interval: breakout_frame("LONG") for interval in ("1min", "5min", "15min", "1h", "4h", "1day")}
        session = SessionState(
            True,
            "POWER_HOUR",
            "Cuối phiên Mỹ",
            datetime(2026, 7, 6, 14, 30, tzinfo=timezone.utc),
            True,
        )
        with patch("strategy_engine.technical_snapshot", side_effect=lambda _frame, interval: snapshot(interval)):
            analysis = analyze_market(
                "AAPL",
                "stock",
                frames,
                {"digits": 2, "close": 100.05, "bid": 100.03, "ask": 100.05},
                session,
                now=datetime(2026, 7, 6, 18, 30, tzinfo=timezone.utc),
            )
        self.assertEqual(analysis.action, "BUY")
        self.assertIsNotNone(analysis.plan)
        self.assertEqual(analysis.plan.setup, "BREAKOUT_RETEST")
        self.assertLess(analysis.plan.stop_loss, analysis.plan.preferred_entry)
        self.assertAlmostEqual(analysis.plan.reward_risk_1, 1.5)
        self.assertAlmostEqual(analysis.plan.reward_risk_2, 2.5)

    def test_opening_range_phase_blocks_new_entry(self):
        frames = {interval: breakout_frame("LONG") for interval in ("1min", "5min", "15min", "1h", "4h", "1day")}
        session = SessionState(
            True,
            "OPENING_RANGE",
            "Đang tạo biên mở cửa",
            datetime(2026, 7, 6, 9, 45, tzinfo=timezone.utc),
            False,
        )
        with patch("strategy_engine.technical_snapshot", side_effect=lambda _frame, interval: snapshot(interval)):
            analysis = analyze_market(
                "AAPL",
                "stock",
                frames,
                {"digits": 2, "close": 100.05, "bid": 100.03, "ask": 100.05},
                session,
                now=datetime(2026, 7, 6, 13, 45, tzinfo=timezone.utc),
            )
        self.assertEqual(analysis.action, "WAIT")
        self.assertIsNone(analysis.plan)
        self.assertIn("biên mở cửa", analysis.reason)

    def test_us_retest_window_keeps_standard_retest_running_in_parallel(self):
        frames = {interval: breakout_frame("LONG") for interval in ("1min", "5min", "15min", "1h", "4h", "1day")}
        session = SessionState(
            True,
            "RETEST_WINDOW",
            "Ưu tiên retest sau mở cửa",
            datetime(2026, 7, 6, 10, 15, tzinfo=timezone.utc),
            True,
        )
        with (
            patch("strategy_engine.technical_snapshot", side_effect=lambda _frame, interval: snapshot(interval)),
            patch("strategy_engine.opening_range_levels", return_value=(105.0, 95.0)),
        ):
            analysis = analyze_market(
                "AAPL",
                "stock",
                frames,
                {"digits": 2, "close": 100.05, "bid": 100.03, "ask": 100.05},
                session,
                now=datetime(2026, 7, 6, 14, 15, tzinfo=timezone.utc),
            )
        self.assertEqual(analysis.action, "BUY")
        self.assertEqual(analysis.plan.setup, "BREAKOUT_RETEST")


if __name__ == "__main__":
    unittest.main()
