from dataclasses import dataclass

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands

# So nen "khoi dong" toi thieu truoc khi tin tuong gia tri indicator (MACD/EMA can nhieu nen nhat, ~26+9).
MIN_WARMUP_BARS = 40


@dataclass
class IndicatorScore:
    name: str
    value: float
    score: int


@dataclass
class SignalResult:
    timestamp: pd.Timestamp
    price: float
    indicators: list[IndicatorScore]
    weighted_score: float
    signal: str
    atr: float
    suggested_stop_distance: float

    def to_dict(self) -> dict:
        return {
            "timestamp": str(self.timestamp),
            "price": self.price,
            "signal": self.signal,
            "weighted_score": self.weighted_score,
            "atr": self.atr,
            "suggested_stop_distance": self.suggested_stop_distance,
            "indicators": {ind.name: {"value": ind.value, "score": ind.score} for ind in self.indicators},
        }


def _compute_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized, causal (right-aligned) indicator computation over the whole series.

    Each indicator at row i only depends on rows <= i, so slicing/iterating this frame
    bar-by-bar later does not leak future information (no look-ahead bias).
    """
    close, high, low = df["close"], df["high"], df["low"]

    rsi = RSIIndicator(close, window=14).rsi()

    macd_ind = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_diff = macd_ind.macd() - macd_ind.macd_signal()
    macd_prev_diff = macd_diff.shift(1)

    ema_fast = EMAIndicator(close, window=9).ema_indicator()
    ema_slow = EMAIndicator(close, window=21).ema_indicator()
    ema_diff = ema_fast - ema_slow
    ema_prev_diff = ema_diff.shift(1)

    bb = BollingerBands(close, window=20, window_dev=2)
    bb_upper, bb_lower = bb.bollinger_hband(), bb.bollinger_lband()

    atr = AverageTrueRange(high, low, close, window=14).average_true_range()

    rsi_score = pd.Series(0, index=df.index)
    rsi_score[rsi < 30] = 1
    rsi_score[rsi > 70] = -1

    macd_score = pd.Series(0, index=df.index)
    macd_score[(macd_prev_diff <= 0) & (macd_diff > 0)] = 1
    macd_score[(macd_prev_diff >= 0) & (macd_diff < 0)] = -1

    ema_score = pd.Series(0, index=df.index)
    ema_score[(ema_prev_diff <= 0) & (ema_diff > 0)] = 1
    ema_score[(ema_prev_diff >= 0) & (ema_diff < 0)] = -1

    bollinger_score = pd.Series(0, index=df.index)
    bollinger_score[close < bb_lower] = 1
    bollinger_score[close > bb_upper] = -1

    return pd.DataFrame(
        {
            "close": close,
            "rsi": rsi,
            "rsi_score": rsi_score,
            "macd_diff": macd_diff,
            "macd_score": macd_score,
            "ema_diff": ema_diff,
            "ema_score": ema_score,
            "bollinger_score": bollinger_score,
            "atr": atr,
        }
    )


def _apply_weights(frame: pd.DataFrame, weights: dict) -> pd.Series:
    return (
        frame["rsi_score"] * weights.get("rsi", 1.0)
        + frame["macd_score"] * weights.get("macd", 1.0)
        + frame["ema_score"] * weights.get("ema_crossover", 1.0)
        + frame["bollinger_score"] * weights.get("bollinger", 1.0)
    )


def _classify(weighted_score: pd.Series, thresholds: dict) -> pd.Series:
    signal = pd.Series("HOLD", index=weighted_score.index)
    signal[weighted_score >= thresholds["buy"]] = "BUY"
    signal[weighted_score <= thresholds["sell"]] = "SELL"
    return signal


def compute_signal(
    df: pd.DataFrame,
    weights: dict,
    thresholds: dict,
    atr_stop_multiplier: float = 1.5,
) -> SignalResult:
    """Compute the signal for the most recent bar in df. df needs >= MIN_WARMUP_BARS rows."""
    if len(df) < MIN_WARMUP_BARS:
        raise ValueError(f"Need at least {MIN_WARMUP_BARS} bars, got {len(df)}")

    frame = _compute_indicator_frame(df)
    weighted_score = _apply_weights(frame, weights)
    signal = _classify(weighted_score, thresholds)

    latest = frame.iloc[-1]
    indicators = [
        IndicatorScore("rsi", float(latest["rsi"]), int(latest["rsi_score"])),
        IndicatorScore("macd", float(latest["macd_diff"]), int(latest["macd_score"])),
        IndicatorScore("ema_crossover", float(latest["ema_diff"]), int(latest["ema_score"])),
        IndicatorScore("bollinger", float(latest["close"]), int(latest["bollinger_score"])),
    ]
    latest_atr = float(latest["atr"])

    return SignalResult(
        timestamp=df.index[-1],
        price=float(latest["close"]),
        indicators=indicators,
        weighted_score=float(weighted_score.iloc[-1]),
        signal=str(signal.iloc[-1]),
        atr=latest_atr,
        suggested_stop_distance=latest_atr * atr_stop_multiplier,
    )


def compute_signal_series(df: pd.DataFrame, weights: dict, thresholds: dict) -> pd.DataFrame:
    """Compute signal/score for every bar (after warmup) — used by the backtester."""
    frame = _compute_indicator_frame(df)
    weighted_score = _apply_weights(frame, weights)
    signal = _classify(weighted_score, thresholds)
    frame = frame.assign(weighted_score=weighted_score, signal=signal)
    return frame.iloc[MIN_WARMUP_BARS:]


@dataclass
class MomentumBias:
    price: float
    composite: float  # -1..+1, continuous
    label: str
    components: dict


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def compute_momentum_bias(df: pd.DataFrame, weights: dict) -> MomentumBias:
    """Continuous short-term momentum lean (-1..+1), always populated.

    compute_signal is intentionally sparse (only fires on crossover/threshold events, tuned
    for discrete backtested trade entries) so it reads as HOLD/0.00 most of the time. This
    function instead grades every indicator continuously as a trend/momentum gauge (RSI>50,
    MACD histogram>0, EMA fast>slow, price above Bollinger mid-band = bullish lean), so there
    is always a readable directional lean for ad-hoc "which way right now" queries. It has not
    been separately backtested — it is a readout of current momentum state, not a proven edge.
    """
    if len(df) < MIN_WARMUP_BARS:
        raise ValueError(f"Need at least {MIN_WARMUP_BARS} bars, got {len(df)}")

    close, high, low = df["close"], df["high"], df["low"]
    atr = AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
    atr = atr if atr and atr > 1e-9 else 1e-9

    rsi = RSIIndicator(close, window=14).rsi().iloc[-1]
    rsi_score = _clip((rsi - 50) / 25)

    macd_ind = MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_hist = (macd_ind.macd() - macd_ind.macd_signal()).iloc[-1]
    macd_score = _clip(macd_hist / atr)

    ema_fast = EMAIndicator(close, window=9).ema_indicator().iloc[-1]
    ema_slow = EMAIndicator(close, window=21).ema_indicator().iloc[-1]
    ema_score = _clip((ema_fast - ema_slow) / atr)

    bb = BollingerBands(close, window=20, window_dev=2)
    mid = bb.bollinger_mavg().iloc[-1]
    upper = bb.bollinger_hband().iloc[-1]
    band_half_width = (upper - mid) if (upper - mid) > 1e-9 else 1e-9
    bollinger_score = _clip((close.iloc[-1] - mid) / band_half_width)

    components = {
        "rsi": rsi_score,
        "macd": macd_score,
        "ema_trend": ema_score,
        "bollinger": bollinger_score,
    }
    component_weights = {
        "rsi": weights.get("rsi", 1.0),
        "macd": weights.get("macd", 1.2),
        "ema_trend": weights.get("ema_crossover", 1.0),
        "bollinger": weights.get("bollinger", 0.8),
    }
    total_weight = sum(component_weights.values())
    composite = sum(components[k] * component_weights[k] for k in components) / total_weight

    if composite >= 0.5:
        label = "TANG manh"
    elif composite >= 0.15:
        label = "TANG nhe"
    elif composite <= -0.5:
        label = "GIAM manh"
    elif composite <= -0.15:
        label = "GIAM nhe"
    else:
        label = "Trung lap / di ngang"

    return MomentumBias(price=float(close.iloc[-1]), composite=float(composite), label=label, components=components)
