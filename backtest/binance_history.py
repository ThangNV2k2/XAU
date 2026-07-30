from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from data_provider.binance_futures_provider import BinanceFuturesProvider
from market_context import interval_duration


CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def fetch_binance_history(
    symbol: str,
    interval: str,
    start: datetime,
    end: datetime,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Page Binance Futures klines without silently reusing a stale end date."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)
    if end <= start:
        raise ValueError("Backtest end must be later than start")

    os.makedirs(CACHE_DIR, exist_ok=True)
    label = f"{symbol.lower()}_{interval}_{start:%Y%m%d}_{end:%Y%m%d}.csv"
    path = os.path.join(CACHE_DIR, label)
    if os.path.exists(path) and not force_refresh:
        cached = pd.read_csv(path, index_col=0, parse_dates=True)
        cached.index = pd.to_datetime(cached.index, utc=True)
        return cached.sort_index()

    provider = BinanceFuturesProvider(symbol=symbol)
    duration = interval_duration(interval)
    cursor = start
    chunks: list[pd.DataFrame] = []
    while cursor < end:
        chunk = provider.get_historical(
            interval=interval,
            outputsize=1500,
            start_date=cursor.isoformat(),
            end_date=end.isoformat(),
        )
        if chunk.empty:
            break
        chunks.append(chunk)
        last_open = chunk.index[-1].to_pydatetime()
        next_cursor = last_open + duration
        if next_cursor <= cursor:
            break
        cursor = next_cursor
        if len(chunk) < 1500:
            break
    if not chunks:
        raise RuntimeError(f"No Binance history returned for {symbol} {interval}")
    frame = pd.concat(chunks).sort_index()
    frame = frame[~frame.index.duplicated(keep="first")]
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    frame.to_csv(path)
    return frame


def resample_ohlcv(frame_15m: pd.DataFrame, rule: str) -> pd.DataFrame:
    aggregation = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    for column in (
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_volume",
        "taker_buy_quote_volume",
    ):
        if column in frame_15m.columns:
            aggregation[column] = "sum"
    result = frame_15m.resample(
        rule,
        label="left",
        closed="left",
        origin="epoch",
    ).agg(aggregation)
    return result.dropna(subset=["open", "high", "low", "close"])


def build_timeframes(frame_15m: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "15min": frame_15m,
        "1h": resample_ohlcv(frame_15m, "1h"),
        "4h": resample_ohlcv(frame_15m, "4h"),
        "1day": resample_ohlcv(frame_15m, "1D"),
    }
