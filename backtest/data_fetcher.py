import os
from datetime import datetime, timedelta

import pandas as pd

from data_provider.twelvedata_provider import TwelveDataProvider

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")

# Twelve Data free tier caps ~5000 bars/request. Heuristic chunk sizes (days) per interval
# so each request stays comfortably under that cap. Adjust if the API returns fewer bars
# than expected for your account tier.
CHUNK_DAYS_BY_INTERVAL = {
    "1h": 180,
    "4h": 700,
    "1day": 365 * 5,
}


def _cache_path(interval: str, years: int) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe_interval = interval.replace("/", "_")
    return os.path.join(CACHE_DIR, f"xauusd_{safe_interval}_{years}y.csv")


def fetch_historical(
    interval: str,
    years: int,
    provider: TwelveDataProvider | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Pull `years` years of `interval` history for XAU/USD, paginating by date range and
    caching the result to CSV so repeated backtest runs don't re-spend API quota."""
    path = _cache_path(interval, years)
    if os.path.exists(path) and not force_refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    provider = provider or TwelveDataProvider()
    end = datetime.utcnow()
    start = end - timedelta(days=365 * years)
    chunk_days = CHUNK_DAYS_BY_INTERVAL.get(interval, 180)

    chunks = []
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end)
        df_chunk = provider.get_historical(
            interval=interval,
            outputsize=5000,
            start_date=chunk_start.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=chunk_end.strftime("%Y-%m-%d %H:%M:%S"),
        )
        chunks.append(df_chunk)
        chunk_start = chunk_end

    full_df = pd.concat(chunks).sort_index()
    full_df = full_df[~full_df.index.duplicated(keep="first")]
    full_df.to_csv(path)
    return full_df
