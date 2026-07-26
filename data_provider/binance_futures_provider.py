from datetime import datetime, timezone

import pandas as pd
import requests

from .base import DataProvider


BINANCE_FUTURES_BASE_URL = "https://fapi.binance.com"
INTERVAL_MAP = {
    "1min": "1m",
    "3min": "3m",
    "5min": "5m",
    "15min": "15m",
    "30min": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1day": "1d",
}


class BinanceFuturesProvider(DataProvider):
    """Public USDⓈ-M Futures market data for the exact traded contract."""

    def __init__(
        self,
        symbol: str = "XAUUSDT",
        base_url: str = BINANCE_FUTURES_BASE_URL,
    ):
        self.symbol = symbol.upper()
        self.base_url = base_url.rstrip("/")

    def _request(self, endpoint: str, params: dict | None = None):
        response = requests.get(
            f"{self.base_url}{endpoint}",
            params=params or {},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict) and "code" in payload and int(payload["code"]) < 0:
            raise RuntimeError(
                f"Binance Futures error {payload['code']}: {payload.get('msg', '')}"
            )
        return payload

    @staticmethod
    def _to_milliseconds(value: str | None) -> int | None:
        if value is None:
            return None
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(timezone.utc)
        else:
            timestamp = timestamp.tz_convert(timezone.utc)
        return int(timestamp.timestamp() * 1000)

    def get_historical(
        self,
        interval: str,
        outputsize: int = 100,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        binance_interval = INTERVAL_MAP.get(interval)
        if binance_interval is None:
            raise ValueError(f"Unsupported Binance Futures interval: {interval}")
        if outputsize < 1 or outputsize > 1500:
            raise ValueError("Binance Futures klines outputsize must be between 1 and 1500")

        params = {
            "symbol": self.symbol,
            "interval": binance_interval,
            "limit": int(outputsize),
        }
        start_ms = self._to_milliseconds(start_date)
        end_ms = self._to_milliseconds(end_date)
        if start_ms is not None:
            params["startTime"] = start_ms
        if end_ms is not None:
            params["endTime"] = end_ms

        rows = self._request("/fapi/v1/klines", params)
        if not rows:
            raise RuntimeError(
                f"Binance Futures returned no klines for {self.symbol} ({interval})"
            )

        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
            "ignore",
        ]
        frame = pd.DataFrame(rows, columns=columns)
        frame["datetime"] = pd.to_datetime(
            frame["open_time"].astype("int64"),
            unit="ms",
            utc=True,
        )
        frame = frame.set_index("datetime").sort_index()
        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        ]
        for column in numeric_columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["trade_count"] = pd.to_numeric(
            frame["trade_count"], errors="coerce"
        ).fillna(0)
        return frame[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trade_count",
                "taker_buy_volume",
                "taker_buy_quote_volume",
            ]
        ]

    def get_latest_price(self) -> float:
        payload = self._request(
            "/fapi/v1/ticker/price",
            {"symbol": self.symbol},
        )
        return float(payload["price"])

    def get_open_interest(self) -> float:
        payload = self._request(
            "/fapi/v1/openInterest",
            {"symbol": self.symbol},
        )
        return float(payload["openInterest"])

    def get_quote(self) -> dict:
        """Return execution, fair-value, funding and positioning metrics."""
        ticker = self._request(
            "/fapi/v1/ticker/price",
            {"symbol": self.symbol},
        )
        book = self._request(
            "/fapi/v1/ticker/bookTicker",
            {"symbol": self.symbol},
        )
        premium = self._request(
            "/fapi/v1/premiumIndex",
            {"symbol": self.symbol},
        )
        open_interest = self._request(
            "/fapi/v1/openInterest",
            {"symbol": self.symbol},
        )
        event_time_ms = int(
            ticker.get("time")
            or book.get("time")
            or premium.get("time")
            or datetime.now(timezone.utc).timestamp() * 1000
        )
        return {
            "symbol": self.symbol,
            "close": float(ticker["price"]),
            "last_quote_at": event_time_ms // 1000,
            "is_market_open": True,
            "source": "Binance Futures REST",
            "bid": float(book["bidPrice"]),
            "ask": float(book["askPrice"]),
            "mark_price": float(premium["markPrice"]),
            "index_price": float(premium["indexPrice"]),
            "funding_rate": float(premium["lastFundingRate"]),
            "next_funding_time": int(premium["nextFundingTime"]) // 1000,
            "open_interest": float(open_interest["openInterest"]),
        }
