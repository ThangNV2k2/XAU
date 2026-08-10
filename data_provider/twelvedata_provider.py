import os

import pandas as pd
import requests

from .base import DataProvider

TWELVEDATA_BASE_URL = "https://api.twelvedata.com"


class TwelveDataProvider(DataProvider):
    def __init__(self, symbol: str = "XAU/USD", api_key: str | None = None):
        self.symbol = symbol
        self.api_key = api_key or os.environ["TWELVEDATA_API_KEY"]

    def _request(self, endpoint: str, params: dict) -> dict:
        resp = requests.get(
            f"{TWELVEDATA_BASE_URL}/{endpoint}",
            params=params,
            headers={"Authorization": f"apikey {self.api_key}"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("status") == "error":
            raise RuntimeError(f"Twelve Data error: {data.get('message')}")
        return data

    def get_historical(
        self,
        interval: str,
        outputsize: int = 100,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        params = {
            "symbol": self.symbol,
            "interval": interval,
            "outputsize": outputsize,
            "format": "JSON",
            "order": "ASC",
            "timezone": "UTC",
        }
        if start_date:
            params["start_date"] = start_date
        if end_date:
            params["end_date"] = end_date

        data = self._request("time_series", params)
        values = data.get("values")
        if not values:
            raise RuntimeError(f"Twelve Data returned no values for {self.symbol} ({interval}): {data}")

        df = pd.DataFrame(values)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].astype(float)
        columns = ["open", "high", "low", "close"]
        if "volume" in df.columns:
            volume = pd.to_numeric(df["volume"], errors="coerce")
            if volume.notna().any() and float(volume.fillna(0).abs().sum()) > 0:
                df["volume"] = volume
                columns.append("volume")
        return df[columns]

    def get_latest_price(self) -> float:
        data = self._request("price", {"symbol": self.symbol})
        return float(data["price"])

    def get_quote(self) -> dict:
        """Return the latest quote together with provider timestamps and market state."""
        quote = self._request("quote", {"symbol": self.symbol})
        normalized = dict(quote)
        normalized["symbol"] = str(quote.get("symbol") or self.symbol)
        normalized["close"] = float(quote["close"])
        normalized["last_quote_at"] = int(quote.get("timestamp") or 0)
        normalized["source"] = "Twelve Data XAU/USD spot"
        return normalized
