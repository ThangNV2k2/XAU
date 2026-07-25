import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from websocket import WebSocketApp

logger = logging.getLogger("gold-query-bot.realtime")


@dataclass
class RealtimeQuote:
    price: float
    market_time: datetime
    received_at: datetime
    source: str
    is_market_open: bool


class TwelveDataPriceStream:
    def __init__(self, api_key: str, symbol: str = "XAU/USD"):
        self.api_key = api_key
        self.symbol = symbol
        self._lock = threading.Lock()
        self._latest: RealtimeQuote | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._socket: WebSocketApp | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="twelve-data-price-stream", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._socket:
            self._socket.close()

    def latest(self, max_received_age_seconds: int = 120) -> RealtimeQuote | None:
        with self._lock:
            quote = self._latest
        if quote is None:
            return None
        age = (datetime.now(timezone.utc) - quote.received_at).total_seconds()
        return quote if age <= max_received_age_seconds else None

    def _on_open(self, socket: WebSocketApp) -> None:
        socket.send(json.dumps({"action": "subscribe", "params": {"symbols": self.symbol}}))
        logger.info("Twelve Data WebSocket connected for %s", self.symbol)

    def _on_message(self, _socket: WebSocketApp, raw_message: str) -> None:
        try:
            message = json.loads(raw_message)
            if message.get("event") != "price" or message.get("symbol") != self.symbol:
                return
            timestamp = int(message["timestamp"])
            now = datetime.now(timezone.utc)
            quote = RealtimeQuote(
                price=float(message["price"]),
                market_time=datetime.fromtimestamp(timestamp, tz=timezone.utc),
                received_at=now,
                source="Twelve Data WebSocket",
                is_market_open=True,
            )
            with self._lock:
                self._latest = quote
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            logger.warning("Ignored malformed Twelve Data WebSocket message")

    def _on_error(self, _socket: WebSocketApp, error) -> None:
        logger.warning("Twelve Data WebSocket error: %s", error)

    def _on_close(self, _socket: WebSocketApp, status_code, message) -> None:
        logger.info("Twelve Data WebSocket closed: %s %s", status_code, message or "")

    def _run(self) -> None:
        retry_seconds = 2
        while not self._stop.is_set():
            url = f"wss://ws.twelvedata.com/v1/quotes/price?apikey={self.api_key}"
            self._socket = WebSocketApp(
                url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            self._socket.run_forever(ping_interval=20, ping_timeout=10)
            if self._stop.wait(retry_seconds):
                break
            retry_seconds = min(30, retry_seconds * 2)
            time.sleep(0)
