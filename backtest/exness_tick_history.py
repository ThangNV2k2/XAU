"""Streaming reader for public Exness Tick History ZIP/CSV files."""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, TextIO

import pandas as pd


TICK_COLUMNS = ["Timestamp", "Bid", "Ask"]


def archive_url(symbol: str, period: str) -> str:
    """Build the public Exness archive URL for YYYY, YYYY-MM or YYYY-MM-DD."""
    clean_symbol = symbol.strip()
    parts = period.split("-")
    if len(parts) == 1:
        year = parts[0]
        name = f"Exness_{clean_symbol}_{year}.zip"
        return f"https://ticks.ex2archive.com/ticks/{clean_symbol}/{year}/{name}"
    if len(parts) == 2:
        year, month = parts
        name = f"Exness_{clean_symbol}_{year}_{month}.zip"
        return f"https://ticks.ex2archive.com/ticks/{clean_symbol}/{year}/{month}/{name}"
    if len(parts) == 3:
        year, month, day = parts
        name = f"Exness_{clean_symbol}_{year}_{month}_{day}.zip"
        return f"https://ticks.ex2archive.com/ticks/{clean_symbol}/{year}/{month}/{day}/{name}"
    raise ValueError("period phải có dạng YYYY, YYYY-MM hoặc YYYY-MM-DD")


def download_archive(symbol: str, period: str, destination: str | Path) -> Path:
    """Download one public archive and reuse a non-empty cached file."""
    target_dir = Path(destination)
    target_dir.mkdir(parents=True, exist_ok=True)
    url = archive_url(symbol, period)
    target = target_dir / url.rsplit("/", 1)[-1]
    if target.exists() and target.stat().st_size > 0:
        return target
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "xau-signal-replay/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(target)
    return target


def _csv_stream(path: Path) -> tuple[TextIO, object | None]:
    if path.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(path)
        members = [item for item in archive.infolist() if item.filename.lower().endswith(".csv")]
        if len(members) != 1:
            archive.close()
            raise ValueError(f"{path} phải chứa đúng một CSV, hiện có {len(members)}")
        stream = archive.open(members[0], "r")
        return stream, archive
    return path.open("rb"), None


def iter_tick_chunks(paths: Iterable[str | Path], chunksize: int = 400_000) -> Iterator[pd.DataFrame]:
    for value in paths:
        path = Path(value)
        stream, owner = _csv_stream(path)
        try:
            for chunk in pd.read_csv(stream, usecols=TICK_COLUMNS, chunksize=chunksize):
                chunk["Timestamp"] = pd.to_datetime(
                    chunk["Timestamp"],
                    utc=True,
                    errors="coerce",
                    format="ISO8601",
                )
                chunk["Bid"] = pd.to_numeric(chunk["Bid"], errors="coerce")
                chunk["Ask"] = pd.to_numeric(chunk["Ask"], errors="coerce")
                chunk = chunk.dropna(subset=TICK_COLUMNS)
                chunk = chunk[(chunk["Bid"] > 0) & (chunk["Ask"] >= chunk["Bid"])]
                if not chunk.empty:
                    yield chunk
        finally:
            stream.close()
            if owner is not None:
                owner.close()


def ticks_to_one_minute(
    paths: Iterable[str | Path],
    chunksize: int = 400_000,
) -> pd.DataFrame:
    """Aggregate large tick archives without loading raw ticks into memory."""
    pieces: list[pd.DataFrame] = []
    for chunk in iter_tick_chunks(paths, chunksize=chunksize):
        minute = chunk["Timestamp"].dt.floor("min")
        chunk = chunk.assign(_minute=minute, _spread=chunk["Ask"] - chunk["Bid"])
        grouped = chunk.groupby("_minute", sort=True).agg(
            bid_open=("Bid", "first"),
            bid_high=("Bid", "max"),
            bid_low=("Bid", "min"),
            bid_close=("Bid", "last"),
            ask_open=("Ask", "first"),
            ask_high=("Ask", "max"),
            ask_low=("Ask", "min"),
            ask_close=("Ask", "last"),
            spread_mean=("_spread", "mean"),
            spread_max=("_spread", "max"),
            volume=("Bid", "size"),
        )
        pieces.append(grouped)
    if not pieces:
        raise ValueError("Không đọc được tick hợp lệ từ các file Exness")
    partial = pd.concat(pieces).sort_index(kind="stable")
    bars = partial.groupby(level=0, sort=True).agg(
        bid_open=("bid_open", "first"),
        bid_high=("bid_high", "max"),
        bid_low=("bid_low", "min"),
        bid_close=("bid_close", "last"),
        ask_open=("ask_open", "first"),
        ask_high=("ask_high", "max"),
        ask_low=("ask_low", "min"),
        ask_close=("ask_close", "last"),
        spread_mean=("spread_mean", "mean"),
        spread_max=("spread_max", "max"),
        volume=("volume", "sum"),
    )
    bars.index.name = "datetime"
    bars[["open", "high", "low", "close"]] = bars[
        ["bid_open", "bid_high", "bid_low", "bid_close"]
    ].to_numpy()
    return bars
