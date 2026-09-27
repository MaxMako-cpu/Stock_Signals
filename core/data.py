"""Market data download with a local Parquet cache.

Tickers use Yahoo Finance notation: US stocks as-is (AAPL), European
listings with an exchange suffix (SAP.DE, ASML.AS, SHEL.L).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# yfinance logs every failed lookup; we raise our own clearer error instead
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"
CACHE_MAX_AGE_HOURS = 12
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


class DataError(RuntimeError):
    pass


def _cache_path(ticker: str, interval: str) -> Path:
    safe = ticker.upper().replace("/", "_").replace("^", "IDX_")
    return CACHE_DIR / f"{safe}_{interval}.parquet"


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # yfinance may return MultiIndex columns like ("Close", "AAPL")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise DataError(f"Downloaded data is missing columns: {missing}")
    df = df[OHLCV].astype(float).dropna(subset=["Open", "High", "Low", "Close"])
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df.index.name = "Date"
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_ohlcv(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return daily OHLCV candles for `ticker`, oldest first."""
    ticker = ticker.strip().upper()
    path = _cache_path(ticker, interval)

    if use_cache and path.exists():
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours < CACHE_MAX_AGE_HOURS:
            return pd.read_parquet(path)

    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise DataError(
            f"No data for '{ticker}'. Check the symbol "
            "(European stocks need a suffix, e.g. SAP.DE, ASML.AS, SHEL.L)."
        )
    df = _normalize(raw)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def fetch_info(ticker: str) -> dict:
    """Best-effort name/currency/exchange lookup; never raises."""
    info = {"ticker": ticker.upper(), "name": ticker.upper(), "currency": "", "exchange": ""}
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        info["currency"] = fi.get("currency") or ""
        info["exchange"] = fi.get("exchange") or ""
        meta = t.get_info() or {}
        name = meta.get("longName") or meta.get("shortName") or info["name"]
        info["name"] = " ".join(name.split())
    except Exception:
        pass
    return info
