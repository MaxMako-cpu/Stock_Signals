"""One-call technical analysis of a ticker — the entry point the CLI,
Streamlit app and Telegram bot all share."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import fetch_info, fetch_ohlcv
from .indicators import add_indicators
from .patterns import detect_patterns
from .scoring import explain, score_components, signal_series, to_signal


@dataclass
class TechnicalAnalysis:
    ticker: str
    name: str
    currency: str
    date: pd.Timestamp
    close: float
    change_pct: float
    score: float
    signal: str
    reasons: list[tuple[float, str]]
    snapshot: dict                 # latest indicator values
    frame: pd.DataFrame            # OHLCV + indicators
    patterns: pd.DataFrame         # boolean pattern hits
    components: pd.DataFrame       # per-bar score components
    signals: pd.Series             # per-bar BUY/HOLD/SELL


def analyze_frame(ticker: str, ohlcv: pd.DataFrame, info: dict | None = None) -> TechnicalAnalysis:
    if len(ohlcv) < 60:
        raise ValueError(f"Only {len(ohlcv)} bars for {ticker}; need at least 60.")
    info = info or {}
    ind = add_indicators(ohlcv)
    pats = detect_patterns(ohlcv)
    comp = score_components(ind, pats)
    last = ind.iloc[-1]
    score = float(comp["total"].iloc[-1])
    snapshot = {k: float(last[k]) for k in
                ["rsi", "macd", "macd_signal", "macd_hist", "ema20", "ema50", "ema200",
                 "bb_pct", "atr", "vol_ratio"]}
    return TechnicalAnalysis(
        ticker=ticker.upper(),
        name=info.get("name", ticker.upper()),
        currency=info.get("currency", ""),
        date=ind.index[-1],
        close=float(last["Close"]),
        change_pct=100 * float(ind["Close"].pct_change().iloc[-1]),
        score=score,
        signal=to_signal(score),
        reasons=explain(ind, pats, comp),
        snapshot=snapshot,
        frame=ind,
        patterns=pats,
        components=comp,
        signals=signal_series(comp["total"]),
    )


def analyze(ticker: str, period: str = "5y", use_cache: bool = True, with_info: bool = True) -> TechnicalAnalysis:
    ohlcv = fetch_ohlcv(ticker, period=period, use_cache=use_cache)
    info = fetch_info(ticker) if with_info else None
    return analyze_frame(ticker, ohlcv, info)
