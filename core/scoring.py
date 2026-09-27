"""Rule-based technical score in [-100, +100] and BUY/HOLD/SELL mapping.

Transparent by design: every point comes from a named component so the
app can explain *why* a signal fired. Phase 5 adds an ML model on top.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .patterns import PATTERNS, pattern_direction_sum

BUY_THRESHOLD = 35
SELL_THRESHOLD = -35

PATTERN_POINTS = 15          # per net pattern on the latest bar
PATTERN_DECAY = (1.0, 0.6, 0.3)  # weight for bars t, t-1, t-2
VOLUME_CONFIRM_RATIO = 1.2
VOLUME_BOOST = 1.3


def score_components(ind: pd.DataFrame, patterns: pd.DataFrame) -> pd.DataFrame:
    """Per-bar score components. `ind` comes from indicators.add_indicators."""
    c = ind["Close"]
    sign = lambda cond_up, cond_dn: np.where(cond_up, 1, np.where(cond_dn, -1, 0))  # noqa: E731

    comp = pd.DataFrame(index=ind.index)
    comp["trend_price_ema50"] = 10 * sign(c > ind["ema50"], c < ind["ema50"])
    comp["trend_ema20_ema50"] = 10 * sign(ind["ema20"] > ind["ema50"], ind["ema20"] < ind["ema50"])
    comp["trend_ema50_ema200"] = 10 * sign(ind["ema50"] > ind["ema200"], ind["ema50"] < ind["ema200"])

    comp["macd"] = 10 * sign(ind["macd_hist"] > 0, ind["macd_hist"] < 0)
    comp["rsi"] = 15 * sign(ind["rsi"] < 30, ind["rsi"] > 70)
    comp["bollinger"] = 10 * sign(ind["bb_pct"] < 0, ind["bb_pct"] > 1)

    net = pattern_direction_sum(patterns).astype(float)
    vol_boost = np.where(ind["vol_ratio"] > VOLUME_CONFIRM_RATIO, VOLUME_BOOST, 1.0)
    weighted = net * vol_boost
    pat = sum(w * weighted.shift(i, fill_value=0.0) for i, w in enumerate(PATTERN_DECAY))
    comp["patterns"] = (PATTERN_POINTS * pat).clip(-30, 30)

    comp["total"] = comp.sum(axis=1).clip(-100, 100)
    # Not enough history for the slow indicators -> no score
    comp.loc[ind["ema50"].isna() | ind["rsi"].isna(), "total"] = np.nan
    return comp


def to_signal(score: float) -> str:
    if pd.isna(score):
        return "N/A"
    if score >= BUY_THRESHOLD:
        return "BUY"
    if score <= SELL_THRESHOLD:
        return "SELL"
    return "HOLD"


def signal_series(total: pd.Series) -> pd.Series:
    return total.map(to_signal)


def explain(ind: pd.DataFrame, patterns: pd.DataFrame, comp: pd.DataFrame) -> list[tuple[float, str]]:
    """Human-readable reasons for the latest bar, strongest first."""
    row, cr = ind.iloc[-1], comp.iloc[-1]
    reasons: list[tuple[float, str]] = []

    def add(points: float, text: str) -> None:
        if points:
            reasons.append((float(points), text))

    above = "above" if row["Close"] > row["ema50"] else "below"
    add(cr["trend_price_ema50"], f"Price {above} 50-day EMA")
    add(cr["trend_ema20_ema50"], "20-EMA " + ("above" if cr["trend_ema20_ema50"] > 0 else "below") + " 50-EMA (short-term trend)")
    add(cr["trend_ema50_ema200"], "Golden-cross regime (50-EMA > 200-EMA)" if cr["trend_ema50_ema200"] > 0
        else "Death-cross regime (50-EMA < 200-EMA)")
    add(cr["macd"], f"MACD histogram {'positive' if cr['macd'] > 0 else 'negative'} ({row['macd_hist']:.2f})")
    add(cr["rsi"], f"RSI {row['rsi']:.0f} — {'oversold' if cr['rsi'] > 0 else 'overbought'}")
    add(cr["bollinger"], "Close " + ("below lower" if cr["bollinger"] > 0 else "above upper") + " Bollinger band")

    recent = patterns.iloc[-len(PATTERN_DECAY):]
    for age, (date, prow) in enumerate(reversed(list(recent.iterrows()))):
        for name in prow.index[prow.values]:
            direction, label = PATTERNS[name]
            when = "today" if age == 0 else f"{age} bar{'s' if age > 1 else ''} ago"
            tag = {1: "bullish", -1: "bearish", 0: "neutral"}[direction]
            reasons.append((direction * PATTERN_POINTS * PATTERN_DECAY[age] or 0.0,
                            f"{label} ({tag}) — {when}, {date:%Y-%m-%d}"))

    if row["vol_ratio"] > VOLUME_CONFIRM_RATIO:
        reasons.append((0.0, f"Volume {row['vol_ratio']:.1f}× its 20-day average (confirms today's candle)"))

    return sorted(reasons, key=lambda r: -abs(r[0]))
