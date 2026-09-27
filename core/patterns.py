"""Candlestick pattern detection (vectorized, causal).

`detect_patterns` returns a boolean DataFrame, one column per pattern,
True on the bar where the pattern completes. `PATTERNS` holds each
pattern's direction (+1 bullish, -1 bearish, 0 neutral) and a label.
"""

from __future__ import annotations

import pandas as pd

PATTERNS: dict[str, tuple[int, str]] = {
    "bullish_engulfing": (1, "Bullish engulfing"),
    "bearish_engulfing": (-1, "Bearish engulfing"),
    "hammer": (1, "Hammer"),
    "inverted_hammer": (1, "Inverted hammer"),
    "hanging_man": (-1, "Hanging man"),
    "shooting_star": (-1, "Shooting star"),
    "morning_star": (1, "Morning star"),
    "evening_star": (-1, "Evening star"),
    "piercing_line": (1, "Piercing line"),
    "dark_cloud_cover": (-1, "Dark cloud cover"),
    "three_white_soldiers": (1, "Three white soldiers"),
    "three_black_crows": (-1, "Three black crows"),
    "doji": (0, "Doji (indecision)"),
}

TREND_LOOKBACK = 5


def _prior_trend(close: pd.Series, bars_before: int) -> tuple[pd.Series, pd.Series]:
    """(downtrend, uptrend) over the TREND_LOOKBACK bars preceding the pattern."""
    end = close.shift(bars_before)
    start = close.shift(bars_before + TREND_LOOKBACK)
    return end < start, end > start


def detect_patterns(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c = df["Open"], df["High"], df["Low"], df["Close"]
    body = c - o
    abody = body.abs()
    rng = (h - l).where(h > l)
    upper = h - pd.concat([o, c], axis=1).max(axis=1)
    lower = pd.concat([o, c], axis=1).min(axis=1) - l
    avg_body = abody.rolling(14, min_periods=5).mean().shift()
    bull, bear = body > 0, body < 0
    long_body = abody > avg_body
    small_body = abody < 0.5 * avg_body

    down1, up1 = _prior_trend(c, 1)
    down3, up3 = _prior_trend(c, 3)

    p = pd.DataFrame(index=df.index)

    # Two-bar engulfing
    po, pc, pbody = o.shift(), c.shift(), body.shift()
    p["bullish_engulfing"] = down1 & (pbody < 0) & bull & (o <= pc) & (c >= po) & (abody > pbody.abs())
    p["bearish_engulfing"] = up1 & (pbody > 0) & bear & (o >= pc) & (c <= po) & (abody > pbody.abs())

    # Single-bar shadow patterns: shape is the same, trend decides meaning
    has_body = abody > 0.05 * rng
    long_lower = has_body & (lower >= 2 * abody) & (upper <= 0.1 * rng + 0.5 * abody)
    long_upper = has_body & (upper >= 2 * abody) & (lower <= 0.1 * rng + 0.5 * abody)
    p["hammer"] = long_lower & down1
    p["hanging_man"] = long_lower & up1
    p["inverted_hammer"] = long_upper & down1
    p["shooting_star"] = long_upper & up1

    # Three-bar stars
    b2, b1 = body.shift(2), body.shift(1)
    long2 = long_body.shift(2, fill_value=False)
    small1 = small_body.shift(1, fill_value=False)
    mid2 = (o.shift(2) + c.shift(2)) / 2
    p["morning_star"] = down3 & (b2 < 0) & long2 & small1 & bull & (c > mid2)
    p["evening_star"] = up3 & (b2 > 0) & long2 & small1 & bear & (c < mid2)

    # Piercing line / dark cloud cover
    pmid = (po + pc) / 2
    plong = long_body.shift(1, fill_value=False)
    p["piercing_line"] = down1 & (pbody < 0) & plong & bull & (o < pc) & (c > pmid) & (c < po)
    p["dark_cloud_cover"] = up1 & (pbody > 0) & plong & bear & (o > pc) & (c < pmid) & (c > po)

    # Three soldiers / crows: 3 solid candles, each opening inside the prior body
    solid = abody > 0.5 * avg_body
    small_upper = upper < 0.3 * abody
    small_lower = lower < 0.3 * abody
    soldier = bull & solid & small_upper
    crow = bear & solid & small_lower
    opens_in_prev_body_up = (o >= o.shift()) & (o <= c.shift())
    opens_in_prev_body_dn = (o <= o.shift()) & (o >= c.shift())
    p["three_white_soldiers"] = (
        soldier & soldier.shift(1, fill_value=False) & soldier.shift(2, fill_value=False)
        & (c > c.shift()) & (c.shift() > c.shift(2))
        & opens_in_prev_body_up & opens_in_prev_body_up.shift(1, fill_value=False)
    )
    p["three_black_crows"] = (
        crow & crow.shift(1, fill_value=False) & crow.shift(2, fill_value=False)
        & (c < c.shift()) & (c.shift() < c.shift(2))
        & opens_in_prev_body_dn & opens_in_prev_body_dn.shift(1, fill_value=False)
    )

    p["doji"] = abody <= 0.1 * rng

    return p.fillna(False).astype(bool)


def pattern_direction_sum(patterns: pd.DataFrame) -> pd.Series:
    """Net bullish(+)/bearish(-) pattern count per bar."""
    weights = pd.Series({k: v[0] for k, v in PATTERNS.items()})
    return patterns[weights.index].astype(int).mul(weights, axis=1).sum(axis=1)
