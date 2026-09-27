"""Offline tests on synthetic candles (no network)."""

import numpy as np
import pandas as pd
import pytest

from core.analysis import analyze_frame
from core.backtest import run_backtest
from core.indicators import add_indicators, rsi
from core.patterns import PATTERNS, detect_patterns


def make_ohlcv(closes, opens=None, highs=None, lows=None, volume=1_000_000):
    closes = np.asarray(closes, dtype=float)
    opens = np.r_[closes[0], closes[:-1]] if opens is None else np.asarray(opens, float)
    highs = np.maximum(opens, closes) * 1.002 if highs is None else np.asarray(highs, float)
    lows = np.minimum(opens, closes) * 0.998 if lows is None else np.asarray(lows, float)
    idx = pd.bdate_range("2023-01-02", periods=len(closes))
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes,
                         "Volume": float(volume)}, index=idx)


def random_walk(n=400, seed=0, drift=0.0005):
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, n)))


def downtrend_then(last_bars):
    """Ten falling bars followed by hand-built candles (o, h, l, c)."""
    base = [(110 - i, 110.5 - i, 108.5 - i, 109 - i) for i in range(10)]  # bearish bodies
    rows = base + list(last_bars)
    o, h, l, c = map(np.array, zip(*rows))
    return make_ohlcv(c, o, h, l)


def uptrend_then(last_bars):
    base = [(90 + i, 91.5 + i, 89.5 + i, 91 + i) for i in range(10)]  # bullish bodies
    rows = base + list(last_bars)
    o, h, l, c = map(np.array, zip(*rows))
    return make_ohlcv(c, o, h, l)


# ---------- indicators ----------

def test_rsi_bounds_and_extremes():
    up = pd.Series(np.linspace(100, 200, 60))
    down = pd.Series(np.linspace(200, 100, 60))
    assert rsi(up).iloc[-1] == pytest.approx(100)
    assert rsi(down).iloc[-1] < 1
    r = rsi(pd.Series(random_walk()))
    assert r.dropna().between(0, 100).all()


def test_indicators_are_causal():
    df = make_ohlcv(random_walk(300))
    full = add_indicators(df)
    cut = add_indicators(df.iloc[:250])
    cols = ["ema20", "ema50", "ema200", "rsi", "macd_hist", "bb_pct", "atr", "vol_ratio"]
    pd.testing.assert_frame_equal(full[cols].iloc[:250], cut[cols])


# ---------- patterns ----------

def test_bullish_engulfing():
    # prev bar bearish 100.5 -> 100, today opens 99.8 closes 101
    df = downtrend_then([(100.5, 100.7, 99.9, 100.0), (99.8, 101.2, 99.7, 101.0)])
    assert detect_patterns(df)["bullish_engulfing"].iloc[-1]


def test_bearish_engulfing():
    df = uptrend_then([(100.0, 100.6, 99.8, 100.5), (100.7, 100.8, 99.3, 99.5)])
    assert detect_patterns(df)["bearish_engulfing"].iloc[-1]


def test_hammer_in_downtrend_and_hanging_man_in_uptrend():
    hammer_shape = (100.0, 100.35, 98.0, 100.3)   # small body on top, long lower shadow
    assert detect_patterns(downtrend_then([hammer_shape]))["hammer"].iloc[-1]
    shifted = tuple(v + 10 for v in hammer_shape)  # sit above the uptrend's last close
    assert detect_patterns(uptrend_then([shifted]))["hanging_man"].iloc[-1]


def test_shooting_star():
    df = uptrend_then([(101.0, 103.0, 100.95, 101.3)])
    assert detect_patterns(df)["shooting_star"].iloc[-1]


def test_morning_star():
    df = downtrend_then([
        (101.0, 101.2, 97.8, 98.0),   # long bearish
        (97.5, 97.9, 97.2, 97.6),     # small body
        (97.8, 100.6, 97.7, 100.4),   # bullish, closes above midpoint of bar 1
    ])
    assert detect_patterns(df)["morning_star"].iloc[-1]


def test_doji():
    df = uptrend_then([(100.0, 101.0, 99.0, 100.05)])
    assert detect_patterns(df)["doji"].iloc[-1]


def test_patterns_are_causal():
    df = make_ohlcv(random_walk(300, seed=3), opens=random_walk(300, seed=3) * 1.003)
    full = detect_patterns(df)
    cut = detect_patterns(df.iloc[:200])
    pd.testing.assert_frame_equal(full.iloc[:200], cut)
    assert set(full.columns) == set(PATTERNS)


# ---------- scoring / analysis ----------

def test_strong_uptrend_scores_positive():
    ta = analyze_frame("TEST", make_ohlcv(np.linspace(50, 150, 300)))
    assert ta.score > 0
    assert -100 <= ta.score <= 100
    assert ta.signal in {"BUY", "HOLD", "SELL"}
    assert ta.reasons


def test_strong_downtrend_scores_negative():
    ta = analyze_frame("TEST", make_ohlcv(np.linspace(150, 50, 300)))
    assert ta.score < 0


def test_too_little_history():
    with pytest.raises(ValueError):
        analyze_frame("TEST", make_ohlcv(random_walk(30)))


# ---------- backtest ----------

def test_backtest_no_lookahead():
    """Signal on day t must only affect returns from the open of t+1."""
    df = make_ohlcv([100.0] * 10, opens=[100, 100, 100, 100, 100, 120, 120, 120, 120, 120])
    sig = pd.Series(["HOLD"] * 10, index=df.index)
    sig.iloc[4] = "BUY"  # decided at close of bar 4 -> enters at open of bar 5 (price 120)
    res = run_backtest(df, sig, fee_pct=0.0)
    assert res.equity.iloc[-1] == pytest.approx(1.0)  # bought after the jump: no gain
    assert res.position.iloc[5] == 1 and res.position.iloc[4] == 0

    sig.iloc[4] = "HOLD"
    sig.iloc[3] = "BUY"  # enters at open of bar 4 (100) -> captures the jump to 120
    res = run_backtest(df, sig, fee_pct=0.0)
    assert res.equity.iloc[-1] == pytest.approx(1.2)


def test_backtest_fees_and_round_trips():
    df = make_ohlcv(random_walk(300, seed=7))
    ta = analyze_frame("TEST", df)
    free = run_backtest(ta.frame, ta.signals, fee_pct=0.0)
    paid = run_backtest(ta.frame, ta.signals, fee_pct=0.5)
    if free.metrics["round_trips"]:
        assert paid.equity.iloc[-1] < free.equity.iloc[-1]
    assert set(free.position.unique()) <= {0.0, 1.0}
