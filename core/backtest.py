"""Long-only backtest of the signal series.

Rules (no lookahead): the signal is computed at the close of day t and
executed at the open of day t+1. BUY -> go long, SELL -> go to cash,
HOLD -> keep the current position. Long-only because that is what a
retail app like Revolut lets you do simply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class BacktestResult:
    equity: pd.Series            # strategy equity curve (starts at 1.0)
    buy_hold: pd.Series          # buy & hold equity curve
    position: pd.Series          # 1 = long, 0 = cash, per bar
    trades: pd.DataFrame         # one row per round trip
    metrics: dict = field(default_factory=dict)
    signal_stats: dict = field(default_factory=dict)


def _max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1).min())


def _cagr(equity: pd.Series) -> float:
    years = len(equity) / TRADING_DAYS
    if years <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    return float(equity.iloc[-1] ** (1 / years) - 1)


def _sharpe(returns: pd.Series) -> float:
    sd = returns.std()
    return float(returns.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def run_backtest(
    ohlc: pd.DataFrame,
    signals: pd.Series,
    fee_pct: float = 0.25,
    horizon: int = 10,
) -> BacktestResult:
    """`fee_pct` is charged per side (commission + FX spread), in percent."""
    valid = signals[signals != "N/A"]
    if valid.empty:
        raise ValueError("No valid signals — not enough history.")
    start = valid.index[0]
    df = ohlc.loc[start:]
    sig = signals.loc[start:]

    target = sig.map({"BUY": 1.0, "SELL": 0.0}).ffill().fillna(0.0)
    position = target.shift(1, fill_value=0.0)          # executed next open
    bar_ret = (df["Open"].shift(-1) / df["Open"] - 1).fillna(0.0)  # open -> next open

    turnover = position.diff().abs().fillna(position.iloc[0])
    strat_ret = position * bar_ret - turnover * fee_pct / 100
    equity = (1 + strat_ret).cumprod()
    buy_hold = (1 + bar_ret).cumprod()

    # Round trips
    rows = []
    entry = None
    for date, pos, prev in zip(position.index, position.values, np.r_[0.0, position.values[:-1]]):
        if pos == 1 and prev == 0:
            entry = date
        elif pos == 0 and prev == 1 and entry is not None:
            rows.append((entry, date))
            entry = None
    if entry is not None:
        rows.append((entry, None))
    trade_list = []
    for e, x in rows:
        seg = bar_ret.loc[e:x].iloc[:-1] if x is not None else bar_ret.loc[e:]
        gross = float((1 + seg).prod() - 1)
        net = (1 + gross) * (1 - fee_pct / 100) ** 2 - 1
        trade_list.append({"entry": e, "exit": x, "bars": len(seg), "return_pct": 100 * net,
                           "open": x is None})
    trades = pd.DataFrame(trade_list, columns=["entry", "exit", "bars", "return_pct", "open"])

    closed = trades[~trades["open"]] if not trades.empty else trades
    metrics = {
        "period": f"{df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d}",
        "bars": len(df),
        "strategy_return_pct": 100 * (equity.iloc[-1] - 1),
        "buy_hold_return_pct": 100 * (buy_hold.iloc[-1] - 1),
        "strategy_cagr_pct": 100 * _cagr(equity),
        "buy_hold_cagr_pct": 100 * _cagr(buy_hold),
        "strategy_max_dd_pct": 100 * _max_drawdown(equity),
        "buy_hold_max_dd_pct": 100 * _max_drawdown(buy_hold),
        "strategy_sharpe": _sharpe(strat_ret),
        "buy_hold_sharpe": _sharpe(bar_ret),
        "exposure_pct": 100 * float(position.mean()),
        "round_trips": int(len(closed)),
        "win_rate_pct": 100 * float((closed["return_pct"] > 0).mean()) if len(closed) else float("nan"),
        "avg_trade_pct": float(closed["return_pct"].mean()) if len(closed) else float("nan"),
        "fee_pct_per_side": fee_pct,
    }

    # Directional hit rate of raw signals over `horizon` bars
    fwd = df["Open"].shift(-(horizon + 1)) / df["Open"].shift(-1) - 1
    known = fwd.notna()
    base_up = float((fwd[known] > 0).mean())
    buys, sells = (sig == "BUY") & known, (sig == "SELL") & known
    signal_stats = {
        "horizon_bars": horizon,
        "baseline_up_pct": 100 * base_up,
        "buy_count": int(buys.sum()),
        "buy_hit_pct": 100 * float((fwd[buys] > 0).mean()) if buys.any() else float("nan"),
        "buy_avg_fwd_pct": 100 * float(fwd[buys].mean()) if buys.any() else float("nan"),
        "sell_count": int(sells.sum()),
        "sell_hit_pct": 100 * float((fwd[sells] < 0).mean()) if sells.any() else float("nan"),
        "sell_avg_fwd_pct": 100 * float(fwd[sells].mean()) if sells.any() else float("nan"),
    }

    return BacktestResult(equity, buy_hold, position, trades, metrics, signal_stats)
