"""Command-line interface for Phase 1.

    python cli.py analyze AAPL            # signal + reasons (+ chart with --chart)
    python cli.py backtest SAP.DE         # how the rules would have done historically
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from core.analysis import analyze
from core.backtest import run_backtest
from core.data import DataError

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DISCLAIMER = "Not financial advice — technical signals only; news analysis arrives in Phase 2."


def _fmt(v: float, suffix: str = "", sign: bool = False) -> str:
    if v != v:  # NaN
        return "n/a"
    return f"{v:+.1f}{suffix}" if sign else f"{v:.1f}{suffix}"


def cmd_analyze(args: argparse.Namespace) -> None:
    ta = analyze(args.ticker, period=args.period, use_cache=not args.refresh)
    bar = "=" * 64
    print(bar)
    print(f" {ta.name} ({ta.ticker})   {ta.date:%Y-%m-%d}")
    print(f" Close {ta.close:.2f} {ta.currency}  ({ta.change_pct:+.2f}% today)")
    print(bar)
    print(f" SIGNAL: {ta.signal}    technical score {ta.score:+.0f} / 100")
    print(f" (BUY ≥ +35, SELL ≤ -35)")
    print("-" * 64)
    print(" Why:")
    for pts, text in ta.reasons:
        print(f"   {pts:+5.0f}  {text}" if pts else f"          {text}")
    s = ta.snapshot
    print("-" * 64)
    print(f" RSI {s['rsi']:.1f} | MACD hist {s['macd_hist']:+.2f} | %B {s['bb_pct']:.2f} | "
          f"ATR {s['atr']:.2f} | Vol× {s['vol_ratio']:.2f}")
    print(f" EMA20 {s['ema20']:.2f} | EMA50 {s['ema50']:.2f} | EMA200 {s['ema200']:.2f}")
    print(bar)
    print(f" {DISCLAIMER}")

    if args.chart:
        from core.chart import build_chart
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / f"{ta.ticker.replace('.', '_')}_chart.html"
        build_chart(ta, bars=args.bars).write_html(path, include_plotlyjs="cdn")
        print(f" Chart saved: {path}")
        if not args.no_open:
            webbrowser.open(path.as_uri())


def cmd_backtest(args: argparse.Namespace) -> None:
    ta = analyze(args.ticker, period=args.period, use_cache=not args.refresh, with_info=False)
    res = run_backtest(ta.frame, ta.signals, fee_pct=args.fee, horizon=args.horizon)
    m, st = res.metrics, res.signal_stats
    bar = "=" * 64
    print(bar)
    print(f" Backtest {ta.ticker}   {m['period']}   ({m['bars']} bars)")
    print(f" Long-only, signal at close → trade next open, fee {m['fee_pct_per_side']}% per side")
    print(bar)
    print(f" {'':22}{'Strategy':>12}{'Buy & hold':>14}")
    print(f" {'Total return':22}{_fmt(m['strategy_return_pct'], '%', True):>12}{_fmt(m['buy_hold_return_pct'], '%', True):>14}")
    print(f" {'CAGR':22}{_fmt(m['strategy_cagr_pct'], '%', True):>12}{_fmt(m['buy_hold_cagr_pct'], '%', True):>14}")
    print(f" {'Max drawdown':22}{_fmt(m['strategy_max_dd_pct'], '%'):>12}{_fmt(m['buy_hold_max_dd_pct'], '%'):>14}")
    print(f" {'Sharpe':22}{m['strategy_sharpe']:>12.2f}{m['buy_hold_sharpe']:>14.2f}")
    print("-" * 64)
    print(f" Time in market {m['exposure_pct']:.0f}% | round trips {m['round_trips']} | "
          f"win rate {_fmt(m['win_rate_pct'], '%')} | avg trade {_fmt(m['avg_trade_pct'], '%', True)}")
    print("-" * 64)
    h = st["horizon_bars"]
    print(f" Signal accuracy over next {h} bars (baseline: up {st['baseline_up_pct']:.0f}% of the time)")
    print(f"   BUY  days {st['buy_count']:5d}  → went up   {_fmt(st['buy_hit_pct'], '%'):>6}, "
          f"avg {_fmt(st['buy_avg_fwd_pct'], '%', True)}")
    print(f"   SELL days {st['sell_count']:5d}  → went down {_fmt(st['sell_hit_pct'], '%'):>6}, "
          f"avg {_fmt(st['sell_avg_fwd_pct'], '%', True)}")
    print(bar)
    print(" Past performance does not predict future results.")

    if args.csv:
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / f"{ta.ticker.replace('.', '_')}_trades.csv"
        res.trades.to_csv(path, index=False)
        print(f" Trades saved: {path}")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser(description="Stock signals — Phase 1 (technical engine)")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("ticker", help="e.g. AAPL, NVDA, SAP.DE, ASML.AS, SHEL.L")
    common.add_argument("--period", default="5y", help="history to download (default 5y)")
    common.add_argument("--refresh", action="store_true", help="ignore the local cache")

    a = sub.add_parser("analyze", parents=[common], help="current signal with reasons")
    a.add_argument("--chart", action="store_true", help="save an interactive HTML chart")
    a.add_argument("--bars", type=int, default=180, help="bars shown on the chart")
    a.add_argument("--no-open", action="store_true", help="don't open the chart in a browser")
    a.set_defaults(func=cmd_analyze)

    b = sub.add_parser("backtest", parents=[common], help="historical performance of the rules")
    b.add_argument("--fee", type=float, default=0.25, help="cost per side in %% (default 0.25)")
    b.add_argument("--horizon", type=int, default=10, help="bars ahead for hit-rate stats")
    b.add_argument("--csv", action="store_true", help="save the trade list as CSV")
    b.set_defaults(func=cmd_backtest)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except (DataError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
