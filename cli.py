"""Command-line interface.

    python cli.py analyze AAPL --news     # technical + AI news -> combined signal
    python cli.py analyze AAPL --chart    # technical signal + interactive chart
    python cli.py news SAP.DE             # AI news analysis only
    python cli.py backtest SAP.DE         # how the technical rules did historically
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import urllib.error
import webbrowser
from pathlib import Path

from dotenv import load_dotenv

from core.ai_news import NewsAIError, NewsAIResult, analyze_news
from core.analysis import analyze
from core.backtest import run_backtest
from core.data import DataError, fetch_info
from core.fusion import NEWS_WEIGHT, TECH_WEIGHT, combine
from core.news import NewsBundle, collect_news

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
DISCLAIMER = "Not financial advice — research signals only. You make the decision."
BAR = "=" * 64
LINE = "-" * 64


def _fmt(v: float, suffix: str = "", sign: bool = False) -> str:
    if v != v:  # NaN
        return "n/a"
    return f"{v:+.1f}{suffix}" if sign else f"{v:.1f}{suffix}"


def _wrap(text: str, indent: str = "   ") -> str:
    return textwrap.fill(text, width=64, initial_indent=indent, subsequent_indent=indent)


def _run_news(ticker: str, company: str, args: argparse.Namespace) -> tuple[NewsBundle, NewsAIResult | None]:
    bundle = collect_news(ticker, company, days=args.days)
    if not bundle.articles:
        return bundle, None
    res = analyze_news(ticker, company, bundle.articles, model=args.model, use_cache=not args.refresh)
    return bundle, res


def _print_news(bundle: NewsBundle, res: NewsAIResult | None, days: int, show_headlines: bool) -> None:
    print(f" NEWS — last {days} days, {len(bundle.articles)} articles")
    for provider, err in bundle.errors.items():
        print(f"   ! {provider} unavailable: {err[:80]}")
    if res is None:
        print("   No recent news found — combined signal uses technicals only.")
        return
    a = res.analysis
    print(f" News view: {a.signal}   sentiment {a.sentiment:+.2f}, confidence {a.confidence:.2f} "
          f"→ news score {res.score:+.0f}")
    print(f" Relevant articles: {a.relevant_articles} of {res.articles_used}   horizon: {a.horizon}")
    print(LINE)
    print(_wrap(a.summary, " "))
    if a.key_events:
        print(" Key events:")
        icon = {"positive": "+", "negative": "-", "neutral": "·"}
        for ev in a.key_events:
            print(_wrap(f"[{icon[ev.impact]}] {ev.headline} ({ev.category}, {ev.importance})", "   "))
    for label, items in (("Catalysts", a.catalysts), ("Risks", a.risks)):
        if items:
            print(f" {label}:")
            for item in items:
                print(_wrap(f"• {item}", "   "))
    if show_headlines:
        print(" Headlines:")
        for art in bundle.articles:
            print(f"   {art.published:%m-%d} {art.source[:18]:18} {art.title[:70]}")
    cost = "cached, free" if res.cached else (
        f"{res.input_tokens + res.cache_read_tokens + res.cache_write_tokens} in / {res.output_tokens} out tokens"
        + (f", ≈ ${res.cost_usd:.3f}" if res.cost_usd is not None else ""))
    print(f" ({res.model}; {cost})")


def cmd_analyze(args: argparse.Namespace) -> None:
    ta = analyze(args.ticker, period=args.period, use_cache=not args.refresh)
    print(BAR)
    print(f" {ta.name} ({ta.ticker})   {ta.date:%Y-%m-%d}")
    print(f" Close {ta.close:.2f} {ta.currency}  ({ta.change_pct:+.2f}% today)")
    print(BAR)
    print(f" TECHNICAL: {ta.signal}    score {ta.score:+.0f} / 100   (BUY ≥ +35, SELL ≤ -35)")
    print(LINE)
    for pts, text in ta.reasons:
        print(f"   {pts:+5.0f}  {text}" if pts else f"          {text}")
    s = ta.snapshot
    print(LINE)
    print(f" RSI {s['rsi']:.1f} | MACD hist {s['macd_hist']:+.2f} | %B {s['bb_pct']:.2f} | "
          f"ATR {s['atr']:.2f} | Vol× {s['vol_ratio']:.2f}")
    print(f" EMA20 {s['ema20']:.2f} | EMA50 {s['ema50']:.2f} | EMA200 {s['ema200']:.2f}")

    if args.news:
        print(BAR)
        news_score = None
        try:
            bundle, res = _run_news(ta.ticker, ta.name, args)
            _print_news(bundle, res, args.days, args.headlines)
            news_score = res.score if res else None
        except NewsAIError as e:
            print(f" News analysis failed: {e}")
        c = combine(ta.score, news_score)
        print(BAR)
        if news_score is None:
            print(f" COMBINED SIGNAL: {c.signal}   (technical only, score {c.score:+.0f})")
        else:
            print(f" COMBINED SIGNAL: {c.signal}   score {c.score:+.0f}  "
                  f"= {TECH_WEIGHT:.0%} × technical {c.technical_score:+.0f} + {NEWS_WEIGHT:.0%} × news {c.news_score:+.0f}")
            if c.agreement == "conflict":
                print(" ⚠ Technicals and news point in opposite directions — treat with extra caution.")

    print(BAR)
    print(f" {DISCLAIMER}")

    if args.chart:
        from core.chart import build_chart
        OUTPUT_DIR.mkdir(exist_ok=True)
        path = OUTPUT_DIR / f"{ta.ticker.replace('.', '_')}_chart.html"
        build_chart(ta, bars=args.bars).write_html(path, include_plotlyjs="cdn")
        print(f" Chart saved: {path}")
        if not args.no_open:
            webbrowser.open(path.as_uri())


def cmd_news(args: argparse.Namespace) -> None:
    info = fetch_info(args.ticker)
    print(BAR)
    print(f" {info['name']} ({info['ticker']})")
    print(BAR)
    bundle, res = _run_news(info["ticker"], info["name"], args)
    _print_news(bundle, res, args.days, args.headlines)
    print(BAR)
    print(f" {DISCLAIMER}")


def _set_env_var(name: str, value: str) -> None:
    """Set NAME=value in .env, replacing an existing line or appending one."""
    path = ROOT / ".env"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for i, line in enumerate(lines):
        if line.strip().startswith(f"{name}="):
            lines[i] = f"{name}={value}"
            break
    else:
        lines.append(f"{name}={value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cmd_gist_setup(args: argparse.Namespace) -> None:
    from core.paper import BOOK_PATH, FileStore, GistStore
    token = os.getenv("GITHUB_TOKEN", "").strip()
    if not token:
        raise ValueError("Set GITHUB_TOKEN in .env first (a GitHub token with Gists read/write).")
    if os.getenv("PAPER_GIST_ID", "").strip() and not args.force:
        raise ValueError("PAPER_GIST_ID is already set in .env (use --force to create a new Gist).")
    local = FileStore(BOOK_PATH).read() or "[]"
    count = len(json.loads(local))
    store = GistStore.create(token, local)
    _set_env_var("PAPER_GIST_ID", store.gist_id)
    print(f" Created private Gist {store.gist_id} with {count} existing trade(s).")
    print(" PAPER_GIST_ID saved to .env. Add the same GITHUB_TOKEN and PAPER_GIST_ID to the")
    print(" Streamlit Cloud Secrets so the cloud dashboard uses the same portfolio.")


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
    load_dotenv(ROOT / ".env")

    p = argparse.ArgumentParser(description="Stock signals — technical + AI news analysis")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("ticker", help="e.g. AAPL, NVDA, SAP.DE, ASML.AS, SHEL.L")
    common.add_argument("--period", default="5y", help="history to download (default 5y)")
    common.add_argument("--refresh", action="store_true", help="ignore the local cache")

    news_opts = argparse.ArgumentParser(add_help=False)
    news_opts.add_argument("--days", type=int, default=7, help="news lookback in days (default 7)")
    news_opts.add_argument("--model", default=None, help="override NEWS_MODEL, e.g. claude-sonnet-5")
    news_opts.add_argument("--headlines", action="store_true", help="also list the raw headlines")

    a = sub.add_parser("analyze", parents=[common, news_opts], help="current signal with reasons")
    a.add_argument("--news", action="store_true", help="add AI news analysis and a combined signal")
    a.add_argument("--chart", action="store_true", help="save an interactive HTML chart")
    a.add_argument("--bars", type=int, default=180, help="bars shown on the chart")
    a.add_argument("--no-open", action="store_true", help="don't open the chart in a browser")
    a.set_defaults(func=cmd_analyze)

    n = sub.add_parser("news", parents=[common, news_opts], help="AI news analysis only")
    n.set_defaults(func=cmd_news)

    b = sub.add_parser("backtest", parents=[common], help="historical performance of the rules")
    b.add_argument("--fee", type=float, default=0.25, help="cost per side in %% (default 0.25)")
    b.add_argument("--horizon", type=int, default=10, help="bars ahead for hit-rate stats")
    b.add_argument("--csv", action="store_true", help="save the trade list as CSV")
    b.set_defaults(func=cmd_backtest)

    g = sub.add_parser("gist-setup", help="store virtual trades in a private GitHub Gist")
    g.add_argument("--force", action="store_true", help="create a new Gist even if one is set")
    g.set_defaults(func=cmd_gist_setup)

    args = p.parse_args(argv)
    try:
        args.func(args)
    except (DataError, NewsAIError, ValueError, urllib.error.URLError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
