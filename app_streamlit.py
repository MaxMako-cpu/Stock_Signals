"""Streamlit dashboard.

    .\\.venv\\Scripts\\streamlit run app_streamlit.py
"""

from __future__ import annotations

from pathlib import Path

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from core.ai_news import DEFAULT_MODEL, NewsAIError, analyze_news
from core.analysis import analyze
from core.backtest import run_backtest
from core.chart import build_chart, build_equity_chart
from core.data import DataError
from core.fusion import NEWS_WEIGHT, TECH_WEIGHT, combine
from core.news import collect_news
from core.paper import (DEFAULT_FEE_PCT, PaperBook, fx_history_to_eur, fx_to_eur, get_quote,
                        normalize_currency, open_trade, realized, value_trade, whatif_last_buy)
from core.scoring import BUY_THRESHOLD, SELL_THRESHOLD

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

st.set_page_config(page_title="Stock Signals", page_icon="📈", layout="wide")

SIGNAL_ICON = {"BUY": "▲", "SELL": "▼", "HOLD": "●", "N/A": "–"}
SIGNAL_BADGE = {"BUY": "green", "SELL": "red", "HOLD": "gray", "N/A": "gray"}
IMPACT_ICON = {"positive": "▲", "negative": "▼", "neutral": "●"}
MODELS = ["claude-sonnet-5", "claude-opus-5", "claude-haiku-4-5"]


# ---------- cached data access ----------

@st.cache_data(ttl=3600, show_spinner=False)
def get_analysis(ticker: str, period: str, refresh: bool):
    return analyze(ticker, period=period, use_cache=not refresh)


@st.cache_data(ttl=1800, show_spinner=False)
def get_news(ticker: str, company: str, days: int, model: str, refresh: bool):
    bundle = collect_news(ticker, company, days=days)
    res = analyze_news(ticker, company, bundle.articles, model=model,
                       use_cache=not refresh) if bundle.articles else None
    return bundle, res


@st.cache_data(ttl=3600, show_spinner=False)
def get_backtest(ticker: str, period: str, fee: float, horizon: int, refresh: bool):
    ta = get_analysis(ticker, period, refresh)
    return run_backtest(ta.frame, ta.signals, fee_pct=fee, horizon=horizon)


@st.cache_data(ttl=300, show_spinner=False)
def get_quote_cached(ticker: str):
    return get_quote(ticker)


@st.cache_data(ttl=900, show_spinner=False)
def get_fx(currency: str) -> float:
    return fx_to_eur(currency)


@st.cache_data(ttl=3600, show_spinner=False)
def get_fx_history(currency: str) -> pd.Series:
    return fx_history_to_eur(currency)


def eur(v: float, sign: bool = False) -> str:
    return f"{v:+,.2f} €" if sign else f"{v:,.2f} €"


def theme_mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def signal_text(sig: str) -> str:
    return f"{SIGNAL_ICON.get(sig, '')} {sig}"


# ---------- sidebar ----------

with st.sidebar:
    st.header("Stock Signals")
    with st.form("controls"):
        ticker = st.text_input("Ticker", value="AAPL",
                               help="Yahoo notation: AAPL, NVDA, SAP.DE, ASML.AS, SHEL.L, MC.PA")
        st.caption("EU stocks need an exchange suffix: .DE Xetra · .AS Amsterdam · .PA Paris · .L London")
        period = st.selectbox("History", ["1y", "2y", "5y", "10y"], index=2)
        bars = st.slider("Chart bars", 60, 500, 180, step=20)
        use_news = st.toggle("AI news analysis", value=True,
                             help="Calls the Claude API (≈ $0.02 per new analysis; cached results are free)")
        env_model = os.getenv("NEWS_MODEL", "").strip() or DEFAULT_MODEL
        models = [env_model] + [m for m in MODELS if m != env_model]
        model = st.selectbox("News model", models, index=0)
        days = st.slider("News lookback (days)", 1, 14, 7)
        refresh = st.checkbox("Ignore caches (fresh download + new AI call)")
        submitted = st.form_submit_button("Analyze", type="primary", width="stretch")

    if submitted:
        if refresh:
            get_analysis.clear()
            get_news.clear()
            get_backtest.clear()
        st.session_state["params"] = dict(ticker=ticker.strip().upper(), period=period, bars=bars,
                                          use_news=use_news, model=model, days=days, refresh=refresh)
    st.caption("Research tool — not financial advice. You place trades yourself.")

params = st.session_state.get("params")
if not params:
    st.title("📈 Stock Signals")
    st.write("Enter a ticker in the sidebar and press **Analyze**.")
    st.stop()

mode = theme_mode()

# ---------- technical analysis ----------

try:
    with st.spinner(f"Loading {params['ticker']}…"):
        ta = get_analysis(params["ticker"], params["period"], params["refresh"])
except (DataError, ValueError) as e:
    st.error(str(e))
    st.stop()

bundle = res = None
news_error = None
if params["use_news"]:
    try:
        with st.spinner("Reading the news with Claude…"):
            bundle, res = get_news(ta.ticker, ta.name, params["days"], params["model"], params["refresh"])
    except NewsAIError as e:
        news_error = str(e)

combined = combine(ta.score, res.score if res else None)

# ---------- header + tiles ----------

st.title(f"{ta.name}")
st.caption(f"{ta.ticker} · last close {ta.date:%a %d %b %Y} · prices in {ta.currency or 'n/a'}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Last close", f"{ta.close:,.2f} {ta.currency}", f"{ta.change_pct:+.2f}% today", border=True)
c2.metric("Combined signal", signal_text(combined.signal),
          f"score {combined.score:+.0f}", delta_color="off", border=True,
          help=f"{TECH_WEIGHT:.0%} technical + {NEWS_WEIGHT:.0%} news. "
               f"BUY ≥ +{BUY_THRESHOLD}, SELL ≤ {SELL_THRESHOLD}.")
c3.metric("Technical", signal_text(ta.signal), f"score {ta.score:+.0f}", delta_color="off", border=True)
if res:
    c4.metric("News", signal_text(res.analysis.signal),
              f"score {res.score:+.0f} · conf {res.analysis.confidence:.2f}", delta_color="off", border=True)
else:
    c4.metric("News", "off" if not params["use_news"] else "n/a",
              "no news analysis", delta_color="off", border=True)

st.badge(f"{signal_text(combined.signal)} — combined", color=SIGNAL_BADGE[combined.signal])
if combined.agreement == "conflict":
    st.warning("Technicals and news point in opposite directions — treat this signal with extra caution.",
               icon="⚠️")
if news_error:
    st.warning(f"News analysis unavailable: {news_error}", icon="📰")

tab_chart, tab_why, tab_news, tab_paper, tab_bt, tab_data = st.tabs(
    ["📈 Chart", "🧮 Why", "📰 News", "💶 Virtual trade", "🧪 Backtest", "📋 Data"])

# ---------- chart ----------

with tab_chart:
    st.plotly_chart(build_chart(ta, bars=params["bars"], mode=mode, title=False),
                    width="stretch", theme=None, config={"displaylogo": False})
    st.caption("▲ bullish / ▼ bearish candle pattern (hover for name) · BUY/SELL labels mark "
               "where the technical signal flipped · shaded band = Bollinger (20, 2)")

# ---------- why ----------

with tab_why:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("What drives the technical score")
        reasons = pd.DataFrame(ta.reasons, columns=["Points", "Reason"])
        st.dataframe(reasons, hide_index=True, width="stretch",
                     column_config={"Points": st.column_config.NumberColumn(format="%+.0f")})
        st.caption(f"Total {ta.score:+.0f} (clipped to ±100). BUY ≥ +{BUY_THRESHOLD}, SELL ≤ {SELL_THRESHOLD}.")
    with right:
        st.subheader("Indicators")
        s = ta.snapshot
        snap = pd.DataFrame([
            ("RSI (14)", f"{s['rsi']:.1f}", "< 30 oversold · > 70 overbought"),
            ("MACD histogram", f"{s['macd_hist']:+.2f}", "> 0 upward momentum"),
            ("Bollinger %B", f"{s['bb_pct']:.2f}", "< 0 below band · > 1 above"),
            ("EMA 20", f"{s['ema20']:.2f}", ""),
            ("EMA 50", f"{s['ema50']:.2f}", ""),
            ("EMA 200", f"{s['ema200']:.2f}", ""),
            ("ATR (14)", f"{s['atr']:.2f}", "typical daily range"),
            ("Volume vs 20-day avg", f"{s['vol_ratio']:.2f}×", "> 1.2 confirms candles"),
        ], columns=["Indicator", "Value", "Reading"])
        st.dataframe(snap, hide_index=True, width="stretch")

# ---------- news ----------

with tab_news:
    if not params["use_news"]:
        st.info("AI news analysis is off — enable it in the sidebar.")
    elif news_error:
        st.error(news_error)
    elif res is None:
        st.info(f"No news found for the last {params['days']} days.")
    else:
        a = res.analysis
        st.subheader(f"News view: {signal_text(a.signal)}")
        st.caption(f"Sentiment {a.sentiment:+.2f} · confidence {a.confidence:.2f} · "
                   f"{a.relevant_articles} of {res.articles_used} articles relevant · horizon: {a.horizon}")
        st.write(a.summary)

        if a.key_events:
            st.markdown("**Key events**")
            events = pd.DataFrame([{
                "Impact": f"{IMPACT_ICON[e.impact]} {e.impact}",
                "Event": e.headline,
                "Type": e.category.replace("_", " "),
                "Importance": e.importance,
            } for e in a.key_events])
            st.dataframe(events, hide_index=True, width="stretch")

        col_c, col_r = st.columns(2)
        with col_c:
            st.markdown("**Catalysts**")
            for item in a.catalysts:
                st.markdown(f"- {item}")
        with col_r:
            st.markdown("**Risks**")
            for item in a.risks:
                st.markdown(f"- {item}")

        with st.expander(f"All {len(bundle.articles)} headlines"):
            for art in bundle.articles:
                src = f" — {art.source}" if art.source else ""
                st.markdown(f"`{art.published:%m-%d %H:%M}` [{art.title}]({art.url}){src}")
        for provider, err in bundle.errors.items():
            st.caption(f"⚠️ {provider} unavailable: {err[:120]}")

        cost = "cached (free)" if res.cached else (
            f"{res.input_tokens + res.cache_read_tokens + res.cache_write_tokens} in / "
            f"{res.output_tokens} out tokens"
            + (f" · ≈ ${res.cost_usd:.3f}" if res.cost_usd is not None else ""))
        st.caption(f"Analyzed by {res.model} · {cost}")

# ---------- virtual trade ----------

with tab_paper:
    book = PaperBook()
    flash = st.session_state.pop("paper_flash", None)
    if flash:
        st.success(flash)

    st.subheader(f"Open a virtual trade — {ta.ticker}")
    st.caption("Practice only: no real money. EUR is converted to the stock's currency like Revolut does, "
               "and a fee is taken on the way in and out.")
    q_price, q_ccy, q_kind = get_quote_cached(ta.ticker)
    price, ccy = normalize_currency(q_price, q_ccy or ta.currency)
    try:
        fx = get_fx(ccy)
    except Exception as e:
        fx = None
        st.error(f"No EUR exchange rate for {ccy}: {e}")

    f1, f2, f3 = st.columns([1, 1, 2])
    amount = f1.number_input("Amount (EUR)", min_value=10.0, max_value=1_000_000.0, value=1000.0, step=100.0)
    fee = f2.number_input("Fee per side (%)", 0.0, 2.0, DEFAULT_FEE_PCT, step=0.05,
                          help="Commission + FX spread. Revolut: often 0 commission within the free "
                               "allowance, plus ~0.5–1% FX markup outside market hours/limits.")
    note = f3.text_input("Note (optional)", placeholder="e.g. testing the news signal")

    if fx:
        preview = open_trade(ta.ticker, ta.name, amount, price, ccy, fx, fee)
        p1, p2, p3, p4 = st.columns(4)
        p1.metric(f"Price ({q_kind})", f"{price:,.2f} {ccy}", border=True)
        p2.metric("EUR per 1 " + ccy, f"{fx:.4f}" if ccy != "EUR" else "1.0000", border=True)
        p3.metric("You'd get", f"{preview.shares:,.4f} shares", border=True)
        p4.metric("Entry fee", eur(amount * fee / 100), border=True)

        if combined.signal != "BUY":
            st.info(f"Current combined signal is {signal_text(combined.signal)} — you can still record "
                    "the trade to test what happens.")
        if st.button(f"💶 Buy virtually for {eur(amount)}", type="primary"):
            book.add(open_trade(ta.ticker, ta.name, amount, price, ccy, fx, fee,
                                signal=combined.signal, score=round(combined.score, 1), note=note))
            st.session_state["paper_flash"] = (f"Virtual buy recorded: {preview.shares:,.4f} {ta.ticker} "
                                               f"at {price:,.2f} {ccy} for {eur(amount)}.")
            st.rerun()

    # ----- what-if on history -----
    st.divider()
    st.subheader("What if… (history)")
    fx_hist = None
    if ccy != "EUR":
        try:
            fx_hist = get_fx_history(ccy)
        except Exception:
            fx_hist = None
    w = whatif_last_buy(ta.frame, ta.signals, amount, ta.currency, fx_hist, fee)
    if w is None:
        st.write(f"No BUY signal flip in the loaded history of {ta.ticker} to test.")
    else:
        v = w["valuation"]
        verb = "gained" if v.pnl_eur >= 0 else "lost"
        st.markdown(
            f"**{eur(amount)} at the last BUY signal** — signal on {w['signal_day']:%d %b %Y}, "
            f"bought at the next open ({w['entry_price']:,.2f} {w['currency']}) and held "
            f"{w['days_held']} days → worth **{eur(v.value_eur)}** today: you'd have {verb} "
            f"**{eur(abs(v.pnl_eur))} ({v.pnl_pct:+.1f}%)**.")
        st.caption(f"Stock {v.price_change_pct:+.1f}% · currency {v.fx_change_pct:+.1f}% vs EUR · "
                   f"fees {eur(v.fees_eur)}"
                   + (" · FX history unavailable, currency effect ignored" if w["fx_ignored"] else ""))
        if w["sell_since"] is not None:
            st.caption(f"Note: the signals said SELL on {w['sell_since']:%d %b %Y} — following them "
                       "you'd have sold earlier.")

    bt_default = get_backtest(ta.ticker, params["period"], fee, 10, params["refresh"])
    m = bt_default.metrics
    strat_eur = amount * bt_default.equity.iloc[-1]
    bh_eur = amount * bt_default.buy_hold.iloc[-1]
    st.markdown(
        f"**{eur(amount)} following every signal** since {bt_default.equity.index[0]:%b %Y} "
        f"→ **{eur(strat_eur)}** ({eur(strat_eur - amount, True)}), versus buy & hold → "
        f"**{eur(bh_eur)}** ({eur(bh_eur - amount, True)}).")
    st.caption("Technical signals only, in the stock's own currency (currency moves ignored). "
               "Details in the Backtest tab.")

    # ----- portfolio -----
    st.divider()
    st.subheader("My virtual portfolio")
    trades = book.load()
    open_trades = [t for t in trades if t.status == "open"]
    closed_trades = [t for t in trades if t.status == "closed"]

    if not trades:
        st.write("No virtual trades yet.")
    if open_trades:
        if st.button("↻ Refresh prices"):
            get_quote_cached.clear()
            get_fx.clear()
            st.rerun()
        rows, marks = [], {}
        for t in open_trades:
            try:
                pq, pc, _ = get_quote_cached(t.ticker)
                pnow, pccy = normalize_currency(pq, pc or t.currency)
                fnow = get_fx(pccy)
                val = value_trade(t, pnow, pccy, fnow)
                marks[t.id] = (pnow, pccy, fnow)
            except Exception as e:
                st.warning(f"Could not price {t.ticker}: {e}")
                continue
            rows.append({
                "ID": t.id, "Ticker": t.ticker, "Opened": t.opened_at[:10],
                "Signal then": t.signal, "Invested €": t.amount_eur,
                "Entry": f"{t.entry_price:,.2f} {t.currency}", "Now": f"{pnow:,.2f} {pccy}",
                "Value €": val.value_eur, "P&L €": val.pnl_eur, "P&L %": val.pnl_pct,
                "Stock %": val.price_change_pct, "FX %": val.fx_change_pct, "Note": t.note,
            })
        if rows:
            df_open = pd.DataFrame(rows)
            inv, value = df_open["Invested €"].sum(), df_open["Value €"].sum()
            t1, t2, t3 = st.columns(3)
            t1.metric("Invested (open)", eur(inv), border=True)
            t2.metric("Value now", eur(value), border=True)
            t3.metric("Unrealized P&L", eur(value - inv, True),
                      f"{100 * (value - inv) / inv:+.2f}%", border=True)
            money = st.column_config.NumberColumn(format="%.2f")
            pct = st.column_config.NumberColumn(format="%+.2f%%")
            st.dataframe(df_open, hide_index=True, width="stretch", column_config={
                "Invested €": money, "Value €": money,
                "P&L €": st.column_config.NumberColumn(format="%+.2f"),
                "P&L %": pct, "Stock %": pct, "FX %": pct})

            labels = {t.id: f"{t.id} · {t.ticker} · {eur(t.amount_eur)} · {t.opened_at[:10]}"
                      for t in open_trades if t.id in marks}
            s1, s2, s3 = st.columns([3, 1, 1])
            chosen = s1.selectbox("Trade", list(labels), format_func=labels.get)
            if s2.button("Sell virtually", width="stretch"):
                pnow, pccy, fnow = marks[chosen]
                closed = book.close(chosen, pnow, pccy, fnow)
                r = realized(closed)
                st.session_state["paper_flash"] = (f"Sold {closed.ticker}: {eur(r.pnl_eur, True)} "
                                                   f"({r.pnl_pct:+.2f}%).")
                st.rerun()
            if s3.button("Delete", width="stretch"):
                book.delete(chosen)
                st.rerun()

    if closed_trades:
        st.markdown("**Closed trades**")
        rows = []
        for t in closed_trades:
            r = realized(t)
            rows.append({"ID": t.id, "Ticker": t.ticker, "Opened": t.opened_at[:10],
                         "Closed": (t.closed_at or "")[:10], "Signal then": t.signal,
                         "Invested €": t.amount_eur, "Returned €": r.value_eur,
                         "P&L €": r.pnl_eur, "P&L %": r.pnl_pct})
        df_closed = pd.DataFrame(rows)
        total = df_closed["P&L €"].sum()
        wins = (df_closed["P&L €"] > 0).mean() * 100
        n = len(df_closed)
        st.caption(f"Realized P&L {eur(total, True)} over {n} trade{'s' if n != 1 else ''} · "
                   f"{wins:.0f}% winners")
        st.dataframe(df_closed, hide_index=True, width="stretch", column_config={
            "Invested €": st.column_config.NumberColumn(format="%.2f"),
            "Returned €": st.column_config.NumberColumn(format="%.2f"),
            "P&L €": st.column_config.NumberColumn(format="%+.2f"),
            "P&L %": st.column_config.NumberColumn(format="%+.2f%%")})

# ---------- backtest ----------

with tab_bt:
    st.subheader("How the technical rules did historically")
    b1, b2, _ = st.columns([1, 1, 2])
    fee = b1.number_input("Cost per side (%)", 0.0, 2.0, 0.25, step=0.05,
                          help="Commission + FX spread per buy or sell")
    horizon = b2.number_input("Hit-rate horizon (bars)", 1, 60, 10)
    bt = get_backtest(ta.ticker, params["period"], fee, int(horizon), params["refresh"])
    m, sst = bt.metrics, bt.signal_stats

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Strategy return", f"{m['strategy_return_pct']:+.1f}%",
              f"{m['strategy_return_pct'] - m['buy_hold_return_pct']:+.1f} pts vs buy & hold", border=True)
    k2.metric("Buy & hold", f"{m['buy_hold_return_pct']:+.1f}%", border=True)
    k3.metric("Max drawdown", f"{m['strategy_max_dd_pct']:.1f}%",
              f"buy & hold {m['buy_hold_max_dd_pct']:.1f}%", delta_color="off", border=True)
    k4.metric("Win rate", f"{m['win_rate_pct']:.0f}%" if m["round_trips"] else "n/a",
              f"{m['round_trips']} round trips", delta_color="off", border=True)

    st.plotly_chart(build_equity_chart(bt.equity, bt.buy_hold, bt.position, mode=mode),
                    width="stretch", theme=None, config={"displaylogo": False})
    st.caption(f"{m['period']} · long-only · signal at close → trade next open · "
               f"shaded = invested ({m['exposure_pct']:.0f}% of the time)")

    left, right = st.columns(2)
    with left:
        st.markdown("**Summary**")
        st.dataframe(pd.DataFrame({
            "Strategy": [f"{m['strategy_return_pct']:+.1f}%", f"{m['strategy_cagr_pct']:+.1f}%",
                         f"{m['strategy_max_dd_pct']:.1f}%", f"{m['strategy_sharpe']:.2f}"],
            "Buy & hold": [f"{m['buy_hold_return_pct']:+.1f}%", f"{m['buy_hold_cagr_pct']:+.1f}%",
                           f"{m['buy_hold_max_dd_pct']:.1f}%", f"{m['buy_hold_sharpe']:.2f}"],
        }, index=["Total return", "CAGR", "Max drawdown", "Sharpe"]), width="stretch")
    with right:
        st.markdown(f"**Signal accuracy over the next {sst['horizon_bars']} bars**")
        st.dataframe(pd.DataFrame([
            ("BUY days", sst["buy_count"], f"{sst['buy_hit_pct']:.0f}% went up",
             f"{sst['buy_avg_fwd_pct']:+.1f}%"),
            ("SELL days", sst["sell_count"], f"{sst['sell_hit_pct']:.0f}% went down",
             f"{sst['sell_avg_fwd_pct']:+.1f}%"),
            ("Any day (baseline)", None, f"{sst['baseline_up_pct']:.0f}% went up", ""),
        ], columns=["", "Count", "Hit rate", "Avg move"]), hide_index=True, width="stretch")

    with st.expander(f"Trades ({len(bt.trades)})"):
        st.dataframe(bt.trades, hide_index=True, width="stretch",
                     column_config={"return_pct": st.column_config.NumberColumn("return %", format="%+.1f")})
    st.caption("Backtest covers the technical rules only — historical news isn't available for free. "
               "Past performance does not predict future results.")

# ---------- data ----------

with tab_data:
    cols = ["Open", "High", "Low", "Close", "Volume", "ema20", "ema50", "ema200", "rsi",
            "macd_hist", "bb_pct", "atr", "vol_ratio"]
    table = ta.frame[cols].copy()
    table["score"] = ta.components["total"]
    table["signal"] = ta.signals
    table = table.iloc[::-1]
    st.dataframe(table.head(250).round(2), width="stretch")
    st.download_button("Download CSV", table.to_csv().encode(), f"{ta.ticker}_signals.csv",
                       "text/csv")
