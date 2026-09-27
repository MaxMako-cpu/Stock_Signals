"""Plotly charts for the CLI (saved as HTML) and the Streamlit dashboard.

Colors follow a validated palette (categorical slots 1-3 checked for
color-vision deficiency in both modes); up/down candles use the status
good/critical steps. Each mode has its own steps rather than an inversion.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analysis import TechnicalAnalysis
from .patterns import PATTERNS

THEMES = {
    "light": {
        "surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7",
        "series": ["#2a78d6", "#eb6834", "#1baf7a"],
        "up": "#0ca30c", "down": "#d03b3b", "neutral": "#898781",
        "band": "rgba(137,135,129,0.10)", "band_line": "rgba(137,135,129,0.45)",
    },
    "dark": {
        "surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835",
        "series": ["#3987e5", "#d95926", "#199e70"],
        "up": "#0ca30c", "down": "#d03b3b", "neutral": "#898781",
        "band": "rgba(195,194,183,0.08)", "band_line": "rgba(195,194,183,0.35)",
    },
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def _base_layout(fig: go.Figure, t: dict, height: int, title: str | None = None) -> None:
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=15, color=t["ink"])) if title else None,
        paper_bgcolor=t["surface"], plot_bgcolor=t["surface"],
        font=dict(family=FONT, color=t["ink2"], size=12),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=t["surface"], font=dict(color=t["ink"]), bordercolor=t["axis"]),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(color=t["ink2"])),
        margin=dict(l=10, r=10, t=60 if title else 30, b=10),
    )
    fig.update_xaxes(gridcolor=t["grid"], linecolor=t["axis"], tickfont=dict(color=t["muted"]),
                     zeroline=False, showspikes=True, spikecolor=t["muted"], spikethickness=1,
                     spikedash="dot", spikemode="across")
    fig.update_yaxes(gridcolor=t["grid"], linecolor=t["axis"], tickfont=dict(color=t["muted"]),
                     zeroline=False)


def build_chart(ta: TechnicalAnalysis, bars: int = 180, mode: str = "light",
                title: bool = True) -> go.Figure:
    t = THEMES[mode]
    df = ta.frame.iloc[-bars:]
    pats = ta.patterns.loc[df.index]
    sig = ta.signals.loc[df.index]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.62, 0.19, 0.19],
                        subplot_titles=("", "RSI (14)", "MACD (12, 26, 9)"))

    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price", increasing=dict(line=dict(color=t["up"], width=1), fillcolor=t["up"]),
        decreasing=dict(line=dict(color=t["down"], width=1), fillcolor=t["down"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], line=dict(width=1, color=t["band_line"]),
                             name="Bollinger band", legendgroup="bb", hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], line=dict(width=1, color=t["band_line"]),
                             fill="tonexty", fillcolor=t["band"], legendgroup="bb",
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    for name, color in zip(("ema20", "ema50", "ema200"), t["series"]):
        fig.add_trace(go.Scatter(x=df.index, y=df[name], name=name.upper(),
                                 line=dict(width=2, color=color),
                                 hovertemplate="%{y:.2f}"), row=1, col=1)

    # Pattern markers: shape + color, so direction never relies on color alone
    for direction, symbol, color, ycol, offset in (
        (1, "triangle-up", t["up"], "Low", 0.985),
        (-1, "triangle-down", t["down"], "High", 1.015),
    ):
        names = [n for n, (d, _) in PATTERNS.items() if d == direction]
        hit = pats[names].any(axis=1)
        if hit.any():
            labels = pats.loc[hit, names].apply(
                lambda r: ", ".join(PATTERNS[n][1] for n in r.index[r.values]), axis=1)
            fig.add_trace(go.Scatter(
                x=df.index[hit], y=df.loc[hit, ycol] * offset, mode="markers",
                marker=dict(symbol=symbol, size=11, color=color,
                            line=dict(width=2, color=t["surface"])),
                name="Bullish pattern" if direction > 0 else "Bearish pattern",
                text=labels, hovertemplate="%{text}<extra></extra>"), row=1, col=1)

    # Position flips only (BUY after SELL and vice versa) — HOLD in between doesn't count
    active = sig[sig.isin(["BUY", "SELL"])]
    changes = active[active != active.shift()]
    for date, s in changes.items():
        sell = s == "SELL"
        fig.add_annotation(x=date, y=df.loc[date, "High" if sell else "Low"], text=s,
                           showarrow=True, arrowhead=2, arrowwidth=1, ay=-28 if sell else 28,
                           arrowcolor=t["down" if sell else "up"],
                           font=dict(size=10, color=t["ink"]), row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", showlegend=False,
                             line=dict(width=2, color=t["series"][0]),
                             hovertemplate="%{y:.1f}"), row=2, col=1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line=dict(dash="dot", width=1, color=t["muted"]), row=2, col=1)

    hist_colors = [t["up"] if v >= 0 else t["down"] for v in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="MACD histogram", showlegend=False,
                         marker=dict(color=hist_colors, line=dict(width=0)),
                         hovertemplate="%{y:.2f}"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", showlegend=False,
                             line=dict(width=2, color=t["series"][0]),
                             hovertemplate="%{y:.2f}"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="MACD signal", showlegend=False,
                             line=dict(width=2, color=t["series"][1]),
                             hovertemplate="%{y:.2f}"), row=3, col=1)

    heading = None
    if title:
        heading = (f"{ta.name} ({ta.ticker}) — {ta.close:.2f} {ta.currency}  ·  "
                   f"technical score {ta.score:+.0f} → {ta.signal}")
    _base_layout(fig, t, height=820, title=heading)
    fig.update_layout(xaxis_rangeslider_visible=False, bargap=0.25)
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(range=[0, 100], tickvals=[30, 50, 70], row=2, col=1)
    for ann in fig.layout.annotations:
        if ann.text in ("RSI (14)", "MACD (12, 26, 9)"):   # subplot titles (paper coords)
            ann.font = dict(size=12, color=t["ink2"])
            ann.x, ann.xanchor = 0, "left"
    return fig


def build_equity_chart(equity: pd.Series, buy_hold: pd.Series, position: pd.Series,
                       mode: str = "light") -> go.Figure:
    """Strategy vs buy-and-hold, both as % return on one axis."""
    t = THEMES[mode]
    fig = go.Figure()
    series = (("Strategy", equity, t["series"][0]), ("Buy & hold", buy_hold, t["series"][1]))
    for name, s, color in series:
        pct = 100 * (s - 1)
        fig.add_trace(go.Scatter(x=pct.index, y=pct, name=name, line=dict(width=2, color=color),
                                 hovertemplate="%{y:+.1f}%"))
        # Direct label at the line end
        fig.add_annotation(x=pct.index[-1], y=pct.iloc[-1], text=f"{name} {pct.iloc[-1]:+.0f}%",
                           showarrow=False, xanchor="left", xshift=6,
                           font=dict(size=11, color=t["ink2"]))

    # Shade the periods where the strategy was invested
    in_market = position.astype(bool)
    starts = position.index[in_market & ~in_market.shift(fill_value=False)]
    ends = position.index[in_market & ~in_market.shift(-1, fill_value=False)]
    for s, e in zip(starts, ends):
        fig.add_vrect(x0=s, x1=e, fillcolor=t["series"][0], opacity=0.06, line_width=0, layer="below")

    _base_layout(fig, t, height=380)
    fig.add_hline(y=0, line=dict(width=1, color=t["axis"]))
    fig.update_yaxes(ticksuffix="%")
    fig.update_layout(margin=dict(r=130))
    return fig
