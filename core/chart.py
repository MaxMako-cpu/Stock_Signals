"""Plotly candlestick chart with indicators and pattern markers.
Returns a Figure so Streamlit can embed it; the CLI saves it as HTML."""

from __future__ import annotations

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .analysis import TechnicalAnalysis
from .patterns import PATTERNS

COLORS = {"up": "#16a34a", "down": "#dc2626", "ema20": "#f59e0b", "ema50": "#3b82f6",
          "ema200": "#8b5cf6", "band": "rgba(148,163,184,0.35)"}


def build_chart(ta: TechnicalAnalysis, bars: int = 180) -> go.Figure:
    df = ta.frame.iloc[-bars:]
    pats = ta.patterns.loc[df.index]
    sig = ta.signals.loc[df.index]

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.62, 0.19, 0.19])

    fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"],
                                 close=df["Close"], name="Price",
                                 increasing_line_color=COLORS["up"],
                                 decreasing_line_color=COLORS["down"]), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_upper"], line=dict(width=1, color=COLORS["band"]),
                             name="Bollinger", legendgroup="bb"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["bb_lower"], line=dict(width=1, color=COLORS["band"]),
                             fill="tonexty", fillcolor="rgba(148,163,184,0.08)",
                             legendgroup="bb", showlegend=False), row=1, col=1)
    for name in ("ema20", "ema50", "ema200"):
        fig.add_trace(go.Scatter(x=df.index, y=df[name], name=name.upper(),
                                 line=dict(width=1.4, color=COLORS[name])), row=1, col=1)

    # Pattern markers
    for direction, symbol, color, ycol, offset in (
        (1, "triangle-up", COLORS["up"], "Low", 0.985),
        (-1, "triangle-down", COLORS["down"], "High", 1.015),
    ):
        names = [n for n, (d, _) in PATTERNS.items() if d == direction]
        hit = pats[names].any(axis=1)
        if hit.any():
            labels = pats.loc[hit, names].apply(
                lambda r: ", ".join(PATTERNS[n][1] for n in r.index[r.values]), axis=1)
            fig.add_trace(go.Scatter(
                x=df.index[hit], y=df.loc[hit, ycol] * offset, mode="markers",
                marker=dict(symbol=symbol, size=10, color=color),
                name="Bullish pattern" if direction > 0 else "Bearish pattern",
                text=labels, hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra></extra>"), row=1, col=1)

    # Signal changes
    changes = sig[(sig != sig.shift()) & sig.isin(["BUY", "SELL"])]
    for date, s in changes.items():
        fig.add_annotation(x=date, y=df.loc[date, "High"] if s == "SELL" else df.loc[date, "Low"],
                           text=s, showarrow=True, arrowhead=2, ay=-30 if s == "SELL" else 30,
                           font=dict(size=9, color=COLORS["down" if s == "SELL" else "up"]),
                           row=1, col=1)

    fig.add_trace(go.Scatter(x=df.index, y=df["rsi"], name="RSI", line=dict(width=1.3, color="#0ea5e9")),
                  row=2, col=1)
    for lvl in (30, 70):
        fig.add_hline(y=lvl, line=dict(dash="dot", width=1, color="gray"), row=2, col=1)

    hist_colors = [COLORS["up"] if v >= 0 else COLORS["down"] for v in df["macd_hist"].fillna(0)]
    fig.add_trace(go.Bar(x=df.index, y=df["macd_hist"], name="MACD hist", marker_color=hist_colors),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd"], name="MACD", line=dict(width=1.2, color="#3b82f6")),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["macd_signal"], name="Signal",
                             line=dict(width=1.2, color="#f59e0b")), row=3, col=1)

    title = (f"{ta.name} ({ta.ticker}) — {ta.close:.2f} {ta.currency}  |  "
             f"Technical score {ta.score:+.0f} → {ta.signal}")
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=800,
                      template="plotly_white", hovermode="x unified",
                      legend=dict(orientation="h", y=1.02, x=0))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    return fig
