# Stock Signals

BUY / HOLD / SELL signals from a candlestick + indicator model, with AI news analysis (Phase 2),
a Streamlit dashboard (Phase 3) and a Telegram bot (Phase 4).

> Research tool, not financial advice. Trades are placed manually (e.g. in Revolut).

## Setup (Windows)

```powershell
cd C:\dev\stock-signals
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Usage

```powershell
.\.venv\Scripts\python cli.py analyze AAPL           # signal + reasons
.\.venv\Scripts\python cli.py analyze SAP.DE --chart # + interactive chart in the browser
.\.venv\Scripts\python cli.py backtest ASML.AS       # how the rules did historically
.\.venv\Scripts\python cli.py backtest NVDA --fee 0.5 --csv
.\.venv\Scripts\python -m pytest -q                  # tests (offline)
```

Tickers use Yahoo notation: `AAPL`, `SAP.DE` (Xetra), `ASML.AS` (Amsterdam), `SHEL.L` (London),
`MC.PA` (Paris). Data is cached in `data/cache/` for 12 h; `--refresh` forces a download.

## How the technical score works (−100 … +100)

| Component | Points | Idea |
|---|---|---|
| Price vs EMA50 | ±10 | medium-term trend |
| EMA20 vs EMA50 | ±10 | short-term trend |
| EMA50 vs EMA200 | ±10 | long-term regime (golden/death cross) |
| MACD histogram | ±10 | momentum |
| RSI < 30 / > 70 | ±15 | oversold / overbought |
| Close outside Bollinger band | ±10 | stretched price |
| Candle patterns (last 3 bars, decaying) | up to ±30 | ×1.3 if volume > 1.2× average |

**BUY ≥ +35, SELL ≤ −35**, otherwise HOLD. Weights live in `core/scoring.py`.

Patterns: engulfing, hammer / hanging man, inverted hammer / shooting star, morning / evening star,
piercing line / dark cloud cover, three white soldiers / black crows, doji.

## Backtest rules

Long-only. Signal at the close of day *t* → trade at the open of day *t+1* (no lookahead).
BUY = go long, SELL = go to cash, HOLD = keep position. Default cost 0.25 % per side
(commission + FX spread). Also reports how often BUY/SELL days were followed by a rise/fall
over the next N bars vs. the baseline.

## Layout

```
core/
  data.py        yfinance download + Parquet cache
  indicators.py  EMA, RSI, MACD, Bollinger, ATR, volume ratio
  patterns.py    candlestick pattern detection
  scoring.py     technical score, signal mapping, explanations
  analysis.py    analyze(ticker) — shared entry point for CLI / Streamlit / Telegram
  backtest.py    long-only backtest + signal hit rates
  chart.py       Plotly candlestick chart with patterns and signals
cli.py
tests/
```

## Roadmap

1. ✅ Core engine + backtester
2. Claude news analysis (Finnhub + Google News RSS → structured sentiment)
3. Streamlit dashboard
4. Telegram bot (`/analyze`, `/watch`)
5. ML model (LightGBM, walk-forward) + signal fusion tuning
