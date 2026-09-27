# Stock Signals

BUY / HOLD / SELL signals from a candlestick + indicator model, with AI news analysis (Phase 2),
a Streamlit dashboard (Phase 3) and a Telegram bot (Phase 4).

> Research tool, not financial advice. Trades are placed manually (e.g. in Revolut).

## Setup (Windows)

```powershell
cd C:\dev\stock-signals
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
copy .env.example .env     # then paste your ANTHROPIC_API_KEY (and optional FINNHUB_API_KEY)
```

## Usage

```powershell
.\.venv\Scripts\python cli.py analyze AAPL --news    # technical + AI news -> combined signal
.\.venv\Scripts\python cli.py news ASML.AS --headlines # AI news analysis only, with raw headlines
.\.venv\Scripts\python cli.py analyze AAPL           # technical signal + reasons
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

## AI news analysis

News for the last 7 days (`--days`) comes from Google News RSS and Yahoo Finance (no key needed),
plus Finnhub for US tickers when `FINNHUB_API_KEY` is set. Up to 30 de-duplicated headlines are sent
to Claude, which returns structured JSON: sentiment (−1…+1), confidence (0…1), a news-only
BUY/HOLD/SELL view, key events, catalysts and risks. Irrelevant pieces (listicles, other companies
with similar names) are told to be ignored.

- **News score** = 100 × sentiment × confidence
- **Combined score** = 60 % technical + 40 % news, same BUY/SELL thresholds (`core/fusion.py`)
- A warning is shown when technicals and news point in opposite directions.

Model: `claude-opus-5` at low effort by default (≈ $0.02–0.05 per analysis). Set `NEWS_MODEL` in
`.env` (or `--model`) to `claude-sonnet-5` or `claude-haiku-4-5` for cheaper runs. Results are cached
per article set in `data/cache/news_ai/`, so re-running on unchanged news is free. Opus requests
enable server-side refusal fallback (`fallbacks: "default"`).

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
  news.py        news collection (Finnhub, Google News RSS, Yahoo)
  ai_news.py     Claude news analysis -> validated NewsAnalysis (structured output + cache)
  fusion.py      technical + news -> combined signal
  analysis.py    analyze(ticker) — shared entry point for CLI / Streamlit / Telegram
  backtest.py    long-only backtest + signal hit rates
  chart.py       Plotly candlestick chart with patterns and signals
cli.py
tests/
```

## Roadmap

1. ✅ Core engine + backtester
2. ✅ Claude news analysis + combined signal
3. Streamlit dashboard
4. Telegram bot (`/analyze`, `/watch`)
5. ML model (LightGBM, walk-forward) + signal fusion tuning
