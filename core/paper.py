"""Virtual (paper) trading in EUR — no real money, just the maths.

You "buy" a stock with an amount in EUR at the current price. The EUR is
converted to the stock's currency at the current FX rate (like Revolut
does), minus a fee. Later the position is valued at the current price and
FX rate, minus the exit fee, so P&L includes both stock and currency moves.

Trades are stored in data/paper_trades.json.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from .data import fetch_ohlcv

BOOK_PATH = Path(__file__).resolve().parent.parent / "data" / "paper_trades.json"
BASE = "EUR"
DEFAULT_FEE_PCT = 0.25


# ---------- prices & FX ----------

def normalize_currency(price: float, currency: str) -> tuple[float, str]:
    """London quotes in pence (GBp/GBX) -> pounds."""
    if currency in ("GBp", "GBX"):
        return price / 100, "GBP"
    return price, (currency or BASE).upper()


def fx_to_eur(currency: str) -> float:
    """EUR per 1 unit of `currency` (latest)."""
    currency = currency.upper()
    if currency == BASE:
        return 1.0
    hist = yf.Ticker(f"{currency}{BASE}=X").history(period="5d")
    if hist.empty:
        raise RuntimeError(f"No FX rate for {currency}->EUR")
    return float(hist["Close"].iloc[-1])


def fx_history_to_eur(currency: str, period: str = "10y") -> pd.Series:
    """Daily EUR per 1 unit of `currency`, tz-naive dates."""
    currency = currency.upper()
    if currency == BASE:
        return pd.Series(dtype=float)
    hist = yf.Ticker(f"{currency}{BASE}=X").history(period=period)
    s = hist["Close"]
    s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
    return s


def get_quote(ticker: str) -> tuple[float, str, str]:
    """(price, currency, 'live'|'last close') — best effort live price."""
    t = yf.Ticker(ticker)
    try:
        fi = t.fast_info
        price, ccy = fi.get("lastPrice"), fi.get("currency") or ""
        if price and price > 0:
            return float(price), ccy, "latest price"
    except Exception:
        pass
    df = fetch_ohlcv(ticker)
    return float(df["Close"].iloc[-1]), "", "last close"


# ---------- trade model ----------

@dataclass
class PaperTrade:
    ticker: str
    name: str
    currency: str            # normalized (GBP, not GBp)
    amount_eur: float        # what you "paid", fees included
    entry_price: float       # in `currency`
    fx_entry: float          # EUR per 1 unit of currency at entry
    fee_pct: float
    shares: float
    signal: str = ""
    score: float = 0.0
    note: str = ""
    opened_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status: str = "open"     # open | closed
    closed_at: str | None = None
    exit_price: float | None = None
    fx_exit: float | None = None


@dataclass
class Valuation:
    value_eur: float         # net of exit fee
    pnl_eur: float
    pnl_pct: float
    price_change_pct: float  # stock move in its own currency
    fx_change_pct: float     # currency move vs EUR
    fees_eur: float          # entry + exit


def open_trade(ticker: str, name: str, amount_eur: float, price: float, currency: str,
               fx: float, fee_pct: float = DEFAULT_FEE_PCT, signal: str = "", score: float = 0.0,
               note: str = "") -> PaperTrade:
    if amount_eur <= 0:
        raise ValueError("Amount must be positive.")
    price, currency = normalize_currency(price, currency)
    invested_eur = amount_eur * (1 - fee_pct / 100)
    shares = invested_eur / (price * fx)
    return PaperTrade(ticker=ticker.upper(), name=name, currency=currency, amount_eur=amount_eur,
                      entry_price=price, fx_entry=fx, fee_pct=fee_pct, shares=shares,
                      signal=signal, score=score, note=note)


def value_trade(trade: PaperTrade, price: float, currency: str, fx: float) -> Valuation:
    price, _ = normalize_currency(price, currency)
    gross_eur = trade.shares * price * fx
    exit_fee = gross_eur * trade.fee_pct / 100
    value = gross_eur - exit_fee
    pnl = value - trade.amount_eur
    return Valuation(
        value_eur=value,
        pnl_eur=pnl,
        pnl_pct=100 * pnl / trade.amount_eur,
        price_change_pct=100 * (price / trade.entry_price - 1),
        fx_change_pct=100 * (fx / trade.fx_entry - 1),
        fees_eur=trade.amount_eur * trade.fee_pct / 100 + exit_fee,
    )


# ---------- storage ----------

class PaperBook:
    def __init__(self, path: Path | None = None):
        self.path = Path(path or BOOK_PATH)

    def load(self) -> list[PaperTrade]:
        if not self.path.exists():
            return []
        return [PaperTrade(**d) for d in json.loads(self.path.read_text(encoding="utf-8"))]

    def _save(self, trades: list[PaperTrade]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps([asdict(t) for t in trades], indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def add(self, trade: PaperTrade) -> None:
        self._save(self.load() + [trade])

    def close(self, trade_id: str, price: float, currency: str, fx: float) -> PaperTrade:
        trades = self.load()
        for t in trades:
            if t.id == trade_id and t.status == "open":
                t.exit_price, _ = normalize_currency(price, currency)
                t.fx_exit = fx
                t.status = "closed"
                t.closed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                self._save(trades)
                return t
        raise KeyError(f"No open trade with id {trade_id}")

    def delete(self, trade_id: str) -> None:
        self._save([t for t in self.load() if t.id != trade_id])


def realized(trade: PaperTrade) -> Valuation:
    """Valuation of a closed trade at its exit price/FX."""
    return value_trade(trade, trade.exit_price, trade.currency, trade.fx_exit)


# ---------- what-if on history ----------

def whatif_last_buy(frame: pd.DataFrame, signals: pd.Series, amount_eur: float, currency: str,
                    fx_hist: pd.Series | None = None, fee_pct: float = DEFAULT_FEE_PCT) -> dict | None:
    """If you had bought `amount_eur` at the open after the most recent BUY flip
    and held until the latest close. Returns None if there was no BUY flip."""
    active = signals[signals.isin(["BUY", "SELL"])]
    flips = active[(active == "BUY") & (active.shift() != "BUY")]
    if flips.empty:
        return None
    signal_day = flips.index[-1]
    pos = frame.index.get_loc(signal_day)
    if pos + 1 >= len(frame):
        return None  # signal fired on the last bar — nothing to measure yet
    entry_day = frame.index[pos + 1]
    price_in, ccy = normalize_currency(float(frame["Open"].iloc[pos + 1]), currency)
    price_now, _ = normalize_currency(float(frame["Close"].iloc[-1]), currency)

    def fx_at(day):
        if ccy == BASE or fx_hist is None or fx_hist.empty:
            return 1.0
        return float(fx_hist.asof(day))

    trade = open_trade("X", "", amount_eur, price_in, ccy, fx_at(entry_day), fee_pct)
    v = value_trade(trade, price_now, ccy, fx_at(frame.index[-1]))
    later = active.loc[signal_day:]
    sold = later[later == "SELL"]
    return {
        "signal_day": signal_day, "entry_day": entry_day, "entry_price": price_in,
        "price_now": price_now, "currency": ccy, "days_held": (frame.index[-1] - entry_day).days,
        "sell_since": sold.index[0] if not sold.empty else None,
        "valuation": v, "fx_ignored": ccy != BASE and (fx_hist is None or fx_hist.empty),
    }
