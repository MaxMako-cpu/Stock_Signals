"""Offline tests for virtual trading maths."""

import numpy as np
import pandas as pd
import pytest

from core.paper import GistStore, PaperBook, open_trade, realized, value_trade, whatif_last_buy


def test_eur_stock_no_fee():
    t = open_trade("SAP.DE", "SAP", 1000, price=200, currency="EUR", fx=1.0, fee_pct=0)
    assert t.shares == pytest.approx(5)
    v = value_trade(t, price=220, currency="EUR", fx=1.0)
    assert v.value_eur == pytest.approx(1100)
    assert v.pnl_eur == pytest.approx(100) and v.pnl_pct == pytest.approx(10)
    assert v.fx_change_pct == 0


def test_usd_stock_with_fx_and_fees():
    # 1000 EUR, 0.25% fee -> 997.50 EUR invested; 1 USD = 0.90 EUR -> 1108.33 USD -> 11.0833 shares @100
    t = open_trade("AAPL", "Apple", 1000, price=100, currency="USD", fx=0.90, fee_pct=0.25)
    assert t.shares == pytest.approx(997.5 / 90)
    # Stock +10%, but USD weakens 5% vs EUR
    v = value_trade(t, price=110, currency="USD", fx=0.855)
    gross = t.shares * 110 * 0.855
    assert v.value_eur == pytest.approx(gross * 0.9975)
    assert v.price_change_pct == pytest.approx(10)
    assert v.fx_change_pct == pytest.approx(-5)
    assert v.pnl_eur == pytest.approx(gross * 0.9975 - 1000)
    assert v.pnl_eur < 100 - 40  # currency drag + fees eat into the 10% gain
    assert v.fees_eur == pytest.approx(2.5 + gross * 0.0025)


def test_london_pence_are_converted():
    t = open_trade("SHEL.L", "Shell", 1000, price=2500, currency="GBp", fx=1.2, fee_pct=0)
    assert t.currency == "GBP" and t.entry_price == 25
    assert t.shares == pytest.approx(1000 / (25 * 1.2))
    assert value_trade(t, 2500, "GBp", 1.2).pnl_eur == pytest.approx(0)


def test_invalid_amount():
    with pytest.raises(ValueError):
        open_trade("AAPL", "Apple", 0, 100, "USD", 0.9)


def test_book_add_close_delete(tmp_path):
    book = PaperBook(tmp_path / "trades.json")
    t = open_trade("AAPL", "Apple", 500, 100, "USD", 0.9, fee_pct=0)
    book.add(t)
    assert [x.id for x in book.load()] == [t.id]
    closed = book.close(t.id, price=120, currency="USD", fx=0.9)
    assert closed.status == "closed"
    assert realized(book.load()[0]).pnl_eur == pytest.approx(100)
    with pytest.raises(KeyError):
        book.close(t.id, 1, "USD", 1)       # already closed
    book.delete(t.id)
    assert book.load() == []


def _frame(opens, closes):
    idx = pd.bdate_range("2026-01-05", periods=len(opens))
    return pd.DataFrame({"Open": opens, "Close": closes}, index=idx)


def test_whatif_last_buy_uses_next_open_and_latest_close():
    frame = _frame([10, 10, 20, 20, 20, 30], [10, 10, 20, 20, 20, 25])
    sig = pd.Series(["HOLD", "BUY", "HOLD", "BUY", "HOLD", "HOLD"], index=frame.index)
    # Only one BUY *flip* (bar 1; bar 3 repeats BUY) -> enter at bar 2 open = 20, now 25
    w = whatif_last_buy(frame, sig, 1000, "EUR", fee_pct=0)
    assert w["entry_day"] == frame.index[2] and w["entry_price"] == 20
    assert w["valuation"].value_eur == pytest.approx(1250)
    assert w["sell_since"] is None


def test_whatif_with_fx_history_and_no_buy():
    frame = _frame([100, 100, 100], [100, 100, 110])
    sig = pd.Series(["BUY", "HOLD", "HOLD"], index=frame.index)
    fx = pd.Series([0.9, 0.9, 0.99], index=frame.index)
    w = whatif_last_buy(frame, sig, 900, "USD", fx_hist=fx, fee_pct=0)
    # 900 EUR -> 1000 USD -> 10 sh; now 10*110 USD * 0.99 = 1089 EUR
    assert w["valuation"].value_eur == pytest.approx(1089)
    assert whatif_last_buy(frame, pd.Series(["HOLD"] * 3, index=frame.index), 900, "USD") is None


# ---------- storage backends ----------

class FakeGist(GistStore):
    """GistStore with the HTTP layer replaced by an in-memory dict."""
    def __init__(self):
        super().__init__("token", "abc123")
        self.files, self.calls = {}, []

    def _request(self, method, url, body=None):
        self.calls.append(method)
        if method == "PATCH":
            for name, f in body["files"].items():
                self.files[name] = f["content"]
        return {"files": {n: {"content": c, "truncated": False} for n, c in self.files.items()}}


def test_gist_store_roundtrip():
    store = FakeGist()
    book = PaperBook(store=store)
    assert book.load() == []
    t = open_trade("AAPL", "Apple", 1000, 100, "USD", 0.9, fee_pct=0)
    book.add(t)
    assert [x.id for x in PaperBook(store=store).load()] == [t.id]
    assert store.calls.count("PATCH") == 1
    assert "gist" in book.describe().lower()


def test_default_store_picks_gist_only_with_both_env_vars(monkeypatch):
    from core import paper
    monkeypatch.delenv("PAPER_GIST_ID", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    assert isinstance(paper.default_store(), paper.FileStore)
    monkeypatch.setenv("PAPER_GIST_ID", "g")
    assert isinstance(paper.default_store(), paper.GistStore)
