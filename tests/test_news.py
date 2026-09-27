"""Offline tests for news collection, Claude news analysis (fake client) and fusion."""

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from core import ai_news
from core.ai_news import NewsAIError, analyze_news, build_user_message
from core.fusion import combine
from core.news import Article, clean_company_name, merge_articles, parse_google_rss, parse_yahoo_items

NOW = datetime.now(timezone.utc)


def art(title, hours_ago=1, summary="", source="Reuters"):
    return Article(title=title, source=source, published=NOW - timedelta(hours=hours_ago),
                   url="https://x", summary=summary)


# ---------- news parsing ----------

GOOGLE_RSS = f"""<?xml version="1.0"?><rss><channel>
<item><title>Apple beats earnings estimates - Reuters</title><link>https://a</link>
<pubDate>{(NOW - timedelta(hours=3)).strftime('%a, %d %b %Y %H:%M:%S GMT')}</pubDate>
<source url="https://reuters.com">Reuters</source></item>
<item><title>Broken date item</title><pubDate>not a date</pubDate></item>
</channel></rss>""".encode()


def test_parse_google_rss_strips_source_suffix():
    items = parse_google_rss(GOOGLE_RSS)
    assert len(items) == 1
    assert items[0].title == "Apple beats earnings estimates"
    assert items[0].source == "Reuters"
    assert items[0].published.tzinfo is not None


def test_parse_yahoo_both_shapes():
    new_shape = [{"content": {"title": "SAP raises guidance", "pubDate": "2026-09-25T08:00:00Z",
                              "summary": "<b>Strong</b> cloud", "provider": {"displayName": "Yahoo"},
                              "canonicalUrl": {"url": "https://y"}}}]
    old_shape = [{"title": "Old style", "providerPublishTime": 1_758_000_000,
                  "publisher": "MarketWatch", "link": "https://m"}]
    a, = parse_yahoo_items(new_shape)
    b, = parse_yahoo_items(old_shape)
    assert a.summary == "Strong cloud" and a.source == "Yahoo"
    assert b.source == "MarketWatch" and b.url == "https://m"


def test_merge_dedupes_filters_and_sorts():
    old = art("Ancient story", hours_ago=24 * 30)
    dup_plain = art("Nvidia unveils new chip!", hours_ago=5)
    dup_rich = art("NVIDIA unveils new chip", hours_ago=4, summary="details")
    newest = art("Newest", hours_ago=1)
    out = merge_articles([[old, dup_plain], [dup_rich, newest]], days=7, limit=10)
    assert [a.title for a in out] == ["Newest", "NVIDIA unveils new chip"]
    assert out[1].summary == "details"


@pytest.mark.parametrize("raw,clean", [
    ("Apple Inc.", "Apple"),
    ("SAP SE", "SAP"),
    ("ASML Holding N.V.", "ASML"),
    ("Shell plc", "Shell"),
    ("Microsoft Corporation", "Microsoft"),
])
def test_clean_company_name(raw, clean):
    assert clean_company_name(raw) == clean


# ---------- Claude news analysis (fake client) ----------

VALID = {
    "sentiment": 0.6, "confidence": 0.8, "signal": "BUY", "horizon": "weeks",
    "summary": "Strong quarter.", "relevant_articles": 2,
    "key_events": [{"headline": "Beat estimates", "category": "earnings",
                    "impact": "positive", "importance": "high"}],
    "catalysts": ["Product launch"], "risks": ["Valuation"],
}


class FakeMessages:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason, self.calls = text, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        content = [SimpleNamespace(type="text", text=self.text)] if self.text is not None else []
        return SimpleNamespace(
            stop_reason=self.stop_reason, content=content, model=kwargs["model"],
            usage=SimpleNamespace(input_tokens=1000, output_tokens=200,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0))


def fake_client(text=json.dumps(VALID), stop_reason="end_turn"):
    msgs = FakeMessages(text, stop_reason)
    return SimpleNamespace(messages=msgs, beta=SimpleNamespace(messages=msgs)), msgs


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(ai_news, "CACHE_DIR", tmp_path / "news_ai")


def test_analyze_news_parses_and_scores():
    client, msgs = fake_client()
    res = analyze_news("AAPL", "Apple", [art("Apple beats"), art("iPhone demand")],
                       model="claude-opus-5", client=client)
    assert res.analysis.signal == "BUY"
    assert res.score == pytest.approx(48.0)          # 100 * 0.6 * 0.8
    assert res.cost_usd == pytest.approx((1000 * 5 + 200 * 25) / 1e6)
    call = msgs.calls[0]
    assert call["fallbacks"] == "default" and call["betas"] == [ai_news.FALLBACK_BETA]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["effort"] == "low"


def test_haiku_gets_no_effort_and_no_fallback():
    client, msgs = fake_client()
    analyze_news("AAPL", "Apple", [art("x")], model="claude-haiku-4-5", client=client)
    call = msgs.calls[0]
    assert "effort" not in call["output_config"]
    assert "fallbacks" not in call


def test_result_is_cached_for_same_articles():
    arts = [art("Apple beats")]
    client, msgs = fake_client()
    first = analyze_news("AAPL", "Apple", arts, model="claude-opus-5", client=client)
    second = analyze_news("AAPL", "Apple", arts, model="claude-opus-5", client=client)
    assert not first.cached and second.cached
    assert len(msgs.calls) == 1
    assert second.analysis == first.analysis


def test_values_are_clamped():
    client, _ = fake_client(json.dumps({**VALID, "sentiment": 3.0, "confidence": -1}))
    res = analyze_news("AAPL", "Apple", [art("x")], client=client)
    assert res.analysis.sentiment == 1.0 and res.analysis.confidence == 0.0


def test_double_escaped_unicode_is_decoded():
    raw = {**VALID, "summary": "Solid \\u2014 strong", "risks": ["EUR\\u00a0risk"]}
    client, _ = fake_client(json.dumps(raw))
    res = analyze_news("AAPL", "Apple", [art("x")], client=client)
    assert res.analysis.summary == "Solid — strong"
    assert res.analysis.risks == ["EUR risk"]


@pytest.mark.parametrize("text,stop,msg", [
    (None, "refusal", "declined"),
    ('{"sentiment": 0.1', "max_tokens", "cut off"),
    ('{"sentiment": "high"}', "end_turn", "schema"),
])
def test_bad_responses_raise(text, stop, msg):
    client, _ = fake_client(text, stop)
    with pytest.raises(NewsAIError, match=msg):
        analyze_news("AAPL", "Apple", [art("x")], client=client)


def test_missing_key_and_empty_articles(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    with pytest.raises(NewsAIError, match="ANTHROPIC_API_KEY"):
        analyze_news("AAPL", "Apple", [art("x")])
    with pytest.raises(NewsAIError, match="No articles"):
        analyze_news("AAPL", "Apple", [])


def test_user_message_contains_articles():
    msg = build_user_message("SAP.DE", "SAP", [art("Cloud growth", summary="s" * 900)])
    assert "SAP.DE" in msg and "Cloud growth" in msg
    assert "s" * 500 + "…" in msg and "s" * 501 not in msg


# ---------- fusion ----------

def test_fusion_weights_and_conflict():
    c = combine(50, 50)
    assert c.score == pytest.approx(50) and c.signal == "BUY" and c.agreement == "agree"
    c = combine(40, -60)                              # tech BUY, news clearly bearish
    assert c.score == pytest.approx(0) and c.signal == "HOLD" and c.agreement == "conflict"
    c = combine(-20, None)
    assert c.signal == "HOLD" and c.agreement == "technical only"
