"""Claude-powered news analysis with structured JSON output.

The model reads recent headlines for one company and returns a validated
`NewsAnalysis`: sentiment, confidence, a news-only BUY/HOLD/SELL view,
key events, risks and catalysts. Results are cached on disk keyed by the
exact article set, so re-running on unchanged news costs nothing.

Configuration (.env):
    ANTHROPIC_API_KEY   required
    NEWS_MODEL          default claude-opus-5 (cheaper: claude-sonnet-5, claude-haiku-4-5)
    NEWS_EFFORT         default low (low | medium | high)
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, ValidationError

from .news import Article

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "low"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "news_ai"
SUMMARY_CHARS = 500   # per-article summary length sent to the model

# Server-side refusal fallback (routes a declined request to another model)
FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5", "claude-fable-5-1"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Haiku 4.5 rejects the effort parameter
NO_EFFORT_MODELS = {"claude-haiku-4-5"}

# USD per million tokens (input, output) — for the cost line in the CLI
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


class NewsAIError(RuntimeError):
    pass


class KeyEvent(BaseModel):
    headline: str
    category: Literal["earnings", "guidance", "analyst", "product", "legal_regulatory",
                      "m_and_a", "management", "macro_sector", "capital", "other"]
    impact: Literal["positive", "negative", "neutral"]
    importance: Literal["high", "medium", "low"]


class NewsAnalysis(BaseModel):
    sentiment: float            # -1 (very bearish) .. +1 (very bullish)
    confidence: float           # 0 .. 1 — how much the news actually says
    signal: Literal["BUY", "HOLD", "SELL"]
    horizon: Literal["days", "weeks", "months"]
    summary: str
    key_events: list[KeyEvent]
    catalysts: list[str]
    risks: list[str]
    relevant_articles: int


@dataclass
class NewsAIResult:
    analysis: NewsAnalysis
    model: str
    articles_used: int
    cached: bool
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None

    @property
    def score(self) -> float:
        """News score in [-100, +100], scaled by confidence."""
        return 100 * self.analysis.sentiment * self.analysis.confidence


_EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "category": {"type": "string", "enum": list(KeyEvent.model_fields["category"].annotation.__args__)},
        "impact": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "importance": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["headline", "category", "impact", "importance"],
    "additionalProperties": False,
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "number"},
        "confidence": {"type": "number"},
        "signal": {"type": "string", "enum": ["BUY", "HOLD", "SELL"]},
        "horizon": {"type": "string", "enum": ["days", "weeks", "months"]},
        "summary": {"type": "string"},
        "key_events": {"type": "array", "items": _EVENT_SCHEMA},
        "catalysts": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "relevant_articles": {"type": "integer"},
    },
    "required": ["sentiment", "confidence", "signal", "horizon", "summary",
                 "key_events", "catalysts", "risks", "relevant_articles"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are an equity news analyst. You receive recent news headlines (and short summaries where \
available) about one listed company, and you judge what the news flow implies for its share \
price over the coming days to weeks.

How to judge:
- Separate company-specific news (earnings, guidance, contracts, products, lawsuits, \
regulation, management changes, M&A, buybacks, dilution, analyst rating/target changes) from \
generic market commentary, listicles, "stocks to watch" pieces and promotional content. Only \
company-specific, material items should move your view; count them in relevant_articles.
- Weigh recency and source quality. Several outlets reporting the same event is one event.
- Ask what is new versus already priced in: a beat that was widely expected, or a stock that \
already moved sharply on the news, deserves less weight.
- Articles may be about other companies that share a name or ticker fragment; ignore them.
- If the news is thin, stale or mixed, say so: keep sentiment near 0, confidence low, signal HOLD.

Fields:
- sentiment: number from -1.0 (clearly bearish) to +1.0 (clearly bullish).
- confidence: number from 0.0 to 1.0 — how strongly the available news supports your view \
(few or irrelevant articles means low confidence).
- signal: BUY, HOLD or SELL based on the news alone. Use BUY/SELL only for clear, material, \
company-specific news; otherwise HOLD.
- horizon: over which timeframe the news most likely matters.
- summary: 2-4 plain-English sentences a retail investor can read quickly.
- key_events: up to 6 most important distinct events, most important first.
- catalysts / risks: up to 4 short phrases each (upcoming events or ongoing factors).

This is research support for a human who makes their own decisions; be calibrated, not \
promotional."""


def _format_articles(articles: list[Article]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(f"[{i}] {a.published:%Y-%m-%d %H:%M} UTC | {a.source or 'unknown'} | {a.title}")
        if a.summary:
            s = a.summary if len(a.summary) <= SUMMARY_CHARS else a.summary[:SUMMARY_CHARS] + "…"
            lines.append(f"    {s}")
    return "\n".join(lines)


def build_user_message(ticker: str, company: str, articles: list[Article]) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return (f"Company: {company} (ticker {ticker})\n"
            f"Today: {today}\n"
            f"Articles ({len(articles)}, newest first):\n\n{_format_articles(articles)}")


def _cache_key(model: str, ticker: str, articles: list[Article]) -> str:
    h = hashlib.sha256()
    h.update(f"{model}|{ticker}|{SYSTEM_PROMPT}".encode())
    for a in articles:
        h.update(f"|{a.published.isoformat()}|{a.title}".encode())
    return h.hexdigest()[:16]


_LITERAL_UNICODE = re.compile(r"\\u([0-9a-fA-F]{4})")


def _unescape_strings(analysis: NewsAnalysis) -> NewsAnalysis:
    """The model occasionally double-escapes characters (a literal "\\u2014"); decode them."""
    def fix(v):
        if isinstance(v, str):
            return _LITERAL_UNICODE.sub(lambda m: chr(int(m.group(1), 16)), v)
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        return v
    return NewsAnalysis.model_validate(fix(analysis.model_dump()))


def _estimate_cost(model: str, usage) -> float | None:
    price = PRICES.get(model)
    if not price:
        return None
    pin, pout = price
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return (usage.input_tokens * pin + cache_write * pin * 1.25 + cache_read * pin * 0.1
            + usage.output_tokens * pout) / 1_000_000


def analyze_news(
    ticker: str,
    company: str,
    articles: list[Article],
    model: str | None = None,
    effort: str | None = None,
    use_cache: bool = True,
    client: anthropic.Anthropic | None = None,
) -> NewsAIResult:
    if not articles:
        raise NewsAIError("No articles to analyze.")
    model = model or os.getenv("NEWS_MODEL", "").strip() or DEFAULT_MODEL
    effort = effort or os.getenv("NEWS_EFFORT", "").strip() or DEFAULT_EFFORT
    ticker = ticker.upper()

    cache_path = CACHE_DIR / f"{ticker.replace('.', '_')}_{_cache_key(model, ticker, articles)}.json"
    if use_cache and cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return NewsAIResult(NewsAnalysis.model_validate(data["analysis"]), data["model"],
                            len(articles), cached=True)

    if client is None:
        if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("ANTHROPIC_AUTH_TOKEN"):
            raise NewsAIError("ANTHROPIC_API_KEY is not set — add it to .env (see .env.example).")
        client = anthropic.Anthropic()

    output_config: dict = {"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}}
    if model not in NO_EFFORT_MODELS:
        output_config["effort"] = effort
    request = dict(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_message(ticker, company, articles)}],
        output_config=output_config,
        cache_control={"type": "ephemeral"},
    )

    try:
        if model in FALLBACK_MODELS:
            response = client.beta.messages.create(**request, betas=[FALLBACK_BETA], fallbacks="default")
        else:
            response = client.messages.create(**request)
    except anthropic.AuthenticationError:
        raise NewsAIError("Anthropic API key was rejected — check ANTHROPIC_API_KEY in .env.")
    except anthropic.RateLimitError:
        raise NewsAIError("Anthropic rate limit hit — wait a minute and retry.")
    except anthropic.NotFoundError:
        raise NewsAIError(f"Model '{model}' not found — check NEWS_MODEL in .env.")
    except anthropic.APIStatusError as e:
        raise NewsAIError(f"Anthropic API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        raise NewsAIError("Could not reach the Anthropic API — check your internet connection.")

    if response.stop_reason == "refusal":
        raise NewsAIError("The model declined to analyze these articles.")
    if response.stop_reason == "max_tokens":
        raise NewsAIError("The model's answer was cut off (max_tokens).")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        raise NewsAIError("The model returned no text.")
    try:
        analysis = NewsAnalysis.model_validate_json(text)
    except ValidationError as e:
        raise NewsAIError(f"Model output did not match the schema: {e}")

    analysis = _unescape_strings(analysis)
    analysis.sentiment = max(-1.0, min(1.0, analysis.sentiment))
    analysis.confidence = max(0.0, min(1.0, analysis.confidence))
    served_by = getattr(response, "model", model) or model

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "model": served_by,
        "created": datetime.now(timezone.utc).isoformat(),
        "request_id": getattr(response, "_request_id", None),
        "analysis": analysis.model_dump(),
    }, indent=2), encoding="utf-8")

    u = response.usage
    return NewsAIResult(
        analysis=analysis,
        model=served_by,
        articles_used=len(articles),
        cached=False,
        input_tokens=u.input_tokens,
        output_tokens=u.output_tokens,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        cost_usd=_estimate_cost(served_by, u),
    )
