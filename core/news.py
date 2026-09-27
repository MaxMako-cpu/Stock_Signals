"""News collection from free sources.

- Finnhub company news (needs FINNHUB_API_KEY; free tier, best for US tickers)
- Google News RSS (no key; covers European companies well)
- Yahoo Finance news via yfinance (no key)

Every source is best-effort: a failing source is reported, not raised.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape

import yfinance as yf

USER_AGENT = "Mozilla/5.0 (stock-signals research tool)"
TIMEOUT = 15
NAME_SUFFIXES = re.compile(
    r"[,.]?\s+(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|se|sa|ag|nv|n\.v|"
    r"ab|asa|oyj|spa|s\.p\.a|holdings?|group|class [a-z])\.?$",
    re.IGNORECASE,
)


@dataclass
class Article:
    title: str
    source: str
    published: datetime          # timezone-aware UTC
    url: str
    summary: str = ""
    provider: str = ""           # finnhub | google | yahoo


@dataclass
class NewsBundle:
    articles: list[Article]
    errors: dict[str, str]       # provider -> error message


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read()


def _clean_html(text: str) -> str:
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", text or "")).split())


def clean_company_name(name: str) -> str:
    prev = None
    name = " ".join(name.split())
    while prev != name:
        prev, name = name, NAME_SUFFIXES.sub("", name).strip()
    return name


# ---------- providers ----------

def fetch_finnhub(ticker: str, days: int, api_key: str) -> list[Article]:
    to = datetime.now(timezone.utc).date()
    frm = to - timedelta(days=days)
    q = urllib.parse.urlencode({"symbol": ticker, "from": frm, "to": to, "token": api_key})
    data = json.loads(_get(f"https://finnhub.io/api/v1/company-news?{q}"))
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data["error"])
    return [
        Article(
            title=item.get("headline", "").strip(),
            source=item.get("source", ""),
            published=datetime.fromtimestamp(item.get("datetime", 0), tz=timezone.utc),
            url=item.get("url", ""),
            summary=_clean_html(item.get("summary", "")),
            provider="finnhub",
        )
        for item in data
        if item.get("headline")
    ]


def parse_google_rss(xml_bytes: bytes) -> list[Article]:
    root = ET.fromstring(xml_bytes)
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        # Google appends " - Source" to titles
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "").astimezone(timezone.utc)
        except (TypeError, ValueError):
            continue
        out.append(Article(title=title, source=source, published=published,
                           url=item.findtext("link") or "", provider="google"))
    return out


def fetch_google(query: str, days: int) -> list[Article]:
    q = urllib.parse.urlencode({"q": f"{query} when:{days}d", "hl": "en-US", "gl": "US", "ceid": "US:en"})
    return parse_google_rss(_get(f"https://news.google.com/rss/search?{q}"))


def parse_yahoo_items(items: list[dict]) -> list[Article]:
    out = []
    for raw in items or []:
        c = raw.get("content", raw)  # yfinance >= 0.2.50 nests under "content"
        title = c.get("title") or ""
        if not title:
            continue
        if c.get("pubDate"):
            published = datetime.fromisoformat(c["pubDate"].replace("Z", "+00:00"))
        elif raw.get("providerPublishTime"):
            published = datetime.fromtimestamp(raw["providerPublishTime"], tz=timezone.utc)
        else:
            continue
        provider = c.get("provider")
        source = provider.get("displayName", "") if isinstance(provider, dict) else ""
        url = (c.get("canonicalUrl") or {}).get("url") or raw.get("link", "")
        out.append(Article(
            title=title.strip(),
            source=source or raw.get("publisher", ""),
            published=published.astimezone(timezone.utc),
            url=url,
            summary=_clean_html(c.get("summary", "")),
            provider="yahoo",
        ))
    return out


def fetch_yahoo(ticker: str) -> list[Article]:
    return parse_yahoo_items(yf.Ticker(ticker).news)


# ---------- aggregation ----------

def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())[:70]


def merge_articles(groups: list[list[Article]], days: int, limit: int) -> list[Article]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    seen: dict[str, Article] = {}
    for group in groups:
        for a in group:
            if a.published < cutoff or not a.title:
                continue
            key = _title_key(a.title)
            # Keep the copy with a summary if we see the same story twice
            if key not in seen or (not seen[key].summary and a.summary):
                seen[key] = a
    return sorted(seen.values(), key=lambda a: a.published, reverse=True)[:limit]


def collect_news(ticker: str, company_name: str = "", days: int = 7, limit: int = 30) -> NewsBundle:
    ticker = ticker.upper()
    groups: list[list[Article]] = []
    errors: dict[str, str] = {}

    finnhub_key = os.getenv("FINNHUB_API_KEY", "").strip()
    if finnhub_key and "." not in ticker:  # Finnhub free tier: US listings only
        try:
            groups.append(fetch_finnhub(ticker, days, finnhub_key))
        except Exception as e:
            errors["finnhub"] = str(e)

    name = clean_company_name(company_name) if company_name and company_name.upper() != ticker else ""
    query = f'"{name}" stock' if name else f"{ticker.split('.')[0]} stock"
    try:
        groups.append(fetch_google(query, days))
    except Exception as e:
        errors["google"] = str(e)

    try:
        groups.append(fetch_yahoo(ticker))
    except Exception as e:
        errors["yahoo"] = str(e)

    return NewsBundle(merge_articles(groups, days, limit), errors)
