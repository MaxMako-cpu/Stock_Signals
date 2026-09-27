"""Combine the technical score with the AI news score into one signal."""

from __future__ import annotations

from dataclasses import dataclass

from .scoring import to_signal

TECH_WEIGHT = 0.6
NEWS_WEIGHT = 0.4


@dataclass
class CombinedSignal:
    score: float
    signal: str
    technical_score: float
    news_score: float | None
    agreement: str        # "agree" | "conflict" | "technical only"


def _direction(signal: str) -> int:
    return {"BUY": 1, "SELL": -1}.get(signal, 0)


def combine(technical_score: float, news_score: float | None) -> CombinedSignal:
    if news_score is None:
        return CombinedSignal(technical_score, to_signal(technical_score), technical_score,
                              None, "technical only")
    score = TECH_WEIGHT * technical_score + NEWS_WEIGHT * news_score
    tech_dir = _direction(to_signal(technical_score))
    news_dir = 1 if news_score >= 15 else -1 if news_score <= -15 else 0
    agreement = "conflict" if tech_dir * news_dir < 0 else "agree"
    return CombinedSignal(score, to_signal(score), technical_score, news_score, agreement)
