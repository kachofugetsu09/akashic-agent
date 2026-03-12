"""
proactive/interest.py — 基于 memory 的兴趣筛选。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from feeds.base import FeedItem

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9+.#-]{1,}|[\u4e00-\u9fff]{2,}")

_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "your",
    "have",
    "will",
    "you",
    "are",
    "not",
    "was",
    "were",
    "been",
    "about",
    "into",
    "over",
    "after",
    "then",
    "than",
    "but",
    "can",
    "could",
    "would",
    "should",
    "我们",
    "你们",
    "他们",
    "这个",
    "那个",
    "以及",
    "然后",
    "就是",
    "还是",
    "如果",
    "因为",
    "所以",
    "已经",
    "一些",
    "一个",
    "不是",
    "没有",
    "可以",
    "需要",
    "进行",
    "相关",
    "内容",
    "消息",
    "信息",
}


@dataclass
class InterestFilterConfig:
    enabled: bool = False
    memory_max_chars: int = 4000
    keyword_max_count: int = 80
    min_token_len: int = 2
    min_score: float = 0.14
    top_k: int = 10
    exploration_ratio: float = 0.20


def _tokenize(text: str, min_len: int = 2) -> list[str]:
    if not text:
        return []
    words = [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]
    return [w for w in words if len(w) >= min_len and w not in _STOPWORDS]


def _build_keyword_weights(
    memory_text: str, cfg: InterestFilterConfig
) -> dict[str, float]:
    tokens = _tokenize(
        memory_text[: max(cfg.memory_max_chars, 0)], min_len=cfg.min_token_len
    )
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    # 频次压缩，避免单个词权重过高
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[
        : max(cfg.keyword_max_count, 1)
    ]
    return {k: 1.0 + math.log1p(v) for k, v in ranked}


def _item_text(item: FeedItem) -> str:
    parts = [
        item.title or "",
        item.content or "",
        item.url or "",
        item.author or "",
        item.source_name or "",
    ]
    return "\n".join(parts)


def score_items_by_memory(
    items: Iterable[FeedItem],
    memory_text: str,
    cfg: InterestFilterConfig,
) -> list[tuple[FeedItem, float]]:
    weights = _build_keyword_weights(memory_text or "", cfg)
    if not weights:
        return [(item, 0.0) for item in items]
    results: list[tuple[FeedItem, float]] = []
    for item in items:
        tokens = set(_tokenize(_item_text(item), min_len=cfg.min_token_len))
        if not tokens:
            results.append((item, 0.0))
            continue
        matched_weight = sum(weights.get(t, 0.0) for t in tokens)
        norm = max(3.0, len(tokens) ** 0.6)
        score = matched_weight / norm
        results.append((item, score))
    return results


def select_interesting_items(
    items: list[FeedItem],
    memory_text: str,
    cfg: InterestFilterConfig,
) -> tuple[list[FeedItem], list[tuple[FeedItem, float]]]:
    """返回筛选后的条目及完整打分列表（按分数降序）。"""
    scored = score_items_by_memory(items, memory_text, cfg)
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    if not ranked:
        return [], []

    above = [pair for pair in ranked if pair[1] >= cfg.min_score]
    keep_target = min(max(cfg.top_k, 1), len(ranked))
    explore_n = int(round(keep_target * max(0.0, min(1.0, cfg.exploration_ratio))))
    if len(above) >= keep_target:
        kept = above[:keep_target]
    else:
        needed = keep_target - len(above)
        tail = [pair for pair in ranked if pair not in above]
        explore = tail[: max(explore_n, needed)]
        kept = (above + explore)[:keep_target]
    return [item for item, _ in kept], ranked
