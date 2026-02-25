"""
NovelKBFeedSource — 将 novel-reader KB 的 chunk 摘要作为 FeedItem 暴露给 ProactiveLoop。

订阅格式（feeds.json）：
    {
        "type": "novel-kb",
        "name": "小说阅读进度",
        "url": "file:///home/huashen/test/kb/project"
    }

每次 fetch() 从 summaries/index.json 取最新 N 条 chunk，
将摘要文本包装成 FeedItem 返回。
去重由 ProactiveLoop 的 seen_items（14天）+ 语义去重统一负责。
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from feeds.base import FeedItem, FeedSource, FeedSubscription

logger = logging.getLogger(__name__)

_SUMMARY_MAX_CHARS = 400   # FeedItem.content 截断长度


class NovelKBFeedSource(FeedSource):
    """从 novel-reader KB 读取最新 chunk 摘要，作为 FeedItem 供 ProactiveLoop 使用。"""

    def __init__(self, sub: FeedSubscription) -> None:
        self._sub = sub
        self._kb_root = _resolve_kb_root(sub.url or "")

    @property
    def name(self) -> str:
        return self._sub.name

    @property
    def source_type(self) -> str:
        return "novel-kb"

    async def fetch(self, limit: int = 5) -> list[FeedItem]:
        if not self._kb_root:
            logger.warning("[novel-kb] 无效的 KB 路径: %r", self._sub.url)
            return []

        index_path = self._kb_root / "summaries" / "index.json"
        if not index_path.exists():
            logger.debug("[novel-kb] index.json 不存在: %s", index_path)
            return []

        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("[novel-kb] 读取 index.json 失败: %s", e)
            return []

        chunks = index.get("chunks", [])
        if not chunks:
            return []

        # 按 created_at 倒序，取最新 N 条
        recent = sorted(chunks, key=lambda c: c.get("created_at", ""), reverse=True)[:limit]

        items: list[FeedItem] = []
        kb_name = self._kb_root.name

        for rec in recent:
            chunk_id = rec.get("chunk_id", "")
            if not chunk_id:
                continue

            summary_rel = rec.get("summary_file", "")
            summary_path = (
                self._kb_root / summary_rel
                if summary_rel
                else self._kb_root / "summaries" / "chunks" / f"{chunk_id}.summary.md"
            )
            if not summary_path.exists():
                logger.debug("[novel-kb] summary 文件不存在: %s", summary_path)
                continue

            try:
                raw = summary_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.warning("[novel-kb] 读取 summary 失败 %s: %s", summary_path, e)
                continue

            content = _extract_body(raw, _SUMMARY_MAX_CHARS)
            if not content:
                continue

            segment = rec.get("segment") or rec.get("route", "")
            published_at = _parse_iso(rec.get("created_at"))

            # url 字段作为去重 key；格式固定，不依赖文件路径
            stable_url = f"novel://{kb_name}/{chunk_id}"

            items.append(FeedItem(
                source_name=self._sub.name,
                source_type="novel-kb",
                title=f"[{self._sub.name}·{segment}] {chunk_id}",
                content=content,
                url=stable_url,
                author=None,
                published_at=published_at,
            ))

        return items


# ── helpers ──────────────────────────────────────────────────────


def _resolve_kb_root(url: str) -> Path | None:
    """将 file:// URL 或裸路径解析为 Path，非法时返回 None。"""
    if not url:
        return None
    if url.startswith("file://"):
        path_str = url[len("file://"):]
    else:
        path_str = url
    p = Path(path_str)
    if not p.is_absolute():
        return None
    return p


def _extract_body(text: str, max_chars: int) -> str:
    """去掉 Markdown 标题行，提取正文并截断。"""
    lines = text.splitlines()
    body_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        # 跳过顶层标题（# ...）和元数据行（- key: value）
        if re.match(r"^#{1,2}\s", stripped):
            continue
        if re.match(r"^-\s+\w[\w\s]*:", stripped):
            continue
        body_lines.append(line)

    body = "\n".join(body_lines).strip()
    body = re.sub(r"\n{3,}", "\n\n", body)

    if len(body) <= max_chars:
        return body
    return body[:max_chars].rstrip() + "…"


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None
