"""
proactive/item_id.py — FeedItem 标识计算公共模块。

统一 item_id / source_key 的计算逻辑，供 proactive 主链路与兼容层共用，
避免各自维护私有副本。
"""

from __future__ import annotations

import hashlib
from urllib.parse import urlsplit, urlunsplit

from feeds.base import FeedItem


def normalize_url(url: str | None) -> str:
    """标准化 URL：小写 scheme/host、去掉尾部 /、保留 query。"""
    if not url:
        return ""
    try:
        p = urlsplit(url.strip())
        scheme = (p.scheme or "").lower()
        netloc = (p.netloc or "").lower()
        path = p.path.rstrip("/")
        return urlunsplit((scheme, netloc, path, p.query, ""))
    except Exception:
        return (url or "").strip()


def compute_item_id(item: FeedItem) -> str:
    """计算 FeedItem 的唯一 ID。

    优先用 URL hash（u_ 前缀）；无 URL 则用内容指纹（h_ 前缀）。
    与 loop.py 旧版 _item_id() 保持完全相同的逻辑，便于状态文件兼容。
    """
    url = normalize_url(item.url)
    if url:
        return "u_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    raw = "|".join(
        [
            (item.source_type or "").strip().lower(),
            (item.source_name or "").strip().lower(),
            (item.title or "").strip().lower(),
            (item.content or "").strip().lower()[:200],
            item.published_at.isoformat() if item.published_at else "",
        ]
    )
    return "h_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def compute_source_key(item: FeedItem) -> str:
    """计算 FeedItem 的 source 标识符，格式：'type:name'（均小写）。"""
    return f"{(item.source_type or '').strip().lower()}:{(item.source_name or '').strip().lower()}"
