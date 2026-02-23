"""
RSS/Atom 信息源。支持 RSS 2.0 和 Atom 1.0 格式。
使用 httpx（已有依赖）+ 标准库 xml.etree.ElementTree。
"""
from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from feeds.base import FeedItem, FeedSource, FeedSubscription

logger = logging.getLogger(__name__)

_ATOM_NS = "http://www.w3.org/2005/Atom"
_TIMEOUT = 15.0
_MAX_CONTENT = 300   # 正文截断字数


class RSSFeedSource(FeedSource):
    """RSS 2.0 / Atom 1.0 信息源。"""

    def __init__(self, sub: FeedSubscription) -> None:
        self._sub = sub

    @property
    def name(self) -> str:
        return self._sub.name

    @property
    def source_type(self) -> str:
        return "rss"

    async def fetch(self, limit: int = 5) -> list[FeedItem]:
        if not self._sub.url:
            return []
        if self._sub.url.startswith("file://"):
            parsed = urlparse(self._sub.url)
            local_path = unquote(parsed.path or "")
            try:
                text = Path(local_path).read_text(encoding="utf-8")
                return self._parse(text, limit)
            except Exception as e:
                logger.warning(f"RSS local file read error [{self._sub.name}] path={local_path!r}: {e}")
                return []
        async with httpx.AsyncClient(
            timeout=_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": "Akasic-Agent/1.0 (feed reader)"},
        ) as client:
            resp = await client.get(self._sub.url)
            resp.raise_for_status()
            return self._parse(resp.text, limit)

    def _parse(self, xml_text: str, limit: int) -> list[FeedItem]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.warning(f"RSS parse error [{self._sub.name}]: {e}")
            return []

        tag = root.tag.lower()
        if "feed" in tag:
            return self._parse_atom(root, limit)
        channel = root.find("channel")
        if channel is not None:
            return self._parse_rss(channel, limit)
        logger.warning(f"未知 XML 格式 root={root.tag!r} [{self._sub.name}]")
        return []

    def _parse_rss(self, channel: ET.Element, limit: int) -> list[FeedItem]:
        items = []
        for item in channel.findall("item")[:limit]:
            title = _text(item, "title")
            link = _text(item, "link")
            desc = _text(item, "description") or ""
            author = _text(item, "author") or _text(item, "dc:creator")
            pub_date = _parse_rfc822(_text(item, "pubDate"))
            content = _strip_html(desc)[:_MAX_CONTENT]
            items.append(FeedItem(
                source_name=self._sub.name,
                source_type="rss",
                title=title,
                content=content,
                url=link,
                author=author,
                published_at=pub_date,
            ))
        return items

    def _parse_atom(self, feed: ET.Element, limit: int) -> list[FeedItem]:
        ns = {"a": _ATOM_NS}
        entries = feed.findall("a:entry", ns) or feed.findall("entry")
        items = []
        for entry in entries[:limit]:
            title = _atom_text(entry, "title", ns)
            link_el = entry.find("a:link", ns) or entry.find("link")
            if link_el is not None:
                link = link_el.get("href") or link_el.text
            else:
                link = None
            summary = (
                _atom_text(entry, "summary", ns)
                or _atom_text(entry, "content", ns)
                or ""
            )
            author_el = entry.find("a:author", ns) or entry.find("author")
            author = None
            if author_el is not None:
                author = _atom_text(author_el, "name", ns) or _text(author_el, "name")
            updated_str = (
                _atom_text(entry, "updated", ns)
                or _atom_text(entry, "published", ns)
                or ""
            )
            published_at = _parse_iso(updated_str)
            content = _strip_html(summary)[:_MAX_CONTENT]
            items.append(FeedItem(
                source_name=self._sub.name,
                source_type="rss",
                title=title,
                content=content,
                url=link,
                author=author,
                published_at=published_at,
            ))
        return items


# ── helpers ──────────────────────────────────────────────────────

def _text(el: ET.Element, tag: str) -> str | None:
    child = el.find(tag)
    return child.text.strip() if child is not None and child.text else None


def _atom_text(el: ET.Element, tag: str, ns: dict) -> str | None:
    child = el.find(f"a:{tag}", ns) or el.find(tag)
    return child.text.strip() if child is not None and child.text else None


def _strip_html(text: str) -> str:
    """去除 HTML 标签，保留纯文本。"""
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&#?\w+;", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_rfc822(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return parsedate_to_datetime(s)
    except Exception:
        return None


def _parse_iso(s: str) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None
