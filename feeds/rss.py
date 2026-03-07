"""
RSS/Atom 信息源。支持 RSS 2.0 和 Atom 1.0 格式。
使用 httpx（已有依赖）+ 标准库 xml.etree.ElementTree。
"""

from __future__ import annotations

import asyncio
import logging
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from core.common.timekit import parse_iso as _parse_iso
from core.net.http import (
    HttpRequester,
    RequestBudget,
    get_default_http_requester,
)
from feeds.base import FeedItem, FeedSource, FeedSubscription

logger = logging.getLogger(__name__)

_ATOM_NS = "http://www.w3.org/2005/Atom"
_TIMEOUT = 15.0
_MAX_CONTENT = 300  # 正文截断字数
_FETCH_RETRY_DELAYS_S = (0.3, 0.8)
_MAX_FETCH_TOTAL_SECONDS = 20.0


class RSSFeedSource(FeedSource):
    """RSS 2.0 / Atom 1.0 信息源。"""

    def __init__(
        self,
        sub: FeedSubscription,
        requester: HttpRequester | None = None,
    ) -> None:
        self._sub = sub
        self._requester = requester

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
                logger.warning(
                    f"RSS local file read error [{self._sub.name}] path={local_path!r}: {e}"
                )
                return []
        if "xcancel.com" in self._sub.url:
            return await self._fetch_via_curl(limit)

        return await self._fetch_http(limit)

    async def _fetch_http(self, limit: int) -> list[FeedItem]:
        requester = self._requester or get_default_http_requester("feed_fetcher")
        parse_retried = False
        last_err: Exception | None = None
        for attempt in range(1, 3):
            try:
                resp = await requester.get(
                    self._sub.url or "",
                    follow_redirects=True,
                    timeout_s=_TIMEOUT,
                    budget=RequestBudget(total_timeout_s=_MAX_FETCH_TOTAL_SECONDS),
                    headers={
                        "User-Agent": "FreshRSS/1.24.0",
                        "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5",
                    },
                )
                resp.raise_for_status()
                return self._parse(resp.text, limit, raise_on_parse_error=True)
            except ET.ParseError as e:
                last_err = e
                if parse_retried or attempt >= 2:
                    break
                parse_retried = True
                delay = _FETCH_RETRY_DELAYS_S[min(attempt - 1, len(_FETCH_RETRY_DELAYS_S) - 1)]
                logger.warning(
                    "RSS 解析失败 [%s] attempt=%d/2 err=%r，%.1fs 后重试一次",
                    self._sub.name,
                    attempt,
                    str(e),
                    delay,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
            except Exception as e:
                last_err = e
                break

        logger.warning(
            "http 请求最终失败 [%s] err_type=%s err=%r",
            self._sub.name,
            type(last_err).__name__ if last_err else "Unknown",
            str(last_err) if last_err else "",
        )
        return []

    async def _fetch_via_curl(self, limit: int) -> list[FeedItem]:
        """对 xcancel 等需要特定 TLS 指纹的源，使用系统 curl 获取。"""
        url = self._sub.url or ""
        attempts = len(_FETCH_RETRY_DELAYS_S) + 1
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _MAX_FETCH_TOTAL_SECONDS
        last_err: str = ""
        parse_retried = False
        for attempt in range(1, attempts + 1):
            remaining = _remaining_budget(deadline, loop)
            if remaining <= 0:
                break
            proc = None
            try:
                proc = await asyncio.create_subprocess_exec(
                    "curl",
                    "-sS",
                    "-L",
                    "--max-time",
                    str(max(1, int(min(_TIMEOUT, remaining)))),
                    "-A",
                    "FreshRSS/1.24.0",
                    "-H",
                    "Accept: */*",
                    url,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=min(_TIMEOUT + 5, remaining + 2)
                )
                if proc.returncode == 0:
                    try:
                        return self._parse(
                            stdout.decode("utf-8", errors="replace"),
                            limit,
                            raise_on_parse_error=True,
                        )
                    except ET.ParseError as e:
                        last_err = f"parse_error: {e}"
                        if parse_retried or attempt >= attempts:
                            break
                        parse_retried = True
                        delay = _FETCH_RETRY_DELAYS_S[attempt - 1]
                        sleep_s = min(delay, _remaining_budget(deadline, loop))
                        if _remaining_budget(deadline, loop) <= 0:
                            break
                        logger.warning(
                            "curl 响应解析失败 [%s] attempt=%d/%d err=%r，%.1fs 后重试一次",
                            self._sub.name,
                            attempt,
                            attempts,
                            str(e),
                            sleep_s,
                        )
                        if sleep_s > 0:
                            await asyncio.sleep(sleep_s)
                        continue

                err_text = stderr.decode("utf-8", errors="replace").strip()
                rc = int(proc.returncode or 0)
                last_err = f"rc={rc} err={err_text[:300]}"
                if not _is_retryable_curl_code(rc):
                    break
                if attempt >= attempts:
                    break
                delay = _FETCH_RETRY_DELAYS_S[attempt - 1]
                sleep_s = min(delay, _remaining_budget(deadline, loop))
                if _remaining_budget(deadline, loop) <= 0:
                    break
                logger.warning(
                    "curl 请求失败 [%s] attempt=%d/%d %s，%.1fs 后重试",
                    self._sub.name,
                    attempt,
                    attempts,
                    last_err,
                    sleep_s,
                )
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)
            except Exception as e:
                if isinstance(e, asyncio.TimeoutError) and proc is not None:
                    try:
                        proc.kill()
                        await proc.communicate()
                    except Exception:
                        pass
                last_err = f"{type(e).__name__}: {e}"
                if attempt >= attempts:
                    break
                delay = _FETCH_RETRY_DELAYS_S[attempt - 1]
                sleep_s = min(delay, _remaining_budget(deadline, loop))
                if _remaining_budget(deadline, loop) <= 0:
                    break
                logger.warning(
                    "curl 子进程异常 [%s] attempt=%d/%d err=%s，%.1fs 后重试",
                    self._sub.name,
                    attempt,
                    attempts,
                    last_err,
                    sleep_s,
                )
                if sleep_s > 0:
                    await asyncio.sleep(sleep_s)

        logger.warning(
            "curl 请求最终失败 [%s] attempts=%d %s",
            self._sub.name,
            attempts,
            last_err,
        )
        return []

    def _parse(
        self,
        xml_text: str,
        limit: int,
        *,
        raise_on_parse_error: bool = False,
    ) -> list[FeedItem]:
        xml_text = _normalize_xml_text(xml_text)
        if _is_xcancel_whitelist_feed(xml_text):
            logger.warning(
                f"RSS source blocked by xcancel whitelist [{self._sub.name}]"
            )
            return []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            if raise_on_parse_error:
                raise
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
            items.append(
                FeedItem(
                    source_name=self._sub.name,
                    source_type="rss",
                    title=title,
                    content=content,
                    url=link,
                    author=author,
                    published_at=pub_date,
                )
            )
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
            items.append(
                FeedItem(
                    source_name=self._sub.name,
                    source_type="rss",
                    title=title,
                    content=content,
                    url=link,
                    author=author,
                    published_at=published_at,
                )
            )
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


def _normalize_xml_text(text: str) -> str:
    if not text:
        return ""
    return text.lstrip("\ufeff\r\n\t ")


def _is_xcancel_whitelist_feed(text: str) -> bool:
    lower = text.lower()
    return "rss reader not yet whitelisted" in lower


def _is_retryable_http_error(err: Exception) -> bool:
    if isinstance(err, httpx.TimeoutException):
        return True
    if isinstance(err, httpx.TransportError):
        return True
    if isinstance(err, httpx.HTTPStatusError):
        code = err.response.status_code
        return code in {408, 409, 425, 429, 500, 502, 503, 504}
    return False


def _is_retryable_curl_code(return_code: int) -> bool:
    return return_code in {6, 7, 18, 28, 35, 47, 52, 55, 56}


def _remaining_budget(deadline: float, loop: asyncio.AbstractEventLoop) -> float:
    return max(0.0, deadline - loop.time())
