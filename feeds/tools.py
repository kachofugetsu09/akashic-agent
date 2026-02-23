"""
信息流订阅管理工具。
FeedSubscribeTool / FeedUnsubscribeTool / FeedListTool

用法示例：
  用户："我很关心 Paul Graham 的博客"
  → Agent 调用 feed_subscribe(name="Paul Graham", url="https://paulgraham.com/rss.html")

  用户："不太关注这个了"
  → Agent 调用 feed_unsubscribe(name="Paul Graham")
"""
from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlparse

import httpx
from agent.tools.base import Tool
from feeds.base import FeedSubscription
from feeds.registry import FeedRegistry
from feeds.store import FeedStore

logger = logging.getLogger(__name__)


class FeedSubscribeTool(Tool):
    name = "feed_subscribe"
    description = (
        "订阅一个 RSS 信息源。当用户表达对某个博客、新闻源感兴趣时调用。\n"
        "订阅成功后，该信息源会纳入主动推送的信息收集范围。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "信息源的人类可读名称，如 'Paul Graham' 或 'Hacker News'",
            },
            "url": {
                "type": "string",
                "description": "RSS/Atom feed 地址",
            },
            "note": {
                "type": "string",
                "description": "备注，记录用户为何关注（可选）",
            },
        },
        "required": ["name", "url"],
    }

    def __init__(self, store: FeedStore) -> None:
        self._store = store

    async def execute(self, **kwargs: Any) -> str:
        name: str = kwargs.get("name", "").strip()
        url: str = kwargs.get("url", "").strip()
        note: str | None = kwargs.get("note")

        if not name:
            return "错误：name 不能为空"
        if not url:
            return "错误：url 不能为空"

        # 检查是否已订阅同一地址
        for s in self._store.load():
            if s.url == url:
                return f"已经订阅过该地址：{s.name!r}（id: {s.id[:8]}），无需重复添加"

        sub = FeedSubscription.new(type="rss", name=name, url=url, note=note)
        self._store.add(sub)
        return f"已订阅 {name!r}（{url}），下次主动巡检时开始收集"


class FeedUnsubscribeTool(Tool):
    name = "feed_unsubscribe"
    description = (
        "取消订阅一个信息源。当用户说不再关心某人/某博客时调用。"
        "按名称模糊匹配，若匹配到多个会全部取消。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "信息源名称（支持模糊匹配）",
            },
        },
        "required": ["name"],
    }

    def __init__(self, store: FeedStore) -> None:
        self._store = store

    async def execute(self, **kwargs: Any) -> str:
        name: str = kwargs.get("name", "").strip()
        if not name:
            return "错误：name 不能为空"

        matches = self._store.find_by_name(name)
        if not matches:
            return f"没有找到名称包含 {name!r} 的订阅"

        for sub in matches:
            self._store.remove(sub.id)
        names = "、".join(f"「{s.name}」" for s in matches)
        return f"已取消订阅：{names}"


class FeedListTool(Tool):
    name = "feed_list"
    description = "列出当前所有订阅的 RSS 信息源"
    parameters = {"type": "object", "properties": {}}

    def __init__(self, store: FeedStore) -> None:
        self._store = store

    async def execute(self, **kwargs: Any) -> str:
        subs = self._store.load()
        if not subs:
            return "当前没有订阅任何信息源"

        lines = [f"RSS 订阅列表（共 {len(subs)} 个）："]
        for sub in subs:
            status = "启用" if sub.enabled else "停用"
            note_part = f"  备注: {sub.note}" if sub.note else ""
            lines.append(f"  [{status}] {sub.name}  {sub.url}{note_part}")
        return "\n".join(lines)


class FeedGenerateSubscribeTool(Tool):
    name = "feed_generate_subscribe"
    description = (
        "使用 Diffbot 官方托管端点把普通网页链接转换成 RSS 并订阅。"
        "适合目标网站没有现成 RSS 地址时调用。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "订阅名称，例如 '某博客更新'",
            },
            "page_url": {
                "type": "string",
                "description": "原始网页地址（列表页/作者页/栏目页）",
            },
            "note": {
                "type": "string",
                "description": "订阅备注（可选）",
            },
            "validate_feed": {
                "type": "boolean",
                "description": "是否先请求生成的 RSS URL 进行可用性校验（默认 true）",
            },
            "auto_discover": {
                "type": "boolean",
                "description": "是否自动发现最适合订阅的页面（默认 true）",
            },
        },
        "required": ["name", "page_url"],
    }

    _DIFFBOT_RSS_ENDPOINT = "https://rss.diffbot.com/rss?url="
    _CRAWL_MAX_DEPTH = 2
    _CRAWL_MAX_PAGES = 160

    def __init__(
        self,
        store: FeedStore,
        validator: Callable[[str], Awaitable[tuple[bool, str]]] | None = None,
        page_fetcher: Callable[[str], Awaitable[tuple[bool, str, str]]] | None = None,
    ) -> None:
        self._store = store
        self._validator = validator or self._validate_generated_feed
        self._page_fetcher = page_fetcher or self._fetch_page_text
        self._generated_dir = self._store.path.parent / "generated_feeds"
        self._generated_dir.mkdir(parents=True, exist_ok=True)

    async def execute(self, **kwargs: Any) -> str:
        name: str = kwargs.get("name", "").strip()
        page_url: str = kwargs.get("page_url", "").strip()
        note: str | None = kwargs.get("note")
        validate_feed: bool = kwargs.get("validate_feed", True)
        auto_discover: bool = kwargs.get("auto_discover", True)

        if not name:
            return "错误：name 不能为空"
        if not page_url:
            return "错误：page_url 不能为空"
        if not page_url.startswith(("http://", "https://")):
            return "错误：page_url 必须是 http/https 链接"

        chosen_url = page_url
        if auto_discover:
            chosen_url = await self._discover_best_page(page_url)
        logger.info(
            "[feed_generate_subscribe] 发现结果 input=%s chosen=%s auto_discover=%s",
            page_url,
            chosen_url,
            auto_discover,
        )

        # 现代优先策略：先尝试站点结构化发现（sitemap / VitePress）并本地生成 RSS。
        local_entries = await self._discover_entries_from_site(page_url)
        if len(local_entries) >= 3:
            local_rss_url = self._write_local_rss(name=name, source_page=page_url, entries=local_entries)
            logger.info(
                "[feed_generate_subscribe] 本地发现成功 entries=%d local_rss=%s",
                len(local_entries),
                local_rss_url,
            )
            sub_result = self._save_subscription_if_new(name=name, url=local_rss_url, note=note)
            if sub_result.startswith("已"):
                return f"{sub_result}（来源：站点结构发现，共 {len(local_entries)} 条）"
            return sub_result
        logger.info(
            "[feed_generate_subscribe] 本地发现不足 entries=%d，回退 Diffbot 方案",
            len(local_entries),
        )

        encoded = quote(chosen_url, safe="")
        rss_url = f"{self._DIFFBOT_RSS_ENDPOINT}{encoded}"
        logger.info(
            "[feed_generate_subscribe] 生成 RSS URL name=%r page_url=%s rss_url=%s",
            name,
            chosen_url,
            rss_url,
        )

        if validate_feed:
            ok, detail = await self._validator(rss_url)
            logger.info(
                "[feed_generate_subscribe] 校验结果 ok=%s detail=%r rss_url=%s",
                ok,
                detail,
                rss_url,
            )
            if not ok:
                return f"生成的 RSS 无法使用：{detail}"

        save_result = self._save_subscription_if_new(name=name, url=rss_url, note=note)
        if not save_result.startswith("已"):
            return save_result
        if chosen_url != page_url:
            return f"{save_result}（自动选择页面：{chosen_url}）"
        return save_result

    def _save_subscription_if_new(self, name: str, url: str, note: str | None) -> str:
        for s in self._store.load():
            if s.url == url:
                logger.info(
                    "[feed_generate_subscribe] 重复订阅拦截 existing_id=%s rss_url=%s",
                    s.id,
                    url,
                )
                return f"已经订阅过该地址：{s.name!r}（id: {s.id[:8]}），无需重复添加"
        sub = FeedSubscription.new(type="rss", name=name, url=url, note=note)
        self._store.add(sub)
        logger.info(
            "[feed_generate_subscribe] 订阅成功 id=%s name=%r rss_url=%s",
            sub.id,
            name,
            url,
        )
        return f"已生成并订阅 {name!r}：{url}"

    async def _validate_generated_feed(self, rss_url: str) -> tuple[bool, str]:
        try:
            async with httpx.AsyncClient(
                timeout=15.0,
                follow_redirects=True,
                headers={"User-Agent": "Akasic-Agent/1.0 (feed generator)"},
            ) as client:
                resp = await client.get(rss_url)
            logger.info(
                "[feed_generate_subscribe] 校验请求完成 status=%d content_type=%r",
                resp.status_code,
                resp.headers.get("content-type"),
            )
            if resp.status_code >= 400:
                return False, f"HTTP {resp.status_code}"
            full_text = resp.text or ""
            text = full_text[:1000].lower()
            if "<rss" in text or "<feed" in text:
                ok, reason = self._assess_feed_quality(full_text)
                if not ok:
                    logger.warning(
                        "[feed_generate_subscribe] 质量门禁未通过 rss_url=%s reason=%s",
                        rss_url,
                        reason,
                    )
                    return False, f"质量不足：{reason}"
                return True, "ok"
            return False, "返回内容不是 RSS/Atom XML"
        except Exception as e:
            logger.warning("[feed_generate_subscribe] 校验请求失败: %s", e)
            return False, str(e)

    def _assess_feed_quality(self, xml_text: str) -> tuple[bool, str]:
        text = xml_text or ""
        lower = text.lower()
        item_count = len(re.findall(r"<item\\b", lower))
        entry_count = len(re.findall(r"<entry\\b", lower))
        total_entries = item_count + entry_count
        links = re.findall(r"<link>\\s*([^<\\s]+)\\s*</link>", text, flags=re.IGNORECASE)
        atom_links = re.findall(r"<link[^>]+href=[\"']([^\"']+)[\"'][^>]*/?>", text, flags=re.IGNORECASE)
        all_links = [x.strip() for x in (links + atom_links) if x.strip()]
        unique_links = len(set(all_links))

        logger.info(
            "[feed_generate_subscribe] 质量评估 total_entries=%d unique_links=%d",
            total_entries,
            unique_links,
        )

        if total_entries < 3:
            return False, f"条目过少（{total_entries}）"
        if unique_links < 3:
            return False, f"有效链接过少（{unique_links}）"
        # 过高重复率也拒绝（例如导航页被误识别）
        if total_entries > 0 and unique_links / total_entries < 0.5:
            ratio = unique_links / total_entries
            return False, f"重复率过高（unique/total={ratio:.2f}）"
        return True, "ok"

    async def _discover_entries_from_site(self, input_url: str) -> list[dict[str, Any]]:
        base = _root_url(input_url)
        logger.info("[feed_generate_subscribe] 本地发现开始 base=%s", base)

        sitemap_entries = await self._discover_from_sitemap(base)
        if len(sitemap_entries) >= 3:
            logger.info("[feed_generate_subscribe] 使用 sitemap 结果 entries=%d", len(sitemap_entries))
            return sitemap_entries

        crawl_entries = await self._discover_by_crawl(start_url=input_url, base=base)
        if len(crawl_entries) >= 3:
            logger.info("[feed_generate_subscribe] 使用树状爬取结果 entries=%d", len(crawl_entries))
            return crawl_entries

        vp_entries = await self._discover_from_vitepress(base)
        if len(vp_entries) >= 3:
            logger.info("[feed_generate_subscribe] 使用 VitePress 结果 entries=%d", len(vp_entries))
            return vp_entries

        logger.info(
            "[feed_generate_subscribe] 本地发现不足 sitemap=%d crawl=%d vitepress=%d",
            len(sitemap_entries),
            len(crawl_entries),
            len(vp_entries),
        )
        best = sitemap_entries
        if len(crawl_entries) > len(best):
            best = crawl_entries
        if len(vp_entries) > len(best):
            best = vp_entries
        return best

    async def _discover_from_sitemap(self, base: str) -> list[dict[str, Any]]:
        sitemap_url = urljoin(base, "/sitemap.xml")
        ok, xml_text, detail = await self._page_fetcher(sitemap_url)
        if not ok:
            logger.info("[feed_generate_subscribe] sitemap 不可用 url=%s detail=%r", sitemap_url, detail)
            return []
        logger.info("[feed_generate_subscribe] sitemap 命中 url=%s", sitemap_url)
        try:
            root = ET.fromstring(xml_text)
        except Exception as e:
            logger.warning("[feed_generate_subscribe] sitemap 解析失败: %s", e)
            return []

        ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        entries: list[dict[str, Any]] = []

        def add_from_urlset(el: ET.Element) -> None:
            urls = el.findall(".//sm:url", ns) or el.findall(".//url")
            for u in urls:
                loc_el = u.find("sm:loc", ns)
                if loc_el is None:
                    loc_el = u.find("loc")
                if loc_el is None or not (loc_el.text or "").strip():
                    continue
                loc = (loc_el.text or "").strip()
                if not _looks_like_article_url(loc, base_host=urlparse(base).netloc):
                    continue
                lm_el = u.find("sm:lastmod", ns)
                if lm_el is None:
                    lm_el = u.find("lastmod")
                entries.append({
                    "title": _title_from_url(loc),
                    "url": loc,
                    "summary": "from sitemap",
                    "published_at": _parse_dt((lm_el.text or "").strip() if lm_el is not None else ""),
                })

        tag = root.tag.lower()
        if "urlset" in tag:
            add_from_urlset(root)
        elif "sitemapindex" in tag:
            smaps = root.findall(".//sm:sitemap", ns) or root.findall(".//sitemap")
            for s in smaps[:6]:
                loc_el = s.find("sm:loc", ns)
                if loc_el is None:
                    loc_el = s.find("loc")
                if loc_el is None or not (loc_el.text or "").strip():
                    continue
                loc = (loc_el.text or "").strip()
                ok2, xml2, detail2 = await self._page_fetcher(loc)
                if not ok2:
                    logger.info("[feed_generate_subscribe] 子 sitemap 抓取失败 url=%s detail=%r", loc, detail2)
                    continue
                try:
                    sub_root = ET.fromstring(xml2)
                    add_from_urlset(sub_root)
                except Exception as e:
                    logger.warning("[feed_generate_subscribe] 子 sitemap 解析失败 url=%s err=%s", loc, e)
                    continue
        dedup = _dedup_entries(entries)
        logger.info("[feed_generate_subscribe] sitemap 发现 entries=%d", len(dedup))
        return dedup

    async def _discover_from_vitepress(self, base: str) -> list[dict[str, Any]]:
        ok, html, detail = await self._page_fetcher(base)
        if not ok:
            logger.info("[feed_generate_subscribe] VitePress 探测失败 base=%s detail=%r", base, detail)
            return []
        site_data = _extract_vp_json(html, "__VP_SITE_DATA__")
        if not site_data:
            logger.info("[feed_generate_subscribe] 未检测到 __VP_SITE_DATA__")
            return []
        entries: list[dict[str, Any]] = []
        theme = site_data.get("themeConfig", {})
        nav = theme.get("nav", [])
        sidebar = theme.get("sidebar", [])
        for n in nav if isinstance(nav, list) else []:
            link = str(n.get("link", "")).strip() if isinstance(n, dict) else ""
            text = str(n.get("text", "")).strip() if isinstance(n, dict) else ""
            url = _vp_link_to_url(base, link)
            if url and _looks_like_article_url(url, base_host=urlparse(base).netloc):
                entries.append({"title": text or _title_from_url(url), "url": url, "summary": "from vitepress nav", "published_at": None})
        self._collect_sidebar_entries(base, sidebar, entries)
        dedup = _dedup_entries(entries)
        logger.info("[feed_generate_subscribe] VitePress 发现 entries=%d", len(dedup))
        return dedup

    async def _discover_by_crawl(self, start_url: str, base: str) -> list[dict[str, Any]]:
        base_host = urlparse(base).netloc
        start = _normalize_page_url(start_url, base)
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        visited: set[str] = set()
        entries: list[dict[str, Any]] = []
        pages = 0

        logger.info(
            "[feed_generate_subscribe] 树状发现开始 start=%s base=%s max_depth=%d max_pages=%d",
            start,
            base,
            self._CRAWL_MAX_DEPTH,
            self._CRAWL_MAX_PAGES,
        )

        while queue and pages < self._CRAWL_MAX_PAGES:
            current, depth = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            ok, html, detail = await self._page_fetcher(current)
            logger.info(
                "[feed_generate_subscribe] 树状抓取 url=%s depth=%d ok=%s detail=%r",
                current,
                depth,
                ok,
                detail,
            )
            if not ok:
                continue
            pages += 1
            links = _extract_anchor_links(current, html)
            logger.info(
                "[feed_generate_subscribe] 树状提取链接 url=%s links=%d",
                current,
                len(links),
            )

            for link_url, link_text in links:
                if not _is_same_host(link_url, base_host):
                    continue
                norm = _normalize_page_url(link_url, base)
                if _looks_like_article_url(norm, base_host=base_host):
                    entries.append({
                        "title": link_text or _title_from_url(norm),
                        "url": norm,
                        "summary": f"from crawl {current}",
                        "published_at": None,
                    })
                if depth < self._CRAWL_MAX_DEPTH and _is_crawlable_page(norm, base_host):
                    if norm not in visited:
                        queue.append((norm, depth + 1))

        dedup = _dedup_entries(entries)
        logger.info(
            "[feed_generate_subscribe] 树状发现完成 visited=%d pages=%d entries=%d",
            len(visited),
            pages,
            len(dedup),
        )
        return dedup

    def _collect_sidebar_entries(self, base: str, node: Any, out: list[dict[str, Any]]) -> None:
        if isinstance(node, list):
            for x in node:
                self._collect_sidebar_entries(base, x, out)
            return
        if isinstance(node, dict):
            link = str(node.get("link", "")).strip()
            text = str(node.get("text", "")).strip()
            url = _vp_link_to_url(base, link)
            if url and _looks_like_article_url(url, base_host=urlparse(base).netloc):
                out.append({
                    "title": text or _title_from_url(url),
                    "url": url,
                    "summary": "from vitepress sidebar",
                    "published_at": None,
                })
            if "items" in node:
                self._collect_sidebar_entries(base, node.get("items"), out)

    def _write_local_rss(self, name: str, source_page: str, entries: list[dict[str, Any]]) -> str:
        key = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:60]
        if not key:
            key = "generated"
        digest = abs(hash(source_page)) % (10**8)
        file_path = self._generated_dir / f"{key}_{digest}.xml"

        rss = ET.Element("rss", version="2.0")
        ch = ET.SubElement(rss, "channel")
        ET.SubElement(ch, "title").text = f"{name} (generated)"
        ET.SubElement(ch, "link").text = source_page
        ET.SubElement(ch, "description").text = f"Auto generated feed from {source_page}"
        ET.SubElement(ch, "lastBuildDate").text = format_datetime(datetime.now(timezone.utc))

        for e in entries[:300]:
            item = ET.SubElement(ch, "item")
            ET.SubElement(item, "title").text = str(e.get("title") or _title_from_url(str(e.get("url") or "")))
            ET.SubElement(item, "link").text = str(e.get("url") or "")
            ET.SubElement(item, "guid").text = str(e.get("url") or "")
            ET.SubElement(item, "description").text = str(e.get("summary") or "")
            dt = e.get("published_at")
            if isinstance(dt, datetime):
                ET.SubElement(item, "pubDate").text = format_datetime(dt.astimezone(timezone.utc))

        xml_text = ET.tostring(rss, encoding="utf-8", xml_declaration=True)
        file_path.write_bytes(xml_text)
        logger.info("[feed_generate_subscribe] 本地 RSS 已写入 path=%s entries=%d", file_path, len(entries))
        return file_path.as_uri()

    async def _discover_best_page(self, input_url: str) -> str:
        candidates = self._build_candidates(input_url)
        logger.info(
            "[feed_generate_subscribe] 自动发现候选页面 count=%d candidates=%s",
            len(candidates),
            candidates,
        )
        best_url = input_url
        best_score = -1
        best_reason = "fallback"
        for url in candidates:
            ok, html, detail = await self._page_fetcher(url)
            if not ok:
                logger.info("[feed_generate_subscribe] 候选抓取失败 url=%s detail=%r", url, detail)
                continue
            score, reason = self._score_page(url, html)
            logger.info(
                "[feed_generate_subscribe] 候选评分 url=%s score=%d reason=%s",
                url,
                score,
                reason,
            )
            if score > best_score:
                best_score = score
                best_url = url
                best_reason = reason
        logger.info(
            "[feed_generate_subscribe] 自动发现完成 chosen=%s score=%d reason=%s",
            best_url,
            best_score,
            best_reason,
        )
        return best_url

    def _build_candidates(self, input_url: str) -> list[str]:
        base = input_url.strip()
        out = [base]
        parsed = urlparse(base)
        root_like = parsed.path in ("", "/")
        if root_like:
            for p in ("/archive.html", "/archives", "/blog", "/posts", "/main.html"):
                out.append(urljoin(base if base.endswith("/") else base + "/", p.lstrip("/")))
        # preserve order and dedupe
        return list(dict.fromkeys(out))

    async def _fetch_page_text(self, url: str) -> tuple[bool, str, str]:
        try:
            async with httpx.AsyncClient(
                timeout=12.0,
                follow_redirects=True,
                headers={"User-Agent": "Akasic-Agent/1.0 (feed discover)"},
            ) as client:
                resp = await client.get(url)
            if resp.status_code >= 400:
                return False, "", f"HTTP {resp.status_code}"
            text = resp.text or ""
            return True, text, "ok"
        except Exception as e:
            return False, "", str(e)

    def _score_page(self, url: str, html: str) -> tuple[int, str]:
        lower = html.lower()
        parsed = urlparse(url)
        host = parsed.netloc
        links = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)

        article_links = 0
        for link in links:
            l = link.strip()
            if not l:
                continue
            if l.startswith(("javascript:", "mailto:", "#")):
                continue
            if l.endswith(".html") or ".html?" in l:
                if l.endswith(("archive.html", "main.html", "index.html")):
                    continue
                if l.startswith("/"):
                    article_links += 1
                    continue
                lp = urlparse(l)
                if lp.netloc == "" or lp.netloc == host:
                    article_links += 1

        date_hits = len(re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{2}-\d{2}\b", html))
        archive_hits = len(re.findall(r"archive|archives|归档|文章|posts", lower))
        score = article_links * 3 + min(date_hits, 40) + archive_hits * 2
        reason = f"article_links={article_links}, date_hits={date_hits}, archive_hits={archive_hits}"
        return score, reason


def _root_url(url: str) -> str:
    p = urlparse(url)
    if not p.scheme or not p.netloc:
        return url
    return f"{p.scheme}://{p.netloc}/"


def _is_same_host(url: str, host: str) -> bool:
    p = urlparse(url)
    return p.netloc == host


def _normalize_page_url(url: str, base: str) -> str:
    raw = urljoin(base, url.strip())
    p = urlparse(raw)
    path = re.sub(r"/{2,}", "/", p.path or "/")
    # 规范掉锚点，保留 query（部分站点依赖）
    q = p.query
    return f"{p.scheme}://{p.netloc}{path}" + (f"?{q}" if q else "")


def _is_crawlable_page(url: str, base_host: str) -> bool:
    p = urlparse(url)
    if not p.scheme.startswith("http"):
        return False
    if p.netloc != base_host:
        return False
    path = (p.path or "").lower()
    if any(path.endswith(ext) for ext in (
        ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".woff", ".woff2", ".ico", ".pdf", ".zip", ".mp4",
    )):
        return False
    if any(path.startswith(prefix) for prefix in ("/assets/", "/images/", "/img/", "/public/", "/static/")):
        return False
    return True


def _parse_dt(text: str) -> datetime | None:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _title_from_url(url: str) -> str:
    p = urlparse(url)
    tail = (p.path or "").strip("/").split("/")[-1]
    tail = re.sub(r"\.html?$", "", tail, flags=re.IGNORECASE)
    tail = tail.replace("-", " ").replace("_", " ")
    return tail or url


def _looks_like_article_url(url: str, base_host: str) -> bool:
    p = urlparse(url)
    if not p.scheme.startswith("http"):
        return False
    if p.netloc and p.netloc != base_host:
        return False
    path = (p.path or "").lower()
    if path in ("", "/", "/index.html", "/main.html", "/labs.html", "/archive.html"):
        return False
    if any(path.startswith(prefix) for prefix in ("/assets/", "/images/", "/img/", "/public/", "/static/")):
        return False
    if path.endswith((".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".woff", ".woff2", ".ico", ".pdf")):
        return False
    # 允许 .html 页面和常见文章路由
    if ".html" in path:
        return True
    if any(seg in path for seg in ("/notes/", "/post/", "/posts/", "/blog/", "/article/", "/tech/", "/cs", "/mysql/")):
        return True
    return False


def _dedup_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for e in entries:
        url = str(e.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(e)
    return out


def _extract_vp_json(html: str, var_name: str) -> dict[str, Any]:
    # window.__VP_SITE_DATA__=JSON.parse("...escaped json...");
    pat = rf"window\.{re.escape(var_name)}\s*=\s*JSON\.parse\(\"([\s\S]*?)\"\);"
    m = re.search(pat, html)
    if not m:
        return {}
    escaped = m.group(1)
    try:
        unescaped = json.loads(f"\"{escaped}\"")
        data = json.loads(unescaped)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _vp_link_to_url(base: str, link: str) -> str:
    if not link:
        return ""
    l = link.strip()
    if l.startswith(("http://", "https://")):
        return l
    if l.startswith("#"):
        return ""
    if l == "/":
        return _root_url(base)
    if not l.startswith("/"):
        l = "/" + l
    # VitePress 常见配置是 cleanUrls=false，页面会落 .html
    if not l.endswith(".html") and "." not in l.split("/")[-1]:
        l = l.rstrip("/") + ".html"
    return urljoin(base, l.lstrip("/"))


def _extract_anchor_links(base_url: str, html: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for m in re.finditer(r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>([\s\S]*?)</a>", html, flags=re.IGNORECASE):
        href = (m.group(1) or "").strip()
        if not href or href.startswith(("javascript:", "mailto:", "#")):
            continue
        text_raw = re.sub(r"<[^>]+>", " ", m.group(2) or "")
        text = re.sub(r"\s+", " ", text_raw).strip()
        url = urljoin(base_url, href)
        pairs.append((url, text))
    return pairs


class FeedManageTool(Tool):
    name = "feed_manage"
    description = (
        "统一管理信息流订阅：发现候选页面、预览、订阅、取消订阅、列出订阅。"
        "用 action 控制行为，减少工具数量同时保留灵活性。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["discover", "preview", "subscribe", "list", "unsubscribe"],
                "description": "操作类型",
            },
            "name": {"type": "string", "description": "订阅名称或用于取消订阅的名称关键词"},
            "url": {"type": "string", "description": "已知 RSS/Atom URL（subscribe 时可选）"},
            "page_url": {"type": "string", "description": "网页地址（discover/preview/subscribe）"},
            "note": {"type": "string", "description": "订阅备注（subscribe 可选）"},
            "validate_feed": {"type": "boolean", "description": "是否校验 feed（默认 true）"},
            "auto_discover": {"type": "boolean", "description": "是否自动发现页面（默认 true）"},
        },
        "required": ["action"],
    }

    def __init__(self, store: FeedStore) -> None:
        self._store = store
        self._list_tool = FeedListTool(store)
        self._sub_tool = FeedSubscribeTool(store)
        self._unsub_tool = FeedUnsubscribeTool(store)
        self._gen_tool = FeedGenerateSubscribeTool(store)

    async def execute(self, **kwargs: Any) -> str:
        action = str(kwargs.get("action", "")).strip().lower()
        logger.info("[feed_manage] action=%s kwargs=%s", action, {k: v for k, v in kwargs.items() if k != "note"})

        if action == "list":
            return await self._list_tool.execute()

        if action == "unsubscribe":
            name = str(kwargs.get("name", "")).strip()
            if not name:
                return "错误：unsubscribe 需要 name"
            return await self._unsub_tool.execute(name=name)

        if action == "discover":
            page_url = str(kwargs.get("page_url", "")).strip()
            if not page_url:
                return "错误：discover 需要 page_url"
            chosen = await self._gen_tool._discover_best_page(page_url)
            entries = await self._gen_tool._discover_entries_from_site(page_url)
            sample = "\n".join([f"- {e.get('title','')} | {e.get('url','')}" for e in entries[:5]]) or "（无）"
            return (
                f"发现完成\n"
                f"输入页面: {page_url}\n"
                f"候选最优: {chosen}\n"
                f"结构化发现条目数: {len(entries)}\n"
                f"示例:\n{sample}"
            )

        if action == "preview":
            page_url = str(kwargs.get("page_url", "")).strip()
            if not page_url:
                return "错误：preview 需要 page_url"
            chosen = await self._gen_tool._discover_best_page(page_url)
            entries = await self._gen_tool._discover_entries_from_site(page_url)
            if len(entries) >= 3:
                local_rss = self._gen_tool._write_local_rss(
                    name=str(kwargs.get("name", "")).strip() or "preview",
                    source_page=page_url,
                    entries=entries,
                )
                return (
                    f"预览结果（结构化发现）\n"
                    f"输入页面: {page_url}\n"
                    f"候选最优: {chosen}\n"
                    f"可生成本地RSS: {local_rss}\n"
                    f"条目数: {len(entries)}"
                )
            encoded = quote(chosen, safe="")
            rss_url = f"{FeedGenerateSubscribeTool._DIFFBOT_RSS_ENDPOINT}{encoded}"
            validate = bool(kwargs.get("validate_feed", True))
            if validate:
                ok, detail = await self._gen_tool._validator(rss_url)
                return (
                    f"预览结果（Diffbot 回退）\n"
                    f"输入页面: {page_url}\n"
                    f"候选最优: {chosen}\n"
                    f"RSS URL: {rss_url}\n"
                    f"校验: ok={ok} detail={detail}"
                )
            return (
                f"预览结果（Diffbot 回退）\n"
                f"输入页面: {page_url}\n"
                f"候选最优: {chosen}\n"
                f"RSS URL: {rss_url}\n"
                f"校验: skipped"
            )

        if action == "subscribe":
            name = str(kwargs.get("name", "")).strip()
            if not name:
                return "错误：subscribe 需要 name"
            url = str(kwargs.get("url", "")).strip()
            if url:
                return await self._sub_tool.execute(name=name, url=url, note=kwargs.get("note"))
            page_url = str(kwargs.get("page_url", "")).strip()
            if not page_url:
                return "错误：subscribe 需要 url 或 page_url"
            return await self._gen_tool.execute(
                name=name,
                page_url=page_url,
                note=kwargs.get("note"),
                validate_feed=kwargs.get("validate_feed", True),
                auto_discover=kwargs.get("auto_discover", True),
            )

        return "错误：action 必须是 discover|preview|subscribe|list|unsubscribe"


class FeedQueryTool(Tool):
    name = "feed_query"
    description = (
        "查询订阅信息流：latest（最近条目）、search（关键词过滤）、summary（订阅概况）。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["latest", "search", "summary", "catalog"],
                "description": "查询动作",
            },
            "source": {"type": "string", "description": "按来源名筛选（可选，模糊匹配）"},
            "keyword": {"type": "string", "description": "search 用关键词"},
            "limit": {"type": "integer", "description": "返回条数（默认 5）", "minimum": 1, "maximum": 30},
            "page": {"type": "integer", "description": "catalog 页码（从 1 开始）", "minimum": 1},
            "page_size": {"type": "integer", "description": "catalog 每页条数（默认 20）", "minimum": 1, "maximum": 100},
        },
        "required": ["action"],
    }

    def __init__(self, store: FeedStore, registry: FeedRegistry) -> None:
        self._store = store
        self._registry = registry

    async def execute(self, **kwargs: Any) -> str:
        action = str(kwargs.get("action", "")).strip().lower()
        source = str(kwargs.get("source", "")).strip().lower()
        keyword = str(kwargs.get("keyword", "")).strip().lower()
        limit = int(kwargs.get("limit", 5) or 5)
        limit = max(1, min(30, limit))
        page = int(kwargs.get("page", 1) or 1)
        page_size = int(kwargs.get("page_size", 20) or 20)
        page = max(1, page)
        page_size = max(1, min(100, page_size))
        logger.info(
            "[feed_query] action=%s source=%r keyword=%r limit=%d page=%d page_size=%d",
            action,
            source,
            keyword,
            limit,
            page,
            page_size,
        )

        subs = self._store.list_enabled()
        if source:
            subs = [s for s in subs if source in s.name.lower()]
        if not subs:
            return "没有匹配的启用订阅"

        fetch_limit = 300 if action == "catalog" else limit
        items = await self._registry.fetch_all(limit_per_source=fetch_limit)
        if source:
            items = [i for i in items if source in (i.source_name or "").lower()]
        items.sort(key=lambda x: x.published_at or datetime(1970, 1, 1, tzinfo=timezone.utc), reverse=True)

        if action == "summary":
            names = "、".join(sorted({i.source_name for i in items if i.source_name})[:10]) or "（无）"
            return f"订阅概况：sources={len(subs)} items={len(items)} 来源={names}"

        if action == "search":
            if not keyword:
                return "错误：search 需要 keyword"
            items = [
                i for i in items
                if keyword in (i.title or "").lower() or keyword in (i.content or "").lower()
            ]

        if action == "catalog":
            total = len(items)
            start = (page - 1) * page_size
            end = start + page_size
            if start >= total and total > 0:
                return json.dumps(
                    {
                        "action": "catalog",
                        "source": source or None,
                        "page": page,
                        "page_size": page_size,
                        "total": total,
                        "has_more": False,
                        "next_page": None,
                        "items": [],
                        "error": "page out of range",
                    },
                    ensure_ascii=False,
                )
            picked = items[start:end]
            payload_items = []
            for i in picked:
                payload_items.append(
                    {
                        "source": i.source_name,
                        "title": i.title or "(无标题)",
                        "url": i.url or "",
                        "published_at": i.published_at.isoformat() if i.published_at else None,
                    }
                )
            has_more = end < total
            return json.dumps(
                {
                    "action": "catalog",
                    "source": source or None,
                    "page": page,
                    "page_size": page_size,
                    "total": total,
                    "has_more": has_more,
                    "next_page": (page + 1) if has_more else None,
                    "items": payload_items,
                },
                ensure_ascii=False,
            )

        if action not in ("latest", "search"):
            return "错误：action 必须是 latest|search|summary|catalog"

        picked = items[:limit]
        if not picked:
            return "没有找到匹配条目"
        lines = []
        for i in picked:
            ts = i.published_at.astimezone().strftime("%Y-%m-%d %H:%M") if i.published_at else "未知时间"
            title = i.title or "(无标题)"
            lines.append(f"- [{i.source_name}] {title} ({ts})")
            if i.url:
                lines.append(f"  {i.url}")
        return "\n".join(lines)
