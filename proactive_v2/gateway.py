"""
proactive_v2/gateway.py — DataGateway

在 agent loop 启动前并行预取所有数据：
- alerts / context：直接透传给 agent（完整内容）
- content：并行 web_fetch，结果存 hashmap，agent 按需通过 get_content 取
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GatewayResult:
    alerts: list[dict] = field(default_factory=list)
    context: list[dict] = field(default_factory=list)
    # 仅含 {id, title, source, url} 的轻量元数据，按顺序
    content_meta: list[dict] = field(default_factory=list)
    # compound_key → 正文（fetch 失败时为空字符串）
    content_store: dict[str, str] = field(default_factory=dict)


class DataGateway:
    def __init__(
        self,
        *,
        alert_fn: Any = None,
        feed_fn: Any = None,
        context_fn: Any = None,
        web_fetch_tool: Any = None,
        max_chars: int = 8_000,
        content_limit: int = 5,
    ) -> None:
        self._alert_fn = alert_fn
        self._feed_fn = feed_fn
        self._context_fn = context_fn
        self._web_fetch_tool = web_fetch_tool
        self._max_chars = max_chars
        self._content_limit = content_limit

    async def run(self) -> GatewayResult:
        """并行拉取所有数据源，返回 GatewayResult。单源失败不影响其他源。"""
        alerts_task = asyncio.create_task(self._fetch_alerts())
        context_task = asyncio.create_task(self._fetch_context())
        content_task = asyncio.create_task(self._fetch_content())

        alerts, ctx_data, (content_meta, content_store) = await asyncio.gather(
            alerts_task, context_task, content_task
        )
        logger.info(
            "[gateway] done: alerts=%d context=%d content=%d (fetched=%d)",
            len(alerts), len(ctx_data), len(content_meta),
            sum(1 for v in content_store.values() if v),
        )
        return GatewayResult(
            alerts=alerts,
            context=ctx_data,
            content_meta=content_meta,
            content_store=content_store,
        )

    # ── private ────────────────────────────────────────────────────────────

    async def _fetch_alerts(self) -> list[dict]:
        try:
            return await self._alert_fn() if self._alert_fn else []
        except Exception as e:
            logger.warning("[gateway] alerts fetch failed: %s", e)
            return []

    async def _fetch_context(self) -> list[dict]:
        try:
            return await self._context_fn() if self._context_fn else []
        except Exception as e:
            logger.warning("[gateway] context fetch failed: %s", e)
            return []

    async def _fetch_content(self) -> tuple[list[dict], dict[str, str]]:
        """拉取 content events，并行 web_fetch，返回 (meta列表, content_store)。"""
        try:
            events = await self._feed_fn(limit=self._content_limit) if self._feed_fn else []
        except Exception as e:
            logger.warning("[gateway] feed fetch failed: %s", e)
            return [], {}

        if not events:
            return [], {}

        # 并行 web_fetch
        fetch_tasks = [asyncio.create_task(self._fetch_one_url(e.get("url", ""))) for e in events]
        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

        content_meta: list[dict] = []
        content_store: dict[str, str] = {}

        for event, result in zip(events, fetch_results):
            item_id = event.get("event_id", "")
            ack_server = event.get("ack_server", "")
            compound_key = f"{ack_server}:{item_id}"

            content_meta.append({
                "id": compound_key,
                "title": event.get("title") or "",
                "source": event.get("source_name") or "",
                "url": event.get("url") or "",
                "published_at": event.get("published_at") or "",
            })

            if isinstance(result, Exception) or not result:
                logger.debug("[gateway] web_fetch failed for %s: %s", compound_key, result)
                content_store[compound_key] = ""
            else:
                content_store[compound_key] = result

        return content_meta, content_store

    async def _fetch_one_url(self, url: str) -> str:
        """抓取单个 URL，返回截断后正文，失败返回空字符串。"""
        if not url or not self._web_fetch_tool:
            return ""
        try:
            import json
            result_json = await self._web_fetch_tool.execute(url=url, format="text")
            result = json.loads(result_json)
            if "error" in result:
                return ""
            text = result.get("text", "")
            return text[:self._max_chars]
        except Exception as e:
            logger.debug("[gateway] _fetch_one_url(%s) error: %s", url, e)
            return ""
