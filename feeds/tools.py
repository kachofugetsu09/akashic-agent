"""
信息流订阅管理工具（Tool 适配层）。

本文件只包含 LLM 可见的工具定义（schema + execute 入口）。
业务逻辑已拆分到：
  feeds/services/subscription_service.py  — 订阅/取消/列表
  feeds/services/query_service.py         — 内容查询
"""

from __future__ import annotations

import logging
from typing import Any

from agent.tools.base import Tool
from feeds.registry import FeedRegistry
from feeds.services.query_service import QueryService
from feeds.services.subscription_service import SubscriptionService
from feeds.store import FeedStore

logger = logging.getLogger(__name__)


class FeedManageTool(Tool):
    """
    统一管理信息流订阅：订阅、取消订阅、列出订阅。

    action 路由：
      list        → SubscriptionService.list_subscriptions()
      unsubscribe → SubscriptionService.unsubscribe()
      subscribe   → SubscriptionService.subscribe()（需提供 url）
    """

    name = "feed_manage"
    description = (
        "统一管理信息流订阅：订阅、取消订阅、列出订阅。"
        '当用户询问"我有哪些订阅/信息来源"时，优先使用 action=list。'
        "用 action 控制行为，减少工具数量同时保留灵活性。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["subscribe", "list", "unsubscribe"],
                "description": "操作类型",
            },
            "name": {
                "type": "string",
                "description": "订阅名称或用于取消订阅的名称关键词",
            },
            "url": {
                "type": "string",
                "description": "RSS/Atom URL，或 novel-kb 时填 file:///path/to/kb",
            },
            "source_type": {
                "type": "string",
                "description": "信息源类型：rss（默认）或 novel-kb",
                "enum": ["rss", "novel-kb"],
            },
            "note": {"type": "string", "description": "订阅备注（subscribe 可选）"},
        },
        "required": ["action"],
    }

    def __init__(self, store: FeedStore, source_scorer: Any | None = None) -> None:
        self._store = store
        self._sub_svc = SubscriptionService(store, source_scorer=source_scorer)

    def set_scorer(self, source_scorer: Any | None) -> None:
        """事后注入 SourceScorer（main.py 在 ProactiveLoop 构建后调用）。"""
        self._sub_svc.set_scorer(source_scorer)

    async def execute(self, **kwargs: Any) -> str:
        action = str(kwargs.get("action", "")).strip().lower()
        logger.info(
            "[feed_manage] action=%s kwargs=%s",
            action,
            {k: v for k, v in kwargs.items() if k != "note"},
        )

        if action == "list":
            return self._sub_svc.list_subscriptions()

        if action == "unsubscribe":
            name = str(kwargs.get("name", "")).strip()
            if not name:
                return "错误：unsubscribe 需要 name"
            return await self._sub_svc.unsubscribe(name)

        if action == "subscribe":
            return await self._handle_subscribe(kwargs)

        return "错误：action 必须是 subscribe|list|unsubscribe"

    async def _handle_subscribe(self, kwargs: dict[str, Any]) -> str:
        name = str(kwargs.get("name", "")).strip()
        if not name:
            return "错误：subscribe 需要 name"
        url = str(kwargs.get("url", "")).strip()
        if not url:
            return "错误：subscribe 需要 url（RSS/Atom 地址）"
        return await self._sub_svc.subscribe(
            name=name,
            url=url,
            source_type=str(kwargs.get("source_type", "rss") or "rss"),
            note=kwargs.get("note"),
        )


class FeedQueryTool(Tool):
    name = "feed_query"
    description = (
        "查询订阅信息流：latest（最近条目）、search（关键词过滤）、summary（订阅概况）。"
        "若需要完整订阅清单，请改用 feed_manage(action=list)。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["latest", "search", "summary", "catalog"],
                "description": "查询动作",
            },
            "source": {
                "type": "string",
                "description": "按来源名筛选（可选，模糊匹配）",
            },
            "keyword": {"type": "string", "description": "search 用关键词"},
            "limit": {
                "type": "integer",
                "description": "返回条数（默认 5）",
                "minimum": 1,
                "maximum": 30,
            },
            "page": {
                "type": "integer",
                "description": "catalog 页码（从 1 开始）",
                "minimum": 1,
            },
            "page_size": {
                "type": "integer",
                "description": "catalog 每页条数（默认 20）",
                "minimum": 1,
                "maximum": 100,
            },
        },
        "required": ["action"],
    }

    def __init__(self, store: FeedStore, registry: FeedRegistry) -> None:
        self._svc = QueryService(store, registry)

    async def execute(self, **kwargs: Any) -> str:
        return await self._svc.query(
            action=str(kwargs.get("action", "")).strip().lower(),
            source=str(kwargs.get("source", "")).strip().lower(),
            keyword=str(kwargs.get("keyword", "")).strip().lower(),
            limit=int(kwargs.get("limit", 5) or 5),
            page=int(kwargs.get("page", 1) or 1),
            page_size=int(kwargs.get("page_size", 20) or 20),
        )
