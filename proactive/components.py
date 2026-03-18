from __future__ import annotations

import json
import logging
from typing import Any, Callable

from agent.tools.web_fetch import WebFetchTool
from core.net.http import get_default_http_requester
from feeds.base import FeedItem
from prompts.proactive import build_compose_prompt_messages
from proactive.composer import (
    Composer,
    _build_proactive_prompt_context,
    build_proactive_memory_query,
    build_proactive_preference_hyde_prompt,
    build_proactive_preference_query,
    classify_content_quality,
)
from proactive.judge import (
    Judge,
    MessageDeduper,
    ProactiveJudgeResult,
)
from proactive.sender import ProactiveSendMeta, ProactiveSender, ProactiveSourceRef

logger = logging.getLogger(__name__)


class ProactiveJudge:
    """兼容旧接口：对外仍暴露 compose + judge，同步桥接到拆分后的服务。"""

    def __init__(
        self,
        *,
        provider,
        model: str,
        max_tokens: int,
        format_items: Callable[[list[FeedItem]], str],
        format_recent: Callable[[list[dict]], str],
        cfg: Any,
    ) -> None:
        self._composer = Composer(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            format_items=format_items,
            format_recent=format_recent,
        )
        self._judge = Judge(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            format_recent=format_recent,
            cfg=cfg,
        )

    async def compose_for_judge(
        self,
        *,
        items: list[FeedItem],
        recent: list[dict],
        preference_block: str = "",
        no_content_token: str = "<no_content/>",
    ):
        items = await self._enrich_items_for_compose(items)
        prompt_context = _build_proactive_prompt_context(
            items=items,
            recent=recent,
            format_items=self._composer._format_items,
            format_recent=self._composer._format_recent,
        )
        system_msg, user_msg = build_compose_prompt_messages(
            prompt_context=prompt_context,
            preference_block=preference_block,
            no_content_token=no_content_token,
        )
        response = await self._composer._provider.chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            tools=[],
            model=self._composer._model,
            max_tokens=min(512, self._composer._max_tokens),
        )
        text = (response.content or "").strip()
        return no_content_token if text.startswith(no_content_token) else text

    async def judge_message(self, **kwargs):
        return await self._judge.judge_message(**kwargs)

    def pre_compose_veto(self, **kwargs):
        return self._judge.pre_compose_veto(**kwargs)

    async def _enrich_items_for_compose(self, items: list[FeedItem]) -> list[FeedItem]:
        candidates = [
            item
            for item in items[:2]
            if item.url and classify_content_quality(item) != "full"
        ]
        if not candidates:
            return items
        fetcher = WebFetchTool(get_default_http_requester("external_default"))
        for item in candidates:
            try:
                raw = await fetcher.execute(url=item.url, format="text", timeout=8)
                data = json.loads(raw or "{}")
                text = str(data.get("text", "") or "").strip()
            except Exception as exc:
                logger.info("[compose] enrich_item_failed url=%s err=%s", item.url, exc)
                setattr(item, "content_status", "fetch_failed")
                continue
            if len(text) <= 400:
                setattr(item, "content_status", "fetch_failed")
                continue
            item.content = text[:4000]
            setattr(item, "content_status", "fetched")
        return items


ProactiveMessageDeduper = MessageDeduper
ComposerService = Composer

__all__ = [
    "Composer",
    "ComposerService",
    "MessageDeduper",
    "ProactiveJudge",
    "ProactiveJudgeResult",
    "ProactiveMessageDeduper",
    "ProactiveSendMeta",
    "ProactiveSender",
    "ProactiveSourceRef",
    "build_proactive_memory_query",
    "build_proactive_preference_hyde_prompt",
    "build_proactive_preference_query",
    "classify_content_quality",
]
