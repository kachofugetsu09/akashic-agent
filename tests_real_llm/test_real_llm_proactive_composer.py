from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from agent.config import load_config
from agent.provider import LLMProvider
from core.net.http import (
    SharedHttpResources,
    clear_default_shared_http_resources,
    configure_default_shared_http_resources,
)
from feeds.base import FeedItem
from proactive.components import ProactiveMessageComposer
from proactive.loop_helpers import _format_items, _format_recent


def _item(title: str, content: str, url: str, minutes_ago: int) -> FeedItem:
    return FeedItem(
        source_name="PC Gamer UK - Games",
        source_type="rss",
        title=title,
        content=content,
        url=url,
        author=None,
        published_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    )


@pytest.mark.asyncio
async def test_real_llm_composer_aggregates_updates_with_per_item_links():
    cfg_path = Path("/mnt/data/coding/akasic-agent/config.json")
    cfg = load_config(str(cfg_path))
    provider = LLMProvider(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        system_prompt=cfg.system_prompt,
        extra_body=cfg.extra_body,
        request_timeout_s=60,
        max_retries=0,
    )
    resources = SharedHttpResources()
    configure_default_shared_http_resources(resources)
    try:
        composer = ProactiveMessageComposer(
            provider=provider,
            model=cfg.model,
            max_tokens=700,
            format_items=_format_items,
            format_recent=_format_recent,
            collect_global_memory=lambda: (
                "用户喜欢有明确信息密度的游戏资讯，不排斥我主动总结同一主题的多条更新。"
            ),
            max_tool_iterations=4,
            fitbit_url=cfg.proactive.fitbit_url,
        )
        items = [
            _item(
                "Banquet for Fools release date confirmed",
                "PC Gamer 报道 Banquet for Fools 的发售窗口已经确认，开发者同时放出一批新截图。",
                "https://www.pcgamer.com/banquet-release",
                8,
            ),
            _item(
                "Banquet for Fools demo is out now",
                "试玩版已经上线，首章内容可直接体验，PC Gamer 还提到战斗系统比预期更复杂。",
                "https://www.pcgamer.com/banquet-demo",
                5,
            ),
            _item(
                "Banquet for Fools gets combat deep dive",
                "最新深度稿细讲了构筑、Boss 节奏和资源管理，整体风格不像一次性快讯，更像连续放料。",
                "https://www.pcgamer.com/banquet-combat",
                2,
            ),
        ]

        message = await composer.compose_message(
            items=items,
            recent=[],
            decision_signals={
                "candidate_items": len(items),
                "minutes_since_last_proactive": 180,
                "user_replied_after_last_proactive": True,
                "proactive_sent_24h": 0,
            },
            retrieved_memory_block="",
            preference_block="",
        )

        assert message.strip()
        assert len(message) <= 400
        assert "https://www.pcgamer.com/banquet-release" in message
        assert "https://www.pcgamer.com/banquet-demo" in message
        assert "https://www.pcgamer.com/banquet-combat" in message
    finally:
        clear_default_shared_http_resources(resources)
        await resources.aclose()
