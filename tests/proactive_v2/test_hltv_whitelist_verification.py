"""
tests/proactive_v2/test_hltv_whitelist_verification.py

真实 LLM 场景测试：验证当 HLTV 相关内容涉及非固定白名单队伍时，
agent 必须通过 web_fetch 查询 HLTV 排名来确认是否在 top15，
而不能直接凭训练数据"常识"跳过查询。

复现的 bug：
- 数据库中有一条 B8 相关的 HLTV 新闻
- B8 不在固定白名单（TyLoo / BC.Game / 100 Thieves）
- 规则要求通过 web_fetch 查 HLTV top15 确认
- 旧行为：LLM 凭训练记忆认为 B8 不是 top15，跳过 web_fetch
- 预期行为：LLM 必须调用 web_fetch 查询 HLTV 排名 URL

运行方式（需真实 LLM + 网络权限）：
    akashic_RUN_HLTV_WHITELIST_TEST=1 pytest tests/proactive_v2/test_hltv_whitelist_verification.py -v -s
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from proactive_v2.gateway import GatewayDeps
from proactive_v2.tools import ToolDeps
from tests.proactive_v2.conftest import FakeLLM, FakeRng, FakeStateStore, make_agent_tick


# ── 环境检查 ───────────────────────────────────────────────────────────────

_RUN = bool(int(os.environ.get("akashic_RUN_HLTV_WHITELIST_TEST", "0")))
_SKIP_REASON = "akashic_RUN_HLTV_WHITELIST_TEST 未开启"

_PROACTIVE_CONTEXT_WITH_HLTV_RULE = """\
# Proactive Context

## CS2 新闻推送过滤规则（动态逻辑）

**触发条件**：当检测到 CS2 相关新闻/赛事动态时，执行以下过滤。

**白名单队伍**：
1. **HLTV Top 15**：每周一更新，取"最近一个周一"的排名数据。
   - 计算逻辑：若今天是周一，用今天数据；若是周二至周日，向前回溯到本周一的数据。
   - 通过webfetch查询源：`https://www.hltv.org/ranking/teams/2026/march/16`（日期动态替换）。
2. **固定关注战队**（无论排名如何）：
   - TyLoo（中国赛区唯一关注）
   - BC.Game
   - 100 Thieves

**过滤逻辑**：
- 仅推送上述白名单队伍及其选手相关的新闻、赛事结果、转会消息。
- 其他未在白名单中的队伍（无论排名多少、是否有明星选手）一律视为"野鸡队"过滤。

**执行注意**：
- 此规则为动态逻辑，但不要为了执行规则而对每条 CS 候选都实时查询 HLTV 排名。
- 只有当当前候选内容看起来可能相关、且仅凭标题/来源/常识无法确认是否属于白名单时，才查询最近一个周一的 HLTV 排名。
- 如果标题或来源已经足够说明该内容明显不在关注范围内，直接过滤，不要额外搜索。
- 选手新闻需反向校验其所属队伍是否在白名单内；同样只在归属不确定时再查询。
- 规则的目标是过滤候选内容，不是主动扩展候选池；不要因为执行白名单规则而去搜索新的新闻主题。
"""

# 模拟 HLTV 排名页返回（top15 里没有 B8）
_FAKE_HLTV_RANKING_PAGE = """
HLTV.org World Ranking
1. NAVI
2. FaZe
3. G2
4. Vitality
5. Liquid
6. Cloud9
7. MOUZ
8. Falcons
9. Complexity
10. NIP
11. Astralis
12. Spirit
13. HEROIC
14. BIG
15. 9z
"""

# 模拟 B8 新闻文章内容（刻意不提 tier / 排名，防止 LLM 凭文章内容跳过 ranking 查询）
_FAKE_B8_ARTICLE = """
B8 announces new lineup for 2026 season
B8 Esports has revealed their new roster for the upcoming competitive season.
The organization announced several player signings and departures.
The team plans to compete in upcoming European CS2 tournaments.
"""

# 模拟 B8 HLTV 新闻 URL（真实格式）
_B8_NEWS_URL = "https://www.hltv.org/news/41199/b8-announces-new-lineup-for-2026"


# ── Fake web_fetch 工具（记录调用 + 返回假数据）─────────────────────────────

class RecordingWebFetch:
    """记录所有 web_fetch 调用，并返回对应假数据。"""

    def __init__(self):
        self.called_urls: list[str] = []

    async def execute(self, url: str, format: str = "text") -> str:
        self.called_urls.append(url)

        if "hltv.org/ranking" in url:
            return json.dumps({"text": _FAKE_HLTV_RANKING_PAGE, "url": url})
        if "hltv.org/news" in url:
            return json.dumps({"text": _FAKE_B8_ARTICLE, "url": url})
        return json.dumps({"text": f"(fake content for {url})", "url": url})

    def called_hltv_ranking(self) -> bool:
        return any("hltv.org/ranking" in u for u in self.called_urls)


# ── 真实 memory 构建 ──────────────────────────────────────────────────────

_WORKSPACE = Path.home() / ".akashic" / "workspace"
_MEMORY_DB = _WORKSPACE / "memory" / "memory2.db"


def _make_fake_memory():
    """
    优先从真实 memory2.db 构建 memory port；
    若环境不可用则退化为 SimpleNamespace（retrieve_related 返回空列表）。
    """
    if not _RUN or not _MEMORY_DB.exists():
        from types import SimpleNamespace
        return SimpleNamespace(
            retrieve_related=AsyncMock(return_value=[]),
            read_long_term=lambda: "",
        )

    # 用真实 db 构建，需要先把 openai stub 替换成真实包
    _patch_real_openai()

    from agent.config import load_config
    from agent.provider import LLMProvider
    from agent.tools.registry import ToolRegistry
    from bootstrap.memory import build_memory_runtime
    from core.net.http import SharedHttpResources

    cfg = load_config("config.toml")
    cfg.memory_v2.db_path = str(_MEMORY_DB)

    provider = LLMProvider(api_key=cfg.api_key, base_url=cfg.base_url)
    light_provider = LLMProvider(
        api_key=(cfg.light_api_key or cfg.api_key),
        base_url=(cfg.light_base_url or cfg.base_url),
    )
    http_resources = SharedHttpResources()
    memory_runtime = build_memory_runtime(
        config=cfg,
        workspace=_WORKSPACE,
        tools=ToolRegistry(),
        provider=provider,
        light_provider=light_provider,
        http_resources=http_resources,
        observe_writer=None,
    )
    return memory_runtime.port


# ── 真实 LLM 构建 ──────────────────────────────────────────────────────────

def _patch_real_openai() -> None:
    for name in list(sys.modules):
        if name == "openai" or name.startswith("openai."):
            del sys.modules[name]
    real_openai = importlib.import_module("openai")
    import agent.provider as provider_mod
    provider_mod.AsyncOpenAI = real_openai.AsyncOpenAI


def _build_real_llm_fn():
    """从 config.toml 读取 API key，构建真实 _llm_fn。"""
    from agent.config import load_config
    from agent.provider import LLMProvider

    cfg = load_config("config.toml")
    provider = LLMProvider(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
    )
    model = cfg.model
    max_tokens = getattr(cfg, "max_tokens", 2048)

    async def _llm_fn(messages: list[dict], schemas: list[dict]) -> dict | None:
        resp = await provider.chat(
            messages=messages,
            tools=schemas,
            model=model,
            max_tokens=max_tokens,
            tool_choice="auto",
        )
        if not resp.tool_calls:
            return None
        tc = resp.tool_calls[0]
        return {"id": tc.id, "name": tc.name, "input": tc.arguments}

    return _llm_fn


# ── 测试：真实 LLM 场景 ────────────────────────────────────────────────────

@pytest.mark.skipif(not _RUN, reason=_SKIP_REASON)
@pytest.mark.asyncio
async def test_b8_hltv_news_triggers_web_fetch_for_ranking():
    """
    B8 HLTV 新闻送入 agent，agent 应调用 web_fetch 查询 HLTV 排名 URL，
    而不是凭训练数据直接跳过排名确认步骤。

    B8 不在固定白名单，但 LLM 训练数据里"知道" B8 不是 top15。
    旧行为（bug）：跳过 web_fetch，直接 mark_not_interesting。
    预期行为（fix）：调用 web_fetch 查 HLTV ranking，确认后再分类。
    """
    _patch_real_openai()

    # B8 新闻 content item
    b8_event = {
        "event_id": "hltv_b8_lineup_2026",
        "id": "hltv_b8_lineup_2026",
        "ack_server": "feed-mcp",
        "url": _B8_NEWS_URL,
        "title": "B8 announces new lineup for 2026 season",
        "source_name": "HLTV",
        "published_at": "2026-03-20T10:00:00Z",
    }

    web_fetch = RecordingWebFetch()
    fake_memory = _make_fake_memory()

    tick = make_agent_tick(
        llm_fn=_build_real_llm_fn(),
        tool_deps=ToolDeps(
            recent_chat_fn=AsyncMock(return_value=[]),
            web_fetch_tool=web_fetch,
            memory=fake_memory,
            max_chars=8000,
        ),
        gateway_deps=GatewayDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(return_value=[b8_event]),
            context_fn=AsyncMock(return_value=[]),
            web_fetch_tool=web_fetch,
            max_chars=8000,
        ),
        workspace_context_fn=lambda: _PROACTIVE_CONTEXT_WITH_HLTV_RULE,
    )

    await tick.tick()

    # ── 断言 ──────────────────────────────────────────────────────────────
    print(f"\n[test] web_fetch called URLs: {web_fetch.called_urls}")
    print(f"[test] terminal_action: {tick.last_ctx.terminal_action}")
    print(f"[test] discarded: {tick.last_ctx.discarded_item_ids}")
    print(f"[test] interesting: {tick.last_ctx.interesting_item_ids}")

    assert web_fetch.called_hltv_ranking(), (
        "bug 重现：agent 未调用 web_fetch 查询 HLTV 排名 URL。"
        f"实际调用的 URLs: {web_fetch.called_urls}\n"
        "期望至少有一次 hltv.org/ranking 的 web_fetch 调用。"
    )

    # B8 不在 top15，应该被过滤
    assert tick.last_ctx.terminal_action in ("skip", None) or (
        tick.last_ctx.terminal_action == "reply"
        and "feed-mcp:hltv_b8_lineup_2026" not in (tick.last_ctx.cited_item_ids or [])
    ), "B8 新闻不应该被 finish_turn(reply) 引用"


@pytest.mark.skipif(not _RUN, reason=_SKIP_REASON)
@pytest.mark.asyncio
async def test_b8_hltv_news_marked_not_interesting_after_ranking_check():
    """
    验证 agent 查询 HLTV 排名后，确认 B8 不在 top15，正确标记为 not_interesting。
    """
    _patch_real_openai()

    b8_event = {
        "event_id": "hltv_b8_lineup_2026",
        "id": "hltv_b8_lineup_2026",
        "ack_server": "feed-mcp",
        "url": _B8_NEWS_URL,
        "title": "B8 announces new lineup for 2026 season",
        "source_name": "HLTV",
        "published_at": "2026-03-20T10:00:00Z",
    }

    web_fetch = RecordingWebFetch()
    fake_memory = _make_fake_memory()

    tick = make_agent_tick(
        llm_fn=_build_real_llm_fn(),
        tool_deps=ToolDeps(
            recent_chat_fn=AsyncMock(return_value=[]),
            web_fetch_tool=web_fetch,
            memory=fake_memory,
            max_chars=8000,
        ),
        gateway_deps=GatewayDeps(
            alert_fn=AsyncMock(return_value=[]),
            feed_fn=AsyncMock(return_value=[b8_event]),
            context_fn=AsyncMock(return_value=[]),
            web_fetch_tool=web_fetch,
            max_chars=8000,
        ),
        workspace_context_fn=lambda: _PROACTIVE_CONTEXT_WITH_HLTV_RULE,
    )

    await tick.tick()

    key = "feed-mcp:hltv_b8_lineup_2026"
    assert key in tick.last_ctx.discarded_item_ids, (
        f"B8 新闻应被标记为 not_interesting（discarded），"
        f"实际 discarded={tick.last_ctx.discarded_item_ids}"
    )


# ── 确定性单元测试（不调用真实 LLM）─────────────────────────────────────────

@pytest.mark.asyncio
async def test_system_prompt_contains_training_data_warning():
    """
    验证系统提示里包含"训练数据记忆"不能替代 web_fetch 的规则。
    这是一个确定性测试，不依赖真实 LLM。
    """
    from proactive_v2.agent_tick import AgentTick, AgentTickContext
    from proactive_v2.gateway import GatewayResult
    from proactive_v2.config import ProactiveConfig
    from datetime import datetime, timezone
    from unittest.mock import MagicMock

    # 构建一个最小化的 AgentTick 来获取系统提示
    tick = make_agent_tick(llm_fn=None)

    ctx = AgentTickContext(session_key="test", now_utc=datetime.now(timezone.utc))
    gw = GatewayResult()

    prompt = tick._build_system_prompt(ctx, gw)

    assert "训练数据" in prompt, (
        "系统提示应包含'训练数据'相关的规则，防止 LLM 用训练记忆跳过 web_fetch 验证"
    )
    assert "web_fetch" in prompt.lower() or "web_fetch" in prompt, (
        "系统提示应包含 web_fetch 工具相关说明"
    )


@pytest.mark.asyncio
async def test_system_prompt_rule8_covers_ranking_verification():
    """
    验证规则第8条明确说明排名/赛况等时效性数据必须 web_fetch，不能用训练记忆。
    """
    from proactive_v2.context import AgentTickContext
    from proactive_v2.gateway import GatewayResult
    from datetime import datetime, timezone

    tick = make_agent_tick(llm_fn=None)
    ctx = AgentTickContext(session_key="test", now_utc=datetime.now(timezone.utc))
    gw = GatewayResult()
    prompt = tick._build_system_prompt(ctx, gw)

    # 规则8的核心断言
    assert "训练数据记忆" in prompt, "规则8应明确提到训练数据记忆不等于常识"
    assert "时效性数据" in prompt, "规则8应涵盖排名等时效性数据的处理要求"
    assert "必须步骤" in prompt, "规则8应明确 web_fetch 是必须步骤而非可选"
