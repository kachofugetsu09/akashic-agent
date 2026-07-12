from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.wake_proactive.context import WakeContext
from plugins.wake_proactive.state import WakeStateStore
from plugins.wake_proactive.tools import TOOL_SCHEMAS, ToolDeps, execute


def _events() -> list[dict]:
    return [
        {
            "id": "feed:a",
            "title": "新架构研究",
            "source": "Research",
            "url": "https://example.com/a",
        },
        {
            "id": "feed:b",
            "title": "普通更新",
            "source": "Feed",
            "url": "https://example.com/b",
        },
    ]


def _plan() -> dict:
    return {
        "items": [
            {
                "item_id": "feed:a",
                "initial_interest": "uncertain",
                "investigate": "both",
                "question": "是否有可复用的唤醒方法",
                "recall_query": "用户是否关心主动唤醒架构",
            },
            {
                "item_id": "feed:b",
                "initial_interest": "not_interesting",
                "investigate": "none",
            },
        ]
    }


def test_tool_schemas_are_independent_three_step_flow():
    names = [schema["function"]["name"] for schema in TOOL_SCHEMAS]
    assert names == [
        "scratchpad",
        "investigate_candidates",
        "share_content",
        "skip_content",
    ]


@pytest.mark.asyncio
async def test_scratchpad_must_cover_every_title_without_training_side_effect():
    ctx = WakeContext(content_events=_events())
    deps = ToolDeps()
    with pytest.raises(ValueError, match="must cover every title"):
        await execute("scratchpad", {"items": _plan()["items"][:1]}, ctx, deps)
    assert ctx.scratchpad == {}

    result = json.loads(await execute("scratchpad", _plan(), ctx, deps))
    assert result == {"ok": True, "planned": 2, "to_investigate": 1}
    assert ctx.terminal_action is None
    assert ctx.cited_item_ids == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("interest", "investigation"),
    [
        ("likely_interesting", "none"),
        ("uncertain", "content"),
        ("not_interesting", "recall"),
    ],
)
async def test_scratchpad_rejects_interest_investigation_contradictions(
    interest, investigation
):
    ctx = WakeContext(content_events=[{"id": "feed:a", "title": "A"}])

    with pytest.raises(ValueError):
        await execute(
            "scratchpad",
            {
                "items": [
                    {
                        "item_id": "feed:a",
                        "initial_interest": interest,
                        "investigate": investigation,
                        "recall_query": "query",
                    }
                ]
            },
            ctx,
            ToolDeps(),
        )


@pytest.mark.asyncio
async def test_investigate_candidates_fetches_and_recalls_concurrently():
    active = 0
    peak = 0

    async def fetch(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return json.dumps({"text": "正文", "url": kwargs["url"]})

    async def recall(_query):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return SimpleNamespace(
            records=[SimpleNamespace(summary="用户关心 agent 架构")]
        )

    web = MagicMock()
    web.execute = AsyncMock(side_effect=fetch)
    memory = MagicMock()
    memory.query = AsyncMock(side_effect=recall)
    ctx = WakeContext(content_events=_events())
    deps = ToolDeps(web_fetch_tool=web, memory=memory)
    await execute("scratchpad", _plan(), ctx, deps)

    result = json.loads(await execute("investigate_candidates", {}, ctx, deps))

    assert result["count"] == 1
    assert result["items"]["feed:a"]["content"]["text"] == "正文"
    assert result["items"]["feed:a"]["memory"]["hits"] == 1
    assert peak == 2
    web.execute.assert_awaited_once()
    memory.query.assert_awaited_once()


@pytest.mark.asyncio
async def test_share_content_renders_one_message_and_stable_mapping(tmp_path):
    store = WakeStateStore(tmp_path / "wake.db")
    ctx = WakeContext(session_key="telegram:1", content_events=_events())
    web = MagicMock()
    web.execute = AsyncMock(return_value=json.dumps({"text": "正文", "url": ""}))
    deps = ToolDeps(state_store=store, web_fetch_tool=web)
    await execute("scratchpad", _plan(), ctx, deps)
    await execute("investigate_candidates", {}, ctx, deps)

    result = json.loads(
        await execute(
            "share_content",
            {
                "opening": "我挑出两条里真正值得看的一条。",
                "items": [
                    {
                        "item_id": "feed:a",
                        "summary": "它把时间因素放进了唤醒判断。",
                        "why_it_matters": "可以直接用于现在的主动 agent。",
                    }
                ],
            },
            ctx,
            deps,
        )
    )

    assert ctx.terminal_action == "reply"
    assert ctx.cited_item_ids == ["feed:a"]
    assert ctx.display_event_map == {1: "feed:a"}
    assert "新架构研究" in result["message"]
    assert "1. 新架构研究" not in result["message"]
    assert "原始来源：https://example.com/a" in result["message"]
    assert ctx.source_refs[0]["display_index"] == 1
    saved = store.get(ctx.wake_id)
    assert saved is not None
    assert json.loads(saved["display_event_map_json"]) == {"1": "feed:a"}
    store.close()


@pytest.mark.asyncio
async def test_investigate_failure_is_evidence_gap_not_negative_label():
    web = MagicMock()
    web.execute = AsyncMock(side_effect=RuntimeError("timeout"))
    memory = MagicMock()
    memory.query = AsyncMock(return_value=SimpleNamespace(records=[]))
    ctx = WakeContext(content_events=_events())
    deps = ToolDeps(web_fetch_tool=web, memory=memory)
    await execute("scratchpad", _plan(), ctx, deps)

    result = json.loads(await execute("investigate_candidates", {}, ctx, deps))

    item = result["items"]["feed:a"]
    assert item["content"]["error"] == "timeout"
    assert item["memory"]["hits"] == 0
    assert ctx.terminal_action is None
    assert ctx.cited_item_ids == []

    with pytest.raises(ValueError, match="successful content evidence"):
        await execute(
            "share_content",
            {
                "items": [
                    {
                        "item_id": "feed:a",
                        "summary": "没有正文证据时不能分享",
                    }
                ]
            },
            ctx,
            deps,
        )
