from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.wake_proactive.context import WakeContext
from plugins.wake_proactive.state import WakeStateStore
from plugins.wake_proactive.tools import (
    MAX_INVESTIGATION_CANDIDATES,
    TOOL_SCHEMAS,
    ToolDeps,
    execute,
)


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
                "item_id": "candidate_1",
                "initial_interest": "uncertain",
                "question": "是否有可复用的唤醒方法",
                "recall_query": "用户是否关心主动唤醒架构",
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
    scratch_items = TOOL_SCHEMAS[0]["function"]["parameters"]["properties"]["items"]
    assert scratch_items["maxItems"] == 8
    share_items = TOOL_SCHEMAS[2]["function"]["parameters"]["properties"]["items"]
    assert share_items["maxItems"] == 5


@pytest.mark.asyncio
async def test_scratchpad_rejects_too_many_candidates():
    events = [
        {"id": f"feed:{index}", "title": f"标题 {index}"}
        for index in range(MAX_INVESTIGATION_CANDIDATES + 1)
    ]
    ctx = WakeContext(content_events=events)

    with pytest.raises(ValueError, match="at most 8"):
        await execute(
            "scratchpad",
            {
                "items": [
                    {
                        "item_id": f"candidate_{index}",
                        "initial_interest": "likely_interesting",
                    }
                    for index, _event in enumerate(events, 1)
                ]
            },
            ctx,
            ToolDeps(),
        )


@pytest.mark.asyncio
async def test_scratchpad_records_only_candidates_without_training_side_effect():
    ctx = WakeContext(content_events=_events())
    deps = ToolDeps()
    result = json.loads(await execute("scratchpad", _plan(), ctx, deps))
    assert result == {"ok": True, "screened": 2, "planned": 1, "to_investigate": 1}
    assert ctx.terminal_action is None
    assert ctx.cited_item_ids == []


@pytest.mark.asyncio
async def test_wake_candidate_ref_is_canonicalized_before_consumption():
    canonical_id = "feed@github:subscriptions:fmcp_ab9a"
    source_event_id = "fmcp_ab9a"
    ctx = WakeContext(
        content_events=[
            {
                "id": canonical_id,
                "_reservoir_source_event_id": source_event_id,
                "title": "GPT-Red",
                "url": "https://example.com/gpt-red",
            }
        ]
    )
    web = MagicMock()
    web.execute = AsyncMock(return_value=json.dumps({"text": "正文", "url": ""}))
    deps = ToolDeps(web_fetch_tool=web)

    await execute(
        "scratchpad",
        {
            "items": [
                {
                    "item_id": "candidate_1",
                    "initial_interest": "likely_interesting",
                }
            ]
        },
        ctx,
        deps,
    )
    await execute("investigate_candidates", {}, ctx, deps)
    await execute(
        "share_content",
        {
            "message": "GPT-Red 发布了。",
            "items": [{"item_id": "candidate_1", "summary": "GPT-Red"}],
        },
        ctx,
        deps,
    )

    assert list(ctx.scratchpad) == [canonical_id]
    assert ctx.cited_item_ids == [canonical_id]
    assert ctx.display_event_map == {1: canonical_id}
    assert ctx.source_refs[0]["event_id"] == canonical_id


@pytest.mark.asyncio
async def test_candidate_refs_disambiguate_equal_source_event_ids():
    github_id = "feed@github:subscriptions:fmcp_same"
    lab_id = "feed@lab:subscriptions:fmcp_same"
    ctx = WakeContext(
        content_events=[
            {
                "id": github_id,
                "_reservoir_source_event_id": "fmcp_same",
                "title": "GitHub Feed",
                "content": "github 正文",
            },
            {
                "id": lab_id,
                "_reservoir_source_event_id": "fmcp_same",
                "title": "Lab Feed",
                "content": "lab 正文",
            },
        ]
    )
    deps = ToolDeps()

    await execute(
        "scratchpad",
        {
            "items": [
                {
                    "item_id": "candidate_2",
                    "initial_interest": "likely_interesting",
                }
            ]
        },
        ctx,
        deps,
    )
    investigation = json.loads(
        await execute("investigate_candidates", {}, ctx, deps)
    )
    await execute(
        "share_content",
        {
            "message": "Lab Feed 发布了。",
            "items": [{"item_id": "candidate_2", "summary": "Lab Feed"}],
        },
        ctx,
        deps,
    )

    assert list(ctx.scratchpad) == [lab_id]
    assert list(investigation["items"]) == ["candidate_2"]
    assert ctx.cited_item_ids == [lab_id]


@pytest.mark.asyncio
async def test_source_event_id_is_not_accepted_as_candidate_ref():
    ctx = WakeContext(content_events=_events())

    with pytest.raises(ValueError, match="unknown item_id"):
        await execute(
            "scratchpad",
            {
                "items": [
                    {
                        "item_id": "feed:a",
                        "initial_interest": "likely_interesting",
                    }
                ]
            },
            ctx,
            ToolDeps(),
        )


@pytest.mark.asyncio
async def test_empty_scratchpad_can_investigate_then_skip():
    ctx = WakeContext(content_events=_events())
    deps = ToolDeps()

    result = json.loads(await execute("scratchpad", {"items": []}, ctx, deps))
    investigation = json.loads(await execute("investigate_candidates", {}, ctx, deps))
    skipped = json.loads(
        await execute("skip_content", {"reason": "没有候选"}, ctx, deps)
    )

    assert result["screened"] == 2
    assert investigation == {"items": {}, "count": 0}
    assert skipped["decision"] == "skip"


@pytest.mark.asyncio
async def test_scratchpad_rejects_invalid_interest():
    ctx = WakeContext(content_events=[{"id": "feed:a", "title": "A"}])

    with pytest.raises(ValueError):
        await execute(
            "scratchpad",
            {
                "items": [
                    {
                        "item_id": "candidate_1",
                        "initial_interest": "maybe",
                        "recall_query": "query",
                    }
                ]
            },
            ctx,
            ToolDeps(),
        )


@pytest.mark.asyncio
async def test_scratchpad_derives_investigation_and_recall_query():
    ctx = WakeContext(content_events=[{"id": "feed:a", "title": "Agent memory"}])

    await execute(
        "scratchpad",
        {"items": [{"item_id": "candidate_1", "initial_interest": "uncertain"}]},
        ctx,
        ToolDeps(),
    )

    assert ctx.scratchpad["feed:a"].investigate == "both"
    assert ctx.scratchpad["feed:a"].recall_query == "Agent memory"


@pytest.mark.asyncio
async def test_scratchpad_ignores_explicit_not_interesting_item():
    ctx = WakeContext(content_events=_events())

    await execute(
        "scratchpad",
        {
            "items": [
                {"item_id": "candidate_1", "initial_interest": "not_interesting"}
            ]
        },
        ctx,
        ToolDeps(),
    )

    assert ctx.screening_completed is True
    assert ctx.scratchpad == {}


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
    assert result["items"]["candidate_1"]["content"]["text"] == "正文"
    assert result["items"]["candidate_1"]["memory"]["hits"] == 1
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
                        "item_id": "candidate_1",
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
    assert "它把时间因素放进了唤醒判断。" in result["message"]
    assert "新架构研究" not in result["message"]
    assert "来源：https://example.com/a" in result["message"]
    assert ctx.source_refs[0]["display_index"] == 1
    saved = store.get(ctx.wake_id)
    assert saved is not None
    assert json.loads(saved["display_event_map_json"]) == {"1": "feed:a"}
    store.close()


@pytest.mark.asyncio
async def test_share_content_accepts_natural_message_with_evidence_sources():
    ctx = WakeContext(content_events=_events())
    web = MagicMock()
    web.execute = AsyncMock(return_value=json.dumps({"text": "正文", "url": ""}))
    deps = ToolDeps(web_fetch_tool=web)
    await execute("scratchpad", _plan(), ctx, deps)
    await execute("investigate_candidates", {}, ctx, deps)

    result = json.loads(
        await execute(
            "share_content",
            {
                "message": "刚看到一个挺对你胃口的设计：它把时间本身放进了唤醒判断。",
                "items": [{"item_id": "candidate_1", "summary": "不会显示的模板摘要"}],
            },
            ctx,
            deps,
        )
    )

    assert "挺对你胃口" in result["message"]
    assert "不会显示的模板摘要" not in result["message"]
    assert "来源：https://example.com/a" in result["message"]


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

    assert result == {"items": {}, "count": 0}
    item = ctx.investigation_results["feed:a"]
    assert item["content"]["error"] == "timeout"
    assert item["memory"]["hits"] == 0
    assert ctx.terminal_action is None
    assert ctx.cited_item_ids == []

    shared = json.loads(
        await execute(
            "share_content",
            {
                "items": [
                    {
                        "item_id": "candidate_1",
                        "summary": "没有正文证据时不能分享",
                    }
                ]
            },
            ctx,
            deps,
        )
    )
    assert shared["decision"] == "skip"
    assert ctx.terminal_action == "skip"
