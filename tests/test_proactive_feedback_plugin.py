from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from agent.plugins.manager import PluginManager
from agent.plugins.registry import plugin_registry
from agent.tools.registry import ToolRegistry
from bus.event_bus import EventBus
from plugins.proactive_feedback.db import FeedbackEvent, insert_feedback, open_db
from plugins.proactive_feedback.scorer import (
    MessageRow,
    classify_pua,
    parse_quote_parts,
    score_followup,
)


@pytest.fixture(autouse=True)
def _clean_plugin_registry():
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()
    yield
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()


def test_classify_pua_uses_strict_topic_follow_thresholds():
    assert classify_pua(0.63) == ("topic_follow", "high", "pua_high")
    assert classify_pua(0.54) == ("topic_follow", "medium", "pua_medium")
    assert classify_pua(0.53) == ("no_topic_follow", "low", "pua_low")


def test_parse_quote_parts_extracts_current_message():
    parts = parse_quote_parts(
        "【你正在回复一条历史消息】 被回复消息（来自 @akasic_bot）："
        "刚看到一个项目，挺适合你。\n"
        "【你当前新消息】 内容是什么"
    )

    assert parts.quoted_text is not None
    assert "刚看到一个项目" in parts.quoted_text
    assert parts.current_text == "内容是什么"


@pytest.mark.asyncio
async def test_explicit_quote_writes_gold_without_embedding():
    async def fail_embed(texts: list[str]) -> list[list[float]]:
        _ = texts
        raise AssertionError("explicit quote should not call embedding")

    user = _message(
        "u1",
        2,
        "user",
        "【你正在回复一条历史消息】 被回复消息（来自 @akasic_bot）："
        "刚看到一个项目，挺适合你。"
        "【你当前新消息】 内容是什么",
    )
    assistant = _message("a2", 3, "assistant", "这是一个项目介绍。")
    proactive = _message("p1", 1, "assistant", "刚看到一个项目，挺适合你。")

    scored = await score_followup(
        embed_batch=fail_embed,
        user=user,
        assistant=assistant,
        candidates=[proactive],
    )

    assert scored is not None
    assert scored.feedback_type == "explicit_quote"
    assert scored.confidence == "gold"
    assert scored.proactive.id == "p1"


@pytest.mark.asyncio
async def test_explicit_quote_matches_markdown_render_difference():
    async def fail_embed(texts: list[str]) -> list[list[float]]:
        _ = texts
        raise AssertionError("explicit quote should not call embedding")

    user = _message(
        "u1",
        4,
        "user",
        "【你正在回复一条历史消息】\n"
        "被回复消息（来自 @akasic_bot）：\n"
        "花月哥哥，今天 arXiv 上有一篇我觉得你会喜欢\n\n"
        "Humans Disengage, Reasoning Models Persist: Separating Difficulty Registration\n"
        "【你当前新消息】\n"
        "这个我也感觉到了",
    )
    assistant = _message("a2", 5, "assistant", "你这个观察很准。")
    older = _message(
        "p_old",
        2,
        "assistant",
        "花月哥哥，今天 arXiv 上有一篇我觉得你会喜欢\n\n"
        "**Humans Disengage, Reasoning Models Persist: Separating Difficulty Registration**",
    )
    latest = _message("p_latest", 3, "assistant", "花月哥哥，另一篇 NebulaExp-8B。")

    scored = await score_followup(
        embed_batch=fail_embed,
        user=user,
        assistant=assistant,
        candidates=[latest, older],
    )

    assert scored is not None
    assert scored.proactive.id == "p_old"
    assert scored.feedback_type == "explicit_quote"


@pytest.mark.asyncio
async def test_pua_score_controls_feedback_type():
    async def embed_batch(texts: list[str]) -> list[list[float]]:
        _ = texts
        return [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ]

    user = _message("u1", 2, "user", "展开讲讲")
    assistant = _message("a2", 3, "assistant", "这个项目的核心是记忆系统。")
    proactive = _message("p1", 1, "assistant", "看到一个记忆系统项目。")

    scored = await score_followup(
        embed_batch=embed_batch,
        user=user,
        assistant=assistant,
        candidates=[proactive],
    )

    assert scored is not None
    assert scored.feedback_type == "topic_follow"
    assert scored.confidence == "high"
    assert scored.matched_by == "recent_pua"


def test_insert_feedback_is_idempotent(tmp_path):
    db_path = tmp_path / "proactive_feedback.db"
    conn = open_db(db_path)
    event = FeedbackEvent(
        session_key="telegram:1",
        user_message_id="u1",
        assistant_message_id="a1",
        proactive_message_id="p1",
        feedback_type="topic_follow",
        confidence="high",
        pa_score=0.61,
        pua_score=0.65,
        lag_seconds=12,
        candidate_count=1,
        matched_by="recent_pua",
        reason="pua_high",
    )

    try:
        insert_feedback(conn, event)
        insert_feedback(conn, event)
        count = conn.execute("SELECT count(*) FROM proactive_feedback_events").fetchone()[0]
    finally:
        conn.close()

    assert count == 1


@pytest.mark.asyncio
async def test_plugin_load_registers_summary_tool(tmp_path):
    root = tmp_path / "plugins"
    source = Path(__file__).parents[1] / "plugins" / "proactive_feedback"
    shutil.copytree(source, root / "proactive_feedback")
    (tmp_path / "config.toml").write_text(
        """
[memory]
enabled = true
engine = "akasha"

[memory.embedding]
model = "text-embedding-v4"
api_key = "test-key"
base_url = "https://example.invalid/v1"
""".strip()
    )
    bus = EventBus()
    tools = ToolRegistry()
    mgr = PluginManager(
        plugin_dirs=[root],
        event_bus=bus,
        tool_registry=tools,
        workspace=tmp_path,
    )

    await mgr.load_all()
    try:
        assert "get_proactive_feedback_summary" in tools._tools
    finally:
        await mgr.terminate_all()
        await bus.aclose()


def _message(
    id_: str,
    seq: int,
    role: str,
    content: str,
) -> MessageRow:
    return MessageRow(
        id=id_,
        seq=seq,
        role=role,
        content=content,
        extra=None,
        ts="2026-06-28T00:00:00+08:00",
    )
