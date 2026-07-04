from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.plugins.manager import PluginManager
from agent.plugins.registry import plugin_registry
from agent.tools.registry import ToolRegistry
from proactive_v2.frame import new_proactive_frame
from plugins.emotion.db import get_state, open_db
from plugins.emotion.plugin import EmotionPlugin
from plugins.proactive_feedback.db import FeedbackEvent
from plugins.proactive_feedback.events import ProactiveFeedbackRecorded
from bus.event_bus import EventBus
from bus.events_lifecycle import DriftFinished
from agent.plugins.jobs import PluginJobContext


def _feedback(feedback_type: str = "topic_follow", confidence: str = "medium") -> FeedbackEvent:
    return FeedbackEvent(
        session_key="cli:test",
        user_message_id="u1",
        assistant_message_id="a1",
        proactive_message_id="p1",
        feedback_type=feedback_type,
        confidence=confidence,
        pa_score=0.5,
        pua_score=0.6,
        lag_seconds=60,
        candidate_count=1,
        matched_by="recent_pua",
        reason="test",
    )


class _FakeLlm:
    async def generate_text(self, **kwargs: Any) -> str:
        assert "effect=boost" in str(kwargs["prompt"])
        return "# Proactive Context\n\n- 提高模型发布解读类推送优先级。\n"


def test_emotion_updates_dominance_from_feedback(tmp_path: Path):
    db_path = tmp_path / "emotion" / "emotion.db"
    plugin = EmotionPlugin()
    plugin._db_path = db_path

    plugin._on_feedback_recorded(
        ProactiveFeedbackRecorded(event_id=1, feedback=_feedback())
    )

    conn = open_db(db_path)
    try:
        state = get_state(conn)
    finally:
        conn.close()
    assert state.valence > 0
    assert state.dominance > 0


def test_emotion_effect_records_threshold_and_prompt(tmp_path: Path):
    db_path = tmp_path / "emotion" / "emotion.db"
    plugin = EmotionPlugin()
    plugin._db_path = db_path
    plugin._on_feedback_recorded(
        ProactiveFeedbackRecorded(event_id=1, feedback=_feedback("explicit_quote", "gold"))
    )

    frame = new_proactive_frame("cli:test")
    frame.slots["proactive:started_at"] = datetime.now(timezone.utc)
    frame.slots["proactive:base_judge_send_threshold"] = 0.60
    frame.slots["proactive:last_user_at"] = None

    effect = plugin.build_proactive_prompt_effect(frame)

    assert effect is not None
    assert effect["provider_name"] == "emotion"
    assert "当前 VAD" in str(effect["prompt_section"])
    assert "final_threshold" in effect["metadata"]


async def test_emotion_plugin_listens_to_feedback_event(tmp_path: Path):
    plugin = EmotionPlugin()
    plugin.context = cast(Any, SimpleNamespace(workspace=tmp_path, event_bus=EventBus()))
    await plugin.initialize()
    await plugin.context.event_bus.fanout(
        ProactiveFeedbackRecorded(event_id=1, feedback=_feedback("explicit_quote", "gold"))
    )

    conn = open_db(tmp_path / "emotion" / "emotion.db")
    try:
        state = get_state(conn)
    finally:
        conn.close()
    assert state.dominance > 0


@pytest.mark.asyncio
async def test_emotion_merges_pending_after_feedback_drift_finished(tmp_path: Path):
    pending_path = tmp_path / "proactive_pending.md"
    context_path = tmp_path / "PROACTIVE_CONTEXT.md"
    pending_path.write_text(
        '# Proactive Pending\n\n'
        '## Batch feedback#1-feedback#1\n\n'
        '- [ ] effect=boost confidence=medium topic="模型发布" '
        'granularity="不扩大到所有科技新闻" inference="用户追问细节" '
        'action="提高优先级" evidence=feedback#1 user_message_id=u1\n',
        encoding="utf-8",
    )
    context_path.write_text("# Proactive Context\n\n- 旧规则。\n", encoding="utf-8")
    plugin = EmotionPlugin()
    plugin.context = cast(Any, SimpleNamespace(workspace=tmp_path))

    await plugin.merge_proactive_pending(
        PluginJobContext(
            plugin_id="emotion",
            event=DriftFinished(
                session_key="cli:test",
                skill_name="emotion:feedback-preference-context",
                status="completed",
                briefing="done",
                message_result="silent",
                timestamp=datetime.now(timezone.utc),
            ),
            reason="event",
            llm=_FakeLlm(),
            plugin_context=plugin.context,
            triggered_at=datetime.now(timezone.utc),
        )
    )

    assert context_path.read_text(encoding="utf-8") == (
        "# Proactive Context\n\n- 提高模型发布解读类推送优先级。\n"
    )
    assert pending_path.read_text(encoding="utf-8") == ""


@pytest.mark.asyncio
async def test_emotion_and_feedback_plugins_load_together(tmp_path: Path):
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()
    plugin_root = tmp_path / "plugins"
    source_root = Path(__file__).parents[1] / "plugins"
    shutil.copytree(source_root / "proactive_feedback", plugin_root / "proactive_feedback")
    shutil.copytree(source_root / "emotion", plugin_root / "emotion")
    bus = EventBus()
    tools = ToolRegistry()
    manager = PluginManager(
        plugin_dirs=[plugin_root],
        event_bus=bus,
        tool_registry=tools,
        workspace=tmp_path,
    )

    await manager.load_all()

    assert len(manager.proactive_modules) == 1
    assert tools.get_tool("get_emotion_state") is not None
    assert tools.get_tool("get_proactive_feedback_summary") is not None
    await manager.terminate_all()
