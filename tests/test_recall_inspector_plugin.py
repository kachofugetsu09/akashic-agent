from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from agent.lifecycle.types import AfterToolResultCtx, BeforeTurnCtx
from agent.plugins.context import PluginContext, PluginKVStore
from plugins.recall_inspector.dashboard import RecallInspectorDashboardReader
from plugins.recall_inspector.plugin import RecallInspector


@pytest.mark.asyncio
async def test_recall_inspector_records_context_and_recall(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins" / "recall_inspector"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text("", encoding="utf-8")

    plugin = RecallInspector()
    plugin.context = PluginContext(
        event_bus=None,
        tool_registry=None,
        plugin_id="recall_inspector",
        plugin_dir=plugin_dir,
        kv_store=PluginKVStore(plugin_dir / ".kv.json"),
    )
    await plugin.initialize()

    ctx = BeforeTurnCtx(
        session_key="cli:1",
        channel="cli",
        chat_id="1",
        content="还记得我喜欢什么吗",
        timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
        skill_names=[],
        retrieved_memory_block="- [m1] 用户喜欢低压力创作\n- [m2] 用户偏好中文回复",
        retrieval_trace_raw={"route_decision": "RETRIEVE"},
        history_messages=(),
    )
    plugin.record_context_prepare(ctx)
    await plugin.record_recall_memory(
        AfterToolResultCtx(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            tool_name="recall_memory",
            arguments={"query": "用户偏好"},
            result=json.dumps(
                {
                    "items": [
                        {
                            "id": "m3",
                            "memory_type": "preference",
                            "summary": "用户喜欢简单方案",
                            "score": 0.7,
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            status="success",
        )
    )

    reader = RecallInspectorDashboardReader(plugin_dir)
    items, total = reader.list_turns()

    assert total == 1
    assert items[0]["context_prepare_count"] == 2
    assert items[0]["recall_memory_count"] == 1
    assert items[0]["context_prepare"]["items"][0]["id"] == "m1"
    assert items[0]["recall_memory_calls"][0]["items"][0]["id"] == "m3"


@pytest.mark.asyncio
async def test_recall_inspector_uses_workspace_data_path(tmp_path: Path) -> None:
    plugin_dir = tmp_path / "plugins" / "recall_inspector"
    workspace = tmp_path / "workspace"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.py").write_text("", encoding="utf-8")

    plugin = RecallInspector()
    plugin.context = PluginContext(
        event_bus=None,
        tool_registry=None,
        plugin_id="recall_inspector",
        plugin_dir=plugin_dir,
        kv_store=PluginKVStore(plugin_dir / ".kv.json"),
        workspace=workspace,
    )
    await plugin.initialize()

    plugin.record_context_prepare(
        BeforeTurnCtx(
            session_key="cli:1",
            channel="cli",
            chat_id="1",
            content="测试召回记录隔离",
            timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
            skill_names=[],
            retrieved_memory_block="",
            retrieval_trace_raw=None,
            history_messages=(),
        )
    )

    reader = RecallInspectorDashboardReader(plugin_dir, workspace)
    _, total = reader.list_turns()

    assert total == 1
    assert (workspace / "observe" / "recall_inspector.jsonl").exists()
    assert not (plugin_dir / ".data" / "recall_turns.jsonl").exists()


def test_recall_inspector_reader_reports_unavailable(tmp_path: Path) -> None:
    reader = RecallInspectorDashboardReader(tmp_path)

    assert reader.get_overview() == {"available": False, "total": 0, "latest_at": None}
    assert reader.list_turns() == ([], 0)
    assert reader.get_turn("missing") is None
