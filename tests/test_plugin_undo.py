from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from agent.plugins.context import PluginContext, PluginKVStore
from plugins.plugin_undo.plugin import PluginUndo, UndoCommandModule


class _SessionManager:
    def __init__(self) -> None:
        self.undo_calls: list[dict[str, object]] = []

    async def undo_last_turn(self, session_key: str, **kwargs):
        resolver = kwargs.get("rollback_source_resolver")
        rollback_source_ids = resolver(["cli:1:0", "cli:1:1", "cli:1:2"]) if callable(resolver) else []
        self.undo_calls.append({"session_key": session_key, **kwargs})
        return SimpleNamespace(
            deleted_ids=["cli:1:0", "cli:1:1", "cli:1:2"],
            rollback_source_ids=rollback_source_ids,
            last_consolidated_before=3,
            last_consolidated_after=0,
        )


class _MemoryEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def undo_by_message_sources(
        self,
        message_ids: list[str],
        *,
        dry_run: bool = False,
    ) -> dict[str, object]:
        self.calls.append({"message_ids": list(message_ids), "dry_run": dry_run})
        return {
            "affected_ids": ["mem1"],
            "restored_ids": ["old1"],
            "rollback_source_ids": ["cli:1:0", "cli:1:1", "cli:1:2"],
        }


@pytest.mark.asyncio
async def test_undo_command_aborts_without_running_llm(tmp_path):
    plugin = PluginUndo()
    session_manager = _SessionManager()
    memory_engine = _MemoryEngine()
    plugin.context = PluginContext(
        event_bus=None,
        tool_registry=None,
        plugin_id="plugin_undo",
        plugin_dir=tmp_path,
        kv_store=PluginKVStore(tmp_path / ".kv.json"),
        session_manager=session_manager,
        memory_engine=memory_engine,
    )
    module = UndoCommandModule(plugin)
    state = SimpleNamespace(
        session_key="cli:1",
        session=object(),
        msg=SimpleNamespace(
            content="/undo",
            channel="cli",
            chat_id="1",
            timestamp=datetime.now(),
        ),
    )
    frame = SimpleNamespace(input=state, slots={"session:session": state.session})

    result = await module.run(frame)

    ctx = result.slots["session:ctx"]
    assert ctx.abort is True
    assert "已撤销上一轮对话" in ctx.abort_reply
    assert callable(session_manager.undo_calls[0]["rollback_source_resolver"])
    assert [call["dry_run"] for call in memory_engine.calls] == [True, False]


def test_undo_plugin_registers_telegram_command():
    assert PluginUndo().telegram_bot_commands() == [("undo", "撤销上一轮对话")]
