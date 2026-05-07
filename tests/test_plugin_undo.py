from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from agent.plugins.context import PluginContext, PluginKVStore
from plugins.plugin_undo.plugin import PluginUndo, UndoCommandModule
from session.manager import SessionManager


class _MemoryEngine:
    def __init__(self, *, fail_real_undo: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_real_undo = fail_real_undo

    def undo_by_message_sources(
        self,
        message_ids: list[str],
        *,
        dry_run: bool = False,
    ) -> dict[str, object]:
        self.calls.append({"message_ids": list(message_ids), "dry_run": dry_run})
        if self.fail_real_undo and not dry_run:
            raise RuntimeError("memory cleanup failed")
        return {
            "affected_ids": ["mem1"],
            "restored_ids": ["old1"],
            "rollback_source_ids": ["cli:1:0", "cli:1:1", "cli:1:2"],
        }


@pytest.mark.asyncio
async def test_undo_command_aborts_without_running_llm(tmp_path):
    plugin = PluginUndo()
    session_manager = SessionManager(tmp_path)
    session = session_manager.get_or_create("cli:1")
    session.add_message("user", '<system-reminder data-system-context-frame="true">内部</system-reminder>')
    session.add_message("user", "u0")
    session.add_message("assistant", "a0")
    session_manager.save(session)
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
        session=session,
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
    assert [call["dry_run"] for call in memory_engine.calls] == [True, False]
    assert session_manager.get_or_create("cli:1").messages == []


@pytest.mark.asyncio
async def test_undo_reports_memory_cleanup_failure_after_session_delete(tmp_path, caplog):
    plugin = PluginUndo()
    session_manager = SessionManager(tmp_path)
    session = session_manager.get_or_create("cli:1")
    session.add_message("user", "u0")
    session.add_message("assistant", "a0")
    session_manager.save(session)
    memory_engine = _MemoryEngine(fail_real_undo=True)
    plugin.context = PluginContext(
        event_bus=None,
        tool_registry=None,
        plugin_id="plugin_undo",
        plugin_dir=tmp_path,
        kv_store=PluginKVStore(tmp_path / ".kv.json"),
        session_manager=session_manager,
        memory_engine=memory_engine,
    )

    with caplog.at_level("ERROR", logger="plugin.undo"):
        reply = await plugin.undo("cli:1")

    assert "已撤销上一轮对话，但记忆清理失败" in reply
    assert session_manager.get_or_create("cli:1").messages == []
    assert [call["dry_run"] for call in memory_engine.calls] == [True, False]
    assert "deleted_ids=['cli:1:0', 'cli:1:1']" in caplog.text
    assert "'affected_ids': ['mem1']" in caplog.text


def test_undo_plugin_registers_telegram_command():
    assert PluginUndo().telegram_bot_commands() == [("undo", "撤销上一轮对话")]
