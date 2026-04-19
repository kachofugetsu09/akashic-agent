from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.provider import LLMResponse
from core.memory.port import DefaultMemoryPort
from proactive_v2.loop import ProactiveLoop
from proactive_v2.memory_optimizer import (
    MemoryOptimizer,
    MemoryOptimizerLoop,
    _parse_cleanup_json,
    _remove_items_from_section,
)
from session.manager import Session, SessionManager, _safe_filename


@pytest.mark.asyncio
async def test_memory_optimizer_loop_and_memory_port_cover_paths(tmp_path: Path):
    memory = MagicMock()
    memory.snapshot_pending.return_value = "- [identity] x"
    memory.read_long_term.return_value = "MEM"
    memory.read_self.return_value = "# Akashic 的自我认知\n## 人格与形象\n- x"
    memory.read_now.return_value = "## 近期进行中\n- 旧任务\n## 待确认事项\n- 已确认"
    memory.read_history.return_value = "history"
    memory.get_memory_context.return_value = "ctx"
    memory.write_long_term = MagicMock()
    memory.append_history = MagicMock()
    memory.commit_pending_snapshot = MagicMock()
    memory.rollback_pending_snapshot = MagicMock()
    memory.write_self = MagicMock()
    memory.write_now = MagicMock()
    memory.read_now_ongoing.return_value = "ongoing"
    provider = MagicMock()
    provider.chat = AsyncMock(
        side_effect=[
            LLMResponse(content="merged"),
            LLMResponse(content="updated self"),
            LLMResponse(content='{"remove_ongoing":["旧任务"],"remove_pending":["已确认"]}'),
        ]
    )
    opt = MemoryOptimizer(memory, provider, "m", max_tokens=100, history_max_chars=20)
    opt._STEP_DELAY_SECONDS = 0
    await opt.optimize()
    memory.write_long_term.assert_called_once_with("merged")
    memory.write_self.assert_called_once()
    memory.write_now.assert_called_once()
    assert _parse_cleanup_json('{"remove_ongoing":["a"],"remove_pending":["b"]}') == (["a"], ["b"])
    assert "近期进行中" in _remove_items_from_section("## 近期进行中\n- a\n", "## 近期进行中", ["a"])

    loop = MemoryOptimizerLoop(opt, interval_seconds=10, _now_fn=lambda: datetime(2025, 1, 1, 0, 0, 1))
    assert loop._seconds_until_next_tick() >= 1.0
    loop.stop()

    store = SimpleNamespace(
        read_long_term=lambda: " mem ",
        write_long_term=lambda content: None,
        read_self=lambda: "self",
        write_self=lambda content: None,
        read_now=lambda: "now",
        write_now=lambda content: None,
        read_now_ongoing=lambda: "ongoing",
        update_now_ongoing=lambda add, remove_keywords: None,
        read_pending=lambda: "pending",
        append_pending=lambda facts: None,
        snapshot_pending=lambda: "snapshot",
        commit_pending_snapshot=lambda: None,
        rollback_pending_snapshot=lambda: None,
        append_history=lambda entry: None,
        get_memory_context=lambda: "ctx",
        history_file=tmp_path / "history.md",
    )
    store.history_file.write_text("abcdef", encoding="utf-8")
    memorizer = SimpleNamespace(
        save_item=AsyncMock(return_value="id"),
        save_from_consolidation=AsyncMock(),
        reinforce_items_batch=MagicMock(),
    )
    retriever = SimpleNamespace(
        retrieve=AsyncMock(return_value=[{"id": "1"}]),
        embed=AsyncMock(return_value=[1.0]),
        retrieve_with_vec=AsyncMock(return_value=[{"id": "2"}]),
        build_injection_block=lambda items: ("block", ["1"]),
        _store=SimpleNamespace(keyword_match_procedures=lambda action_tokens: [{"id": "p1"}]),
    )
    port = DefaultMemoryPort(store, memorizer, retriever)
    assert port.read_history(3) == "def"
    assert port.has_long_term_memory() is True
    assert await port.retrieve_related("q") == [{"id": "1"}]
    assert await port.embed_query("q") == [1.0]
    assert await port.retrieve_related_vec([1.0]) == [{"id": "2"}]
    assert port.build_injection_block([]) == ("block", ["1"])
    assert await port.save_item("s", "procedure", {}, "src") == "id"
    await port.save_from_consolidation("h", [], "src", "c", "id")
    port.reinforce_items_batch(["1"])
    memorizer.reinforce_items_batch.assert_called_once_with(["1"])
    assert port.keyword_match_procedures(["shell"]) == [{"id": "p1"}]

    broken = DefaultMemoryPort(
        SimpleNamespace(
            read_long_term=lambda: (_ for _ in ()).throw(RuntimeError("x")),
            write_long_term=lambda content: None,
            read_self=lambda: "",
            write_self=lambda content: None,
            read_now=lambda: "",
            write_now=lambda content: None,
            read_now_ongoing=lambda: "",
            update_now_ongoing=lambda add, remove_keywords: None,
            read_pending=lambda: "",
            append_pending=lambda facts: None,
            snapshot_pending=lambda: "",
            commit_pending_snapshot=lambda: None,
            rollback_pending_snapshot=lambda: None,
            append_history=lambda entry: None,
            get_memory_context=lambda: "",
            history_file=tmp_path / "missing.txt",
        ),
        memorizer=SimpleNamespace(
            save_item=AsyncMock(side_effect=RuntimeError("x")),
            save_from_consolidation=AsyncMock(side_effect=RuntimeError("x")),
            reinforce_items_batch=MagicMock(side_effect=RuntimeError("x")),
        ),
        retriever=SimpleNamespace(
            retrieve=AsyncMock(side_effect=RuntimeError("x")),
            embed=AsyncMock(side_effect=RuntimeError("x")),
            retrieve_with_vec=AsyncMock(side_effect=RuntimeError("x")),
            build_injection_block=lambda items: (_ for _ in ()).throw(RuntimeError("x")),
            _store=SimpleNamespace(keyword_match_procedures=lambda action_tokens: (_ for _ in ()).throw(RuntimeError("x"))),
        ),
    )
    assert broken.has_long_term_memory() is False
    assert await broken.retrieve_related("q") == []
    assert await broken.embed_query("q") == []
    assert await broken.retrieve_related_vec([1.0]) == []
    assert broken.build_injection_block([]) == ("", [])
    assert await broken.save_item("s", "procedure", {}, "src") == ""
    await broken.save_from_consolidation("h", [], "src", "c", "id")
    broken.reinforce_items_batch(["1"])
    assert broken.keyword_match_procedures(["shell"]) == []


@pytest.mark.asyncio
async def test_session_manager_and_proactive_loop_cover_paths(tmp_path: Path):
    session = Session("telegram:1")
    session.add_message("user", "hi", media=["/tmp/a.png"])
    session.add_message(
        "assistant",
        "reply",
        proactive=True,
        state_summary_tag="tag",
        source_refs=[{"source_name": "Feed", "title": "T", "url": "https://x"}],
    )
    session.messages[-1]["tool_chain"] = [
        {"calls": [{"call_id": "1", "name": "tool", "arguments": {}, "result": "ok"}]}
    ]
    history = session.get_history()
    assert history[0]["role"] == "user"
    assert history[-1]["role"] == "assistant"
    assert _safe_filename("telegram:1") == "telegram_1"

    manager = SessionManager(tmp_path)
    manager.save(session)
    loaded = manager.get_or_create("telegram:1")
    assert loaded.key == "telegram:1"
    await manager.append_messages(session, [{"role": "user", "content": "next"}])
    assert manager.list_sessions()
    assert manager.get_channel_metadata("telegram")[0]["chat_id"] == "1"
    manager.invalidate("telegram:1")

    loop = ProactiveLoop.__new__(ProactiveLoop)
    loop._cfg = SimpleNamespace(
        interval_seconds=10,
        score_weight_energy=0.5,
        tick_interval_s3=1,
        tick_interval_s2=2,
        tick_interval_s1=3,
        tick_interval_s0=4,
        tick_jitter=0.0,
    )
    loop._presence = None
    loop._trace_proactive_rate_decision = MagicMock()
    assert loop._next_interval() == 10
    loop._presence = SimpleNamespace(
        get_last_user_at=lambda session_key: datetime.now(timezone.utc)
    )
    loop._sense = SimpleNamespace(
        target_session_key=lambda: "telegram:1",
        has_global_memory=lambda: True,
        read_memory_text=lambda: "mem",
        compute_energy=lambda: 0.5,
        compute_interruptibility=lambda **kwargs: (0.5, {"x": 1}),
    )
    loop._rng = None
    loop._memory = SimpleNamespace(read_long_term_context=lambda: "remember", get_memory_context=lambda: "ctx")
    loop._sessions = SimpleNamespace(workspace=tmp_path)
    (tmp_path / "AGENTS.md").write_text("guide", encoding="utf-8")
    loop._sender = SimpleNamespace(send=AsyncMock(return_value=True))
    loop._engine = SimpleNamespace(tick=AsyncMock(return_value=0.2))
    loop._feed_poll_lock = asyncio.Lock()
    loop._mcp_pool = SimpleNamespace(
        connect_all=AsyncMock(return_value=None),
        disconnect_all=AsyncMock(return_value=None),
    )
    loop._poll_feeds_once = AsyncMock(return_value=None)
    assert loop._sample_random_memory(1)


def test_session_get_history_returns_empty_when_window_is_zero():
    session = Session("cli:1")
    session.add_message("user", "hello")
    session.add_message("assistant", "world")

    assert session.get_history(max_messages=0) == []


def test_session_get_history_does_not_inject_inference_tag():
    session = Session("cli:1")
    session.add_message("user", "hello")
    session.add_message("assistant", "world")

    history = session.get_history()

    assert history[-1] == {"role": "assistant", "content": "world"}


@pytest.mark.asyncio
async def test_proactive_loop_wrapper_methods_cover_paths(tmp_path: Path):
    loop = ProactiveLoop.__new__(ProactiveLoop)
    loop._cfg = SimpleNamespace(
        interval_seconds=10,
        score_weight_energy=0.5,
        tick_interval_s3=1,
        tick_interval_s2=2,
        tick_interval_s1=3,
        tick_interval_s0=4,
        tick_jitter=0.0,
        threshold=0.5,
        default_channel="telegram",
        default_chat_id="42",
    )
    loop._running = False
    loop._trace_proactive_rate_decision = MagicMock()
    loop._presence = SimpleNamespace(
        get_last_user_at=lambda session_key: datetime.now(timezone.utc)
    )
    loop._sense = SimpleNamespace(
        target_session_key=lambda: "telegram:1",
        has_global_memory=lambda: True,
        read_memory_text=lambda: "mem",
        compute_energy=lambda: 0.5,
        compute_interruptibility=lambda **kwargs: (0.5, {"x": 1}),
    )
    loop._rng = None
    loop._memory = SimpleNamespace(read_long_term=lambda: "remember", get_memory_context=lambda: "ctx")
    loop._sessions = SimpleNamespace(workspace=tmp_path)
    (tmp_path / "AGENTS.md").write_text("guide", encoding="utf-8")
    loop._sender = SimpleNamespace(send=AsyncMock(return_value=True))
    loop._agent_tick = SimpleNamespace(tick=AsyncMock(return_value=0.2))
    loop._feed_poll_lock = asyncio.Lock()
    loop._mcp_pool = SimpleNamespace(
        connect_all=AsyncMock(return_value=None),
        disconnect_all=AsyncMock(return_value=None),
    )
    loop._run_loop = AsyncMock(return_value=None)

    assert loop._has_global_memory() is True
    assert loop._read_memory_text() == "mem"
    assert loop._compute_energy() == 0.5
    assert loop._compute_interruptibility(
        now_hour=10,
        now_utc=datetime.now(timezone.utc),
        recent_msg_count=0,
    ) == (0.5, {"x": 1})
    assert await loop._tick() == 0.2
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("proactive_v2.loop.compute_energy", lambda last_user_at: 0.8)
        mp.setattr("proactive_v2.loop.d_energy", lambda energy: 0.5)
        mp.setattr("proactive_v2.loop.next_tick_from_score", lambda *args, **kwargs: 7)
        assert loop._next_interval() == 7
    await loop.run()
    loop._mcp_pool.connect_all.assert_awaited_once()
    loop._run_loop.assert_awaited_once()
    loop._mcp_pool.disconnect_all.assert_awaited_once()
