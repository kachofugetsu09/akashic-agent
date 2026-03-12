from __future__ import annotations

import asyncio
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.provider import LLMResponse
from core.memory.port import DefaultMemoryPort
from feeds.base import FeedSubscription
from feeds.novel import NovelKBFeedSource, _extract_body, _resolve_kb_root
from feeds.rss import (
    RSSFeedSource,
    _atom_text,
    _is_retryable_curl_code,
    _is_retryable_http_error,
    _normalize_xml_text,
    _parse_rfc822,
    _remaining_budget,
    _strip_html,
    _text,
)
from memory2.sop_indexer import SopIndexer, _parse_sop_chunks
from proactive.loop import ProactiveLoop
from proactive.memory_optimizer import (
    MemoryOptimizer,
    MemoryOptimizerLoop,
    _parse_cleanup_json,
    _remove_items_from_section,
)
from session.manager import Session, SessionManager, _safe_filename


@pytest.mark.asyncio
async def test_rss_novel_and_sop_indexer_cover_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    rss_xml = """<rss><channel><item><title>T</title><link>https://x</link><description><![CDATA[<b>Hello</b>]]></description><pubDate>Tue, 01 Jan 2025 00:00:00 +0000</pubDate></item></channel></rss>"""
    file_path = tmp_path / "feed.xml"
    file_path.write_text(rss_xml, encoding="utf-8")
    sub = FeedSubscription.new(type="rss", name="Feed", url=f"file://{file_path}")
    items = await RSSFeedSource(sub).fetch()
    assert items[0].title == "T"
    assert items[0].content == "Hello"

    sub = FeedSubscription.new(type="rss", name="Feed", url="https://xcancel.com/rss")

    class _Proc:
        def __init__(self, rc: int, out: bytes = b"", err: bytes = b"") -> None:
            self.returncode = rc
            self._out = out
            self._err = err

        async def communicate(self):
            return self._out, self._err

        def kill(self):
            return None

    monkeypatch.setattr(
        "feeds.rss.asyncio.create_subprocess_exec",
        AsyncMock(return_value=_Proc(0, rss_xml.encode())),
    )
    items = await RSSFeedSource(sub)._fetch_via_curl(limit=1)
    assert items[0].title == "T"
    assert RSSFeedSource(sub)._parse("rss reader not yet whitelisted", 1) == []
    assert RSSFeedSource(sub)._parse("<root />", 1) == []

    root = ET.fromstring(rss_xml)
    assert _text(root.find("channel/item"), "title") == "T"  # type: ignore[arg-type]
    atom_xml = ET.fromstring(
        '<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>A</title></entry></feed>'
    )
    assert _atom_text(atom_xml.find("{http://www.w3.org/2005/Atom}entry"), "title", {"a": "http://www.w3.org/2005/Atom"}) == "A"  # type: ignore[arg-type]
    assert _strip_html("<p>a &amp; b</p>") == "a & b"
    assert _parse_rfc822("Tue, 01 Jan 2025 00:00:00 +0000") is not None
    assert _normalize_xml_text("\ufeff\n<a/>") == "<a/>"
    assert _is_retryable_curl_code(28) is True
    err = __import__("httpx").TimeoutException("x")
    assert _is_retryable_http_error(err) is True
    loop = asyncio.get_running_loop()
    assert _remaining_budget(loop.time() + 1, loop) > 0

    kb = tmp_path / "kb"
    summary_dir = kb / "summaries" / "chunks"
    summary_dir.mkdir(parents=True)
    (kb / "summaries" / "index.json").write_text(
        json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "segment": "s1",
                        "created_at": "2025-01-01T00:00:00+00:00",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (summary_dir / "c1.summary.md").write_text("# T\n- key: value\n正文" * 10, encoding="utf-8")
    source = NovelKBFeedSource(
        FeedSubscription.new(type="novel-kb", name="KB", url=f"file://{kb}")
    )
    items = await source.fetch()
    assert items and items[0].url == "novel://kb/c1"
    assert _resolve_kb_root("relative") is None
    assert _extract_body("# T\n- a: b\n正文", 10) == "正文"

    sop_dir = tmp_path / "sop"
    sop_dir.mkdir()
    sop_file = sop_dir / "core-rules.md"
    sop_file.write_text("## Rule\nUse `shell_tool` here", encoding="utf-8")
    chunks = _parse_sop_chunks(sop_file, "procedure")
    assert chunks and "trigger_keywords" in chunks[0]["extra"]
    store = MagicMock()
    store.delete_by_source_ref.return_value = 1
    store.upsert_item.return_value = "new:1"
    embedder = MagicMock()
    embedder.embed_batch = AsyncMock(return_value=[[1.0]])
    indexer = SopIndexer(store, embedder, sop_dir)
    assert indexer.is_sop_file(sop_file) is True
    summary = await indexer.reindex(sop_file)
    assert "重索引" in summary
    assert "README" in await indexer.reindex(sop_dir / "README.md")


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
        supersede_batch=MagicMock(),
    )
    retriever = SimpleNamespace(
        retrieve=AsyncMock(return_value=[{"id": "1"}]),
        embed=AsyncMock(return_value=[1.0]),
        retrieve_with_vec=AsyncMock(return_value=[{"id": "2"}]),
        format_injection_block=lambda items: "block",
        format_injection_with_ids=lambda items: ("block", ["1"]),
        select_for_injection=lambda items: items[:1],
        _store=SimpleNamespace(keyword_match_procedures=lambda action_tokens: [{"id": "p1"}]),
    )
    port = DefaultMemoryPort(store, memorizer, retriever)
    assert port.read_history(3) == "def"
    assert port.has_long_term_memory() is True
    assert await port.retrieve_related("q") == [{"id": "1"}]
    assert await port.embed_query("q") == [1.0]
    assert await port.retrieve_related_vec([1.0]) == [{"id": "2"}]
    assert port.format_injection_block([]) == "block"
    assert port.format_injection_with_ids([]) == ("block", ["1"])
    assert port.select_for_injection([{"id": "1"}, {"id": "2"}]) == [{"id": "1"}]
    assert await port.save_item("s", "procedure", {}, "src") == "id"
    await port.save_from_consolidation("h", [], "src", "c", "id")
    port.supersede_batch(["1"])
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
            supersede_batch=MagicMock(),
        ),
        retriever=SimpleNamespace(
            retrieve=AsyncMock(side_effect=RuntimeError("x")),
            embed=AsyncMock(side_effect=RuntimeError("x")),
            retrieve_with_vec=AsyncMock(side_effect=RuntimeError("x")),
            format_injection_block=lambda items: "block",
            format_injection_with_ids=lambda items: (_ for _ in ()).throw(RuntimeError("x")),
            select_for_injection=lambda items: (_ for _ in ()).throw(RuntimeError("x")),
            _store=SimpleNamespace(keyword_match_procedures=lambda action_tokens: (_ for _ in ()).throw(RuntimeError("x"))),
        ),
    )
    assert broken.has_long_term_memory() is False
    assert await broken.retrieve_related("q") == []
    assert await broken.embed_query("q") == []
    assert await broken.retrieve_related_vec([1.0]) == []
    assert broken.format_injection_with_ids([]) == ("block", [])
    assert broken.select_for_injection([{"id": "1"}]) == [{"id": "1"}]
    assert await broken.save_item("s", "procedure", {}, "src") == ""
    await broken.save_from_consolidation("h", [], "src", "c", "id")
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
        use_global_memory=True,
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
    loop._memory = SimpleNamespace(read_long_term=lambda: "remember", get_memory_context=lambda: "ctx")
    loop._sessions = SimpleNamespace(workspace=tmp_path)
    (tmp_path / "AGENTS.md").write_text("guide", encoding="utf-8")
    loop._sender = SimpleNamespace(send=AsyncMock(return_value=True))
    loop._engine = SimpleNamespace(tick=AsyncMock(return_value=0.2))
    assert loop._sample_random_memory(1)
    assert "Workspace 导航" in loop._build_context_block()
    assert await loop._send("hi") is True
    assert await loop._tick() == 0.2
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("proactive.loop.compute_energy", lambda last_user_at: 0.8)
        mp.setattr("proactive.loop.d_energy", lambda energy: 0.5)
        mp.setattr("proactive.loop.next_tick_from_score", lambda *args, **kwargs: 7)
        assert loop._next_interval() == 7
    loop._cfg.use_global_memory = False
    assert "禁用" in loop._collect_global_memory()
    loop._cfg.use_global_memory = True
    loop._memory = None
    assert "无全局记忆" in loop._collect_global_memory()
    loop._manual_trigger_event = asyncio.Event()
    loop._cfg.threshold = 0.5
    loop._cfg.default_channel = "telegram"
    loop._cfg.default_chat_id = "42"
    wait_results = iter([None, asyncio.TimeoutError()])

    async def _fake_wait_for(awaitable, timeout):
        if hasattr(awaitable, "close"):
            awaitable.close()
        result = next(wait_results)
        if isinstance(result, Exception):
            raise result
        return result

    tick_calls = {"count": 0}

    async def _fake_tick():
        tick_calls["count"] += 1
        loop._running = False
        return 0.1

    loop._tick = _fake_tick
    loop._manual_trigger_event.set()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("proactive.loop.asyncio.wait_for", _fake_wait_for)
        await loop.run()
    assert tick_calls["count"] == 1
