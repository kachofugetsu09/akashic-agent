from __future__ import annotations

import asyncio
import json
import sqlite3
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.core.runtime_support import ToolDiscoveryState, TurnRunResult
from agent.provider import ContentSafetyError, ContextLengthError, LLMResponse
from agent.tools.shell import ShellTool, _MAX_OUTPUT, _truncate, _validate_network_command
from agent.tools.web_fetch import WebFetchTool, _to_markdown, _to_text, _validate_url_target
from memory2.procedure_tagger import ProcedureTagger, _validate
from memory2.store import MemoryStore2
from proactive_v2.event import GenericContentEvent


class _ReasonerHarness:
    def __init__(self, outcomes):
        self.tools = SimpleNamespace(get_always_on_names=lambda: {"always"})
        self._outcomes = list(outcomes)
        self.discovery = ToolDiscoveryState()
        self.discovery._unlocked = {"s:1": OrderedDict({"old": None})}
        self.reasoner = SimpleNamespace(
            run_turn=AsyncMock(side_effect=self._run_reasoner)
        )

    async def _run_reasoner(self, **kwargs):
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return TurnRunResult(
            reply=outcome[0],
            tools_used=outcome[1],
            tool_chain=outcome[2],
            thinking=outcome[4],
        )


@pytest.mark.asyncio
async def test_reasoner_wrapper_and_shell_cover_branches(tmp_path: Path):
    msg = SimpleNamespace(
        content="hello",
        media=[],
        channel="telegram",
        chat_id="1",
        timestamp=datetime.now(timezone.utc),
    )
    session = SimpleNamespace(
        key="s:1",
        messages=[{"role": "u", "content": str(i)} for i in range(6)],
        get_history=lambda max_messages: [{"role": "u", "content": str(i)} for i in range(6)],
        last_consolidated=3,
    )
    harness = _ReasonerHarness(
        [
            ("ok", ["tool_search", "x", "y"], [{"calls": []}], None, None),
        ]
    )
    result = await harness.reasoner.run_turn(msg=msg, session=session)
    assert result.reply == "ok"
    assert result.tools_used == ["tool_search", "x", "y"]

    harness = _ReasonerHarness(
        [("上下文过长无法处理，请尝试新建对话。", [], [], None, None)]
    )
    result = await harness.reasoner.run_turn(msg=msg, session=session)
    assert "上下文过长" in str(result.reply)
    assert result.tools_used == []
    assert result.tool_chain == []

    harness.reasoner = SimpleNamespace(
        run_turn=AsyncMock(return_value=TurnRunResult(reply="ok"))
    )
    result = await harness.reasoner.run_turn(msg=msg, session=session)
    assert result.reply == "ok"
    assert result.tools_used == []
    assert result.tool_chain == []

    harness = _ReasonerHarness([("ok", ["always", "tool_search", "a", "b", "c", "d", "e", "f"], [], None, None)])
    harness.discovery.update("s:1", ["always", "tool_search", "a", "b", "c", "d", "e", "f"], harness.tools.get_always_on_names())
    assert "always" not in harness.discovery._unlocked["s:1"]
    assert len(harness.discovery._unlocked["s:1"]) == 5

    tool = ShellTool()
    assert "命令不能为空" in await tool.execute(command="")
    assert "不被允许" in await tool.execute(command="nc localhost 1")
    assert "URL" in _validate_network_command("curl ftp://x")
    assert "上传/写文件" in _validate_network_command("curl -o out http://x.com")
    assert _validate_network_command("echo hi") is None
    assert "禁止访问内网" in _validate_network_command("curl http://127.0.0.1")
    truncated = _truncate("HEAD\n" + ("a" * 31000) + "\nTAIL")
    assert truncated["truncated"] is True
    assert truncated["strategy"] == "tail"
    assert "HEAD" not in truncated["text"]
    assert "TAIL" in truncated["text"]
    assert len(truncated["text"]) <= _MAX_OUTPUT

    from types import SimpleNamespace as _SN

    async def _fake_subprocess(command, **kwargs):
        class _P:
            returncode = 2
            pid = 0

            async def wait(self_):
                return 2

        p = _P()
        buf_out = [b"out"]
        buf_err = [b"err"]

        async def _read_out(_size=-1):
            return buf_out.pop(0) if buf_out else b""

        async def _read_err(_size=-1):
            return buf_err.pop(0) if buf_err else b""

        p.stdout = _SN(read=_read_out)
        p.stderr = _SN(read=_read_err)
        return p

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("agent.tools.shell.asyncio.create_subprocess_shell", _fake_subprocess)
        result = json.loads(await tool.execute(command="echo 1", timeout=999))
    assert result["exit_code"] == 2
    assert "Exit code 2" in result["output"]

async def test_web_fetch_procedure_tagger_and_store_cover_core_paths(tmp_path: Path):
    class _Resp:
        def __init__(self, *, status=200, headers=None, content=b"", encoding="utf-8", url="https://x"):
            self.status_code = status
            self.headers = headers or {}
            self.content = content
            self.encoding = encoding
            self.url = url

    requester = MagicMock()
    requester.get = AsyncMock(
        return_value=_Resp(
            headers={"content-type": "text/html", "content-length": "20"},
            content=b"<html><body><script>x</script><p>Hello <b>world</b></p></body></html>",
        )
    )
    tool = WebFetchTool(requester=requester)
    result = json.loads(await tool.execute(url="https://example.com", format="text"))
    assert result["text"] == "Hello world"
    result = json.loads(await tool.execute(url="https://example.com", format="markdown"))
    assert "Hello" in result["text"]

    requester.get = AsyncMock(return_value=_Resp(status=404))
    assert "HTTP 404" in json.loads(await tool.execute(url="https://example.com"))["error"]
    requester.get = AsyncMock(return_value=_Resp(headers={"content-type": "application/pdf"}))
    assert "二进制内容" in json.loads(await tool.execute(url="https://example.com"))["error"]
    requester.get = AsyncMock(
        side_effect=__import__("httpx").TimeoutException("slow")
    )
    assert "请求超时" in json.loads(await tool.execute(url="https://example.com"))["error"]
    assert "http:// 或 https://" in json.loads(await tool.execute(url="ftp://x"))["error"]
    assert _validate_url_target("http://127.0.0.1")
    assert _to_text(b"<html><body><style>x</style><p>Hi</p></body></html>") == "Hi"
    assert "Title" in _to_markdown("<h1>Title</h1>")

    provider = MagicMock()
    provider.chat = AsyncMock(
        return_value=LLMResponse(
            content='```json\n{"tools":["shell","bad"],"skills":["rsshub-route-finder"],"keywords":["pacman","x"],"scope":"global"}\n```'
        )
    )
    tagger = ProcedureTagger(provider, "m", lambda: ["rsshub-route-finder"])
    tag = await tagger.tag("测试")
    assert tag == {
        "tools": ["shell"],
        "skills": ["rsshub-route-finder"],
        "keywords": ["pacman"],
        "scope": "tool_triggered",
    }
    provider.chat = AsyncMock(side_effect=RuntimeError("x"))
    assert await tagger.tag("测试") is None
    assert (
        _validate({"scope": "bad", "keywords": ["okay"]}, {"shell"}, set())["scope"]
        == "tool_triggered"
    )

    store = MemoryStore2(tmp_path / "mem.db")
    first = store.upsert_item("procedure", "Hello   world", [1.0, 0.0], source_ref="s1")
    assert first.startswith("new:")
    item_id = first.split(":", 1)[1]
    assert store.upsert_item("procedure", "hello world", [1.0, 0.0]).startswith("reinforced:")
    store.mark_superseded(item_id)
    assert store.upsert_item("procedure", "hello world", [1.0, 0.0]).startswith("reinforced:")
    store.mark_superseded_batch([item_id])
    assert store.get_all_with_embedding(include_superseded=True)
    assert store.has_item_by_source_ref("s1", "procedure") is True
    assert store.delete_by_source_ref("s1") >= 1
    event = store.upsert_consolidation_event(source_ref="r1", summary="Event A", embedding=[0.0, 1.0])
    assert event.startswith("new:")
    assert store.upsert_consolidation_event(source_ref="r1", summary="Event A", embedding=[0.0, 1.0]).startswith("skipped:")
    store.upsert_item(
        "procedure",
        "Use pacman",
        [1.0, 0.0],
        extra={"trigger_tags": {"scope": "tool_triggered", "tools": [], "skills": [], "keywords": ["pacman"]}},
    )
    hits = store.keyword_match_procedures(["shell", "pacman"])
    assert hits and hits[0]["memory_type"] == "procedure"
    results = store.vector_search([0.0, 1.0], top_k=2, memory_types=["event"])
    assert results and results[0]["memory_type"] == "event"
    assert store.list_by_type("event")
    old_res = store.upsert_item(
        "procedure",
        "旧流程：查 Steam 时直接用 web_search",
        [1.0, 0.0],
        source_ref="old-rule",
        extra={"tool_requirement": "web_search"},
    )
    new_res = store.upsert_item(
        "procedure",
        "新流程：查 Steam 时必须先用 steam_mcp",
        [0.9, 0.1],
        source_ref="new-rule",
        extra={"tool_requirement": "steam_mcp"},
    )
    old_item = store.get_items_by_ids([old_res.split(":", 1)[1]])[0]
    new_item = store.get_items_by_ids([new_res.split(":", 1)[1]])[0]
    assert store.record_replacements(
        old_items=[old_item],
        new_item=new_item,
        source_ref="test@replace",
    ) == 1
    replacements = store.list_replacements()
    assert replacements[0]["old_summary"] == "旧流程：查 Steam 时直接用 web_search"
    assert replacements[0]["new_summary"] == "新流程：查 Steam 时必须先用 steam_mcp"
    assert replacements[0]["old_extra_json"]["tool_requirement"] == "web_search"
    assert replacements[0]["new_extra_json"]["tool_requirement"] == "steam_mcp"
    store.close()


def test_memory_store_runtime_migrates_emotional_weight_column(tmp_path: Path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE memory_items (
            id            TEXT PRIMARY KEY,
            memory_type   TEXT NOT NULL,
            summary       TEXT NOT NULL,
            content_hash  TEXT NOT NULL,
            embedding     TEXT,
            reinforcement INTEGER NOT NULL DEFAULT 1,
            extra_json    TEXT,
            source_ref    TEXT,
            happened_at   TEXT,
            status        TEXT NOT NULL DEFAULT 'active',
            created_at    TEXT NOT NULL,
            updated_at    TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()

    store = MemoryStore2(db_path)
    cols = {
        row[1] for row in store._db.execute("PRAGMA table_info(memory_items)").fetchall()
    }
    assert "emotional_weight" in cols
    store.close()
