from __future__ import annotations

import asyncio
import json
import runpy
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.context import ContextBuilder
from agent.tools.base import Tool
from agent.tools.memorize import MemorizeTool, _append_to_sop_file
from agent.tools.message_push import MessagePushTool
from agent.tools.notify_owner import NotifyOwnerTool
from agent.tools.registry import ToolMeta, ToolRegistry
from agent.tools.web_search import WebSearchTool
from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from core.common import timekit
from core.memory.port import DefaultMemoryPort
from infra.persistence.json_store import atomic_save_json, load_json, save_json
from memory2.memorizer import Memorizer
from memory2.store import MemoryStore2


class _DummyTool(Tool):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def description(self) -> str:
        return "dummy description"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 2},
                "count": {"type": "integer", "minimum": 1, "maximum": 3},
                "mode": {"type": "string", "enum": ["a", "b"]},
                "items": {"type": "array", "items": {"type": "number"}},
            },
            "required": ["name", "count"],
        }

    async def execute(self, **kwargs) -> str:
        return json.dumps(kwargs, ensure_ascii=False)


@pytest.mark.asyncio
async def test_message_push_tool_covers_success_failure_and_fallbacks():
    tool = MessagePushTool()
    sent = {"text": [], "file": [], "image": []}

    async def text(chat_id: str, message: str) -> None:
        sent["text"].append((chat_id, message))

    async def file(chat_id: str, path: str, name: str | None) -> None:
        sent["file"].append((chat_id, path, name))

    async def image(chat_id: str, path: str) -> None:
        sent["image"].append((chat_id, path))

    tool.register_channel("telegram", text=text, file=file, image=image)
    result = await tool.execute(
        channel="telegram",
        chat_id=123,
        message="hello",
        file="/tmp/demo.txt",
        image="https://img",
    )

    assert "文本已发送" in result
    assert "文件 'demo.txt' 已发送" in result
    assert "图片已发送" in result
    assert sent["text"] == [("123", "hello")]
    assert sent["file"] == [("123", "/tmp/demo.txt", "demo.txt")]
    assert sent["image"] == [("123", "https://img")]

    assert await tool.execute(channel="telegram", chat_id=1) == (
        "错误：message、file、image 至少提供一个"
    )
    assert "未注册" in await tool.execute(channel="qq", chat_id=1, message="x")

    tool.register_channel("limited", text=text)
    limited = await tool.execute(
        channel="limited", chat_id=1, file="/tmp/a.txt", image="/tmp/a.png"
    )
    assert "不支持发送文件" in limited
    assert "不支持发送图片" in limited

    async def broken(chat_id: str, message: str) -> None:
        raise RuntimeError("send failed")

    tool.register_channel("broken", text=broken)
    assert "发送失败" in await tool.execute(channel="broken", chat_id=1, message="x")


@pytest.mark.asyncio
async def test_notify_owner_tool_and_memorize_tool_cover_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    push = AsyncMock()
    push.execute = AsyncMock(return_value="ok")
    notify = NotifyOwnerTool(push, "telegram", "42")
    assert await notify.execute("  hi  ") == "ok"
    push.execute.assert_awaited_once_with(
        channel="telegram", chat_id="42", message="hi"
    )

    assert "跳过发送" in await NotifyOwnerTool(push, "", "").execute("hi")
    assert "消息内容为空" in await notify.execute("   ")

    push.execute = AsyncMock(side_effect=RuntimeError("nope"))
    assert "发送失败" in await notify.execute("hi")

    home = tmp_path / "home"
    monkeypatch.setattr("pathlib.Path.home", lambda: home)
    _append_to_sop_file("user.md", "规则", ["一步", "二步"])
    content = (home / ".akasic" / "workspace" / "sop" / "user.md").read_text(
        encoding="utf-8"
    )
    assert "## 规则" in content
    assert "- 一步" in content

    memory = MagicMock()
    memory.save_item_with_supersede = AsyncMock(return_value="mem-1")

    class _Tagger:
        async def tag(self, summary: str) -> dict[str, str]:
            assert summary == "记住这条流程"
            return {"scope": "task"}

    tool = MemorizeTool(memory, tagger=_Tagger())
    result = await tool.execute(
        summary="记住这条流程",
        memory_type="procedure",
        steps=["先查", "再做"],
        persist_file="ops.md",
    )

    assert "已记住（mem-1）" in result
    extra = memory.save_item_with_supersede.await_args.kwargs["extra"]
    assert extra["trigger_tags"] == {"scope": "task"}
    assert extra["persist_file"] == "ops.md"
    assert extra["rule_schema"]["required_tools"] == []
    assert extra["rule_schema"]["forbidden_tools"] == []

    class _BadTagger:
        async def tag(self, summary: str) -> dict[str, str]:
            raise RuntimeError("bad")

    bad = MemorizeTool(memory, tagger=_BadTagger())
    await bad.execute(summary="普通偏好", memory_type="procedure")
    await bad.execute(summary="偏好", memory_type="preference")


@pytest.mark.asyncio
async def test_memorize_tool_should_not_create_second_active_procedure_when_incremental_update():
    class _Embedder:
        async def embed(self, text: str) -> list[float]:
            return [1.0, 0.0]

    store = MemoryStore2(":memory:")
    memorizer = Memorizer(store, _Embedder())
    memory = DefaultMemoryPort(MagicMock(), memorizer=memorizer, retriever=None)
    tool = MemorizeTool(memory)

    await memorizer.save_item(
        summary="查询 Steam 游戏信息时，必须先使用 steam_mcp 工具查询游戏详情，再用 web_search 补充验证价格和评价信息。",
        memory_type="procedure",
        extra={
            "steps": ["使用 steam_mcp 工具查询游戏详情", "使用 web_search 补充验证价格和评价"],
            "tool_requirement": "steam_mcp",
        },
        source_ref="seed",
    )

    await tool.execute(
        summary="查询 Steam 游戏信息时，先判断区服（大陆区/港区/美区），再使用 steam_mcp 工具查询游戏详情。",
        memory_type="procedure",
        tool_requirement="steam_mcp",
        steps=["判断目标区服", "使用 steam_mcp 工具查询游戏详情"],
    )

    rows = store._db.execute(
        "SELECT id, summary FROM memory_items WHERE memory_type='procedure' AND status='active'"
    ).fetchall()
    assert len(rows) == 1
    assert "steam_mcp" in rows[0][1]
    assert "区服" in rows[0][1]


@pytest.mark.asyncio
async def test_memorize_tool_should_coerce_language_reply_rule_to_preference():
    memory = MagicMock()
    memory.save_item_with_supersede = AsyncMock(return_value="mem-1")
    tool = MemorizeTool(memory)

    await tool.execute(
        summary="之后跟我说话只用中文，不要夹杂英文，专有名词也尽量翻译。",
        memory_type="procedure",
    )

    assert memory.save_item_with_supersede.await_args.kwargs["memory_type"] == "preference"


@pytest.mark.asyncio
async def test_web_search_covers_filters(monkeypatch: pytest.MonkeyPatch):
    class _Response:
        def __init__(self, text: str) -> None:
            self.text = text

        def raise_for_status(self) -> None:
            return None

    class _Client:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, json: dict, headers: dict) -> _Response:
            assert json["params"]["arguments"]["numResults"] == 20
            assert json["params"]["arguments"]["livecrawl"] == "preferred"
            assert json["params"]["arguments"]["type"] == "deep"
            return _Response(
                'data: {"result":{"content":[{"text":"hello world"}]}}\n\n'
            )

    monkeypatch.setattr("httpx.AsyncClient", _Client)
    result = json.loads(
        await WebSearchTool().execute(
            query="搜索 网络",
            num_results=99,
            livecrawl="preferred",
            type="deep",
        )
    )
    assert result["result"] == "hello world"

    class _BadClient(_Client):
        async def post(self, url: str, json: dict, headers: dict) -> _Response:
            raise RuntimeError("net down")

    monkeypatch.setattr("httpx.AsyncClient", _BadClient)
    result = json.loads(await WebSearchTool().execute(query="x"))
    assert "搜索失败" in result["error"]

    class _EmptyClient(_Client):
        async def post(self, url: str, json: dict, headers: dict) -> _Response:
            return _Response("data: not-json\n\ndata: {}")

    monkeypatch.setattr("httpx.AsyncClient", _EmptyClient)
    result = json.loads(await WebSearchTool().execute(query="x"))
    assert result["count"] == 0


def test_tool_base_and_timekit_and_json_store_cover_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    tool = _DummyTool()
    errors = tool.validate_params(
        {"name": "x", "count": 5, "mode": "c", "items": ["a"]}
    )
    assert "name 最短 2 个字符" in errors
    assert "count 须 <= 3" in errors
    assert "mode 须为以下值之一" in errors[2]
    assert "[0] 应为 number 类型" in errors[3]
    assert tool.validate_params({})[:2] == ["缺少必填字段：name", "缺少必填字段：count"]
    assert tool.to_schema()["function"]["name"] == "dummy"

    class _BadSchemaTool(_DummyTool):
        @property
        def parameters(self) -> dict:
            return {"type": "array"}

    with pytest.raises(ValueError):
        _BadSchemaTool().validate_params({})

    path = tmp_path / "data.json"
    assert load_json(path, default={"a": 1}) == {"a": 1}
    save_json(path, {"x": "中"})
    assert load_json(path)["x"] == "中"
    path.write_text("{bad", encoding="utf-8")
    assert load_json(path, default=[]) == []
    atomic_save_json(path, {"y": 2})
    assert load_json(path)["y"] == 2

    class _BadPath:
        parent = tmp_path
        suffix = ".json"

        def with_suffix(self, suffix: str):
            return tmp_path / "bad.json.tmp"

    bad = _BadPath()
    monkeypatch.setattr(
        "pathlib.Path.write_text",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    with pytest.raises(RuntimeError):
        save_json(tmp_path / "x.json", {"x": 1})
    with pytest.raises(RuntimeError):
        atomic_save_json(bad, {"x": 1})  # type: ignore[arg-type]

    parsed = timekit.parse_iso("2025-06-01T09:00:00Z")
    assert parsed and parsed.tzinfo is not None
    assert timekit.parse_iso("bad") is None
    assert timekit.format_iso(datetime(2025, 1, 1)).endswith("+00:00")
    logger = MagicMock()
    assert str(timekit.safe_zone("bad/zone", logger=logger)) == "UTC"
    logger.warning.assert_called_once()
    assert timekit.local_now("UTC").tzinfo is not None
    assert timekit.utcnow().tzinfo is not None


def test_context_builder_builds_prompt_messages_and_assistant_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    class _Skills:
        def __init__(self, workspace: Path) -> None:
            self.workspace = workspace

        def get_always_skills(self) -> list[str]:
            return ["always"]

        def load_skills_for_context(self, names: list[str]) -> str:
            return ",".join(names)

        def build_skills_summary(self) -> str:
            return "skill summary"

    class _Memory:
        def get_memory_context(self) -> str:
            return "memory block"

        def read_self(self) -> str:
            return "self note"

    monkeypatch.setattr("agent.context.SkillsLoader", _Skills)
    monkeypatch.setattr("agent.context.build_agent_identity_prompt", lambda **_: "identity")
    monkeypatch.setattr(
        "agent.context.build_current_session_prompt", lambda **_: "\nsession prompt"
    )
    monkeypatch.setattr(
        "agent.context.build_telegram_rendering_prompt", lambda: "\ntelegram prompt"
    )
    monkeypatch.setattr("agent.context.build_sop_index_prompt", lambda text: f"sop:{text}")
    monkeypatch.setattr(
        "agent.context.build_skills_catalog_prompt", lambda text: f"catalog:{text}"
    )

    (tmp_path / "sop").mkdir()
    (tmp_path / "sop" / "README.md").write_text("index", encoding="utf-8")
    (tmp_path / "memes").mkdir()
    (tmp_path / "memes" / "manifest.json").write_text(
        '{"version":1,"categories":{"shy":{"desc":"害羞","enabled":true}}}',
        encoding="utf-8",
    )
    image = tmp_path / "a.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")

    builder = ContextBuilder(tmp_path, _Memory())  # type: ignore[arg-type]
    prompt = builder.build_system_prompt(
        skill_names=["extra"],
        message_timestamp=datetime.now(timezone.utc),
        retrieved_memory_block="retrieved",
    )
    assert "identity" in prompt
    assert "retrieved" in prompt
    assert "memory block" in prompt
    assert "Akashic 自我认知" in prompt
    assert "# Memes" in prompt
    assert "<meme:shy>" in prompt
    assert "catalog:skill summary" in prompt

    messages = builder.build_messages(
        history=[{"role": "assistant", "content": "hi"}],
        current_message="hello",
        media=["https://img", str(image), str(tmp_path / "bad.txt")],
        skill_names=["extra"],
        channel="telegram",
        chat_id="42",
    )
    assert messages[0]["role"] == "system"
    assert "session prompt" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert len(messages[-1]["content"]) == 3

    msgs = builder.add_tool_result([], "call-1", "dummy", "ok")
    assert msgs[-1]["role"] == "tool"
    msgs = builder.add_assistant_message(msgs, None, [{"id": "1"}], "thinking")
    assert msgs[-1]["tool_calls"] == [{"id": "1"}]
    assert msgs[-1]["reasoning_content"] == "thinking"


@pytest.mark.asyncio
async def test_message_bus_covers_flows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    bus = MessageBus()
    await bus.publish_inbound(InboundMessage("telegram", "u", "1", "hello"))
    inbound = await bus.consume_inbound()
    assert inbound.session_key == "telegram:1"

    sent: list[str] = []
    attempts = {"count": 0}

    async def callback(msg: OutboundMessage) -> None:
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("first")
        sent.append(msg.content)

    bus.subscribe_outbound("telegram", callback)
    task = asyncio.create_task(bus.dispatch_outbound())
    await bus.publish_outbound(OutboundMessage("telegram", "1", "payload"))
    for _ in range(300):
        if sent:
            break
        await asyncio.sleep(0.01)
    bus.stop()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sent == ["payload"]
    assert bus.inbound_size == 0
    assert bus.outbound_size == 0


@pytest.mark.asyncio
async def test_loop_trigger_and_main_entry_cover_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    module = __import__("main")
    monkeypatch.setattr(module.Config, "load", classmethod(lambda cls, path="config.json": SimpleNamespace(channels=SimpleNamespace(socket="/tmp/sock"))))
    monkeypatch.setitem(sys.modules, "infra.channels.cli_tui", SimpleNamespace(run_tui=MagicMock()))
    module.connect_cli("config.json")
    sys.modules["infra.channels.cli_tui"].run_tui.assert_called_once_with("/tmp/sock")

    real_import = __import__

    def _fake_import(name, *args, **kwargs):
        if name == "infra.channels.cli_tui":
            raise RuntimeError("bad tui")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "infra.channels.cli_tui", raising=False)
    monkeypatch.setattr("builtins.__import__", _fake_import)
    cli_run = AsyncMock()
    monkeypatch.setitem(
        sys.modules,
        "infra.channels.cli",
        SimpleNamespace(CLIClient=lambda sock: SimpleNamespace(run=cli_run)),
    )

    def _fake_asyncio_run(coro):
        coro.close()
        return None

    monkeypatch.setattr("asyncio.run", _fake_asyncio_run)
    module.connect_cli("config.json")

    runtime = SimpleNamespace(run=AsyncMock())
    monkeypatch.setattr(module.Config, "load", classmethod(lambda cls, path="config.json": SimpleNamespace()))
    monkeypatch.setattr(module, "build_app_runtime", lambda config, workspace: runtime)
    await module.serve("config.json")
    runtime.run.assert_awaited_once()
