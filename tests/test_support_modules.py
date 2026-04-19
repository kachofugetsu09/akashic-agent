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

from agent.context import ContextBuilder, ContextRequest
from agent.tools.base import Tool
from agent.tools.memorize import MemorizeTool
from agent.tools.message_push import MessagePushTool
from agent.tools.registry import ToolMeta, ToolRegistry
from agent.tools.web_search import WebSearchTool
from bus.events import InboundMessage, OutboundMessage
from bus.queue import MessageBus
from core.common import timekit
from core.memory.default_engine import DefaultMemoryEngine
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
    sent = {"text": [], "stream_text": [], "file": [], "image": []}

    async def text(chat_id: str, message: str) -> None:
        sent["text"].append((chat_id, message))

    async def stream_text(chat_id: str, message: str) -> None:
        sent["stream_text"].append((chat_id, message))

    async def file(chat_id: str, path: str, name: str | None) -> None:
        sent["file"].append((chat_id, path, name))

    async def image(chat_id: str, path: str) -> None:
        sent["image"].append((chat_id, path))

    tool.register_channel(
        "telegram",
        text=text,
        stream_text=stream_text,
        file=file,
        image=image,
    )
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
    assert sent["text"] == []
    assert sent["stream_text"] == [("123", "hello")]
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
async def test_memorize_tool_cover_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    memorizer = MagicMock()
    memorizer.save_item_with_supersede = AsyncMock(return_value="new:mem-1")

    class _Tagger:
        async def tag(self, summary: str) -> dict[str, str]:
            assert summary == "记住这条流程"
            return {"scope": "task"}

    tool = MemorizeTool(
        DefaultMemoryEngine(
            retriever=MagicMock(),
            memorizer=memorizer,
            tagger=_Tagger(),
        )
    )
    result = await tool.execute(
        summary="记住这条流程",
        memory_type="procedure",
        steps=["先查", "再做"],
    )

    assert "item_id=mem-1" in result
    assert "status=new" in result
    extra = memorizer.save_item_with_supersede.await_args.kwargs["extra"]
    assert extra["trigger_tags"] == {"scope": "task"}
    assert extra["rule_schema"]["required_tools"] == []
    assert extra["rule_schema"]["forbidden_tools"] == []

    class _BadTagger:
        async def tag(self, summary: str) -> dict[str, str]:
            raise RuntimeError("bad")

    bad = MemorizeTool(
        DefaultMemoryEngine(
            retriever=MagicMock(),
            memorizer=memorizer,
            tagger=_BadTagger(),
        )
    )
    await bad.execute(summary="普通偏好", memory_type="procedure")
    await bad.execute(summary="偏好", memory_type="preference")


@pytest.mark.asyncio
async def test_memorize_tool_should_not_create_second_active_procedure_when_incremental_update():
    class _Embedder:
        async def embed(self, text: str) -> list[float]:
            return [1.0, 0.0]

    store = MemoryStore2(":memory:")
    memorizer = Memorizer(store, _Embedder())
    tool = MemorizeTool(
        DefaultMemoryEngine(
            retriever=MagicMock(),
            memorizer=memorizer,
        )
    )

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
async def test_memorizer_profile_supersede_keeps_high_emotional_weight_item_under_092():
    class _Embedder:
        async def embed(self, text: str) -> list[float]:
            mapping = {
                "用户仍在等待 offer": [1.0, 0.0],
                "用户开始等待新的 offer": [0.91, 0.4146],
            }
            return mapping[text]

    store = MemoryStore2(":memory:")
    memorizer = Memorizer(store, _Embedder())

    await memorizer.save_item(
        summary="用户仍在等待 offer",
        memory_type="profile",
        extra={"category": "status"},
        source_ref="old",
        emotional_weight=8,
    )
    await memorizer.save_item_with_supersede(
        summary="用户开始等待新的 offer",
        memory_type="profile",
        extra={"category": "status"},
        source_ref="new",
    )

    rows = store._db.execute(
        "SELECT source_ref, status FROM memory_items WHERE memory_type='profile' ORDER BY source_ref"
    ).fetchall()
    assert rows == [("new", "active"), ("old", "active")]


@pytest.mark.asyncio
async def test_memorizer_profile_supersede_retires_low_emotional_weight_item_at_091():
    class _Embedder:
        async def embed(self, text: str) -> list[float]:
            mapping = {
                "用户仍在等待 offer": [1.0, 0.0],
                "用户开始等待新的 offer": [0.91, 0.4146],
            }
            return mapping[text]

    store = MemoryStore2(":memory:")
    memorizer = Memorizer(store, _Embedder())

    await memorizer.save_item(
        summary="用户仍在等待 offer",
        memory_type="profile",
        extra={"category": "status"},
        source_ref="old",
        emotional_weight=0,
    )
    await memorizer.save_item_with_supersede(
        summary="用户开始等待新的 offer",
        memory_type="profile",
        extra={"category": "status"},
        source_ref="new",
    )

    rows = store._db.execute(
        "SELECT source_ref, status FROM memory_items WHERE memory_type='profile' ORDER BY source_ref"
    ).fetchall()
    assert rows == [("new", "active"), ("old", "superseded")]


@pytest.mark.asyncio
async def test_memorize_tool_should_coerce_language_reply_rule_to_preference():
    memorizer = MagicMock()
    memorizer.save_item_with_supersede = AsyncMock(return_value="new:mem-1")
    tool = MemorizeTool(
        DefaultMemoryEngine(
            retriever=MagicMock(),
            memorizer=memorizer,
        )
    )

    await tool.execute(
        summary="之后跟我说话只用中文，不要夹杂英文，专有名词也尽量翻译。",
        memory_type="procedure",
    )

    assert memorizer.save_item_with_supersede.await_args.kwargs["memory_type"] == "preference"


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

    with pytest.raises(TypeError, match="必须定义字段：description, parameters"):
        class _MissingTool(Tool):
            name = "bad"

            async def execute(self, **kwargs) -> str:
                return "ok"

    with pytest.raises(TypeError, match="字段不能为空：name, description, parameters"):
        class _EmptyTool(Tool):
            name = ""
            description = ""
            parameters = {}

            async def execute(self, **kwargs) -> str:
                return "ok"

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
        def read_profile(self) -> str:
            return "memory block"

        def read_self(self) -> str:
            return "self note"

        def read_recent_context(self) -> str:
            return ""

    monkeypatch.setattr("agent.context.SkillsLoader", _Skills)
    monkeypatch.setattr(
        "agent.context.build_agent_static_identity_prompt", lambda **_: "identity"
    )
    monkeypatch.setattr(
        "agent.context.build_telegram_rendering_prompt", lambda: "\ntelegram prompt"
    )
    monkeypatch.setattr(
        "agent.context.build_skills_catalog_prompt", lambda text: f"catalog:{text}"
    )

    (tmp_path / "memes").mkdir()
    (tmp_path / "memes" / "manifest.json").write_text(
        '{"version":1,"categories":{"shy":{"desc":"害羞","enabled":true}}}',
        encoding="utf-8",
    )
    image = tmp_path / "a.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    now = datetime.now(timezone.utc)

    builder = ContextBuilder(tmp_path, _Memory())  # type: ignore[arg-type]
    result = builder.render(
        ContextRequest(
            history=[],
            current_message="",
            skill_names=["extra"],
            message_timestamp=now,
            retrieved_memory_block="retrieved",
        )
    )
    prompt = result.system_prompt
    assert "identity" in prompt
    assert "## 行为规范" in prompt
    assert "retrieved" in prompt
    assert "memory block" in prompt
    assert "Akashic 自我认知" in prompt
    assert "request_time=" in prompt
    assert "今天=" in prompt
    assert "明天=" in prompt
    assert "## 环境" in prompt
    assert "# Memes" in prompt
    assert "<meme:shy>" in prompt
    assert "catalog:skill summary" in prompt
    assert [item.name for item in builder.last_debug_breakdown][:1] == ["identity"]

    result2 = builder.render(
        ContextRequest(
            history=[],
            current_message="",
            skill_names=["extra"],
            message_timestamp=now,
            retrieved_memory_block="retrieved",
        )
    )
    assert result2.system_prompt
    identity_meta = next(
        item for item in builder.last_debug_breakdown if item.name == "identity"
    )
    assert identity_meta.cache_hit is True

    messages = builder.render(
        ContextRequest(
            history=[{"role": "assistant", "content": "hi"}],
            current_message="hello",
            media=["https://img", str(image), str(tmp_path / "bad.txt")],
            skill_names=["extra"],
            channel="telegram",
            chat_id="42",
        )
    ).messages
    assert messages[0]["role"] == "system"
    assert "request_time=" in messages[0]["content"]
    assert "今天=" in messages[0]["content"]
    assert "明天=" in messages[0]["content"]
    assert "## 环境" in messages[0]["content"]
    assert "## Current Session" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert len(messages[-1]["content"]) == 3
    assert messages[-1]["content"][-1]["text"].startswith("[当前消息时间:")
    assert builder.last_assembled_contexts["turn_injection_context"] == {}

    turn_injection = builder.build_turn_injection_context(turn_injection_prompt="pref")
    render_result = builder.render(
        ContextRequest(
            history=[{"role": "assistant", "content": "hi"}],
            current_message="hello",
            media=["https://img", str(image), str(tmp_path / "bad.txt")],
            skill_names=["extra"],
            channel="telegram",
            chat_id="42",
            message_timestamp=now,
            turn_injection_prompt="pref",
        )
    )
    assert render_result.system_prompt
    assert render_result.turn_injection_context == turn_injection
    assert render_result.messages


def test_context_builder_reproduces_temporal_conflict_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    class _Skills:
        def __init__(self, workspace: Path) -> None:
            self.workspace = workspace

        def get_always_skills(self) -> list[str]:
            return []

        def load_skills_for_context(self, names: list[str]) -> str:
            return ""

        def build_skills_summary(self) -> str:
            return ""

    class _Memory:
        def read_profile(self) -> str:
            return ""

        def read_self(self) -> str:
            return ""

        def read_recent_context(self) -> str:
            return ""

    monkeypatch.setattr("agent.context.SkillsLoader", _Skills)
    monkeypatch.setattr(
        "agent.context.build_agent_static_identity_prompt", lambda **_: "identity"
    )
    monkeypatch.setattr(
        "agent.context.build_telegram_rendering_prompt", lambda: ""
    )
    monkeypatch.setattr(
        "agent.context.build_skills_catalog_prompt", lambda text: text
    )

    (tmp_path / "memes").mkdir()
    (tmp_path / "memes" / "manifest.json").write_text(
        '{"version":1,"categories":{}}',
        encoding="utf-8",
    )

    builder = ContextBuilder(tmp_path, _Memory())  # type: ignore[arg-type]
    request_time = datetime.fromisoformat("2026-04-08T17:57:00+08:00")
    retrieved_memory_block = """
[item_5a9c8d59f77c] [2026-03-29 12:44] 用户表示明天下午三点有面试，因当前感到疲惫想小睡，但担心此举会打乱明天的生物钟。
证据: 用户消息「明天我下午三点面试 我现在睡一会会打乱明天发生物钟吗有点疲惫」

[item_87aa0364de9e] [2026-03-29 14:42] 用户因午睡未成功，转为练习力扣题目以准备次日下午三点的字节跳动面试。
证据: 用户消息「没睡着做会力扣准备明天面试了」

[item_recent_interview] [2026-04-07 23:10] 用户提到 4 月 9 日（周四）下午 3 点的面试安排。
证据: 可回源原文「4 月 9 日（周四）下午 3 点」
""".strip()

    result = builder.render(
        ContextRequest(
            history=[],
            current_message="你还记得明天什么时候面试吗",
            channel="telegram",
            chat_id="7674283004",
            message_timestamp=request_time,
            retrieved_memory_block=retrieved_memory_block,
        )
    )

    system_prompt = result.messages[0]["content"]
    user_message = result.messages[-1]["content"]

    assert "request_time=2026-04-08T17:57:00+08:00" in system_prompt
    assert "local_date=2026-04-08" in system_prompt
    assert "今天=2026-04-08" in system_prompt
    assert "明天=2026-04-09" in system_prompt
    assert "用户表示明天下午三点有面试" in system_prompt
    assert "准备次日下午三点的字节跳动面试" in system_prompt
    assert "4 月 9 日（周四）下午 3 点" in system_prompt
    assert user_message.startswith("[当前消息时间: 2026-04-08 17:57")
    assert "今天=2026-04-08" in user_message
    assert "明天=2026-04-09" in user_message
    assert user_message.endswith("你还记得明天什么时候面试吗")


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
    monkeypatch.setattr(module.Config, "load", classmethod(lambda cls, path="config.toml": SimpleNamespace(channels=SimpleNamespace(socket="/tmp/sock"))))
    monkeypatch.setitem(sys.modules, "infra.channels.cli_tui", SimpleNamespace(run_tui=MagicMock()))
    module.connect_cli("config.toml")
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
    module.connect_cli("config.toml")

    runtime = SimpleNamespace(run=AsyncMock())
    monkeypatch.setattr(module.Config, "load", classmethod(lambda cls, path="config.toml": SimpleNamespace()))
    monkeypatch.setattr(module, "build_app_runtime", lambda config, workspace: runtime)
    await module.serve("config.toml")
    runtime.run.assert_awaited_once()
