from __future__ import annotations
import asyncio
import json
import runpy
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.context import ContextBuilder, ContextRequest
from agent.persona import reset_veda
from agent.prompting import PromptSectionRender, SYSTEM_CONTEXT_FRAME_MARKER
from agent.tools.base import Tool
from agent.tools.message_push import MessagePushTool
from agent.plugin_composition.channels import (
    ChannelDeliveryReceipt,
    DeliveryStatus as ChannelDeliveryStatus,
)
from agent.tools.registry import ToolMeta, ToolRegistry
from agent.tools.web_search import WebSearchTool
from bus.events import (
    AttachmentKind,
    ChannelAttachment,
    ChannelMessage,
    InboundMessage,
    OutboundMessage,
    TurnTerminalStatus,
)
from bus.queue import ChatLane, MessageBus
from core.common import timekit
from infra.persistence.json_store import atomic_save_json, load_json, save_json
from prompts.agent import build_agent_behavior_rules_prompt
from prompts.completion import VERIFIABLE_COMPLETION_RULES


class _MemoryProfileStub:
    def read_long_term(self) -> str:
        return ""

    def write_long_term(self, content: str) -> None:
        pass

    def read_self(self) -> str:
        return ""

    def write_self(self, content: str) -> None:
        pass

    def backup_long_term(self, backup_name: str = "MEMORY.bak.md") -> None:
        pass

    def backup_self(self, backup_name: str = "SELF.bak.md") -> None:
        pass

    def get_memory_context(self) -> str:
        return ""

    def has_long_term_memory(self) -> bool:
        return False


def test_inbound_message_default_timestamp_is_aware_utc() -> None:
    message = InboundMessage(
        channel="test",
        sender="user",
        chat_id="one",
        content="hello",
    )

    assert message.timestamp.tzinfo is timezone.utc


def test_agent_prompt_uses_authoritative_completion_rules(tmp_path: Path) -> None:
    prompt = build_agent_behavior_rules_prompt(workspace=tmp_path)

    assert VERIFIABLE_COMPLETION_RULES in prompt
    assert "每个主要工具结果后都要把新增证据对应到用户明确提出的要求" in prompt
    assert 'transport_status="success"' in prompt
    assert "只补尚未证明要求的最小缺口" in prompt


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
async def test_message_push_dispatches_exact_v3_receipt_and_media():
    tool = MessagePushTool()
    seen: list[tuple[ChannelMessage, bool]] = []

    async def dispatch(
        message: ChannelMessage, passive: bool
    ) -> ChannelDeliveryReceipt:
        seen.append((message, passive))
        return ChannelDeliveryReceipt(
            delivery_id="delivery-1",
            status=ChannelDeliveryStatus.DELIVERED,
            provider_ids=("provider-1",),
        )

    tool.bind_v3_channel_dispatcher(dispatch)
    result = json.loads(
        await tool.execute(
            target_channel="telegram",
            target_chat_id=123,
            message="hello",
            file="/tmp/demo.txt",
            image="https://img",
        )
    )

    assert result == {
        "delivery_id": "delivery-1",
        "status": "delivered",
        "retryable": False,
        "provider_ids": ["provider-1"],
        "error": None,
    }
    assert not hasattr(tool, "register_channel")
    assert seen[0][1] is False
    assert seen[0][0].attachments == (
        ChannelAttachment(AttachmentKind.FILE, "/tmp/demo.txt", "demo.txt"),
        ChannelAttachment(AttachmentKind.IMAGE, "https://img"),
    )
    assert seen[0][0].metadata == {"source": "message_push"}


@pytest.mark.asyncio
async def test_message_push_missing_committed_dispatcher_fails_loud() -> None:
    tool = MessagePushTool()

    with pytest.raises(RuntimeError, match="committed Channel dispatcher 未绑定"):
        await tool.execute(
            target_channel="telegram",
            target_chat_id="1",
            message="hello",
        )


@pytest.mark.asyncio
async def test_message_push_passive_role_is_forwarded_to_committed_dispatcher() -> None:
    tool = MessagePushTool()
    passive_roles: list[bool] = []
    messages: list[ChannelMessage] = []

    async def dispatch(
        _message: ChannelMessage,
        passive: bool,
    ) -> ChannelDeliveryReceipt:
        passive_roles.append(passive)
        messages.append(_message)
        return ChannelDeliveryReceipt(
            delivery_id="delivery-passive",
            status=ChannelDeliveryStatus.UNKNOWN,
            error="provider outcome unknown",
        )

    tool.bind_v3_channel_dispatcher(dispatch)
    result = json.loads(
        await tool.execute(
            target_channel="mobile",
            target_chat_id="1",
            message="final",
            _commit_role="passive",
        )
    )

    assert passive_roles == [True]
    assert messages[0].metadata == {"source": "message_push"}
    assert result["status"] == "unknown"
    assert result["retryable"] is False


@pytest.mark.asyncio
async def test_passive_terminal_dispatch_does_not_become_message_push() -> None:
    tool = MessagePushTool()
    messages: list[ChannelMessage] = []

    async def dispatch(
        message: ChannelMessage,
        _passive: bool,
    ) -> ChannelDeliveryReceipt:
        messages.append(message)
        return ChannelDeliveryReceipt(
            delivery_id="delivery-terminal",
            status=ChannelDeliveryStatus.DELIVERED,
        )

    tool.bind_v3_channel_dispatcher(dispatch)
    await tool.dispatch(ChannelMessage(
        channel="akashic",
        chat_id="session",
        content="普通最终回复",
        control_turn_id="turn:normal",
        terminal_status=TurnTerminalStatus.COMPLETED,
    ), commit_role="passive")

    assert messages[0].metadata == {}
    assert messages[0].terminal_status is TurnTerminalStatus.COMPLETED


@pytest.mark.asyncio
async def test_chat_lane_cancelled_non_passive_waiter_does_not_wedge_lane():
    lane = ChatLane()
    ran: list[str] = []

    async def first_send() -> None:
        ran.append("first")

    async def second_send() -> None:
        ran.append("second")

    await lane.mark_passive_pending("cli", "1")
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            lane.run_non_passive("cli", "1", first_send),
            timeout=0.01,
        )

    await lane.mark_passive_done("cli", "1")
    await asyncio.wait_for(
        lane.run_non_passive("cli", "1", second_send),
        timeout=1,
    )

    assert ran == ["second"]
    assert lane._states == {}


@pytest.mark.asyncio
async def test_chat_lane_releases_idle_state_after_send_error():
    lane = ChatLane()

    await lane.mark_passive_pending("cli", "1")
    await lane.mark_passive_done("cli", "1")
    assert lane._states == {}

    async def failed_send() -> None:
        raise RuntimeError("send failed")

    with pytest.raises(RuntimeError, match="send failed"):
        await lane.run_passive("cli", "1", failed_send)

    assert lane._states == {}


@pytest.mark.asyncio
async def test_message_push_passive_send_does_not_consume_queued_outbound_pending():
    lane = ChatLane()
    events: list[str] = []

    async def record(value: str) -> None:
        events.append(value)

    await lane.mark_passive_send_pending("cli", "1")
    await lane.run_passive("cli", "1", lambda: record("push"))
    active = asyncio.create_task(
        lane.run_non_passive("cli", "1", lambda: record("active"))
    )

    await asyncio.sleep(0.01)
    assert events == ["push"]
    assert not active.done()

    await lane.run_passive(
        "cli",
        "1",
        lambda: record("outbound"),
        pending_registered=True,
    )
    await asyncio.wait_for(active, timeout=1)

    assert events == ["push", "outbound", "active"]
    assert lane._states == {}


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

    numeric_type_errors = tool.validate_params(
        {"name": "ok", "count": True, "items": [False]}
    )
    assert numeric_type_errors == [
        "count 应为 integer 类型",
        "items[0] 应为 number 类型",
    ]

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
    with pytest.raises(RuntimeError, match=r"\[json_store\].*data\.json"):
        load_json(path, default=[])
    atomic_save_json(path, {"y": 2})
    assert load_json(path)["y"] == 2

    monkeypatch.setattr(
        "pathlib.Path.write_text",
        lambda self, *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    with pytest.raises(RuntimeError):
        save_json(tmp_path / "x.json", {"x": 1})

    monkeypatch.setattr(
        "infra.persistence.json_store.os.fsync",
        lambda _fd: (_ for _ in ()).throw(RuntimeError("bad")),
    )
    with pytest.raises(RuntimeError):
        atomic_save_json(tmp_path / "x.json", {"x": 1})

    parsed = timekit.parse_iso("2025-06-01T09:00:00Z")
    assert parsed and parsed.tzinfo is not None
    assert timekit.parse_iso("bad") is None
    assert timekit.format_iso(datetime(2025, 1, 1)).endswith("+00:00")
    logger = MagicMock()
    assert str(timekit.safe_zone("bad/zone", logger=logger)) == "UTC"
    logger.warning.assert_called_once()
    assert timekit.local_now("UTC").tzinfo is not None
    assert timekit.utcnow().tzinfo is not None


@pytest.mark.asyncio
async def test_context_builder_debug_projection_is_turn_local(tmp_path: Path) -> None:
    """并发 render 只暴露调用 task 自己的诊断投影。"""

    class _Memory(_MemoryProfileStub):
        pass

    _ = reset_veda(tmp_path)
    builder = ContextBuilder(tmp_path, _Memory())
    first_rendered = asyncio.Event()
    second_rendered = asyncio.Event()

    async def render(marker: str) -> tuple[list[object], list[object], dict[str, str]]:
        # 1. 让两个 task 写入不同的 debug 与 turn injection 投影。
        result = builder.render(
            ContextRequest(
                history=[],
                current_message=marker,
                turn_injection_prompt=marker,
            ),
            system_sections_top=[
                PromptSectionRender(
                    name=f"marker-{marker}",
                    content=marker,
                    is_static=False,
                )
            ],
        )
        if marker == "first":
            first_rendered.set()
            await second_rendered.wait()
        else:
            await first_rendered.wait()
            second_rendered.set()

        # 2. 在另一 task 已完成 render 后读取，必须仍得到本 task 的值。
        return (
            list(result.debug_breakdown),
            list(builder.last_debug_breakdown),
            builder.last_assembled_contexts["turn_injection_context"],
        )

    first, second = await asyncio.gather(render("first"), render("second"))

    assert first[1] == first[0]
    assert second[1] == second[0]
    assert first[2] == {"turn_injection": "first"}
    assert second[2] == {"turn_injection": "second"}


def test_context_builder_builds_prompt_messages_and_assistant_blocks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _ = reset_veda(tmp_path)

    class _Skills:
        def __init__(self, workspace: Path, **_: object) -> None:
            self.workspace = workspace

        def get_always_skills(self) -> list[str]:
            return ["always"]

        def load_skills_for_context(self, names: list[str]) -> str:
            return ",".join(names)

        def build_skills_summary(self) -> str:
            return "skill summary"

    class _Memory(_MemoryProfileStub):
        def read_long_term(self) -> str:
            return "memory block"

        def read_self(self) -> str:
            return "self note"

        def get_memory_context(self) -> str:
            return "memory block"

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

    image = tmp_path / "a.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    document = tmp_path / "view.pdf"
    document.write_bytes(b"%PDF-1.4\n")
    now = datetime.now(timezone.utc)
    (tmp_path / "memory").mkdir(exist_ok=True)
    (tmp_path / "memory" / "SELF.md").write_text("self note", encoding="utf-8")

    builder = ContextBuilder(tmp_path, _Memory())
    result = builder.render(
        ContextRequest(
            history=[],
            current_message="",
            skill_names=["extra"],
            message_timestamp=now,
            turn_injection_prompt="retrieved",
        )
    )
    prompt = result.system_prompt
    context_frame = result.messages[-2]["content"]
    assert "identity" in prompt
    assert "## 行为规范" in prompt
    assert "最终回复前逐项核对用户明确提出的要求" in result.messages[0]["content"]
    assert "超过验收标准的完美继续调用工具" in result.messages[0]["content"]
    assert "retrieved" not in prompt
    assert context_frame.startswith(SYSTEM_CONTEXT_FRAME_MARKER)
    assert "retrieved" in context_frame
    assert "memory block" in prompt
    assert "Akashic 自我认知" in prompt
    assert "## 环境" in prompt
    assert "# Memes" not in prompt
    assert "<meme:shy>" not in prompt
    assert "catalog:skill summary" in prompt
    assert [item.name for item in builder.last_debug_breakdown][:2] == [
        "veda",
        "identity",
    ]

    result2 = builder.render(
        ContextRequest(
            history=[],
            current_message="",
            skill_names=["extra"],
            message_timestamp=now,
            turn_injection_prompt="retrieved",
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
            media=["https://img", str(image), str(document), str(tmp_path / "bad.txt")],
            skill_names=["extra"],
            channel="telegram",
            chat_id="42",
        )
    ).messages
    assert messages[0]["role"] == "system"
    assert "## 环境" in messages[0]["content"]
    assert "## Current Session" in messages[0]["content"]
    assert messages[-1]["role"] == "user"
    assert len(messages[-1]["content"]) == 3
    stamped_message = messages[-1]["content"][-1]["text"]
    assert stamped_message.startswith("[当前消息时间:")
    assert "[附加媒体]" in stamped_message
    assert f"- 文件路径: {document}" in stamped_message
    assert f"- 不可用媒体路径: {tmp_path / 'bad.txt'}" in stamped_message
    assert "request_time=" in stamped_message
    assert "今天=" in stamped_message
    assert "昨天=" in stamped_message
    assert "明天=" in stamped_message
    assert "后天=" in stamped_message
    assert "weekday=" in stamped_message
    assert builder.last_assembled_contexts["turn_injection_context"] == {}

    turn_injection = builder.build_turn_injection_context(turn_injection_prompt="pref")
    render_result = builder.render(
        ContextRequest(
            history=[{"role": "assistant", "content": "hi"}],
            current_message="hello",
            media=["https://img", str(image), str(document), str(tmp_path / "bad.txt")],
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
    assert render_result.messages[-2]["role"] == "user"
    assert render_result.messages[-2]["content"].startswith(SYSTEM_CONTEXT_FRAME_MARKER)
    assert "pref" in render_result.messages[-2]["content"]

    custom_telegram = builder.render(
        ContextRequest(
            history=[],
            current_message="hello",
            channel="telegram_work",
            chat_id="42",
            message_timestamp=now,
        )
    )
    assert "telegram prompt" in custom_telegram.messages[0]["content"]

    media_only_messages = builder.render(
        ContextRequest(
            history=[],
            current_message="",
            media=["https://img"],
            skill_names=["extra"],
            message_timestamp=now,
        )
    ).messages
    media_only_text = media_only_messages[-1]["content"][-1]["text"]
    assert media_only_text.startswith("[当前消息时间:")
    assert "request_time=" in media_only_text
    assert "今天=" in media_only_text

    text_media_builder = ContextBuilder(
        tmp_path,
        _Memory(),
        multimodal=False,
        vl_available=True,
    )
    text_media_messages = text_media_builder.render(
        ContextRequest(
            history=[],
            current_message="看看这张图",
            media=[str(image), str(document), str(tmp_path / "bad.txt")],
            skill_names=["extra"],
            message_timestamp=now,
        )
    ).messages
    text_media_content = text_media_messages[-1]["content"]
    assert isinstance(text_media_content, str)
    assert str(image) in text_media_content
    assert str(document) in text_media_content
    assert f"- 不可用媒体路径: {tmp_path / 'bad.txt'}" in text_media_content
    assert "read_image_vision" in text_media_content
    assert "image_url" not in text_media_content

    missing_media_content = text_media_builder.render(
        ContextRequest(
            history=[],
            current_message="附件呢",
            media=[str(tmp_path / "bad.txt")],
        )
    ).messages[-1]["content"]
    assert f"- 不可用媒体路径: {tmp_path / 'bad.txt'}" in missing_media_content
    assert "没有可供 read_image_vision 读取的本地图片" in missing_media_content
    assert "read_image_vision(path=" not in missing_media_content


def test_context_builder_reproduces_temporal_conflict_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _ = reset_veda(tmp_path)

    class _Skills:
        def __init__(self, workspace: Path, **_: object) -> None:
            self.workspace = workspace

        def get_always_skills(self) -> list[str]:
            return []

        def load_skills_for_context(self, names: list[str]) -> str:
            return ""

        def build_skills_summary(self) -> str:
            return ""

    class _Memory(_MemoryProfileStub):
        pass

    monkeypatch.setattr("agent.context.SkillsLoader", _Skills)
    monkeypatch.setattr(
        "agent.context.build_agent_static_identity_prompt", lambda **_: "identity"
    )
    monkeypatch.setattr("agent.context.build_telegram_rendering_prompt", lambda: "")
    monkeypatch.setattr("agent.context.build_skills_catalog_prompt", lambda text: text)

    (tmp_path / "memes").mkdir()
    (tmp_path / "memes" / "manifest.json").write_text(
        '{"version":1,"categories":{}}',
        encoding="utf-8",
    )

    builder = ContextBuilder(tmp_path, _Memory())
    request_time = datetime.fromisoformat("2026-04-08T17:57:00+08:00")
    local_request_time = request_time.astimezone()
    turn_injection_prompt = """
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
            turn_injection_prompt=turn_injection_prompt,
        )
    )

    system_prompt = result.messages[0]["content"]
    context_frame = result.messages[-2]["content"]
    user_message = result.messages[-1]["content"]

    assert "request_time=2026-04-08T17:57:00+08:00" not in system_prompt
    assert "local_date=2026-04-08" not in system_prompt
    assert "今天=2026-04-08" not in system_prompt
    assert "明天=2026-04-09" not in system_prompt
    assert context_frame.startswith(SYSTEM_CONTEXT_FRAME_MARKER)
    assert "用户表示明天下午三点有面试" in context_frame
    assert "准备次日下午三点的字节跳动面试" in context_frame
    assert "4 月 9 日（周四）下午 3 点" in context_frame
    assert user_message.startswith(
        f"[当前消息时间: {local_request_time:%Y-%m-%d %H:%M:%S}"
    )
    assert f"request_time={local_request_time.isoformat()}" in user_message
    assert f"今天={local_request_time:%Y-%m-%d}" in user_message
    assert f"昨天={local_request_time - timedelta(days=1):%Y-%m-%d}" in user_message
    assert f"明天={local_request_time + timedelta(days=1):%Y-%m-%d}" in user_message
    assert f"后天={local_request_time + timedelta(days=2):%Y-%m-%d}" in user_message
    assert f"weekday={local_request_time:%A}" in user_message
    assert "相对时间以此为准" in user_message
    assert user_message.endswith("你还记得明天什么时候面试吗")


@pytest.mark.asyncio
async def test_message_bus_rejects_removed_legacy_outbound_paths():
    bus = MessageBus()
    with pytest.raises(RuntimeError, match="legacy publish_outbound 已删除"):
        await bus.publish_outbound(OutboundMessage("telegram", "1", "payload"))
    with pytest.raises(RuntimeError, match="legacy publish_outbound_awaited 已删除"):
        await bus.publish_outbound_awaited(OutboundMessage("telegram", "1", "payload"))
    assert bus.inbound_size == 0
    assert bus.outbound_size == 0
