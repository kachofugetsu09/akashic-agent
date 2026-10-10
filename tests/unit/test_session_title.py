import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.plugin_composition.models import (
    BoundChatModel,
    LLMResponse,
    ModelRequest,
    TransportError,
)
from agent.plugin_contracts import ContentPart, Control, Input, Message
from plugins.session_title.plugin import (
    Config,
    derive_fallback_title,
    extract_user_text,
    generate_title_with_model,
)


from datetime import UTC, datetime

def _make_input_message(session_id: str, text: str, source: str = "conversation") -> Message:
    return Message(
        message_id="m1",
        session_id=session_id,
        seq=1,
        recorded_at=datetime.now(UTC),
        author="user",
        source=source,
        body=Input(
            parts=(ContentPart(kind="text", value=text),),
        ),
    )


def test_extract_user_text():
    msg = _make_input_message("s1", "  你好，我想了解一下 Python 的异步编程。  ")
    assert extract_user_text(msg) == "你好，我想了解一下 Python 的异步编程。"

    # 非 Input 消息返回空
    other = Message(
        message_id="m2",
        session_id="s1",
        seq=2,
        recorded_at=datetime.now(UTC),
        author="system",
        source="conversation",
        body=Control(action="pause", through_seq=1),
    )
    assert extract_user_text(other) == ""


def test_derive_fallback_title():
    assert derive_fallback_title("") == "新会话"
    assert derive_fallback_title("   ") == "新会话"
    short_text = "帮我写个脚本"
    assert derive_fallback_title(short_text) == short_text

    long_text = "这是一个非常非常非常非常非常长的问题，关于分布式的并发控制与一致性哈希算法在生产环境中的落地与调优"
    fallback = derive_fallback_title(long_text, max_chars=20)
    assert len(fallback) <= 23
    assert fallback.endswith("...")


@pytest.mark.asyncio
async def test_generate_title_with_model_success():
    mock_model = AsyncMock(spec=BoundChatModel)
    mock_model.complete.return_value = LLMResponse(
        content="\"Python 异步编程探索\"",
        usage=None,
    )

    title = await generate_title_with_model(
        mock_model,
        "你好，我想了解一下 Python 的异步编程",
        max_chars=36,
    )
    assert title == "Python 异步编程探索"


@pytest.mark.asyncio
async def test_generate_title_with_model_failure_returns_none():
    mock_model = AsyncMock(spec=BoundChatModel)
    mock_model.complete.side_effect = TransportError("Connection refused")

    title = await generate_title_with_model(
        mock_model,
        "测试提示词",
        max_chars=36,
    )
    assert title is None


@pytest.mark.asyncio
async def test_generate_title_with_model_empty_response():
    mock_model = AsyncMock(spec=BoundChatModel)
    mock_model.complete.return_value = LLMResponse(
        content="   ",
        usage=None,
    )

    title = await generate_title_with_model(
        mock_model,
        "测试提示词",
        max_chars=36,
    )
    assert title is None


from contextlib import asynccontextmanager
from plugins.session_title.plugin import apply
from agent.plugin_contracts.sources import SourceChangedV3


@pytest.mark.asyncio
async def test_apply_flow_success_generates_and_sets_title():
    session_id = "sess_123"
    msg = _make_input_message(session_id, "帮我写一个 Python 贪吃蛇游戏")

    mock_reader = MagicMock()
    mock_reader.session_id = session_id
    mock_reader.title = None
    mock_reader.read.return_value = (msg,)

    mock_catalog = MagicMock()
    mock_catalog.reader.return_value = mock_reader

    mock_session_admin = AsyncMock()

    mock_bound_model = AsyncMock(spec=BoundChatModel)
    mock_bound_model.complete.return_value = LLMResponse(
        content="Python 贪吃蛇游戏",
        usage=None,
    )

    mock_execution = MagicMock()
    mock_execution.chat.return_value = mock_bound_model

    @asynccontextmanager
    async def fake_independent_execution():
        yield mock_execution

    mock_chat_models = MagicMock()
    mock_chat_models.independent_execution = fake_independent_execution

    listeners = {}

    async def mock_on(event_key, handler):
        listeners[event_key.name] = handler

    mock_ctx = MagicMock()
    mock_ctx.config = {}
    mock_ctx.on = mock_on
    mock_ctx.require.side_effect = lambda key: {
        "core.session_admin": mock_session_admin,
        "core.message_catalog": mock_catalog,
        "models.chat.v1": mock_chat_models,
    }[key.name]

    await apply(mock_ctx)

    source_changed_handler = listeners["source.changed.v3"]
    event = SourceChangedV3(reader=mock_reader, source="conversation", pending=False)
    await source_changed_handler(event)

    await asyncio.sleep(0.05)

    mock_session_admin.set_title.assert_awaited_once_with(session_id, "Python 贪吃蛇游戏")


@pytest.mark.asyncio
async def test_apply_flow_skips_if_title_already_set():
    session_id = "sess_456"
    msg = _make_input_message(session_id, "你好")

    mock_reader = MagicMock()
    mock_reader.session_id = session_id
    mock_reader.title = "用户自定义标题"
    mock_reader.read.return_value = (msg,)

    mock_catalog = MagicMock()
    mock_catalog.reader.return_value = mock_reader

    mock_session_admin = AsyncMock()
    mock_chat_models = MagicMock()

    listeners = {}

    async def mock_on(event_key, handler):
        listeners[event_key.name] = handler

    mock_ctx = MagicMock()
    mock_ctx.config = {}
    mock_ctx.on = mock_on
    mock_ctx.require.side_effect = lambda key: {
        "core.session_admin": mock_session_admin,
        "core.message_catalog": mock_catalog,
        "models.chat.v1": mock_chat_models,
    }[key.name]

    await apply(mock_ctx)

    source_changed_handler = listeners["source.changed.v3"]
    event = SourceChangedV3(reader=mock_reader, source="conversation", pending=False)
    await source_changed_handler(event)

    await asyncio.sleep(0.05)

    mock_session_admin.set_title.assert_not_called()


@pytest.mark.asyncio
async def test_apply_flow_fallback_on_model_error():
    session_id = "sess_789"
    msg = _make_input_message(session_id, "分析一下这个系统的架构")

    mock_reader = MagicMock()
    mock_reader.session_id = session_id
    mock_reader.title = None
    mock_reader.read.return_value = (msg,)

    mock_catalog = MagicMock()
    mock_catalog.reader.return_value = mock_reader

    mock_session_admin = AsyncMock()

    mock_bound_model = AsyncMock(spec=BoundChatModel)
    mock_bound_model.complete.side_effect = TransportError("LLM offline")

    mock_execution = MagicMock()
    mock_execution.chat.return_value = mock_bound_model

    @asynccontextmanager
    async def fake_independent_execution():
        yield mock_execution

    mock_chat_models = MagicMock()
    mock_chat_models.independent_execution = fake_independent_execution

    listeners = {}

    async def mock_on(event_key, handler):
        listeners[event_key.name] = handler

    mock_ctx = MagicMock()
    mock_ctx.config = {}
    mock_ctx.on = mock_on
    mock_ctx.require.side_effect = lambda key: {
        "core.session_admin": mock_session_admin,
        "core.message_catalog": mock_catalog,
        "models.chat.v1": mock_chat_models,
    }[key.name]

    await apply(mock_ctx)

    source_changed_handler = listeners["source.changed.v3"]
    event = SourceChangedV3(reader=mock_reader, source="conversation", pending=False)
    await source_changed_handler(event)

    await asyncio.sleep(0.05)

    mock_session_admin.set_title.assert_awaited_once_with(session_id, "分析一下这个系统的架构")
