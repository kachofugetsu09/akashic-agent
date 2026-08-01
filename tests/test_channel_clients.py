from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import types
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from bus.event_bus import EventBus
from bus.events import OutboundMessage
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from infra.channels.base import AttachmentStore
from infra.channels.contract import ChannelContext


class _Bus:
    def __init__(self) -> None:
        self.inbound = []
        self.outbound = []

    async def publish_inbound(self, msg) -> None:
        self.inbound.append(msg)

    def subscribe_outbound(self, channel, callback) -> None:
        self.outbound.append((channel, callback))


class _SessionManager:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.sessions = {}
        self.saved = []

    def get_or_create(self, key: str):
        return self.sessions.setdefault(key, SimpleNamespace(key=key, metadata={}))

    async def save_async(self, session) -> None:
        self.saved.append(session.key)

    def get_channel_metadata(self, channel: str):
        return []


def _import_telegram_channel(monkeypatch: pytest.MonkeyPatch):
    telegram = types.ModuleType("telegram")
    telegram_constants = types.ModuleType("telegram.constants")
    telegram_error = types.ModuleType("telegram.error")
    telegram_ext = types.ModuleType("telegram.ext")

    class Update:
        ALL_TYPES = ["message"]

    class Bot:
        async def edit_message_text(self, *args, **kwargs):
            return True

    class BotCommand:
        def __init__(self, command, description):
            self.command = command
            self.description = description

    class MessageEntity:
        def __init__(self, *, type, offset, length):
            self.type = type
            self.offset = offset
            self.length = length

    class TelegramError(Exception):
        pass

    class Conflict(TelegramError):
        pass

    class BadRequest(TelegramError):
        pass

    class RetryAfter(TelegramError):
        def __init__(self, retry_after=1.0):
            super().__init__(retry_after)
            self.retry_after = retry_after

    class NetworkError(TelegramError):
        pass

    class TimedOut(TelegramError):
        pass

    class _Filter:
        def __and__(self, other):
            return self

        def __invert__(self):
            return self

    class _Document:
        ALL = _Filter()

    class MessageHandler:
        def __init__(self, flt, callback):
            self.filter = flt
            self.callback = callback

    class CommandHandler:
        def __init__(self, command, callback):
            self.command = command
            self.callback = callback

    class _Updater:
        def __init__(self):
            self.running = False
            self.error_callback = None

        async def start_polling(self, **kwargs):
            self.running = True
            self.error_callback = kwargs.get("error_callback")

        async def stop(self):
            self.running = False

    class _Builder:
        def __init__(self):
            self._token = None

        def token(self, token):
            self._token = token
            return self

        def build(self):
            return _Application(self._token)

    class _Application:
        def __init__(self, token):
            self.token = token
            self.bot = SimpleNamespace(
                send_message=AsyncMock(return_value=SimpleNamespace(message_id=99)),
                edit_message_text=AsyncMock(),
                send_document=AsyncMock(),
                send_photo=AsyncMock(),
                send_chat_action=AsyncMock(),
                delete_message=AsyncMock(),
                get_file=AsyncMock(),
                set_my_commands=AsyncMock(),
            )
            self.updater = _Updater()
            self.handlers = []

        @classmethod
        def builder(cls):
            return _Builder()

        async def initialize(self):
            return None

        async def start(self):
            return None

        async def stop(self):
            return None

        async def shutdown(self):
            return None

        def add_handler(self, handler):
            self.handlers.append(handler)

    telegram.Bot = Bot
    telegram.BotCommand = BotCommand
    telegram.MessageEntity = MessageEntity
    telegram.Update = Update
    telegram_constants.ChatAction = SimpleNamespace(TYPING="typing")
    telegram_error.Conflict = Conflict
    telegram_error.BadRequest = BadRequest
    telegram_error.NetworkError = NetworkError
    telegram_error.RetryAfter = RetryAfter
    telegram_error.TelegramError = TelegramError
    telegram_error.TimedOut = TimedOut
    telegram_ext.Application = _Application
    telegram_ext.ContextTypes = SimpleNamespace(DEFAULT_TYPE=object)
    telegram_ext.CommandHandler = CommandHandler
    telegram_ext.MessageHandler = MessageHandler
    telegram_ext.filters = SimpleNamespace(
        TEXT=_Filter(),
        COMMAND=_Filter(),
        PHOTO=_Filter(),
        Document=_Document(),
    )
    monkeypatch.setitem(sys.modules, "telegram", telegram)
    monkeypatch.setitem(sys.modules, "telegram.constants", telegram_constants)
    monkeypatch.setitem(sys.modules, "telegram.error", telegram_error)
    monkeypatch.setitem(sys.modules, "telegram.ext", telegram_ext)
    sys.modules.pop("infra.channels.telegram_channel", None)
    return importlib.import_module("infra.channels.telegram_channel")


def _import_qq_channel(monkeypatch: pytest.MonkeyPatch):
    ncatbot_core = types.ModuleType("ncatbot.core")
    ncatbot_core_adapter = types.ModuleType("ncatbot.core.adapter")
    ncatbot_core_adapter_adapter = types.ModuleType("ncatbot.core.adapter.adapter")
    ncatbot_utils = types.ModuleType("ncatbot.utils")
    captured_connect_calls = []

    class _Api:
        def __init__(self):
            self.calls = []

        async def send_group_text(self, group_id, content):
            self.calls.append(("group_text", group_id, content))

        async def send_private_text(self, user_id, content):
            self.calls.append(("private_text", user_id, content))

        async def send_group_file(self, group_id, uri, name):
            self.calls.append(("group_file", group_id, uri, name))

        async def send_private_file(self, user_id, uri, name):
            self.calls.append(("private_file", user_id, uri, name))

        async def send_group_image(self, group_id, image):
            self.calls.append(("group_image", group_id, image))

        async def send_private_image(self, user_id, image):
            self.calls.append(("private_image", user_id, image))

    class BotClient:
        def __init__(self):
            self.api = _Api()
            self.private_handler = None
            self.group_handler = None
            self.startup_handler = None

        def on_private_message(self):
            def _wrap(fn):
                self.private_handler = fn
                return fn

            return _wrap

        def on_group_message(self):
            def _wrap(fn):
                self.group_handler = fn
                return fn

            return _wrap

        def on_startup(self):
            def _wrap(fn):
                self.startup_handler = fn
                return fn

            return _wrap

        def run_backend(self):
            return self.api

        def exit(self):
            return None

    class ForwardConstructor:
        def __init__(self, user_id, nickname):
            self.user_id = user_id
            self.nickname = nickname
            self.nodes = []

        def attach_text(self, text, nickname=None):
            self.nodes.append(
                {
                    "type": "text",
                    "data": {"text": text},
                    "nickname": nickname or self.nickname,
                    "user_id": self.user_id,
                }
            )

        def to_forward(self):
            class _Forward:
                def __init__(self, nodes):
                    self._nodes = nodes

                def to_forward_dict(self):
                    return {
                        "messages": list(self._nodes),
                        "news": [],
                        "prompt": "",
                        "summary": "",
                        "source": "",
                    }

            return _Forward(self.nodes)

    def _fake_connect(*args, **kwargs):
        captured_connect_calls.append(kwargs.copy())
        return ("connect", args, kwargs)

    ncatbot_core.BotClient = BotClient
    ncatbot_core.ForwardConstructor = ForwardConstructor
    ncatbot_core_adapter_adapter.websockets = SimpleNamespace(connect=_fake_connect)
    ncatbot_core_adapter_adapter._captured_connect_calls = captured_connect_calls
    ncatbot_utils.ncatbot_config = SimpleNamespace(
        bt_uin="",
        root="",
        check_ncatbot_update=True,
        skip_ncatbot_install_check=False,
        napcat=SimpleNamespace(remote_mode=False, enable_webui=True),
        enable_webui_interaction=True,
        plugin=SimpleNamespace(plugins_dir=""),
    )
    monkeypatch.setitem(sys.modules, "ncatbot.core", ncatbot_core)
    monkeypatch.setitem(sys.modules, "ncatbot.core.adapter", ncatbot_core_adapter)
    monkeypatch.setitem(
        sys.modules,
        "ncatbot.core.adapter.adapter",
        ncatbot_core_adapter_adapter,
    )
    monkeypatch.setitem(sys.modules, "ncatbot.utils", ncatbot_utils)
    sys.modules.pop("infra.channels.qq_channel", None)
    return importlib.import_module("infra.channels.qq_channel")


def test_qq_channel_ws_timeout_patch_is_best_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mod = _import_qq_channel(monkeypatch)
    monkeypatch.delitem(sys.modules, "ncatbot.core.adapter.adapter", raising=False)

    mod._patch_ncatbot_ws_open_timeout(7.5)


@pytest.mark.asyncio
async def test_telegram_channel_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mod = _import_telegram_channel(monkeypatch)
    bus = _Bus()
    event_bus = EventBus()
    session_manager = _SessionManager(tmp_path)
    interrupt_controller = MagicMock()
    interrupt_controller.request_interrupt.return_value = SimpleNamespace(
        status="interrupted",
        session_key="telegram:123",
        message="已中断",
    )
    channel = mod.TelegramChannel(
        "token",
        bus,
        session_manager,
        allow_from=["1", "Alice"],
        bot_commands=[
            ("memorystatus", "查看记忆整理状态"),
            ("kvcache", "查看 KVCache 状态"),
        ],
        event_bus=event_bus,
        interrupt_controller=interrupt_controller,
    )
    channel._telegram_outbound_limiter = mod.TelegramOutboundLimiter(
        send_interval_s=0.0,
        edit_interval_s=0.0,
        typing_interval_s=0.0,
        global_interval_s=0.0,
        retry_padding_s=0.0,
    )
    channel._live_edit_queue = mod.TelegramLiveEditQueue(
        min_interval_s=0.0,
        limiter=channel._telegram_outbound_limiter,
    )
    monkeypatch.setattr(mod, "send_markdown", AsyncMock())
    monkeypatch.setattr(mod, "send_stream_markdown", AsyncMock())
    monkeypatch.setattr(mod, "send_thinking_block", AsyncMock())
    await channel.start()
    assert len(channel._app.handlers) == 5
    assert [cmd.command for cmd in channel._app.bot.set_my_commands.await_args.args[0]] == [
        "memorystatus",
        "kvcache",
        "stop",
    ]
    assert bus.outbound[0][0] == "telegram"

    class _File:
        def __init__(self, suffix):
            self.suffix = suffix

        async def download_to_drive(self, path):
            Path(path).write_text("x", encoding="utf-8")

    channel._app.bot.get_file = AsyncMock(
        side_effect=[_File(".jpg"), _File(".txt"), _File(".jpg"), _File(".txt"), _File(".md")]
    )
    context = SimpleNamespace(bot=channel._app.bot)
    reply_photo = [SimpleNamespace(file_id="p1")]
    reply_doc = SimpleNamespace(file_id="d1", file_name="note.txt")
    reply_user = SimpleNamespace(id=2, username="other")
    reply_msg = SimpleNamespace(
        text="原消息",
        caption="",
        photo=reply_photo,
        document=reply_doc,
        from_user=reply_user,
        message_id=9,
    )
    update = SimpleNamespace(
        effective_message=SimpleNamespace(
            text="你好",
            message_id=1,
            reply_to_message=reply_msg,
            photo=None,
            document=None,
        ),
        effective_chat=SimpleNamespace(id=123),
        effective_user=SimpleNamespace(id=1, username="Alice"),
    )
    await channel._on_message(update, context)
    assert len(bus.inbound) == 1
    assert bus.inbound[0].metadata["reply_to_sender"] == "@other"
    assert len(bus.inbound[0].media) == 2

    stop_update = SimpleNamespace(
        effective_message=SimpleNamespace(text="/stop", message_id=99),
        effective_chat=SimpleNamespace(id=123),
        effective_user=SimpleNamespace(id=1, username="Alice"),
    )
    await channel._on_stop_command(stop_update, context)
    interrupt_controller.request_interrupt.assert_called_once_with(
        session_key="telegram:123",
        sender="1",
        command="/stop",
    )
    assert len(bus.inbound) == 1

    status_update = SimpleNamespace(
        effective_message=SimpleNamespace(text="/memorystatus", message_id=100),
        effective_chat=SimpleNamespace(id=123),
        effective_user=SimpleNamespace(id=1, username="Alice"),
    )
    await channel._on_command(status_update, context)
    assert len(bus.inbound) == 2
    assert bus.inbound[1].content == "/memorystatus"
    assert bus.inbound[1].metadata["username"] == "Alice"

    kvcache_update = SimpleNamespace(
        effective_message=SimpleNamespace(text="/kvcache 5", message_id=101),
        effective_chat=SimpleNamespace(id=123),
        effective_user=SimpleNamespace(id=1, username="Alice"),
    )
    await channel._on_command(kvcache_update, context)
    assert len(bus.inbound) == 3
    assert bus.inbound[2].content == "/kvcache 5"
    assert bus.inbound[2].metadata["username"] == "Alice"

    photo_update = SimpleNamespace(
        effective_message=SimpleNamespace(
            photo=[SimpleNamespace(file_id="main"), SimpleNamespace(file_id="main2")],
            message_id=2,
            caption="图说",
            reply_to_message=SimpleNamespace(
                photo=[SimpleNamespace(file_id="rp")],
                text="",
                caption="",
                from_user=reply_user,
                message_id=10,
            ),
        ),
        effective_chat=SimpleNamespace(id=123),
        effective_user=SimpleNamespace(id=1, username="Alice"),
    )
    await channel._on_photo(photo_update, context)

    doc_update = SimpleNamespace(
        effective_message=SimpleNamespace(
            document=SimpleNamespace(file_id="doc1", file_name="a.md", mime_type="text/plain"),
            caption="",
            reply_to_message=None,
        ),
        effective_chat=SimpleNamespace(id=123),
        effective_user=SimpleNamespace(id=1, username="Alice"),
    )
    await channel._on_document(doc_update, context)
    assert len(bus.inbound) == 5
    assert bus.inbound[-1].metadata["document_filename"] == "a.md"

    assert channel._resolve_chat_id("123") == "123"
    channel.user_map["alice"] = "456"
    assert channel._resolve_chat_id("@Alice") == "456"
    with pytest.raises(ValueError):
        channel._resolve_chat_id("@missing")

    await channel.send("123", "hi")
    await channel.send_stream("123", "stream hi")
    sample = tmp_path / "doc.txt"
    sample.write_text("x", encoding="utf-8")
    await channel.send_file("123", str(sample), name="doc.txt", caption="cap")
    await channel.send_image("123", "https://example.com/img.jpg")
    await channel.send_image("123", str(sample))
    await channel._on_response(OutboundMessage(channel="telegram", chat_id="123", content="pong"))
    assert mod.send_markdown.await_count == 3
    assert mod.send_stream_markdown.await_count == 1
    sender = channel.create_stream_sender("123")
    assert sender is not None
    await sender({"thinking_delta": "先想一点"})
    await sender("流式片段")
    await sender("继续补充一大段内容继续补充一大段内容继续补充一大段内容继续补充一大段内容")
    assert channel._app.bot.send_message.await_count >= 1
    before_send = channel._app.bot.send_message.await_count
    before_edit = channel._app.bot.edit_message_text.await_count
    live = mod.TelegramLiveTextMessage(
        channel._app.bot,
        mod.TelegramLiveEditQueue(min_interval_s=0.0),
        123,
    )
    await asyncio.gather(
        live.update("工具调用\na"),
        live.update("工具调用\nb"),
        live.update("工具调用\nc"),
    )
    assert channel._app.bot.send_message.await_count == before_send + 1
    assert channel._app.bot.edit_message_text.await_count >= before_edit + 1
    await event_bus.observe(
        StreamDeltaReady(
            session_key="telegram:456",
            channel="telegram",
            chat_id="456",
            content_delta="事件片段",
        )
    )
    assert channel._active_streams.get("456") is None
    await asyncio.sleep(0)
    assert channel._live_messages.get("telegram:456") is not None
    channel._thinking_live_next_at["telegram:456"] = 0.0
    await event_bus.observe(
        StreamDeltaReady(
            session_key="telegram:456",
            channel="telegram",
            chat_id="456",
            thinking_delta="事件思考",
        )
    )
    await asyncio.sleep(0)
    live_texts = [
        call.kwargs.get("text", "")
        for call in (
            channel._app.bot.send_message.await_args_list
            + channel._app.bot.edit_message_text.await_args_list
        )
    ]
    assert any(
        "临时回复" in text and "事件片段" in text and "思考过程" in text and "事件思考" in text
        for text in live_texts
    )
    assert any(
        text.find("思考过程") < text.find("临时回复")
        for text in live_texts
        if "思考过程" in text and "临时回复" in text
    )
    before_threshold_edit = channel._app.bot.edit_message_text.await_count
    await event_bus.observe(
        StreamDeltaReady(
            session_key="telegram:456",
            channel="telegram",
            chat_id="456",
            thinking_delta="继续分析" * 60,
        )
    )
    await asyncio.sleep(0)
    assert channel._app.bot.edit_message_text.await_count > before_threshold_edit
    await event_bus.observe(
        ToolCallStarted(
            session_key="telegram:456",
            channel="telegram",
            chat_id="456",
            iteration=1,
            call_id="call-1",
            tool_name="shell",
            arguments={"cmd": "df -h", "description": "查看磁盘空间"},
        )
    )
    await event_bus.observe(
        ToolCallCompleted(
            session_key="telegram:456",
            channel="telegram",
            chat_id="456",
            iteration=1,
            call_id="call-1",
            tool_name="shell",
            arguments={"cmd": "df -h", "description": "查看磁盘空间"},
            final_arguments={"cmd": "df -h", "description": "查看磁盘空间"},
            status="ok",
            result_preview="exit=0",
        )
    )
    await asyncio.sleep(0)
    if channel._live_tasks:
        await asyncio.gather(*list(channel._live_tasks))
    assert channel._live_messages.get("telegram:456") is not None
    assert any(
        "工具调用" in call.kwargs.get("text", "")
        for call in channel._app.bot.send_message.await_args_list
    )
    tool_texts = [
        call.kwargs.get("text", "")
        for call in (
            channel._app.bot.send_message.await_args_list
            + channel._app.bot.edit_message_text.await_args_list
        )
        if "工具调用" in call.kwargs.get("text", "")
    ]
    assert any(
        "shell: 查看磁盘空间" in text and "df -h" in text and "✅" in text
        for text in tool_texts
    )
    assert all("exit=0" not in text for text in tool_texts)
    long_text, long_html = mod._format_turn_live(
        [
            mod._ToolLiveLine(
                call_id="long",
                tool_name="shell",
                intent="查看长输出",
                target="工具开头" + "x" * 1300 + "工具结尾",
                status="done",
            )
        ],
        "回复开头" + "y" * 1300 + "回复结尾",
        "思考开头" + "z" * 1600 + "思考结尾",
    )
    assert "思考结尾" in long_text and "思考开头" not in long_text
    assert "工具结尾" in long_text and "工具开头" not in long_text
    assert "回复结尾" in long_text and "回复开头" not in long_text
    assert "<blockquote>" in long_html and "<pre>" in long_html
    channel.user_map["group"] = "-1001"
    assert channel.create_stream_sender("@group") is None
    await channel._on_response(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="final",
            metadata={"streamed_reply": True},
        )
    )
    assert channel._app.bot.edit_message_text.await_count >= 1
    assert mod.send_markdown.await_count == 3
    assert mod.send_stream_markdown.await_count == 1
    mod.send_thinking_block.reset_mock()
    before_final_markdown = mod.send_markdown.await_count
    before_delete = channel._app.bot.delete_message.await_count
    await channel._on_response(
        OutboundMessage(
            channel="telegram",
            chat_id="456",
            content="事件最终回复",
            thinking="继续分析",
        )
    )
    assert channel._app.bot.delete_message.await_count == before_delete + 1
    assert "telegram:456" not in channel._tool_lines
    assert "telegram:456" not in channel._thinking_live_next_at
    assert "telegram:456" not in channel._live_last_lengths
    mod.send_thinking_block.assert_awaited_once()
    assert mod.send_markdown.await_count == before_final_markdown + 2
    snapshot_text = mod.send_markdown.await_args_list[-2].args[2]
    assert "工具调用" in snapshot_text
    assert "事件思考继续分析" not in snapshot_text
    assert "临时回复" not in snapshot_text
    assert snapshot_text.startswith("```")

    mod.send_thinking_block.reset_mock()
    sender = channel.create_stream_sender("123")
    assert sender is not None
    await sender({"thinking_delta": "分析中"})
    await channel._on_response(
        OutboundMessage(
            channel="telegram",
            chat_id="123",
            content="final",
            thinking="分析中",
            metadata={"streamed_reply": True},
        )
    )
    mod.send_thinking_block.assert_awaited_once()
    last_edit = channel._app.bot.edit_message_text.await_args_list[-1].kwargs["text"]
    assert last_edit == "final"

    channel._app.bot.send_chat_action = AsyncMock(side_effect=[mod.TimedOut("x"), mod.NetworkError("x"), None])
    monkeypatch.setattr(mod.asyncio, "sleep", AsyncMock(return_value=None))
    await channel._safe_send_typing(context, 123)
    channel._app.bot.send_chat_action = AsyncMock(side_effect=RuntimeError("boom"))
    await channel._safe_send_typing(context, 123)

    created = []
    real_create_task = asyncio.create_task

    def _capture_task(coro):
        task = real_create_task(coro)
        created.append(task)
        return task

    monkeypatch.setattr(mod.asyncio, "create_task", _capture_task)
    channel._on_polling_error(mod.Conflict("conflict"))
    if created:
        await asyncio.gather(*created)
    channel._on_polling_error(mod.TelegramError("warn"))
    await channel.stop()

    merged, meta = mod._build_inbound_text_with_reply("hi", None)
    assert (merged, meta) == ("hi", {})
    merged, meta = mod._build_inbound_text_with_reply(
        "hi",
        SimpleNamespace(text="", caption="", photo=[1], from_user=None, message_id=11),
    )
    assert "[图片]" in merged


@pytest.mark.asyncio
async def test_telegram_live_task_index_releases_finished_session(monkeypatch: pytest.MonkeyPatch):
    mod = _import_telegram_channel(monkeypatch)
    channel = object.__new__(mod.TelegramChannel)
    channel._live_tasks = set()
    channel._live_tasks_by_session = {}

    async def complete() -> None:
        return None

    channel._start_live_task("telegram:stale", complete())
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert channel._live_tasks == set()
    assert channel._live_tasks_by_session == {}


@pytest.mark.asyncio
async def test_telegram_live_message_is_retained_when_delete_fails(monkeypatch):
    mod = _import_telegram_channel(monkeypatch)
    channel = object.__new__(mod.TelegramChannel)
    message = SimpleNamespace(delete=AsyncMock(return_value=False))
    channel._live_messages = {"telegram:stale": message}

    await channel._delete_live_message("telegram:stale")

    assert channel._live_messages["telegram:stale"] is message


@pytest.mark.asyncio
async def test_qq_channel_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mod = _import_qq_channel(monkeypatch)
    bus = _Bus()
    session_manager = _SessionManager(tmp_path)
    class _Response:
        status_code = 200
        headers = {"content-type": "image/png"}

        async def aiter_bytes(self, *, chunk_size: int):
            _ = chunk_size
            yield b"img"

    class _Requester:
        @asynccontextmanager
        async def stream(self, method: str, url: str, **kwargs: object):
            _ = (method, kwargs)
            if not (url.endswith("a.jpg") or url.endswith("a.png")):
                raise RuntimeError("boom")
            yield _Response()

    requester = _Requester()
    group_filter = SimpleNamespace(should_process=AsyncMock(return_value=True))
    group_cfg = SimpleNamespace(group_id="100")
    channel = mod.QQChannel(
        "42",
        bus,
        session_manager,
        allow_from=["1"],
        groups=[group_cfg],
        websocket_open_timeout_seconds=7.5,
        group_filter=group_filter,
        http_requester=requester,
        interrupt_controller=SimpleNamespace(
            request_interrupt=MagicMock(
                return_value=SimpleNamespace(
                    status="interrupted",
                    session_key="qq:1",
                    message="已中断",
                )
            )
        ),
    )
    adapter_mod = sys.modules["ncatbot.core.adapter.adapter"]
    adapter_mod.websockets.connect("ws://example.invalid", open_timeout=1)
    assert adapter_mod._captured_connect_calls[-1]["open_timeout"] == 7.5
    assert sys.modules["ncatbot.utils"].ncatbot_config.root == "1"
    assert channel._is_allowed("1") is True
    assert channel._is_allowed("2") is False
    assert mod._extract_cq_images("hello [CQ:image,url=http://x/a.jpg]") == ("hello", ["http://x/a.jpg"])

    scheduled = []
    real_create_task = asyncio.create_task

    def _run_coroutine_threadsafe(coro, loop):
        scheduled.append(real_create_task(coro))
        return SimpleNamespace(result=lambda timeout=None: True)

    monkeypatch.setattr(mod.asyncio, "run_coroutine_threadsafe", _run_coroutine_threadsafe)
    await channel.start()
    assert bus.outbound[0][0] == "qq"

    async def _drain(coro):
        return await coro

    channel._run_on_bot_loop = AsyncMock(side_effect=_drain)

    await channel._bot.startup_handler(SimpleNamespace())
    await channel._bot.private_handler(SimpleNamespace(user_id="1", raw_message="hi [CQ:image,url=http://x/a.jpg]"))
    await channel._bot.group_handler(SimpleNamespace(group_id="100", user_id="1", raw_message="hello"))
    await channel._bot.private_handler(SimpleNamespace(user_id="1", raw_message="/stop"))
    await channel._bot.group_handler(SimpleNamespace(group_id="100", user_id="1", raw_message="/stop"))
    if scheduled:
        await asyncio.gather(*scheduled)
    assert len(bus.inbound) == 2
    assert bus.inbound[0].metadata["chat_type"] == "private"
    assert bus.inbound[1].metadata["chat_type"] == "group"
    assert channel._interrupt_controller.request_interrupt.call_count == 2

    channel._run_on_bot_loop = AsyncMock(side_effect=_drain)
    sample = tmp_path / "image.bin"
    sample.write_bytes(b"abc")
    await channel.send("1", "pong")
    await channel.send("gqq:100", "group pong")
    await channel.send_file("1", str(sample), name="x.bin")
    await channel.send_image("1", str(sample))
    await channel._on_response(OutboundMessage(channel="qq", chat_id="gqq:100", content="reply"))
    assert channel._api.calls
    assert mod._is_local(str(sample)) is True
    assert mod._is_local("https://example.com/x.jpg") is False
    assert mod._local_to_base64(str(sample)).startswith("base64://")
    oversized = tmp_path / "oversized.bin"
    oversized.write_bytes(b"x" * (mod.MAX_QQ_IMAGE_BYTES + 1))
    with pytest.raises(ValueError, match="QQ 图片不能超过"):
        mod._local_to_base64(str(oversized))

    test_attachments = mod.AttachmentStore(tmp_path / "uploads")
    paths = await mod._download_to_temp(
        ["http://x/a.png", "http://x/b.png"],
        requester,
        test_attachments,
    )
    assert len(paths) == 1

    channel._bot_loop = None
    pending = asyncio.sleep(0)
    with pytest.raises(RuntimeError):
        await mod.QQChannel._run_on_bot_loop(channel, pending)
    pending.close()
    await channel.stop()


@pytest.mark.asyncio
async def test_qq_private_trace_sends_forward_then_final_and_clears_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    mod = _import_qq_channel(monkeypatch)
    bus = _Bus()
    session_manager = _SessionManager(tmp_path)
    event_bus = EventBus()
    channel = mod.QQChannel(
        "42",
        bus,
        session_manager,
        allow_from=["1"],
        event_bus=event_bus,
        http_requester=SimpleNamespace(get=AsyncMock()),
    )
    await channel.start()

    calls: list[tuple[str, object, object]] = []

    async def _drain(coro):
        return await coro

    async def _fake_send_private_forward_msg(user_id, **payload):
        calls.append(("forward", user_id, payload))

    async def _fake_send_private_text(user_id, content):
        calls.append(("text", user_id, content))

    async def _fake_get_login_info():
        return SimpleNamespace(user_id="42", nickname="Bot")

    channel._run_on_bot_loop = AsyncMock(side_effect=_drain)
    channel._api.send_private_forward_msg = _fake_send_private_forward_msg
    channel._api.send_private_text = _fake_send_private_text
    channel._api.get_login_info = _fake_get_login_info
    channel._workspace = tmp_path
    (tmp_path / "memory").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory" / "SELF.md").write_text(
        "# Akashic 的自我认知\n- 我是 Steria，负责陪伴和协作。\n",
        encoding="utf-8",
    )

    await event_bus.observe(
        TurnStarted(
            session_key="qq:1",
            channel="qq",
            chat_id="1",
            content="帮我看看最近的提交",
            timestamp=__import__("datetime").datetime.now(),
        )
    )
    await event_bus.observe(
        ToolCallStarted(
            session_key="qq:1",
            channel="qq",
            chat_id="1",
            iteration=1,
            call_id="call-1",
            tool_name="fetch_messages",
            arguments={"description": "查最近消息", "query": "最近提交"},
        )
    )
    await event_bus.observe(
        ToolCallCompleted(
            session_key="qq:1",
            channel="qq",
            chat_id="1",
            iteration=1,
            call_id="call-1",
            tool_name="fetch_messages",
            arguments={"description": "查最近消息", "query": "最近提交"},
            final_arguments={"description": "查最近消息", "query": "最近提交"},
            status="ok",
            result_preview='{"count": 21, "matched_count": 1}',
        )
    )

    await channel._on_response(
        OutboundMessage(
            channel="qq",
            chat_id="1",
            content="我看到了，最近主要是 QQ tracing 的改动。",
            thinking="先确认这轮是否有工具调用，再组织结论。",
        )
    )

    assert [item[0] for item in calls] == ["forward", "text"]
    forward_payload = cast(dict[str, Any], calls[0][2])
    assert forward_payload["news"] == [
        {"text": "Steria：【模型思路】"},
        {"text": "Steria：【工具链】"},
    ]
    assert "fetch_messages" in str(forward_payload)
    assert "命中 1 条，返回上下文 21 条" in str(forward_payload)
    assert calls[1] == ("text", 1, "我看到了，最近主要是 QQ tracing 的改动。")
    assert "qq:1" not in channel._trace_states


@pytest.mark.asyncio
async def test_qq_private_trace_skips_empty_trace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    mod = _import_qq_channel(monkeypatch)
    bus = _Bus()
    session_manager = _SessionManager(tmp_path)
    event_bus = EventBus()
    channel = mod.QQChannel(
        "42",
        bus,
        session_manager,
        allow_from=["1"],
        event_bus=event_bus,
        http_requester=SimpleNamespace(get=AsyncMock()),
    )
    await channel.start()

    calls: list[tuple[str, object, object]] = []

    async def _drain(coro):
        return await coro

    async def _fake_send_private_forward_msg(user_id, **payload):
        calls.append(("forward", user_id, payload))

    async def _fake_send_private_text(user_id, content):
        calls.append(("text", user_id, content))

    async def _fake_get_login_info():
        return SimpleNamespace(user_id="42", nickname="Bot")

    channel._run_on_bot_loop = AsyncMock(side_effect=_drain)
    channel._api.send_private_forward_msg = _fake_send_private_forward_msg
    channel._api.send_private_text = _fake_send_private_text
    channel._api.get_login_info = _fake_get_login_info

    await event_bus.observe(
        TurnStarted(
            session_key="qq:1",
            channel="qq",
            chat_id="1",
            content="好",
            timestamp=__import__("datetime").datetime.now(),
        )
    )

    await channel._on_response(
        OutboundMessage(
            channel="qq",
            chat_id="1",
            content="嗯，收到。",
            thinking=None,
        )
    )

    assert [item[0] for item in calls] == ["text"]
    assert calls[0] == ("text", 1, "嗯，收到。")
