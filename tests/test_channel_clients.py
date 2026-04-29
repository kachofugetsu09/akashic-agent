from __future__ import annotations

import asyncio
import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from bus.event_bus import EventBus
from bus.events import OutboundMessage
from bus.events_lifecycle import StreamDeltaReady, ToolCallCompleted, ToolCallStarted


class _Bus:
    def __init__(self) -> None:
        self.inbound = []
        self.outbound = []

    async def publish_inbound(self, msg) -> None:
        self.inbound.append(msg)

    def subscribe_outbound(self, channel, callback) -> None:
        self.outbound.append((channel, callback))


class _SessionManager:
    def __init__(self) -> None:
        self.sessions = {}
        self.saved = []

    def get_or_create(self, key: str):
        return self.sessions.setdefault(key, SimpleNamespace(key=key, metadata={}))

    async def save_async(self, session) -> None:
        self.saved.append(session.key)

    def get_channel_metadata(self, channel: str):
        return []


def _import_cli_tui(monkeypatch: pytest.MonkeyPatch):
    rich_mod = types.ModuleType("rich")
    rich_markdown = types.ModuleType("rich.markdown")
    rich_text = types.ModuleType("rich.text")
    textual = types.ModuleType("textual")
    textual_app = types.ModuleType("textual.app")
    textual_containers = types.ModuleType("textual.containers")
    textual_widgets = types.ModuleType("textual.widgets")

    class Markdown:
        def __init__(self, text, **kwargs):
            self.text = text

    class Text:
        def __init__(self, text, style=None):
            self.text = text
            self.style = style

    def on(*args, **kwargs):
        return lambda fn: fn

    class App:
        @classmethod
        def __class_getitem__(cls, item):
            return cls

        def __init__(self):
            self.exited = False

        def run_worker(self, *args, **kwargs):
            self.worker = (args, kwargs)

        def exit(self):
            self.exited = True

        def run(self, **kwargs):
            self.run_kwargs = kwargs

    class _Ctx:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class Horizontal:
        pass

    class Vertical(_Ctx):
        pass

    class Input:
        class Submitted:
            pass

        def __init__(self, placeholder="", id=""):
            self.placeholder = placeholder
            self.id = id
            self.value = ""
            self.focused = False

        def focus(self):
            self.focused = True

    class RichLog:
        def __init__(self, **kwargs):
            self.items = []

        def write(self, item):
            self.items.append(item)

        def clear(self):
            self.items.clear()

    class Static:
        def __init__(self, text="", id=""):
            self.text = text
            self.id = id

        def update(self, text):
            self.text = text

    rich_markdown.Markdown = Markdown
    rich_text.Text = Text
    textual.on = on
    textual_app.App = App
    textual_app.ComposeResult = list
    textual_containers.Horizontal = Horizontal
    textual_containers.Vertical = Vertical
    textual_widgets.Input = Input
    textual_widgets.RichLog = RichLog
    textual_widgets.Static = Static
    monkeypatch.setitem(sys.modules, "rich", rich_mod)
    monkeypatch.setitem(sys.modules, "rich.markdown", rich_markdown)
    monkeypatch.setitem(sys.modules, "rich.text", rich_text)
    monkeypatch.setitem(sys.modules, "textual", textual)
    monkeypatch.setitem(sys.modules, "textual.app", textual_app)
    monkeypatch.setitem(sys.modules, "textual.containers", textual_containers)
    monkeypatch.setitem(sys.modules, "textual.widgets", textual_widgets)
    sys.modules.pop("infra.channels.cli_tui", None)
    return importlib.import_module("infra.channels.cli_tui")


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
    ncatbot_utils = types.ModuleType("ncatbot.utils")

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

    ncatbot_core.BotClient = BotClient
    ncatbot_utils.ncatbot_config = SimpleNamespace(
        bt_uin="",
        check_ncatbot_update=True,
        skip_ncatbot_install_check=False,
        napcat=SimpleNamespace(remote_mode=False),
        enable_webui_interaction=True,
        plugin=SimpleNamespace(plugins_dir=""),
    )
    monkeypatch.setitem(sys.modules, "ncatbot.core", ncatbot_core)
    monkeypatch.setitem(sys.modules, "ncatbot.utils", ncatbot_utils)
    sys.modules.pop("infra.channels.qq_channel", None)
    return importlib.import_module("infra.channels.qq_channel")


@pytest.mark.asyncio
async def test_cli_tui_paths(monkeypatch: pytest.MonkeyPatch):
    mod = _import_cli_tui(monkeypatch)
    app = mod.CLITextualApp("/tmp/test.sock")
    log = mod.RichLog()
    meta = mod.Static()
    input_widget = mod.Input()
    app.query_one = lambda selector, *_: {
        mod.RichLog: log,
        "#meta": meta,
        mod.Input: input_widget,
    }[selector]
    def _run_worker(coro, **kwargs):
        coro.close()

    app.run_worker = MagicMock(side_effect=_run_worker)

    await app.on_mount()
    assert input_widget.focused is True
    assert app.run_worker.called

    app.action_clear_log()
    assert any(getattr(item, "text", "") == "  日志已清空。" for item in log.items)

    event = SimpleNamespace(value="", input=input_widget)
    await app._on_input_submitted(event)
    event = SimpleNamespace(value="quit", input=input_widget)
    await app._on_input_submitted(event)
    assert app.exited is True

    app.exited = False
    event = SimpleNamespace(value="hello", input=input_widget)
    await app._on_input_submitted(event)
    assert any("尚未连接到 agent" in getattr(item, "text", "") for item in log.items)

    class _Writer:
        def __init__(self):
            self.payloads = []
            self.closed = False

        def write(self, payload):
            self.payloads.append(payload)

        async def drain(self):
            return None

        def close(self):
            self.closed = True

        async def wait_closed(self):
            return None

    writer = _Writer()
    app._writer = writer
    event = SimpleNamespace(value="你好", input=input_widget)
    await app._on_input_submitted(event)
    assert writer.payloads
    assert app.stats.sent == 1

    async def _open_fail(path):
        raise FileNotFoundError

    monkeypatch.setattr(mod.asyncio, "open_unix_connection", _open_fail)
    await app._connect_and_receive()
    assert "connected: no" in meta.text

    class _Reader:
        def __init__(self):
            self.lines = iter(
                [
                    b'{"content":"hello","metadata":{"tool_chain":[{"calls":[{"name":"search"}]}]}}\n',
                    b"bad-json\n",
                    b"",
                ]
            )

        async def readline(self):
            return next(self.lines)

    writer2 = _Writer()

    async def _open_ok(path):
        return _Reader(), writer2

    monkeypatch.setattr(mod.asyncio, "open_unix_connection", _open_ok)
    await app._connect_and_receive()
    assert app.stats.received == 1
    assert app.stats.tool_calls == 1

    app.connected = True
    app._writer = writer2
    await app.on_unmount()
    assert writer2.closed is True
    assert mod._env_bool("akashic_TUI_MOUSE", True) is True
    monkeypatch.setenv("akashic_TUI_MOUSE", "0")
    assert mod._env_bool("akashic_TUI_MOUSE", True) is False

    captured = {}

    class _FakeApp:
        def __init__(self, socket_path):
            captured["socket"] = socket_path

        def run(self, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(mod, "CLITextualApp", _FakeApp)
    mod.run_tui("/tmp/x.sock")
    assert captured == {"socket": "/tmp/x.sock", "kwargs": {"mouse": False}}


@pytest.mark.asyncio
async def test_telegram_channel_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mod = _import_telegram_channel(monkeypatch)
    bus = _Bus()
    event_bus = EventBus()
    session_manager = _SessionManager()
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
    assert len(channel._app.handlers) == 6
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
        effective_message=SimpleNamespace(text="/memory_status", message_id=100),
        effective_chat=SimpleNamespace(id=123),
        effective_user=SimpleNamespace(id=1, username="Alice"),
    )
    await channel._on_memory_status_command(status_update, context)
    assert len(bus.inbound) == 2
    assert bus.inbound[1].content == "/memory_status"
    assert bus.inbound[1].metadata["username"] == "Alice"

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
    assert len(bus.inbound) == 4
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
async def test_qq_channel_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    mod = _import_qq_channel(monkeypatch)
    bus = _Bus()
    session_manager = _SessionManager()
    async def _request_get(url, **kwargs):
        if url.endswith("a.jpg") or url.endswith("a.png"):
            return SimpleNamespace(
                headers={"content-type": "image/png"},
                content=b"img",
                raise_for_status=lambda: None,
            )
        raise RuntimeError("boom")

    requester = SimpleNamespace(get=AsyncMock(side_effect=_request_get))
    group_filter = SimpleNamespace(should_process=AsyncMock(return_value=True))
    group_cfg = SimpleNamespace(group_id="100")
    channel = mod.QQChannel(
        "42",
        bus,
        session_manager,
        allow_from=["1"],
        groups=[group_cfg],
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

    paths = await mod._download_to_temp(["http://x/a.png", "http://x/b.png"], requester)
    assert len(paths) == 1

    channel._bot_loop = None
    pending = asyncio.sleep(0)
    with pytest.raises(RuntimeError):
        await mod.QQChannel._run_on_bot_loop(channel, pending)
    pending.close()
    await channel.stop()
