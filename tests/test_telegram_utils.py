from types import SimpleNamespace

import pytest
from unittest.mock import AsyncMock

from infra.channels.telegram_utils import (
    TelegramStreamMessage,
    render_telegram_preview_html,
    send_markdown,
    send_thinking_block,
)


class BotStub:
    def __init__(self):
        self.messages = []
        self.edits = []
        self.document_calls = 0
        self.photo_calls = 0

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)
        return SimpleNamespace(message_id=len(self.messages))

    async def edit_message_text(self, **kwargs):
        self.edits.append(kwargs)

    async def send_document(self, **kwargs):
        self.document_calls += 1

    async def send_photo(self, **kwargs):
        self.photo_calls += 1


@pytest.mark.asyncio
async def test_send_markdown_splits_long_code_block_into_multiple_messages():
    bot = BotStub()
    code = "print('x')\n" * 800
    markdown = f"```python\n{code}```"

    await send_markdown(bot, "123", markdown)

    assert len(bot.messages) >= 2
    assert bot.document_calls == 0
    assert bot.photo_calls == 0
    assert all(call["chat_id"] == 123 for call in bot.messages)
    assert all(call["text"].strip() for call in bot.messages)
    assert any(entity["type"] == "pre" for entity in bot.messages[0]["entities"])
    assert all(len(call["text"]) <= 4090 for call in bot.messages)


@pytest.mark.asyncio
async def test_send_markdown_falls_back_to_plain_text(monkeypatch):
    bot = BotStub()

    def fake_convert_with_segments(text):
        raise TypeError("boom")

    monkeypatch.setattr(
        "infra.channels.telegram_utils.convert_with_segments", fake_convert_with_segments
    )

    await send_markdown(bot, 456, "line1\nline2")

    assert bot.messages == [{"chat_id": 456, "text": "line1\nline2"}]


def test_render_telegram_preview_html_renders_markdown():
    html = render_telegram_preview_html("### 标题\n\n**重点**\n\n- 一\n- 二")
    assert "<b>标题</b>" in html
    assert "<b>重点</b>" in html
    assert "• 一" in html
    assert "• 二" in html


def test_render_telegram_preview_html_supports_links_strike_and_spoiler():
    html = render_telegram_preview_html("[官网](https://example.com) 和 ~~删除~~ 以及 ||隐藏||")
    assert '<a href="https://example.com">' in html
    assert "<s>删除</s>" in html
    assert "<tg-spoiler>隐藏</tg-spoiler>" in html


def test_render_telegram_preview_html_keeps_spacing_compact():
    html = render_telegram_preview_html(
        "我在。\n\n### 呼吸\n\n1. 吸气\n\n1. 呼气\n\n> 慢一点"
    )
    assert "<b>呼吸</b>" in html
    assert "• 吸气" in html
    assert "• 呼气" in html
    assert "<blockquote>慢一点</blockquote>" in html
    assert "\n\n\n" not in html


@pytest.mark.asyncio
async def test_stream_message_falls_back_to_plain_text_on_html_parse_error():
    bot = BotStub()

    async def broken_edit_message_text(**kwargs):
        if kwargs.get("parse_mode") == "HTML":
            raise RuntimeError("can't parse entities")
        bot.edits.append(kwargs)

    bot.edit_message_text = broken_edit_message_text
    stream = TelegramStreamMessage(bot, 123)
    await stream.push_delta("**hello**")
    await stream.finalize("**hello**\n\n- a\n- b")

    assert bot.messages[0]["parse_mode"] == "HTML"
    assert bot.edits[-1]["text"] == "**hello**\n\n- a\n- b"


@pytest.mark.asyncio
async def test_stream_message_ignores_message_not_modified_error():
    bot = BotStub()

    class MessageNotModifiedError(Exception):
        pass

    async def unchanged_edit_message_text(**kwargs):
        raise MessageNotModifiedError(
            "Message is not modified: specified new message content and reply markup "
            "are exactly the same as a current content and reply markup of the message"
        )

    bot.edit_message_text = unchanged_edit_message_text
    stream = TelegramStreamMessage(bot, 123)

    await stream.push_delta("hello")
    await stream.finalize("hello")

    assert len(bot.messages) == 1


@pytest.mark.asyncio
async def test_stream_message_skips_duplicate_truncated_preview():
    bot = BotStub()
    stream = TelegramStreamMessage(bot, 123)
    first = "a" * 4096 + "X"
    second = "a" * 4096 + "Y"

    await stream.push_delta(first, force=True)
    await stream.finalize(second)

    assert len(bot.messages) == 1
    assert bot.edits == []


@pytest.mark.asyncio
async def test_stream_message_retry_after_enters_cooldown_without_blocking(monkeypatch):
    bot = BotStub()
    from infra.channels import telegram_utils as mod

    values = [10.0, 20.0, 30.0, 70.0, 80.0]

    class _Loop:
        def __init__(self):
            self._index = 0

        def time(self):
            value = values[min(self._index, len(values) - 1)]
            self._index += 1
            return value

    async def limited_edit_message_text(**kwargs):
        raise mod.RetryAfter(48.0)

    bot.edit_message_text = limited_edit_message_text
    sleep_mock = AsyncMock()
    monkeypatch.setattr("infra.channels.telegram_utils.asyncio.sleep", sleep_mock)
    monkeypatch.setattr(
        "infra.channels.telegram_utils.asyncio.get_running_loop",
        lambda: _Loop(),
    )

    stream = TelegramStreamMessage(bot, 123)
    await stream.push_delta("hello", force=True)
    await stream.push_delta(" world", force=True)
    assert stream._edit_cooldown_until > 30.0
    await stream.push_delta(" again")
    await stream.push_delta(" after cooldown")

    assert sleep_mock.await_count == 0
    assert len(bot.messages) == 1
    assert len(bot.edits) == 0


@pytest.mark.asyncio
async def test_send_thinking_block_splits_long_content():
    bot = BotStub()
    # 每个中文字符占 1 个 UTF-16 code unit，构造超长 thinking
    thinking = "思" * 5000
    await send_thinking_block(bot, 123, thinking)
    assert len(bot.messages) >= 2
    # 每条消息都应该有 expandable_blockquote entity
    for msg in bot.messages:
        entities = msg.get("entities", [])
        assert len(entities) == 1
        assert entities[0].type == "expandable_blockquote"
    # 第一条包含 header
    assert bot.messages[0]["text"].startswith("💭 思考过程")
    # 拼合所有 text 应还原完整内容
    combined = "".join(m["text"] for m in bot.messages)
    assert "思" * 5000 in combined


@pytest.mark.asyncio
async def test_send_thinking_block_short_content_single_message():
    bot = BotStub()
    await send_thinking_block(bot, 123, "短思考")
    assert len(bot.messages) == 1
    assert "短思考" in bot.messages[0]["text"]
