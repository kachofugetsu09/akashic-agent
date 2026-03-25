import pytest

from infra.channels.telegram_utils import send_markdown


class BotStub:
    def __init__(self):
        self.messages = []
        self.document_calls = 0
        self.photo_calls = 0

    async def send_message(self, **kwargs):
        self.messages.append(kwargs)

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
