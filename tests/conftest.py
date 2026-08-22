"""Shared fixtures and test bootstrap helpers."""

import sys
import types
from datetime import datetime, timezone, timedelta

import pytest

# Provide lightweight telegram stubs so optional messaging deps do not block
# unrelated test collection.
if "telegram" not in sys.modules:
    telegram_stub = types.ModuleType("telegram")
    telegram_error_stub = types.ModuleType("telegram.error")

    class Bot:
        async def edit_message_text(self, *args, **kwargs):
            return True

    class TelegramMessageEntity:
        def __init__(self, *, type, offset, length):
            self.type = type
            self.offset = offset
            self.length = length

    class RetryAfter(Exception):
        def __init__(self, retry_after=1.0):
            super().__init__(f"retry after {retry_after}")
            self.retry_after = retry_after

    class NetworkError(Exception):
        pass

    class BadRequest(Exception):
        pass

    class TimedOut(Exception):
        pass

    telegram_stub.Bot = Bot
    telegram_stub.MessageEntity = TelegramMessageEntity
    telegram_error_stub.BadRequest = BadRequest
    telegram_error_stub.RetryAfter = RetryAfter
    telegram_error_stub.NetworkError = NetworkError
    telegram_error_stub.TimedOut = TimedOut
    sys.modules["telegram"] = telegram_stub
    sys.modules["telegram.error"] = telegram_error_stub

if "telegramify_markdown.converter" not in sys.modules:
    telegramify_stub = types.ModuleType("telegramify_markdown")
    converter_stub = types.ModuleType("telegramify_markdown.converter")
    entity_stub = types.ModuleType("telegramify_markdown.entity")

    class MessageEntity:
        def __init__(
            self,
            *,
            type,
            offset,
            length,
            url=None,
            language=None,
            custom_emoji_id=None,
        ):
            self.type = type
            self.offset = offset
            self.length = length
            self.url = url
            self.language = language
            self.custom_emoji_id = custom_emoji_id

        def to_dict(self):
            data = {
                "type": self.type,
                "offset": self.offset,
                "length": self.length,
            }
            if self.url is not None:
                data["url"] = self.url
            if self.language is not None:
                data["language"] = self.language
            if self.custom_emoji_id is not None:
                data["custom_emoji_id"] = self.custom_emoji_id
            return data

    def convert_with_segments(text):
        if text.startswith("```") and text.endswith("```"):
            first_newline = text.find("\n")
            code = text[first_newline + 1 : -3] if first_newline != -1 else ""
            entity = MessageEntity(type="pre", offset=0, length=len(code))
            return code, [entity], []
        return text, [], []

    def split_entities(text, entities, limit):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + limit, len(text))
            chunk_text = text[start:end]
            chunk_entities = []
            for entity in entities:
                entity_start = entity.offset
                entity_end = entity.offset + entity.length
                overlap_start = max(start, entity_start)
                overlap_end = min(end, entity_end)
                if overlap_end <= overlap_start:
                    continue
                chunk_entities.append(
                    MessageEntity(
                        type=entity.type,
                        offset=overlap_start - start,
                        length=overlap_end - overlap_start,
                        url=entity.url,
                        language=entity.language,
                        custom_emoji_id=entity.custom_emoji_id,
                    )
                )
            chunks.append((chunk_text, chunk_entities))
            start = end
        return chunks or [("", [])]

    converter_stub.convert_with_segments = convert_with_segments
    entity_stub.MessageEntity = MessageEntity
    entity_stub.split_entities = split_entities
    sys.modules["telegramify_markdown"] = telegramify_stub
    sys.modules["telegramify_markdown.converter"] = converter_stub
    sys.modules["telegramify_markdown.entity"] = entity_stub

from agent.scheduler import ScheduledJob


def make_job(
    trigger="at",
    tier="instant",
    fire_at=None,
    channel="telegram",
    chat_id="123",
    message: str | None = "hello",
    prompt=None,
    name=None,
    interval_seconds=None,
    cron_expr=None,
    timezone_="UTC",
) -> ScheduledJob:
    if fire_at is None:
        fire_at = datetime.now(timezone.utc) + timedelta(minutes=5)
    return ScheduledJob(
        trigger=trigger,
        tier=tier,
        fire_at=fire_at,
        channel=channel,
        chat_id=chat_id,
        message=message,
        prompt=prompt,
        name=name,
        interval_seconds=interval_seconds,
        cron_expr=cron_expr,
        timezone=timezone_,
    )
