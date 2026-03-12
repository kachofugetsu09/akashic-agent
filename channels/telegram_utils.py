"""
Telegram Markdown 发送工具

将 Markdown 文本转换成 Telegram text+entities 后发送：
- 自动分段（超出 4096 字符时）
- 长代码块拆成多条富文本消息
- 转换失败时降级为纯文本
"""

import asyncio
import logging

from telegram import Bot
from telegram.error import NetworkError, RetryAfter, TimedOut
from telegramify_markdown.converter import convert_with_segments
from telegramify_markdown.entity import MessageEntity, split_entities

logger = logging.getLogger(__name__)


async def _send_with_retry(
    send_coro_factory,
    *,
    label: str,
    max_attempts: int = 3,
    base_delay: float = 0.8,
) -> None:
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            await send_coro_factory()
            return
        except RetryAfter as e:
            last_err = e
            if attempt >= max_attempts:
                break
            delay = max(float(getattr(e, "retry_after", 1.0) or 1.0), base_delay)
            logger.warning(
                "[telegram] %s 命中限流，准备重试 attempt=%d/%d delay=%.1fs err=%s",
                label,
                attempt,
                max_attempts,
                delay,
                e,
            )
            await asyncio.sleep(delay)
        except (TimedOut, NetworkError) as e:
            last_err = e
            if attempt >= max_attempts:
                break
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "[telegram] %s 发送失败，准备重试 attempt=%d/%d delay=%.1fs err=%s",
                label,
                attempt,
                max_attempts,
                delay,
                e,
            )
            await asyncio.sleep(delay)
    if last_err is not None:
        raise last_err


def _serialize_entities(entities: list[MessageEntity]) -> list[dict] | None:
    return [entity.to_dict() for entity in entities] if entities else None


def _strip_chunk(
    text: str,
    entities: list[MessageEntity],
) -> tuple[str, list[MessageEntity]]:
    leading = len(text) - len(text.lstrip("\n"))
    trailing = len(text) - len(text.rstrip("\n"))
    if leading == 0 and trailing == 0:
        return text, entities

    end = len(text) - trailing if trailing else len(text)
    stripped = text[leading:end]
    if not stripped:
        return "", []

    stripped_utf16_len = len(stripped.encode("utf-16-le")) // 2
    adjusted: list[MessageEntity] = []
    for entity in entities:
        new_offset = entity.offset - leading
        new_end = new_offset + entity.length
        if new_end <= 0 or new_offset >= stripped_utf16_len:
            continue
        new_offset = max(0, new_offset)
        new_end = min(new_end, stripped_utf16_len)
        new_length = new_end - new_offset
        if new_length <= 0:
            continue
        adjusted.append(
            MessageEntity(
                type=entity.type,
                offset=new_offset,
                length=new_length,
                url=entity.url,
                language=entity.language,
                custom_emoji_id=entity.custom_emoji_id,
            )
        )
    return stripped, adjusted


async def send_markdown(bot: Bot, chat_id: int | str, text: str) -> None:
    cid = int(chat_id)
    try:
        rendered_text, entities, _segments = convert_with_segments(text)
        chunks = split_entities(rendered_text, entities, 4090)
        for chunk_text, chunk_entities in chunks:
            chunk_text, chunk_entities = _strip_chunk(chunk_text, chunk_entities)
            if not chunk_text:
                continue
            await _send_with_retry(
                lambda: bot.send_message(
                    chat_id=cid,
                    text=chunk_text,
                    entities=_serialize_entities(chunk_entities),
                ),
                label="send_message(markdown)",
            )
    except Exception as e:
        logger.warning(f"[telegram] Markdown 转换失败，降级纯文本: {e}")
        for chunk in _split_text(text, 4090):
            await _send_with_retry(
                lambda: bot.send_message(chat_id=cid, text=chunk),
                label="send_message(plain)",
            )


def _split_text(text: str, limit: int) -> list[str]:
    """按行切分文本，每段不超过 limit 字符。"""
    chunks, current = [], []
    current_len = 0
    for line in text.splitlines(keepends=True):
        if current_len + len(line) > limit and current:
            chunks.append("".join(current))
            current, current_len = [], 0
        # 单行本身超限时强制切断
        while len(line) > limit:
            chunks.append(line[:limit])
            line = line[limit:]
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks
