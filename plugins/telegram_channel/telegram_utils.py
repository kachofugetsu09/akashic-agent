"""Telegram 出站限流与消息分段实体处理。"""

import asyncio
import logging
import weakref
from collections.abc import Awaitable, Callable
from typing import TypeVar

from telegram.error import NetworkError, RetryAfter, TimedOut
from telegramify_markdown.entity import MessageEntity

logger = logging.getLogger(__name__)
_THROTTLE_PRUNE_THRESHOLD = 256
_T = TypeVar("_T")


class TelegramOutboundLimiter:
    def __init__(
        self,
        *,
        send_interval_s: float = 2.0,
        edit_interval_s: float = 5.0,
        typing_interval_s: float = 8.0,
        global_interval_s: float = 0.25,
        retry_padding_s: float = 1.0,
        max_attempts: int = 5,
    ) -> None:
        self._send_interval_s = send_interval_s
        self._edit_interval_s = edit_interval_s
        self._typing_interval_s = typing_interval_s
        self._global_interval_s = global_interval_s
        self._retry_padding_s = retry_padding_s
        self._max_attempts = max_attempts
        self._chat_locks: weakref.WeakValueDictionary[int, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._typing_locks: weakref.WeakValueDictionary[int, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._global_lock = asyncio.Lock()
        self._next_chat_at: dict[int, float] = {}
        self._next_typing_at: dict[int, float] = {}
        self._next_global_at = 0.0

    async def run(
        self,
        chat_id: int | str,
        *,
        kind: str,
        label: str,
        action: Callable[[], Awaitable[_T]],
        max_attempts: int | None = None,
    ) -> _T:
        cid = int(chat_id)
        self._prune_expired(asyncio.get_running_loop().time())
        if kind == "typing":
            return await self._run_typing(cid, label=label, action=action)
        attempts = max_attempts or self._max_attempts
        lock = self._chat_locks.get(cid)
        if lock is None:
            lock = asyncio.Lock()
            self._chat_locks[cid] = lock
        async with lock:
            last_err: Exception | None = None
            for attempt in range(1, attempts + 1):
                await self._wait_for_chat_slot(cid)
                try:
                    result = await self._run_with_global_slot(action)
                    self._mark_used(cid, kind)
                    return result
                except RetryAfter as e:
                    last_err = e
                    delay = max(
                        float(getattr(e, "retry_after", 1.0) or 1.0) + self._retry_padding_s,
                        self._interval(kind),
                    )
                    self._cooldown(cid, delay)
                    logger.warning(
                        "[telegram] %s 命中限流，按 retry_after 冷却 chat_id=%s attempt=%d/%d delay=%.1fs",
                        label,
                        cid,
                        attempt,
                        attempts,
                        delay,
                    )
                except (TimedOut, NetworkError) as e:
                    last_err = e
                    delay = min(0.8 * (2 ** (attempt - 1)), 8.0)
                    self._cooldown(cid, delay)
                    logger.warning(
                        "[telegram] %s 网络失败，准备重试 chat_id=%s attempt=%d/%d delay=%.1fs err=%s",
                        label,
                        cid,
                        attempt,
                        attempts,
                        delay,
                        e,
                    )
                if attempt >= attempts:
                    break
                await self._wait_for_chat_slot(cid)
            if last_err is not None:
                raise last_err
            raise RuntimeError(f"{label} failed without exception")

    async def _run_typing(
        self,
        chat_id: int,
        *,
        label: str,
        action: Callable[[], Awaitable[_T]],
    ) -> _T:
        lock = self._typing_locks.get(chat_id)
        if lock is None:
            lock = asyncio.Lock()
            self._typing_locks[chat_id] = lock
        async with lock:
            now = asyncio.get_running_loop().time()
            wait_s = self._next_typing_at.get(chat_id, 0.0) - now
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            try:
                result = await action()
                self._next_typing_at[chat_id] = (
                    asyncio.get_running_loop().time() + self._typing_interval_s
                )
                return result
            except RetryAfter as e:
                delay = (
                    float(getattr(e, "retry_after", 1.0) or 1.0)
                    + self._retry_padding_s
                )
                self._next_typing_at[chat_id] = asyncio.get_running_loop().time() + delay
                raise

    async def _wait_for_chat_slot(self, chat_id: int) -> None:
        now = asyncio.get_running_loop().time()
        wait_s = self._next_chat_at.get(chat_id, 0.0) - now
        if wait_s > 0:
            await asyncio.sleep(wait_s)

    async def _run_with_global_slot(
        self,
        action: Callable[[], Awaitable[_T]],
    ) -> _T:
        async with self._global_lock:
            now = asyncio.get_running_loop().time()
            wait_s = self._next_global_at - now
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            try:
                return await action()
            finally:
                self._next_global_at = (
                    asyncio.get_running_loop().time() + self._global_interval_s
                )

    def _mark_used(self, chat_id: int, kind: str) -> None:
        now = asyncio.get_running_loop().time()
        self._next_chat_at[chat_id] = now + self._interval(kind)

    def _cooldown(self, chat_id: int, delay: float) -> None:
        now = asyncio.get_running_loop().time()
        self._next_chat_at[chat_id] = max(
            self._next_chat_at.get(chat_id, 0.0),
            now + delay,
        )
        self._next_global_at = max(self._next_global_at, now + self._global_interval_s)

    def _interval(self, kind: str) -> float:
        if kind == "edit":
            return self._edit_interval_s
        if kind == "typing":
            return self._typing_interval_s
        return self._send_interval_s

    def _prune_expired(self, now: float) -> None:
        if (
            max(len(self._next_chat_at), len(self._next_typing_at))
            <= _THROTTLE_PRUNE_THRESHOLD
        ):
            return
        for deadlines in (self._next_chat_at, self._next_typing_at):
            for chat_id, deadline in list(deadlines.items()):
                if deadline <= now:
                    deadlines.pop(chat_id, None)


def strip_chunk(
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
