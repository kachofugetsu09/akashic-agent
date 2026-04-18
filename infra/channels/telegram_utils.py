"""
Telegram Markdown 发送工具

将 Markdown 文本转换成 Telegram text+entities 后发送：
- 自动分段（超出 4096 字符时）
- 长代码块拆成多条富文本消息
- 转换失败时降级为纯文本
"""

import asyncio
import html
import logging
import re
from typing import Any, cast

from telegram import Bot, MessageEntity as TgEntity
from telegram.error import BadRequest, NetworkError, RetryAfter, TimedOut
from telegramify_markdown.converter import convert_with_segments
from telegramify_markdown.entity import MessageEntity, split_entities

logger = logging.getLogger(__name__)
_STREAM_CHUNK_STEP = 120
_STREAM_PUSH_MIN_INTERVAL_S = 2.5
_STREAM_PUSH_MIN_CHARS = 200
_TELEGRAM_MSG_LIMIT = 4096
_THINKING_CAP = 800
_THINKING_MIN = 100
_PREVIEW_OVERHEAD = 80
_PARSE_ERR_RE = re.compile(r"can't parse entities|parse entities|find end of the entity", re.I)
_SPOILER_RE = re.compile(r"\|\|(.+?)\|\|", re.S)
_STRIKE_RE = re.compile(r"~~(.+?)~~", re.S)
_FENCE_RE = re.compile(r"^\s*```")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
_LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+(.*)$")
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_LINK_RE = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1", re.S)
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", re.S)


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
                    entities=cast(Any, _serialize_entities(chunk_entities)),
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


async def send_thinking_block(bot: Bot, chat_id: int | str, thinking: str) -> None:
    """Send thinking content as expandable blockquote message(s).

    Telegram 单条消息限制 4096 UTF-16 code units。超长 thinking 按行分段，
    每段独立包裹为 expandable_blockquote。
    """
    cid = int(chat_id)
    header = "💭 思考过程\n\n"
    # 4096 UTF-16 code units, 留一点余量
    max_utf16 = 4080
    header_utf16 = len(header.encode("utf-16-le")) // 2

    chunks = _split_thinking(thinking, max_utf16 - header_utf16)
    for i, chunk in enumerate(chunks):
        text = (header if i == 0 else "") + chunk
        utf16_len = len(text.encode("utf-16-le")) // 2
        entity = TgEntity(type="expandable_blockquote", offset=0, length=utf16_len)
        try:
            await _send_with_retry(
                lambda text=text, entity=entity: bot.send_message(
                    chat_id=cid,
                    text=text,
                    entities=[entity],
                ),
                label="send_message(thinking_block)",
            )
        except Exception as e:
            logger.warning("[telegram] failed to send thinking block chunk %d, skipping: %s", i, e)
            return
    logger.info("[telegram] thinking block sent, chunks=%d, length=%d", len(chunks), len(thinking))


def _split_thinking(text: str, max_utf16: int) -> list[str]:
    """按行切分 thinking 文本，每段不超过 max_utf16 个 UTF-16 code units。"""
    if len(text.encode("utf-16-le")) // 2 <= max_utf16:
        return [text]
    chunks: list[str] = []
    current_lines: list[str] = []
    current_utf16 = 0
    for line in text.splitlines(keepends=True):
        line_utf16 = len(line.encode("utf-16-le")) // 2
        if current_utf16 + line_utf16 > max_utf16 and current_lines:
            chunks.append("".join(current_lines))
            current_lines, current_utf16 = [], 0
        # 单行本身超限时强制切断
        while line_utf16 > max_utf16:
            # 按字符逼近切点
            cut = _utf16_cut(line, max_utf16)
            chunks.append(line[:cut])
            line = line[cut:]
            line_utf16 = len(line.encode("utf-16-le")) // 2
        current_lines.append(line)
        current_utf16 += line_utf16
    if current_lines:
        chunks.append("".join(current_lines))
    return chunks


def _utf16_cut(text: str, max_utf16: int) -> int:
    """返回 text 中前 max_utf16 个 UTF-16 code units 对应的 Python str 切点。"""
    utf16_count = 0
    for i, ch in enumerate(text):
        utf16_count += 2 if ord(ch) > 0xFFFF else 1
        if utf16_count > max_utf16:
            return i
    return len(text)


async def send_stream_markdown(bot: Bot, chat_id: int | str, text: str) -> None:
    """主动推送场景的简化流式展示。"""
    cid = int(chat_id)
    stripped = text.strip()
    if not stripped:
        return

    if cid > 0:
        try:
            stream = TelegramStreamMessage(bot, cid)
            for chunk in _iter_stream_chunks(stripped):
                await stream.push_delta(chunk, force=True)
            await stream.finalize(text)
        except Exception as e:
            logger.warning("[telegram] stream edit 失败，降级普通发送: %s", e)

    else:
        await send_markdown(bot, cid, text)


def _ring_tail(text: str, cap: int) -> str:
    """保留文本最后 cap 个字符，超出部分用省略号标记。"""
    if cap <= 0:
        return ""
    if len(text) <= cap:
        return text
    return "…" + text[-(cap - 1):]


class TelegramStreamMessage:
    def __init__(self, bot: Bot, chat_id: int) -> None:
        self._bot = bot
        self._chat_id = int(chat_id)
        self._message_id: int | None = None
        self._reply_buffer = ""
        self._thinking_buffer = ""
        self._last_sent_plain = ""
        self._last_sent_at = 0.0
        self._edit_cooldown_until = 0.0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def push_delta(
        self,
        delta: str | dict[str, str],
        *,
        force: bool = False,
    ) -> None:
        if self._chat_id <= 0:
            return
        if isinstance(delta, str):
            self._reply_buffer += delta
        else:
            self._reply_buffer += delta.get("content_delta", "")
            self._thinking_buffer += delta.get("thinking_delta", "")
        result = self._build_stream_preview()
        if result is None:
            return
        html_text, plain_text = result
        now = asyncio.get_running_loop().time()
        if not force and now < self._edit_cooldown_until:
            return
        if not force:
            grown = len(plain_text) - len(self._last_sent_plain)
            if (
                self._last_sent_plain
                and grown < _STREAM_PUSH_MIN_CHARS
                and now - self._last_sent_at < _STREAM_PUSH_MIN_INTERVAL_S
            ):
                return
        await self._send_or_edit(html_text, plain_text)
        self._last_sent_at = now

    async def finalize(self, text: str) -> None:
        self._reply_buffer = text or ""
        current = (text or "").strip()
        if not current:
            return
        await self._push_reply_text(current)

    # ------------------------------------------------------------------
    # preview 构建
    # ------------------------------------------------------------------

    def _build_stream_preview(self) -> tuple[str, str] | None:
        """构建流式预览 (html, plain)。thinking 用环形缓冲，reply 优先占预算。"""
        reply = self._reply_buffer.strip()
        thinking = self._thinking_buffer.strip()
        if not reply and not thinking:
            return None
        limit = _TELEGRAM_MSG_LIMIT

        # ---- 仅回复 ----
        if not thinking:
            trimmed = reply[:limit]
            return render_telegram_preview_html(trimmed), trimmed

        # ---- 仅思考 ----
        if not reply:
            cap = limit - _PREVIEW_OVERHEAD
            tail = _ring_tail(thinking, cap)
            plain = f"💭 {tail}"
            h = f"<blockquote>💭 <i>{html.escape(tail)}</i></blockquote>"
            return h, plain

        # ---- 双区域：reply 优先，thinking 取剩余 ----
        reply_need = min(len(reply), limit - _PREVIEW_OVERHEAD - _THINKING_MIN)
        t_budget = max(
            min(limit - reply_need - _PREVIEW_OVERHEAD, _THINKING_CAP),
            _THINKING_MIN,
        )
        r_budget = limit - t_budget - _PREVIEW_OVERHEAD

        tail = _ring_tail(thinking, t_budget)
        reply_trimmed = reply[:r_budget]

        plain = f"💭 {tail}\n\n{reply_trimmed}"
        h = (
            f"<blockquote>💭 <i>{html.escape(tail)}</i></blockquote>"
            f"\n{render_telegram_preview_html(reply_trimmed)}"
        )
        return h, plain

    # ------------------------------------------------------------------
    # 底层发送 / 编辑
    # ------------------------------------------------------------------

    async def _push_reply_text(self, text: str) -> None:
        """finalize 专用：发送纯回复文本（无思考前缀）。"""
        preview = text if len(text) <= _TELEGRAM_MSG_LIMIT else text[:_TELEGRAM_MSG_LIMIT]
        if preview == self._last_sent_plain:
            return
        html_text = render_telegram_preview_html(preview)
        await self._send_or_edit(html_text, preview)

    async def _send_or_edit(self, html_text: str, plain_text: str) -> None:
        """首次调用 send，后续 edit。成功后更新 _last_sent_plain。"""
        if plain_text == self._last_sent_plain and self._message_id is not None:
            return
        if self._message_id is None:
            sent = await _send_with_retry_result(
                lambda: _send_preview_message(
                    self._bot, self._chat_id, html_text, plain_text
                ),
                label="send_message(stream_start)",
            )
            self._message_id = int(getattr(sent, "message_id", 0) or 0) or None
            self._last_sent_plain = plain_text
        else:
            if await self._try_edit_preview_message(html_text, plain_text):
                self._last_sent_plain = plain_text

    async def _try_edit_preview_message(
        self,
        html_text: str,
        plain_text: str,
    ) -> bool:
        try:
            await _edit_preview_message(
                self._bot,
                self._chat_id,
                self._message_id,
                html_text,
                plain_text,
            )
            return True
        except RetryAfter as e:
            delay = max(float(getattr(e, "retry_after", 1.0) or 1.0), 1.0)
            now = asyncio.get_running_loop().time()
            self._edit_cooldown_until = now + delay
            logger.warning(
                "[telegram] edit_message_text(stream) 命中限流，进入冷却 %.1fs err=%s",
                delay,
                e,
            )
            return False
        except (TimedOut, NetworkError) as e:
            logger.warning("[telegram] edit_message_text(stream) 失败 err=%s", e)
            return False


async def _send_with_retry_result(
    send_coro_factory,
    *,
    label: str,
    max_attempts: int = 3,
    base_delay: float = 0.8,
):
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await send_coro_factory()
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
    raise RuntimeError(f"{label} failed without exception")


def _iter_stream_chunks(text: str) -> list[str]:
    if len(text) <= _STREAM_CHUNK_STEP:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + _STREAM_CHUNK_STEP, len(text))
        if end < len(text):
            newline = text.rfind("\n", start, end)
            if newline > start:
                end = newline + 1
        chunks.append(text[start:end])
        start = end
    return chunks


async def _send_preview_message(bot: Bot, chat_id: int, html_text: str, plain_text: str):
    try:
        return await bot.send_message(
            chat_id=chat_id,
            text=html_text,
            parse_mode="HTML",
        )
    except Exception as e:
        if not _is_telegram_html_parse_error(e):
            raise
        logger.warning("[telegram] preview HTML 解析失败，降级纯文本: %s", e)
        return await bot.send_message(chat_id=chat_id, text=plain_text)


async def _edit_preview_message(
    bot: Bot,
    chat_id: int,
    message_id: int | None,
    html_text: str,
    plain_text: str,
) -> None:
    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=html_text,
            parse_mode="HTML",
        )
    except BadRequest as e:
        if _is_telegram_message_not_modified_error(e):
            logger.debug("[telegram] preview edit skipped: %s", e)
            return
        if not _is_telegram_html_parse_error(e):
            raise
        logger.warning("[telegram] preview edit HTML 解析失败，降级纯文本: %s", e)
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=plain_text,
        )
    except Exception as e:
        if not _is_telegram_html_parse_error(e):
            raise
        logger.warning("[telegram] preview edit HTML 解析失败，降级纯文本: %s", e)
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=plain_text,
        )


def _is_telegram_html_parse_error(err: Exception) -> bool:
    return bool(_PARSE_ERR_RE.search(str(err)))


def _is_telegram_message_not_modified_error(err: Exception) -> bool:
    return "message is not modified" in str(err).lower()


def render_telegram_preview_html(text: str) -> str:
    prepared = _prepare_preview_markdown(text or "")
    rendered = _render_preview_blocks(prepared)
    return rendered.strip() or html.escape(text or "")


def _prepare_preview_markdown(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"(?m)^\s*([-*_])\1{2,}\s*$", "", text)
    return text


def _render_preview_blocks(text: str) -> str:
    lines = text.split("\n")
    parts: list[str] = []
    prev_kind: str | None = None
    pending_blank = False
    in_fence = False
    fence_lines: list[str] = []
    blockquote_lines: list[str] = []

    def flush_blockquote() -> None:
        nonlocal blockquote_lines, prev_kind, pending_blank
        if not blockquote_lines:
            return
        _append_preview_part(
            parts,
            "<blockquote>" + "\n".join(_render_inline(line) for line in blockquote_lines) + "</blockquote>",
            kind="blockquote",
            prev_kind=prev_kind,
            pending_blank=pending_blank,
        )
        prev_kind = "blockquote"
        pending_blank = False
        blockquote_lines = []

    def flush_fence() -> None:
        nonlocal fence_lines, in_fence, prev_kind, pending_blank
        if not in_fence:
            return
        code = "\n".join(fence_lines).strip("\n")
        _append_preview_part(
            parts,
            f"<pre><code>{html.escape(code)}</code></pre>",
            kind="pre",
            prev_kind=prev_kind,
            pending_blank=pending_blank,
        )
        prev_kind = "pre"
        pending_blank = False
        fence_lines = []
        in_fence = False

    for line in lines:
        if _FENCE_RE.match(line):
            flush_blockquote()
            if in_fence:
                flush_fence()
            else:
                in_fence = True
                fence_lines = []
            continue
        if in_fence:
            fence_lines.append(line)
            continue
        blockquote_match = _BLOCKQUOTE_RE.match(line)
        if blockquote_match:
            blockquote_lines.append(blockquote_match.group(1))
            continue

        flush_blockquote()

        stripped = line.strip()
        if not stripped:
            pending_blank = True
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            _append_preview_part(
                parts,
                f"<b>{_render_inline(heading_match.group(1).strip())}</b>",
                kind="heading",
                prev_kind=prev_kind,
                pending_blank=pending_blank,
            )
            prev_kind = "heading"
            pending_blank = False
            continue

        list_match = _LIST_RE.match(line)
        if list_match:
            _append_preview_part(
                parts,
                f"• {_render_inline(list_match.group(1).strip())}",
                kind="list_item",
                prev_kind=prev_kind,
                pending_blank=pending_blank,
            )
            prev_kind = "list_item"
            pending_blank = False
            continue

        _append_preview_part(
            parts,
            _render_inline(stripped),
            kind="paragraph",
            prev_kind=prev_kind,
            pending_blank=pending_blank,
        )
        prev_kind = "paragraph"
        pending_blank = False

    flush_blockquote()
    flush_fence()
    return "\n".join(parts).strip()


def _append_preview_part(
    parts: list[str],
    text: str,
    *,
    kind: str,
    prev_kind: str | None,
    pending_blank: bool,
) -> None:
    if not text:
        return
    if parts and pending_blank and prev_kind in {"paragraph", "blockquote", "pre"} and kind in {"paragraph", "blockquote", "pre"}:
        parts.append("")
    parts.append(text)


def _render_inline(text: str) -> str:
    if not text:
        return ""
    pieces: list[str] = []
    idx = 0
    patterns = [
        ("link", _LINK_RE),
        ("code", _CODE_SPAN_RE),
        ("spoiler", _SPOILER_RE),
        ("strike", _STRIKE_RE),
        ("bold", _BOLD_RE),
        ("italic", _ITALIC_RE),
    ]

    while idx < len(text):
        earliest_kind = None
        earliest_match = None
        for kind, pattern in patterns:
            match = pattern.search(text, idx)
            if match is None:
                continue
            if earliest_match is None or match.start() < earliest_match.start():
                earliest_kind = kind
                earliest_match = match
        if earliest_match is None:
            pieces.append(html.escape(text[idx:]))
            break
        if earliest_match.start() > idx:
            pieces.append(html.escape(text[idx:earliest_match.start()]))
        pieces.append(_render_inline_match(earliest_kind or "", earliest_match))
        idx = earliest_match.end()
    return "".join(pieces)


def _render_inline_match(kind: str, match: re.Match[str]) -> str:
    if kind == "link":
        label = _render_inline(match.group(1))
        href = html.escape(match.group(2), quote=True)
        return f'<a href="{href}">{label}</a>'
    if kind == "code":
        return f"<code>{html.escape(match.group(1))}</code>"
    if kind == "spoiler":
        return f"<tg-spoiler>{_render_inline(match.group(1))}</tg-spoiler>"
    if kind == "strike":
        return f"<s>{_render_inline(match.group(1))}</s>"
    if kind == "bold":
        return f"<b>{_render_inline(match.group(2))}</b>"
    if kind == "italic":
        inner = match.group(1) or match.group(2) or ""
        return f"<i>{_render_inline(inner)}</i>"
    return html.escape(match.group(0))
