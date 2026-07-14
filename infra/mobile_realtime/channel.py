from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast
from uuid import UUID

from bus.events import InboundMessage, OutboundMessage
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from infra.channels.contract import ChannelContext
from infra.mobile_realtime.protocol import GenericCommand, MessageSendCommand
from infra.mobile_realtime.storage import CommandReceipt

if TYPE_CHECKING:
    from infra.mobile_realtime.gateway import MobileGatewayRuntime


class MobileCommandError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CommandReply:
    type: str
    payload: dict[str, object]
    session_id: str | None = None
    turn_id: str | None = None


@dataclass(slots=True)
class _DeltaBatch:
    segments: list[tuple[str, str, str | None, int | None]]
    byte_count: int
    timer: asyncio.Task[None]


@dataclass(slots=True)
class _ProcessTurnState:
    next_ordinal: int
    thinking_block: tuple[str, int] | None
    tool_blocks: dict[str, tuple[str, int]]


_DELTA_FLUSH_SECONDS = 0.05
_DELTA_FLUSH_BYTES = 4 * 1024
_MAX_DELTA_BATCHES = 256


class MobileRealtimeChannel:
    """把移动协议接入现有消息、生命周期和主动推送总线。"""

    name = "mobile"

    def __init__(self, runtime: MobileGatewayRuntime) -> None:
        self._runtime = runtime
        self._ctx: ChannelContext | None = None
        self._active_turn_ids: dict[str, str] = {}
        self._process_turns: dict[tuple[str, str], _ProcessTurnState] = {}
        self._delta_batches: dict[tuple[str, str], _DeltaBatch] = {}
        self._delta_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._delta_failure: BaseException | None = None

    async def start(self, ctx: ChannelContext) -> None:
        """注册移动渠道的出站、流事件和主动推送入口。"""

        if self._ctx is not None:
            raise RuntimeError("MobileRealtimeChannel 已启动")
        self._ctx = ctx
        _ = ctx.bus.subscribe_outbound(self.name, self._on_response)
        _ = ctx.event_bus.on(TurnStarted, self._on_turn_started)
        _ = ctx.event_bus.on(StreamDeltaReady, self._on_stream_delta)
        _ = ctx.event_bus.on(ToolCallStarted, self._on_tool_call_started)
        _ = ctx.event_bus.on(ToolCallCompleted, self._on_tool_call_completed)
        _ = ctx.push_tool.register_channel(
            self.name,
            text=self.send,
            stream_text=self.send_stream,
        )

    async def stop(self) -> None:
        for batch in self._delta_batches.values():
            _ = batch.timer.cancel()
        if self._delta_batches:
            _ = await asyncio.gather(
                *(batch.timer for batch in self._delta_batches.values()),
                return_exceptions=True,
            )
        self._delta_batches.clear()
        self._delta_locks.clear()
        self._delta_failure = None
        self._ctx = None
        self._active_turn_ids.clear()
        self._process_turns.clear()

    async def handle_command(
        self,
        *,
        device_id: str,
        frame: GenericCommand | MessageSendCommand,
    ) -> CommandReply:
        """幂等执行业务命令，并持久化可跨重连复用的回复。"""

        # 1. 先持久化命令占用，避免重连重复触发 Agent turn
        self._raise_delta_failure()
        receipt, created = self._runtime.storage.reserve_command(
            device_id=device_id,
            command_id=frame.id,
            command_type=frame.type,
            request_hash=_command_hash(frame),
            created_at=_utc_now(),
        )
        if not created:
            return _reply_from_receipt(receipt)

        # 2. 只把可恢复的客户端错误写成稳定 error reply
        try:
            reply = await self._execute_command(device_id=device_id, frame=frame)
        except MobileCommandError as error:
            reply = CommandReply(
                type=f"{frame.type}.error",
                payload={"code": error.code, "message": str(error)},
                session_id=frame.session_id,
                turn_id=frame.turn_id,
            )
        # 3. 成功副作用完成后固化回复；内部异常保持 processing 并向上暴露
        completed = self._runtime.storage.complete_command(
            device_id=device_id,
            command_id=frame.id,
            reply_type=reply.type,
            reply_payload_json=json.dumps(
                reply.payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ),
            session_id=reply.session_id,
            turn_id=reply.turn_id,
            completed_at=_utc_now(),
        )
        return _reply_from_receipt(completed)

    async def send(self, chat_id: str, message: str) -> None:
        self._raise_delta_failure()
        session_id = self._session_id(chat_id)
        await self._runtime.publish_event(
            event_type="message.proactive",
            session_id=session_id,
            payload={
                "content": message,
                "media": [],
                "metadata": {"source": "message_push"},
            },
        )

    async def send_stream(self, chat_id: str, message: str) -> None:
        await self.send(chat_id, message)

    async def _execute_command(
        self,
        *,
        device_id: str,
        frame: GenericCommand | MessageSendCommand,
    ) -> CommandReply:
        if frame.type == "session.list":
            return await self._list_sessions(device_id, frame)
        if frame.type == "session.create":
            raise MobileCommandError(
                "unsupported_command",
                "当前版本由手机本地生成 mobile session_id",
            )
        if frame.type == "session.open":
            return await self._open_session(device_id, frame)
        if frame.type == "history.get":
            return await self._get_history(device_id, frame)
        if frame.type == "message.send":
            return await self._send_message(device_id, frame)
        if frame.type == "turn.stop":
            return await self._stop_turn(device_id, frame)
        raise MobileCommandError("unsupported_command", f"尚不支持命令: {frame.type}")

    async def _list_sessions(
        self,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        """发布全部移动会话索引，供手机按需分页拉取缺失历史。"""

        # 1. 所有已认证手机共享 mobile 渠道的完整会话空间
        _expect_keys(frame.payload, set())
        ctx = self._require_ctx()
        session_rows = {item["key"]: item for item in ctx.session_manager.list_sessions()}
        session_ids = tuple(
            session_id
            for session_id in session_rows
            if session_id.startswith(f"{self.name}:")
        )

        # 2. 补充抽屉标题和历史消息总数
        items: list[dict[str, object]] = []
        for session_id in session_ids:
            session = session_rows.get(session_id)
            if session is None:
                raise RuntimeError(f"已绑定移动会话在 session store 中不存在: {session_id}")
            messages, total = ctx.session_manager.control_store.list_messages_for_dashboard(
                session_key=session_id,
                page=1,
                page_size=1,
                sort_by="seq",
                sort_order="asc",
            )
            first_content = str(messages[0]["content"]).strip() if messages else ""
            items.append(
                {
                    "session_id": session_id,
                    "title": first_content.splitlines()[0][:32] or "新对话",
                    "updated_at": str(session["updated_at"]),
                    "message_count": total,
                }
            )

        # 3. 索引也走 durable event，断线后仍会重放
        await self._runtime.publish_event(
            event_type="session.list",
            device_id=device_id,
            payload={"items": cast(list[object], items)},
        )
        return CommandReply(type="session.list.ok", payload={"total": len(items)})

    async def _open_session(
        self,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        _expect_keys(frame.payload, set())
        session_id = self._require_mobile_session(frame.session_id)
        await self._runtime.publish_event(
            event_type="session.updated",
            session_id=session_id,
            payload={"session_id": session_id, "state": "opened"},
        )
        return CommandReply(
            type="session.open.ok",
            session_id=session_id,
            payload={"session_id": session_id},
        )

    async def _get_history(
        self,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        session_id = self._require_mobile_session(frame.session_id)
        pagination = _pagination_payload(frame.payload)
        (
            items,
            total,
        ) = self._require_ctx().session_manager.control_store.list_messages_for_dashboard(
            session_key=session_id,
            page=pagination["page"],
            page_size=pagination["page_size"],
            sort_by="seq",
            sort_order="asc",
        )
        mobile_items = [_mobile_history_item(item) for item in items]
        page_payload: dict[str, object] = {
            "items": cast(list[object], mobile_items),
            "total": total,
            **pagination,
        }
        await self._runtime.publish_event(
            event_type="history.page",
            session_id=session_id,
            device_id=device_id,
            payload=page_payload,
        )
        return CommandReply(
            type="history.get.ok",
            session_id=session_id,
            payload={"total": total, **pagination},
        )

    async def _send_message(
        self,
        device_id: str,
        frame: MessageSendCommand,
    ) -> CommandReply:
        session_id = self._normalize_session_id(frame.session_id)
        if frame.payload.media_refs:
            raise MobileCommandError(
                "attachments_not_ready",
                "当前版本尚未开放移动端附件发送",
            )
        if not frame.payload.text.strip():
            raise MobileCommandError("empty_message", "消息内容不能为空")
        ctx = self._require_ctx()
        self._runtime.storage.claim_session(
            device_id=device_id,
            session_id=session_id,
            created_at=_utc_now(),
        )
        _ = ctx.session_manager.get_or_create(session_id)
        await ctx.bus.publish_inbound(
            InboundMessage(
                channel=self.name,
                sender=f"device:{device_id}",
                chat_id=self._chat_id(session_id),
                content=frame.payload.text,
                metadata={
                    "client_request_id": frame.id,
                    "client_message_id": frame.payload.client_message_id,
                    "client_created_at": frame.payload.client_created_at,
                    "device_id": device_id,
                },
            )
        )
        return CommandReply(
            type="message.send.ok",
            session_id=session_id,
            payload={
                "accepted": True,
                "client_message_id": frame.payload.client_message_id,
            },
        )

    async def _stop_turn(
        self,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        _expect_keys(frame.payload, set())
        session_id = self._normalize_session_id(frame.session_id)
        self._runtime.storage.require_session_owner(
            device_id=device_id,
            session_id=session_id,
        )
        interrupt = self._require_ctx().interrupt_controller
        if interrupt is None:
            raise MobileCommandError("interrupt_unavailable", "当前未启用中断功能")
        result = interrupt.request_interrupt(
            session_key=session_id,
            sender=f"device:{device_id}",
            command="/stop",
        )
        await self._runtime.publish_event(
            event_type="turn.interrupted",
            session_id=session_id,
            turn_id=frame.turn_id,
            payload={"status": result.status, "message": result.message},
        )
        return CommandReply(
            type="turn.stop.ok",
            session_id=session_id,
            turn_id=frame.turn_id,
            payload={"status": result.status, "message": result.message},
        )

    async def _on_turn_started(self, event: TurnStarted) -> None:
        self._raise_delta_failure()
        if event.channel != self.name:
            return
        turn_id = event.turn_id or event.session_key
        self._active_turn_ids[event.session_key] = turn_id
        process_key = (event.session_key, turn_id)
        if process_key in self._process_turns:
            raise RuntimeError(
                f"mobile turn.started 重复: {event.session_key}/{turn_id}"
            )
        self._process_turns[process_key] = _ProcessTurnState(
            next_ordinal=0,
            thinking_block=None,
            tool_blocks={},
        )
        await self._runtime.publish_event(
            event_type="turn.started",
            session_id=event.session_key,
            turn_id=turn_id,
            payload={"content": event.content},
        )

    async def _on_stream_delta(self, event: StreamDeltaReady) -> None:
        self._raise_delta_failure()
        if event.channel != self.name:
            return
        turn_id = event.turn_id or self._current_turn_id(event.session_key)
        if event.thinking_delta:
            block_id, ordinal = self._thinking_block(event.session_key, turn_id)
            await self._buffer_delta(
                session_id=event.session_key,
                turn_id=turn_id,
                event_type="react.thinking.delta",
                delta=event.thinking_delta,
                block_id=block_id,
                ordinal=ordinal,
            )
        if event.content_delta:
            await self._buffer_delta(
                session_id=event.session_key,
                turn_id=turn_id,
                event_type="answer.delta",
                delta=event.content_delta,
                block_id=None,
                ordinal=None,
            )

    async def _on_tool_call_started(self, event: ToolCallStarted) -> None:
        self._raise_delta_failure()
        if event.channel != self.name:
            return
        turn_id = event.turn_id or self._current_turn_id(event.session_key)
        await self._flush_deltas(event.session_key, turn_id)
        state = self._require_process_state(event.session_key, turn_id)
        state.thinking_block = None
        if event.call_id in state.tool_blocks:
            raise RuntimeError(f"mobile tool call_id 重复开始: {event.call_id}")
        ordinal = state.next_ordinal
        state.next_ordinal += 1
        block_id = f"tool:{event.call_id}"
        state.tool_blocks[event.call_id] = (block_id, ordinal)
        await self._runtime.publish_event(
            event_type="react.tool.started",
            session_id=event.session_key,
            turn_id=turn_id,
            payload={
                "call_id": event.call_id,
                "block_id": block_id,
                "ordinal": ordinal,
                "tool_name": event.tool_name,
                "arguments": cast(dict[str, object], event.arguments),
            },
        )

    async def _on_tool_call_completed(self, event: ToolCallCompleted) -> None:
        self._raise_delta_failure()
        if event.channel != self.name:
            return
        turn_id = event.turn_id or self._current_turn_id(event.session_key)
        await self._flush_deltas(event.session_key, turn_id)
        state = self._require_process_state(event.session_key, turn_id)
        block = state.tool_blocks.get(event.call_id)
        if block is None:
            raise RuntimeError(f"mobile tool completed 缺少 started: {event.call_id}")
        block_id, ordinal = block
        await self._runtime.publish_event(
            event_type="react.tool.completed",
            session_id=event.session_key,
            turn_id=turn_id,
            payload={
                "call_id": event.call_id,
                "block_id": block_id,
                "ordinal": ordinal,
                "tool_name": event.tool_name,
                "status": event.status,
                "result_preview": event.result_preview,
            },
        )

    async def _on_response(self, message: OutboundMessage) -> None:
        self._raise_delta_failure()
        session_id = self._session_id(message.chat_id)
        turn_id = message.control_turn_id or self._current_turn_id(session_id)
        await self._flush_deltas(session_id, turn_id)
        await self._runtime.publish_event(
            event_type="message.final",
            session_id=session_id,
            turn_id=turn_id,
            payload={
                "content": message.content,
                "thinking": message.thinking or "",
                "media": list(message.media),
                "metadata": dict(message.metadata),
            },
        )
        _ = self._active_turn_ids.pop(session_id, None)
        _ = self._process_turns.pop((session_id, turn_id), None)

    async def _buffer_delta(
        self,
        *,
        session_id: str,
        turn_id: str,
        event_type: str,
        delta: str,
        block_id: str | None,
        ordinal: int | None,
    ) -> None:
        """按 50ms 或 4KiB 合并连续 delta，限制 SQLite 写入频率。"""

        key = (session_id, turn_id)
        lock = self._delta_locks.setdefault(key, asyncio.Lock())
        flush_now = False
        async with lock:
            batch = self._delta_batches.get(key)
            if batch is None:
                if len(self._delta_batches) >= _MAX_DELTA_BATCHES:
                    raise RuntimeError("mobile delta batch 已达到 256 个活跃 turn 上限")
                timer = asyncio.create_task(
                    self._flush_after_delay(key),
                    name=f"mobile-delta-flush:{turn_id}",
                )
                timer.add_done_callback(self._on_delta_timer_done)
                batch = _DeltaBatch(segments=[], byte_count=0, timer=timer)
                self._delta_batches[key] = batch
            segment_identity = (event_type, block_id, ordinal)
            if (
                batch.segments
                and (
                    batch.segments[-1][0],
                    batch.segments[-1][2],
                    batch.segments[-1][3],
                )
                == segment_identity
            ):
                previous_type, previous_delta, previous_block, previous_ordinal = (
                    batch.segments[-1]
                )
                batch.segments[-1] = (
                    previous_type,
                    previous_delta + delta,
                    previous_block,
                    previous_ordinal,
                )
            else:
                batch.segments.append((event_type, delta, block_id, ordinal))
            batch.byte_count += len(delta.encode("utf-8"))
            flush_now = batch.byte_count >= _DELTA_FLUSH_BYTES
        if flush_now:
            await self._flush_deltas(session_id, turn_id)

    async def _flush_after_delay(self, key: tuple[str, str]) -> None:
        await asyncio.sleep(_DELTA_FLUSH_SECONDS)
        await self._flush_deltas(*key)

    async def _flush_deltas(self, session_id: str, turn_id: str) -> None:
        """按原始顺序发布一个 turn 当前已聚合的 delta 段。"""

        key = (session_id, turn_id)
        lock = self._delta_locks.setdefault(key, asyncio.Lock())
        async with lock:
            batch = self._delta_batches.pop(key, None)
            if batch is None:
                return
            current = asyncio.current_task()
            if batch.timer is not current:
                _ = batch.timer.cancel()
            for event_type, delta, block_id, ordinal in batch.segments:
                payload: dict[str, object] = {"delta": delta}
                if block_id is not None:
                    if ordinal is None:
                        raise AssertionError("thinking delta block 缺少 ordinal")
                    payload["block_id"] = block_id
                    payload["ordinal"] = ordinal
                await self._runtime.publish_event(
                    event_type=event_type,
                    session_id=session_id,
                    turn_id=turn_id,
                    payload=payload,
                )
        _ = self._delta_locks.pop(key, None)

    def _on_delta_timer_done(self, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        self._delta_failure = error
        ctx = self._ctx
        if ctx is not None:
            ctx.log.error(
                "mobile delta flush 失败",
                exc_info=(type(error), error, error.__traceback__),
            )

    def _raise_delta_failure(self) -> None:
        if self._delta_failure is None:
            return
        error = self._delta_failure
        self._delta_failure = None
        raise error

    def _thinking_block(self, session_id: str, turn_id: str) -> tuple[str, int]:
        state = self._require_process_state(session_id, turn_id)
        if state.thinking_block is None:
            ordinal = state.next_ordinal
            state.next_ordinal += 1
            state.thinking_block = (f"thinking:{turn_id}:{ordinal}", ordinal)
        return state.thinking_block

    def _require_process_state(
        self,
        session_id: str,
        turn_id: str,
    ) -> _ProcessTurnState:
        state = self._process_turns.get((session_id, turn_id))
        if state is None:
            raise RuntimeError(f"mobile process turn 未开始: {session_id}/{turn_id}")
        return state

    def _require_mobile_session(self, value: str | None) -> str:
        session_id = self._normalize_session_id(value)
        if not self._require_ctx().session_manager.session_exists(session_id):
            raise MobileCommandError("session_not_found", f"会话不存在: {session_id}")
        return session_id

    def _normalize_session_id(self, value: object) -> str:
        if not isinstance(value, str) or not value.startswith(f"{self.name}:"):
            raise MobileCommandError(
                "invalid_session", "session_id 必须属于 mobile 渠道"
            )
        raw_id = value[len(self.name) + 1 :]
        try:
            parsed = UUID(raw_id)
        except ValueError as error:
            raise MobileCommandError(
                "invalid_session",
                "mobile session_id 必须包含 UUID",
            ) from error
        if raw_id not in {str(parsed), parsed.hex}:
            raise MobileCommandError(
                "invalid_session",
                "mobile session_id 必须使用规范小写 UUID",
            )
        return value

    def _session_id(self, chat_id: str) -> str:
        text = str(chat_id).strip()
        if not text:
            raise ValueError("chat_id 不能为空")
        if text.startswith(f"{self.name}:"):
            return self._normalize_session_id(text)
        return f"{self.name}:{text}"

    def _chat_id(self, session_id: str) -> str:
        return self._normalize_session_id(session_id)[len(self.name) + 1 :]

    def _current_turn_id(self, session_id: str) -> str:
        return self._active_turn_ids.get(session_id, session_id)

    def _require_ctx(self) -> ChannelContext:
        if self._ctx is None:
            raise RuntimeError("MobileRealtimeChannel 尚未启动")
        return self._ctx


def _command_hash(frame: GenericCommand | MessageSendCommand) -> str:
    payload = frame.model_dump(mode="json", exclude_none=True)
    _ = payload.pop("connection_epoch")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reply_from_receipt(receipt: CommandReceipt) -> CommandReply:
    if receipt.status != "completed":
        return CommandReply(
            type=f"{receipt.command_type}.error",
            payload={
                "code": "command_outcome_unknown",
                "message": "该命令上次执行时中断，请使用新的命令 ID 核对状态",
            },
        )
    if receipt.reply_type is None or receipt.reply_payload_json is None:
        raise AssertionError("completed 命令收据缺少回复")
    return CommandReply(
        type=receipt.reply_type,
        payload=_decode_reply_payload(receipt.reply_payload_json),
        session_id=receipt.session_id,
        turn_id=receipt.turn_id,
    )


def _decode_reply_payload(raw: str) -> dict[str, object]:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"命令回复包含重复字段: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"命令回复包含非标准常量: {value}")

    decoded = json.loads(
        raw,
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    if not isinstance(decoded, dict):
        raise TypeError("命令回复 payload 必须是 JSON object")
    return cast(dict[str, object], decoded)


def _pagination_payload(payload: Mapping[str, object]) -> dict[str, int]:
    _expect_keys(payload, {"page", "page_size"})
    page = payload.get("page", 1)
    page_size = payload.get("page_size", 50)
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        raise MobileCommandError("invalid_pagination", "page 必须是正整数")
    if (
        not isinstance(page_size, int)
        or isinstance(page_size, bool)
        or not 1 <= page_size <= 200
    ):
        raise MobileCommandError("invalid_pagination", "page_size 必须在 1..200")
    return {"page": page, "page_size": page_size}


def _expect_keys(payload: Mapping[str, object], allowed: set[str]) -> None:
    unexpected = set(payload) - allowed
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise MobileCommandError("invalid_payload", f"payload 包含未知字段: {names}")


def _mobile_history_item(item: Mapping[str, object]) -> dict[str, object]:
    """裁剪服务端内部字段，只向手机同步可展示历史。"""

    mobile_extra: dict[str, object] = {}
    for field in ("reasoning_content", "turn_duration_ms"):
        value = item.get(field)
        if isinstance(value, (str, int, float)):
            mobile_extra[field] = value

    return {
        "id": str(item["id"]),
        "session_key": str(item["session_key"]),
        "seq": cast(int, item["seq"]),
        "role": str(item["role"]),
        "content": str(item["content"]),
        "tool_chain": _mobile_tool_chain(item.get("tool_chain")),
        "extra": mobile_extra,
        "ts": str(item["timestamp"]),
    }


def _mobile_tool_chain(value: object) -> list[dict[str, object]] | None:
    if not isinstance(value, list):
        return None
    groups: list[dict[str, object]] = []
    for raw_group in cast(list[object], value):
        if not isinstance(raw_group, dict):
            continue
        group_record = cast(dict[str, object], raw_group)
        group: dict[str, object] = {}
        for field in ("reasoning_content", "text"):
            group_text = group_record.get(field)
            if isinstance(group_text, str) and group_text:
                group[field] = group_text
        raw_calls = group_record.get("calls")
        calls: list[dict[str, object]] = []
        if isinstance(raw_calls, list):
            for raw_call in cast(list[object], raw_calls):
                if not isinstance(raw_call, dict):
                    continue
                call_record = cast(dict[str, object], raw_call)
                name = call_record.get("name")
                if not isinstance(name, str) or not name:
                    continue
                arguments = call_record.get("final_arguments", call_record.get("arguments"))
                arguments_record = cast(dict[str, object], arguments) if isinstance(arguments, dict) else None
                description = arguments_record.get("description") if arguments_record is not None else None
                call: dict[str, object] = {
                    "call_id": str(call_record.get("call_id") or ""),
                    "name": name,
                    "status": str(call_record.get("status") or "success"),
                }
                if isinstance(description, str) and description:
                    call["description"] = description
                result = call_record.get("result")
                if result is not None:
                    call["result_preview"] = str(result)[:2000]
                calls.append(call)
        group["calls"] = calls
        groups.append(group)
    return groups


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = ["CommandReply", "MobileRealtimeChannel"]
