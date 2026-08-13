from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import sqlite3
from collections import defaultdict
from collections.abc import AsyncGenerator, Iterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from time import monotonic
from typing import TYPE_CHECKING, cast
from uuid import UUID, uuid4

from bus.events import (
    ChannelMessage,
    DeliveryReceipt,
    DeliveryStatus,
    InboundMessage,
    OutboundMessage,
    TurnTerminalStatus,
    channel_message_from_outbound,
)
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnOutputCompleted,
    TurnStarted,
)
from agent.plugins.mobile_ui import (
    MobileUiPluginUnavailable,
    MobileUiQueryOverloaded,
    MobileUiQueryTimeout,
    MobileUiRpcExecutionError,
    MobileUiRpcInvalidRequest,
    MobileUiStaleRevision,
)
from agent.control.models import TurnStatus
from agent.model_runtime.session_selection import read_session_model_selection
from core.common.diagnostic_log import turn_milestone
from infra.mobile_realtime.runtime_inspection import (
    RuntimeInspectionError,
    RuntimeInspectionService,
)
from infra.channels.contract import ChannelContext
from infra.channels.reply_context import build_reply_inbound_text
from infra.mobile_realtime.attachments import (
    AttachmentChunk,
    AttachmentRequestError,
    AttachmentTransferService,
    MAX_ATTACHMENT_CHUNK_BYTES,
    attachment_descriptor,
)
from infra.mobile_realtime.protocol import (
    AttachmentBeginCommand,
    AttachmentDownloadCommand,
    AttachmentFinishCommand,
    ClientCommand,
    GenericCommand,
    MessageReplyReference,
    MessageSendCommand,
    MAX_JSON_FRAME_BYTES,
    TURN_OUTPUT_COMPLETED_CAPABILITY,
)
from infra.mobile_realtime.plugin_ui import PluginUiQuery, PluginUiQueryScheduler
from infra.mobile_realtime.remote_media import (
    RemoteMediaError,
    RemoteMediaSnapshot,
    snapshot_remote_media,
)
from infra.mobile_realtime.storage import (
    AttachmentRecord,
    AttachmentStateError,
    CommandReceipt,
    CommandReceiptCapacityError,
    MobileStorageError,
)

if TYPE_CHECKING:
    from agent.model_runtime.registry import ModelRegistry
    from agent.plugins.mobile_ui import MobileUiProvider
    from infra.mobile_realtime.gateway import MobileGatewayRuntime


logger = logging.getLogger(__name__)


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
    binary: AttachmentChunk | None = None
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class _ResolvedReply:
    message_id: str
    role: str
    content: str
    preview: str


@dataclass(slots=True)
class _DeltaBatch:
    segments: list[tuple[str, str, str | None, int | None]]
    byte_count: int
    timer: asyncio.Task[None]


@dataclass(slots=True)
class _ProcessTurnState:
    next_ordinal: int
    thinking_block: tuple[str, int] | None
    tool_blocks: dict[str, tuple[str, int, float]]
    answer_segments: list[str]
    control_turn_id: str = ""
    first_thinking_received: bool = False
    first_answer_received: bool = False
    first_thinking_published: bool = False
    first_answer_published: bool = False
    client_message_id: str = ""
    final_suffix_emitted: str = ""


_DELTA_FLUSH_BYTES = 4 * 1024
_DELTA_FLUSH_INTERVAL_SECONDS = 1.0 / 60.0
_MAX_DELTA_BATCHES = 256
_MAX_DEVICE_CAPABILITIES = 128
_MAX_DEVICE_CAPABILITY_LENGTH = 512
_BOT_COMMAND_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_PLUGIN_SEGMENT = r"[A-Za-z0-9][A-Za-z0-9._-]*"
_PLUGIN_ID_PATTERN = re.compile(rf"^{_PLUGIN_SEGMENT}(?:@{_PLUGIN_SEGMENT})?$")
_MOBILE_TOOL_ARGUMENT_MAX_DEPTH = 5
_MOBILE_TOOL_ARGUMENT_MAX_ITEMS = 256
_MOBILE_TOOL_ARGUMENT_MAX_CONTAINER_ITEMS = 64
_MOBILE_TOOL_ARGUMENT_MAX_STRING_CHARS = 2_000
_MOBILE_TOOL_ARGUMENT_MAX_BYTES = 8 * 1024
_MOBILE_HISTORY_TOOL_ARGUMENT_MAX_BYTES = 8 * 1024
_MOBILE_HISTORY_PAYLOAD_MAX_BYTES = 240 * 1024
_MOBILE_TOOL_ARGUMENT_REDACTED = "[已隐藏]"
_MOBILE_TOOL_ARGUMENT_TRUNCATED = "[已截断]"
_MOBILE_HISTORY_DETAIL_OMITTED = "[历史同步时已省略过长详情]"


def _utf8_chunks(text: str, max_bytes: int) -> Iterator[str]:
    """按字符边界生成不超过指定 UTF-8 字节数的非空片段。"""

    start = 0
    chunk_bytes = 0
    for index, character in enumerate(text):
        character_bytes = len(character.encode("utf-8"))
        if chunk_bytes and chunk_bytes + character_bytes > max_bytes:
            yield text[start:index]
            start = index
            chunk_bytes = 0
        chunk_bytes += character_bytes
    if start < len(text):
        yield text[start:]


_MOBILE_TOOL_SECRET_KEYS = frozenset(
    {
        "secret",
        "auth",
        "token",
        "password",
        "passwd",
        "authorization",
        "cookie",
        "apikey",
        "privatekey",
        "secretaccesskey",
        "accesstoken",
        "refreshtoken",
        "clientsecret",
        "credential",
        "credentials",
    }
)
_MOBILE_TOOL_SECRET_TEXT_PATTERN = re.compile(
    r"(?ix)"
    r"(?<![a-z0-9_])(?:[a-z][a-z0-9]*[-_])*(?:authorization|"
    r"proxy[-_ ]?authorization|secret[-_ ]?access[-_ ]?key|api[-_ ]?key|"
    r"access[-_ ]?token|refresh[-_ ]?token|client[-_ ]?secret|token|secret|"
    r"password|passwd|cookie)\s*[:=]"
    r"|--?(?:api[-_]?key|access[-_]?token|refresh[-_]?token|client[-_]?secret|"
    r"password|passwd|cookie|token)\s+(?:[^\s]|$)"
    r"|\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"
)


class MobileRealtimeChannel:
    """把移动协议接入现有消息、生命周期和主动推送总线。"""

    name = "mobile"

    def __init__(self, runtime: MobileGatewayRuntime) -> None:
        self._runtime = runtime
        self._ctx: ChannelContext | None = None
        self._processing_commands: set[tuple[str, str]] = set()
        self._receipt_completion_failures: set[tuple[str, str]] = set()
        self._active_turn_ids: dict[str, str] = {}
        self._process_turns: dict[tuple[str, str], _ProcessTurnState] = {}
        self._send_received_at: dict[tuple[str, str], float] = {}
        self._turn_started_at: dict[tuple[str, str], float] = {}
        self._delta_batches: dict[tuple[str, str], _DeltaBatch] = {}
        self._delta_locks = defaultdict[tuple[str, str], asyncio.Lock](asyncio.Lock)
        self._turn_terminals: dict[tuple[str, str], str] = {}
        self._delta_failure: BaseException | None = None
        self._attachments: AttachmentTransferService | None = None
        self._mobile_ui_provider: MobileUiProvider | None = None
        self._mobile_ui_scheduler: PluginUiQueryScheduler | None = None
        self._mobile_ui_catalog_identity = ""
        self._mobile_ui_hot_connections: dict[str, int] = {}
        self._runtime_inspection: RuntimeInspectionService | None = None
        self._model_registry: ModelRegistry | None = None

    def bind_runtime_inspection(self, service: RuntimeInspectionService) -> None:
        """绑定只读运行时检查服务。"""

        if self._runtime_inspection is not None:
            raise RuntimeError("Runtime inspection service 已绑定")
        self._runtime_inspection = service

    def bind_model_registry(self, registry: ModelRegistry) -> None:
        """绑定 Core 模型目录，移动端只消费该权威快照。"""

        if self._model_registry is not None:
            raise RuntimeError("Model runtime registry 已绑定")
        self._model_registry = registry

    def bind_mobile_ui_provider(self, provider: MobileUiProvider) -> None:
        """绑定读取当前插件快照的移动 UI 提供器。"""

        if self._mobile_ui_provider is not None:
            raise RuntimeError("Mobile UI provider 已绑定")
        self._mobile_ui_provider = provider
        self._mobile_ui_scheduler = PluginUiQueryScheduler(provider)
        self._mobile_ui_catalog_identity = _mobile_ui_catalog_identity(
            provider.catalog()
        )

    async def refresh_mobile_ui_catalog(self) -> None:
        """目录内容变化时通知所有手机重新拉取插件 UI。"""

        provider = self._mobile_ui_provider
        if provider is None:
            return
        catalog = provider.catalog()
        identity = _mobile_ui_catalog_identity(catalog)
        if identity == self._mobile_ui_catalog_identity:
            return
        _ = await asyncio.gather(
            *(
                self._runtime.publish_connection_control(
                    control_type="plugin.ui.changed",
                    payload={"catalog_revision": identity},
                    device_id=device_id,
                    connection_epoch=connection_epoch,
                )
                for device_id, connection_epoch in tuple(
                    self._mobile_ui_hot_connections.items()
                )
            )
        )
        self._mobile_ui_catalog_identity = identity

    async def start(self, ctx: ChannelContext) -> None:
        """注册移动渠道的出站、流事件和主动推送入口。"""

        if self._ctx is not None:
            raise RuntimeError("MobileRealtimeChannel 已启动")
        self._ctx = ctx
        self._attachments = AttachmentTransferService(
            self._runtime.storage,
            ctx.attachment_store,
            max_attachment_bytes=self._runtime.config.max_attachment_mb * 1024 * 1024,
        )
        _ = ctx.bus.subscribe_outbound(self.name, self._on_response)
        _ = ctx.event_bus.on(TurnStarted, self._on_turn_started)
        _ = ctx.event_bus.on(StreamDeltaReady, self._on_stream_delta)
        _ = ctx.event_bus.on(ToolCallStarted, self._on_tool_call_started)
        _ = ctx.event_bus.on(ToolCallCompleted, self._on_tool_call_completed)
        _ = ctx.event_bus.on(TurnOutputCompleted, self._on_output_completed)
        _ = ctx.push_tool.register_channel(
            self.name,
            deliver=self._deliver_message,
        )

    async def stop(self) -> None:
        """先发布活动 turn 的终态，再释放移动渠道运行态。"""

        # 1. 计划停机必须把已向手机公开的活动 turn 收敛为终态；
        #    每个 turn 都走同一 owner 的终态 barrier，避免与 in-flight final 竞态。
        store = self._require_ctx().session_manager.control_store
        for session_id, turn_id in tuple(self._active_turn_ids.items()):
            turn = store.read_turn(turn_id)
            if turn is not None and turn.status is TurnStatus.COMPLETED:
                continue
            payload = self._interrupt_payload(
                session_id,
                turn_id,
                status=(
                    turn.status.value
                    if turn is not None and turn.status.is_terminal
                    else TurnStatus.INTERRUPTED.value
                ),
                message="服务端正在维护，本轮生成已中断",
                reason="runtime_shutdown",
            )
            _ = await self._publish_terminal(
                session_id=session_id,
                turn_id=turn_id,
                event_type="turn.interrupted",
                payload=payload,
            )

        # 2. 终态已持久化后再取消批处理并清理进程内状态
        for batch in self._delta_batches.values():
            _ = batch.timer.cancel()
        if self._delta_batches:
            _ = await asyncio.gather(
                *(batch.timer for batch in self._delta_batches.values()),
                return_exceptions=True,
            )
        self._delta_batches.clear()
        self._delta_locks.clear()
        self._turn_terminals.clear()
        self._delta_failure = None
        self._attachments = None
        self._ctx = None
        self._active_turn_ids.clear()
        self._process_turns.clear()
        self._send_received_at.clear()
        self._turn_started_at.clear()
        self._processing_commands.clear()
        self._receipt_completion_failures.clear()

    async def reconcile_active_turns(
        self,
        *,
        device_id: str,
        active_turns: tuple[str, ...],
    ) -> None:
        """把客户端残留的活动 turn 与 SessionDB 权威终态对账。"""

        # 1. 只接受属于移动会话且已被手机声明过的权威 turn
        if not active_turns:
            return
        store = self._require_ctx().session_manager.control_store
        for turn_id in active_turns:
            turn = store.read_turn(turn_id)
            if (
                turn is None
                or not turn.thread_id.startswith("mobile:")
                or not self._runtime.storage.has_session_claim(turn.thread_id)
            ):
                continue

            # 2. completed 由 durable message.final/history 恢复，其余终态补发关闭信号；
            #    统一走每 turn 唯一 owner 的终态 barrier，保证 delta→terminal 顺序。
            if not turn.status.is_terminal or turn.status is TurnStatus.COMPLETED:
                continue
            payload = self._interrupt_payload(
                turn.thread_id,
                turn.id,
                status=turn.status.value,
                message="服务端已确认本轮生成结束",
                reason="resume_reconciliation",
            )
            _ = await self._publish_terminal(
                session_id=turn.thread_id,
                turn_id=turn.id,
                event_type="turn.interrupted",
                payload=payload,
                device_id=device_id,
            )

    async def handle_command(
        self,
        *,
        device_id: str,
        frame: ClientCommand,
    ) -> CommandReply:
        """幂等执行业务命令，并持久化可跨重连复用的回复。"""

        # 1. 先持久化命令占用，避免重连重复触发 Agent turn
        self._raise_delta_failure()
        try:
            receipt, created = self._runtime.storage.reserve_command(
                device_id=device_id,
                command_id=frame.id,
                command_type=frame.type,
                request_hash=_command_hash(frame),
                created_at=_utc_now(),
            )
        except CommandReceiptCapacityError as error:
            raise MobileCommandError(
                "mobile_command_receipt_capacity_reached",
                str(error),
            ) from error
        if not created:
            replay = self._recover_message_send_receipt(
                device_id=device_id,
                frame=frame,
                receipt=receipt,
            )
            if (
                isinstance(frame, AttachmentDownloadCommand)
                and replay.type == "attachment.download.ok"
            ):
                return self._download_attachment(frame, replay)
            return CommandReply(
                type=replay.type,
                payload=replay.payload,
                session_id=replay.session_id,
                turn_id=replay.turn_id,
                binary=replay.binary,
                replayed=True,
            )

        # 2. 当前实例只在命令实际执行期间拥有 processing 收据
        command_key = (device_id, frame.id)
        self._processing_commands.add(command_key)
        try:
            try:
                reply = await self._execute_command(device_id=device_id, frame=frame)
            except (MobileCommandError, RuntimeInspectionError) as error:
                reply = CommandReply(
                    type=f"{frame.type}.error",
                    payload={"code": error.code, "message": str(error)},
                    session_id=frame.session_id,
                    turn_id=frame.turn_id,
                )

            # 3. 只有收据完成后才释放当前进程对未决副作用的所有权
            _validate_reply_frame_size(frame, reply)
            try:
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
            except Exception:
                self._receipt_completion_failures.add(command_key)
                raise
            stored = _reply_from_receipt(completed)
            return CommandReply(
                type=stored.type,
                payload=stored.payload,
                session_id=stored.session_id,
                turn_id=stored.turn_id,
                binary=reply.binary,
            )
        finally:
            self._processing_commands.discard(command_key)

    async def handle_plugin_ui_command(
        self,
        *,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        """执行不写 command receipt 的 Mobile Plugin UI v2 临时请求。"""

        try:
            if frame.type == "plugin.ui.catalog":
                return self._plugin_ui_catalog(device_id, frame)
            if frame.type == "plugin.ui.asset.get":
                return self._plugin_ui_asset(frame)
            if frame.type == "plugin.ui.query":
                return await self._plugin_ui_query(device_id, frame)
            if frame.type == "plugin.ui.cancel":
                return await self._cancel_plugin_ui(device_id, frame)
        except MobileUiPluginUnavailable as error:
            raise MobileCommandError("plugin_unavailable", str(error)) from error
        except MobileUiStaleRevision as error:
            raise MobileCommandError("stale_revision", str(error)) from error
        except MobileUiRpcInvalidRequest as error:
            raise MobileCommandError("plugin_invalid_request", str(error)) from error
        except MobileUiRpcExecutionError as error:
            raise MobileCommandError("plugin_failed", str(error)) from error
        except MobileUiQueryTimeout as error:
            raise MobileCommandError("plugin_timeout", str(error)) from error
        except MobileUiQueryOverloaded as error:
            raise MobileCommandError("plugin_overloaded", str(error)) from error
        raise MobileCommandError("unsupported_command", f"尚不支持命令: {frame.type}")

    def prepare_message_content(self, frame: GenericCommand) -> dict[str, object]:
        """校验正文下载请求并返回当前不可变内容描述。"""

        # 1. 请求必须精确引用 history.page 宣告的正文版本
        _expect_keys(frame.payload, {"message_id", "byte_length", "sha256"})
        session_id = self._require_mobile_session(frame.session_id)
        message_id = _expect_nonempty_string(frame.payload["message_id"], "message_id")
        byte_length = _expect_nonnegative_int(
            frame.payload["byte_length"], "byte_length"
        )
        sha256 = _expect_sha256(frame.payload["sha256"], "sha256")

        # 2. 重新读取 SessionDB 权威正文，拒绝客户端猜测或过期 manifest
        content = self.read_message_content(
            session_id=session_id,
            message_id=message_id,
            byte_length=byte_length,
            sha256=sha256,
        )
        return {
            "message_id": message_id,
            "byte_length": len(content),
            "sha256": sha256,
        }

    def read_message_content(
        self,
        *,
        session_id: str,
        message_id: str,
        byte_length: int,
        sha256: str,
    ) -> bytes:
        """从 SessionDB 读取并核对票据绑定的完整 UTF-8 正文。"""

        message = self._require_ctx().session_manager.control_store.get_message(
            message_id
        )
        if message is None or message["session_key"] != session_id:
            raise MobileCommandError("message_not_found", "消息正文不存在")
        content = str(message["content"]).encode("utf-8")
        if len(content) != byte_length or hashlib.sha256(content).hexdigest() != sha256:
            raise MobileCommandError(
                "content_changed", "消息正文与历史 manifest 不一致"
            )
        return content

    async def cancel_plugin_ui_device(self, device_id: str) -> None:
        """断线时取消设备的全部临时插件查询。"""

        scheduler = self._mobile_ui_scheduler
        if scheduler is not None:
            await scheduler.cancel_device(device_id)
        _ = self._mobile_ui_hot_connections.pop(device_id, None)

    def _recover_message_send_receipt(
        self,
        *,
        device_id: str,
        frame: ClientCommand,
        receipt: CommandReceipt,
    ) -> CommandReply:
        """从已持久化消息修复中断的 message.send 收据。"""

        # 1. 已完成或非消息命令继续复用稳定回复
        if receipt.status in {"completed", "outcome_unknown"}:
            self._processing_commands.discard((device_id, frame.id))
            return _reply_from_receipt(receipt)
        if (device_id, frame.id) in self._processing_commands:
            return CommandReply(
                type=f"{receipt.command_type}.error",
                payload={
                    "code": "command_in_progress",
                    "message": "该命令仍在执行，请等待原命令 ID 的最终收据",
                },
            )
        if not isinstance(frame, MessageSendCommand):
            unknown = self._runtime.storage.mark_command_outcome_unknown(
                device_id=device_id,
                command_id=frame.id,
            )
            return _reply_from_receipt(unknown)

        # 2. 只有同一 client_message_id 已落库时，才能确认副作用已完成
        session_id = self._normalize_session_id(frame.session_id)
        message = (
            self._require_ctx().session_manager.control_store.get_message_by_client_id(
                session_id,
                frame.payload.client_message_id,
            )
        )
        if message is None:
            if (device_id, frame.id) in self._processing_commands:
                return CommandReply(
                    type=f"{receipt.command_type}.error",
                    payload={
                        "code": "command_in_progress",
                        "message": "该命令仍在执行，请等待原命令 ID 的最终收据",
                    },
                )
            if self._require_ctx().bus.has_pending_mobile_handoff(
                session_key=session_id,
                client_message_id=frame.payload.client_message_id,
            ):
                return CommandReply(
                    type=f"{receipt.command_type}.error",
                    payload={
                        "code": "command_in_progress",
                        "message": "该命令已进入持久化队列，请等待原命令 ID 的最终收据",
                    },
                )
            if (device_id, frame.id) in self._receipt_completion_failures:
                self._receipt_completion_failures.discard((device_id, frame.id))
                unknown = self._runtime.storage.mark_command_outcome_unknown(
                    device_id=device_id,
                    command_id=frame.id,
                )
                return _reply_from_receipt(unknown)
            return self._complete_interrupted_message_send(
                device_id=device_id,
                frame=frame,
                session_id=session_id,
            )
        if message["role"] != "user":
            raise RuntimeError(
                f"client_message_id 绑定了非用户消息: {frame.payload.client_message_id}"
            )

        # 3. 原子补写成功收据，后续重放继续得到相同 ACK
        completed = self._runtime.storage.complete_command(
            device_id=device_id,
            command_id=frame.id,
            reply_type="message.send.ok",
            reply_payload_json=json.dumps(
                {
                    "accepted": True,
                    "client_message_id": frame.payload.client_message_id,
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ),
            session_id=session_id,
            turn_id=None,
            completed_at=_utc_now(),
        )
        self._receipt_completion_failures.discard((device_id, frame.id))
        self._processing_commands.discard((device_id, frame.id))
        return _reply_from_receipt(completed)

    def _complete_interrupted_message_send(
        self,
        *,
        device_id: str,
        frame: MessageSendCommand,
        session_id: str,
    ) -> CommandReply:
        """把服务重启前未落库的发送收据收束为可安全重试。"""

        completed = self._runtime.storage.complete_command(
            device_id=device_id,
            command_id=frame.id,
            reply_type="message.send.error",
            reply_payload_json=json.dumps(
                {
                    "code": "command_interrupted",
                    "message": "上次发送在服务重启时中断，可以安全重试",
                },
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
                allow_nan=False,
            ),
            session_id=session_id,
            turn_id=None,
            completed_at=_utc_now(),
        )
        return _reply_from_receipt(completed)

    async def send(self, chat_id: str, message: str) -> None:
        await self._send_proactive(chat_id, message, delivery_id=None)

    async def send_with_metadata(
        self,
        chat_id: str,
        message: str,
        metadata: dict[str, object],
    ) -> None:
        """发送携带核心主动投递身份的消息。"""

        delivery_id = metadata.get("delivery_id")
        if (
            not isinstance(delivery_id, str)
            or not delivery_id
            or len(delivery_id) > 128
        ):
            raise ValueError("mobile proactive delivery_id 无效")
        await self._send_proactive(chat_id, message, delivery_id=delivery_id)

    async def _send_proactive(
        self,
        chat_id: str,
        message: str,
        *,
        delivery_id: str | None,
    ) -> None:
        """把主动文字发布为移动端持久事件。"""

        self._raise_delta_failure()
        session_id = self._session_id(chat_id)
        payload: dict[str, object] = {
            "content": message,
            "attachments": [],
            "metadata": {"source": "message_push"},
            "control_turn_id": f"turn:{uuid4().hex}",
        }
        if delivery_id is not None:
            payload["delivery_id"] = delivery_id
        await self._runtime.publish_event(
            event_type="message.proactive",
            session_id=session_id,
            payload=payload,
        )

    async def send_stream(self, chat_id: str, message: str) -> None:
        await self.send(chat_id, message)

    async def _deliver_message(self, message: ChannelMessage) -> DeliveryReceipt:
        """把完整主动消息原子提交为一个 Mobile durable event。"""

        self._raise_delta_failure()
        if message.metadata.get("_channel_commit_role") == "passive":
            return await self._deliver_passive_message(message)
        if message.control_turn_id is None:
            message = replace(message, control_turn_id=f"turn:{uuid4().hex}")
        session_id = self._session_id(message.chat_id)
        if not self._runtime.storage.list_active_devices():
            return DeliveryReceipt(
                DeliveryStatus.FAILED,
                detail="Mobile 没有可接收消息的已配对设备",
            )
        delivery_id = message.metadata.get("delivery_id")
        if delivery_id is not None and (
            not isinstance(delivery_id, str)
            or not delivery_id
            or len(delivery_id) > 128
        ):
            raise ValueError("mobile proactive delivery_id 无效")
        if not message.attachments:
            payload: dict[str, object] = {
                "content": message.content,
                "attachments": [],
                "metadata": {"source": "message_push"},
                "control_turn_id": message.control_turn_id,
            }
            if delivery_id is not None:
                payload["delivery_id"] = delivery_id
            await self._runtime.publish_event(
                event_type="message.proactive",
                session_id=session_id,
                payload=payload,
            )
            return DeliveryReceipt(DeliveryStatus.SUCCESS)

        snapshots: list[RemoteMediaSnapshot] = []
        candidates: tuple[AttachmentRecord, ...] = ()
        try:
            # 1. 所有源先成为受限本地快照，尚不写数据库
            paths: list[str] = []
            overrides: list[tuple[str, str] | None] = []
            for attachment in message.attachments:
                if attachment.source.startswith(("http://", "https://")):
                    snapshot = await snapshot_remote_media(
                        attachment.source,
                        self._require_ctx().attachment_store,
                        max_bytes=(
                            self._runtime.config.max_attachment_mb * 1024 * 1024
                        ),
                    )
                    snapshots.append(snapshot)
                    paths.append(str(snapshot.path))
                    overrides.append((snapshot.filename, snapshot.content_type))
                else:
                    paths.append(attachment.source)
                    overrides.append(None)
            candidates = await asyncio.to_thread(
                self._require_attachments().snapshot_outbound_batch,
                session_id=session_id,
                local_media_paths=tuple(paths),
                metadata_overrides=tuple(overrides),
            )

            # 2. 附件记录与所有设备 inbox 行在同一事务提交
            resolved = await self._runtime.publish_event_with_outbound_attachments(
                candidates=candidates,
                session_id=session_id,
                payload_builder=lambda records: self._proactive_attachment_payload(
                    message,
                    records,
                    delivery_id=cast(str | None, delivery_id),
                ),
            )
            return DeliveryReceipt(
                DeliveryStatus.SUCCESS,
                canonical_media=tuple(record.local_path for record in resolved),
            )
        except (
            AttachmentRequestError,
            AttachmentStateError,
            MobileStorageError,
            RemoteMediaError,
            OSError,
            sqlite3.Error,
        ) as error:
            if candidates:
                self._require_attachments().cleanup_outbound_candidates(candidates)
            logger.warning(
                "mobile proactive 附件提交失败: session=%s error=%s",
                session_id,
                error,
            )
            return DeliveryReceipt(DeliveryStatus.FAILED, detail=str(error))
        except BaseException:
            if candidates:
                self._require_attachments().cleanup_outbound_candidates(candidates)
            raise
        finally:
            for snapshot in snapshots:
                try:
                    snapshot.path.unlink(missing_ok=True)
                except OSError as cleanup_error:
                    logger.error(
                        "mobile proactive cleanup_degraded: 远程媒体临时快照清理失败 "
                        "path=%s error=%s",
                        snapshot.path,
                        cleanup_error,
                    )

    @staticmethod
    def _proactive_attachment_payload(
        message: ChannelMessage,
        records: tuple[AttachmentRecord, ...],
        *,
        delivery_id: str | None,
    ) -> dict[str, object]:
        """用事务内解析出的 canonical 附件构造主动事件。"""

        payload: dict[str, object] = {
            "content": message.content,
            "attachments": [attachment_descriptor(record) for record in records],
            "metadata": {"source": "message_push"},
            "control_turn_id": message.control_turn_id,
        }
        if delivery_id is not None:
            payload["delivery_id"] = delivery_id
        return payload

    async def _execute_command(
        self,
        *,
        device_id: str,
        frame: ClientCommand,
    ) -> CommandReply:
        if frame.type == "device.update":
            return await self._update_device_capabilities(device_id, frame)
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
        if frame.type == "command.list":
            return self._list_commands(frame)
        if frame.type == "runtime.document.list":
            _expect_keys(frame.payload, set())
            return CommandReply(
                type="runtime.document.list.ok",
                payload=self._require_runtime_inspection().list_documents(),
            )
        if frame.type == "runtime.document.get":
            _expect_keys(frame.payload, {"document_id"})
            document_id = _expect_nonempty_string(
                frame.payload["document_id"],
                "document_id",
            )
            return CommandReply(
                type="runtime.document.get.ok",
                payload=self._require_runtime_inspection().get_document(document_id),
            )
        if frame.type == "scheduler.job.list":
            _expect_keys(frame.payload, set())
            return CommandReply(
                type="scheduler.job.list.ok",
                payload=self._require_runtime_inspection().list_jobs(),
            )
        if frame.type == "scheduler.job.get":
            _expect_keys(frame.payload, {"job_id"})
            job_id = _expect_nonempty_string(frame.payload["job_id"], "job_id")
            return CommandReply(
                type="scheduler.job.get.ok",
                payload=self._require_runtime_inspection().get_job(job_id),
            )
        if frame.type == "runtime.capability.list":
            _expect_keys(frame.payload, set())
            return CommandReply(
                type="runtime.capability.list.ok",
                payload=await self._require_runtime_inspection().list_capabilities(),
            )
        if frame.type == "runtime.mcp.get":
            _expect_keys(frame.payload, {"owner_id", "server_name"})
            owner_id = _expect_nonempty_string(frame.payload["owner_id"], "owner_id")
            server_name = _expect_nonempty_string(
                frame.payload["server_name"],
                "server_name",
            )
            return CommandReply(
                type="runtime.mcp.get.ok",
                payload=await self._require_runtime_inspection().get_mcp(
                    owner_id,
                    server_name,
                ),
            )
        if frame.type == "model.catalog.get":
            return await self._model_catalog(frame)
        if frame.type == "message.send":
            return await self._send_message(device_id, frame)
        if frame.type == "turn.stop":
            return await self._stop_turn(device_id, frame)
        if frame.type == "attachment.begin":
            return await self._begin_attachment(device_id, frame)
        if frame.type == "attachment.finish":
            return await self._finish_attachment(device_id, frame)
        if frame.type == "attachment.download":
            return self._download_attachment(frame)
        raise MobileCommandError("unsupported_command", f"尚不支持命令: {frame.type}")

    async def _update_device_capabilities(
        self,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        """设备升级后刷新持久化能力声明，无需重新配对。

        复用配对协议的边界约束：最多 128 项、每项 1..512 字符，杜绝通过
        命令帧把超长 capability 集合写入 mobile_devices。
        """

        _expect_keys(frame.payload, {"capabilities"})
        raw_capabilities = frame.payload["capabilities"]
        if not isinstance(raw_capabilities, list):
            raise MobileCommandError(
                "invalid_payload",
                "device.update capabilities 必须是字符串数组",
            )
        if len(raw_capabilities) > _MAX_DEVICE_CAPABILITIES:
            raise MobileCommandError(
                "invalid_payload",
                f"device.update capabilities 最多 {_MAX_DEVICE_CAPABILITIES} 项",
            )
        capabilities: list[str] = []
        for item in raw_capabilities:
            if (
                not isinstance(item, str)
                or not item
                or len(item) > _MAX_DEVICE_CAPABILITY_LENGTH
            ):
                raise MobileCommandError(
                    "invalid_payload",
                    f"device.update capability 必须是 1..{_MAX_DEVICE_CAPABILITY_LENGTH} 字符的非空字符串",
                )
            capabilities.append(item)
        if len(set(capabilities)) != len(capabilities):
            raise MobileCommandError(
                "invalid_payload",
                "device.update capabilities 不能包含重复项",
            )
        await self._runtime.refresh_device_capabilities(
            device_id=device_id,
            capabilities=tuple(capabilities),
        )
        return CommandReply(type="device.update.ok", payload={})

    async def _model_catalog(self, frame: GenericCommand) -> CommandReply:
        """返回当前模型 generation 和指定会话已经提交的选择。"""

        _expect_keys(frame.payload, set())
        session_id = self._normalize_session_id(frame.session_id)
        registry = self._model_registry
        if registry is None:
            raise MobileCommandError("model_registry_unavailable", "模型注册表尚未绑定")
        current = await registry.refresh()
        runtimes = [
            {
                key: runtime[key]
                for key in (
                    "id",
                    "provider",
                    "model",
                    "sourceId",
                    "sourceName",
                    "reasoningEffort",
                    "supportedReasoningEfforts",
                    "roles",
                    "contextWindow",
                    "inputModalities",
                )
            }
            for runtime in registry.list_runtimes()
        ]
        selection = (
            read_session_model_selection(
                self._require_ctx().session_manager.get_existing(session_id).metadata
            )
            if self._require_ctx().session_manager.session_exists(session_id)
            else None
        )
        return CommandReply(
            type="model.catalog.get.ok",
            session_id=session_id,
            payload={
                "generation_id": current.generation_id,
                "default_runtime": current.role_runtime_ids["default"],
                "selected_runtime_id": selection.model_ref if selection else "",
                "selected_reasoning_effort": (
                    selection.reasoning_effort if selection else ""
                ),
                "runtimes": runtimes,
            },
        )

    def _require_runtime_inspection(self) -> RuntimeInspectionService:
        service = self._runtime_inspection
        if service is None:
            raise RuntimeInspectionError(
                "runtime_inspection_unavailable",
                "运行时检查服务尚未绑定",
            )
        return service

    def _list_commands(self, frame: GenericCommand) -> CommandReply:
        """返回当前已启用插件的快捷命令目录。"""

        # 1. 命令目录由 ChannelContext 统一提供
        _expect_keys(frame.payload, set())
        items: list[dict[str, str]] = []
        seen: set[str] = set()
        for raw_command, raw_description in self._require_ctx().mobile_bot_commands:
            command = raw_command.strip().removeprefix("/")
            description = raw_description.strip()
            if not _BOT_COMMAND_PATTERN.fullmatch(command):
                raise RuntimeError(f"插件命令名无效: {raw_command!r}")
            if not description or len(description) > 256:
                raise RuntimeError(f"插件命令描述无效: {command}")
            if command == "stop" or command in seen:
                continue
            seen.add(command)
            items.append({"command": command, "description": description})

        # 2. 保持插件注册顺序，便于管理高频命令的位置
        return CommandReply(type="command.list.ok", payload={"items": items})

    def _plugin_ui_catalog(self, device_id: str, frame: GenericCommand) -> CommandReply:
        """返回不含源码的 Mobile Plugin UI v2 catalog。"""

        _expect_keys(frame.payload, {"subscribe", "if_revision"})
        subscribe = frame.payload.get("subscribe", False)
        if not isinstance(subscribe, bool):
            raise MobileCommandError("invalid_payload", "subscribe 必须是布尔值")
        if subscribe:
            self._mobile_ui_hot_connections[device_id] = frame.connection_epoch
        else:
            _ = self._mobile_ui_hot_connections.pop(device_id, None)
        if_revision = frame.payload.get("if_revision")
        if if_revision is not None and (
            not isinstance(if_revision, str)
            or not re.fullmatch(r"[0-9a-f]{64}", if_revision)
        ):
            raise MobileCommandError("invalid_revision", "if_revision 无效")
        provider = self._mobile_ui_provider
        catalog: dict[str, object]
        if provider is None:
            catalog = {
                "catalog_revision": hashlib.sha256(b"[]").hexdigest(),
                "items": [],
            }
        else:
            catalog = provider.catalog()
        if catalog["catalog_revision"] == if_revision:
            return CommandReply(
                type="plugin.ui.catalog.not_modified",
                payload={"catalog_revision": if_revision},
            )
        return CommandReply(type="plugin.ui.catalog.ok", payload=catalog)

    def _plugin_ui_asset(self, frame: GenericCommand) -> CommandReply:
        """按 revision 和摘要返回一个未缓存资源。"""

        _expect_keys(
            frame.payload,
            {"plugin_id", "plugin_revision", "kind", "sha256"},
        )
        plugin_id = frame.payload["plugin_id"]
        plugin_revision = frame.payload["plugin_revision"]
        kind = frame.payload["kind"]
        sha256 = frame.payload["sha256"]
        if not isinstance(plugin_id, str) or not _PLUGIN_ID_PATTERN.fullmatch(
            plugin_id
        ):
            raise MobileCommandError("invalid_plugin", "plugin_id 无效")
        if not isinstance(plugin_revision, str) or not 1 <= len(plugin_revision) <= 128:
            raise MobileCommandError("invalid_revision", "plugin_revision 无效")
        if kind not in {"module", "stylesheet"}:
            raise MobileCommandError("invalid_asset", "kind 无效")
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise MobileCommandError("invalid_asset", "sha256 无效")
        asset = self._require_mobile_ui_provider().asset(
            plugin_id,
            plugin_revision,
            cast(str, kind),
            sha256,
        )
        return CommandReply(type="plugin.ui.asset.get.ok", payload=asset)

    async def _plugin_ui_query(
        self,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        """校验并调度一个有 owner 的只读插件查询。"""

        query = self.prepare_plugin_ui_query(device_id=device_id, frame=frame)
        result = await self._require_mobile_ui_scheduler().execute(
            device_id,
            query,
        )
        return CommandReply(type="plugin.ui.query.ok", payload={"result": result})

    def prepare_plugin_ui_query(
        self,
        *,
        device_id: str,
        frame: GenericCommand,
    ) -> PluginUiQuery:
        """校验一次插件查询并冻结其 HTTP/WS 共用的调度参数。"""

        # 1. 在协议边界验证插件、owner、参数和槽位
        _expect_keys(
            frame.payload,
            {"owner_id", "plugin_id", "plugin_revision", "method", "payload", "slot"},
        )
        owner_id = frame.payload["owner_id"]
        plugin_id = frame.payload["plugin_id"]
        plugin_revision = frame.payload["plugin_revision"]
        method = frame.payload["method"]
        payload = frame.payload["payload"]
        slot = frame.payload["slot"]
        if not isinstance(owner_id, str) or not 1 <= len(owner_id) <= 128:
            raise MobileCommandError("invalid_owner", "owner_id 无效")
        if not isinstance(plugin_id, str) or not _PLUGIN_ID_PATTERN.fullmatch(
            plugin_id
        ):
            raise MobileCommandError("invalid_plugin", "plugin_id 无效")
        if not isinstance(plugin_revision, str) or not 1 <= len(plugin_revision) <= 128:
            raise MobileCommandError("invalid_revision", "plugin_revision 无效")
        if not isinstance(method, str) or not re.fullmatch(
            r"[a-z][a-z0-9_.-]{0,63}", method
        ):
            raise MobileCommandError("invalid_method", "插件方法无效")
        if not isinstance(payload, dict):
            raise MobileCommandError("invalid_payload", "插件参数必须是对象")
        if (
            _mobile_tool_argument_encoded_size(cast(dict[str, object], payload))
            > 64 * 1024
        ):
            raise MobileCommandError("invalid_payload", "插件参数超过 64 KiB")
        if slot not in {
            "dashboard.main",
            "turn.before_reasoning",
            "turn.before_tool",
            "turn.after_answer",
            "drawer.panel",
        }:
            raise MobileCommandError("invalid_slot", "插件 slot 无效")
        session_id = (
            None
            if frame.session_id is None
            else self._require_mobile_session(frame.session_id)
        )
        _ = self._require_mobile_ui_scheduler()

        # 2. 后续执行只消费这份已验证的不可变查询
        return PluginUiQuery(
            request_id=frame.id,
            owner_id=owner_id,
            plugin_id=plugin_id,
            plugin_revision=plugin_revision,
            method=method,
            payload=cast(dict[str, object], payload),
            slot=cast(str, slot),
            session_id=session_id,
            turn_id=frame.turn_id,
        )

    async def _cancel_plugin_ui(
        self,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        """取消一个已卸载 owner 的全部查询。"""

        _expect_keys(frame.payload, {"owner_id"})
        owner_id = frame.payload["owner_id"]
        if not isinstance(owner_id, str) or not 1 <= len(owner_id) <= 128:
            raise MobileCommandError("invalid_owner", "owner_id 无效")
        cancelled = await self._require_mobile_ui_scheduler().cancel_owner(
            device_id,
            owner_id,
        )
        return CommandReply(
            type="plugin.ui.cancel.ok",
            payload={"cancelled": cancelled},
        )

    def _require_mobile_ui_provider(self) -> MobileUiProvider:
        provider = self._mobile_ui_provider
        if provider is None:
            raise MobileCommandError("plugin_unavailable", "服务端没有启用移动 UI 插件")
        return provider

    def _require_mobile_ui_scheduler(self) -> PluginUiQueryScheduler:
        scheduler = self._mobile_ui_scheduler
        if scheduler is None:
            raise MobileCommandError("plugin_unavailable", "服务端没有启用移动 UI 插件")
        return scheduler

    async def handle_attachment_chunk(
        self,
        *,
        device_id: str,
        chunk: AttachmentChunk,
    ) -> None:
        """落盘一个二进制分片，并按稀疏确认发布进度。"""

        try:
            record, should_report = await asyncio.to_thread(
                self._require_attachments().append_chunk,
                device_id=device_id,
                chunk=chunk,
            )
        except (AttachmentRequestError, AttachmentStateError) as error:
            raise MobileCommandError("attachment_chunk_rejected", str(error)) from error
        if should_report:
            await self._runtime.publish_event(
                event_type="attachment.progress",
                device_id=device_id,
                session_id=record.session_id,
                payload={
                    "attachment_id": record.attachment_id,
                    "transferred_bytes": record.transferred_bytes,
                    "size_bytes": record.size_bytes,
                },
            )

    async def _begin_attachment(
        self,
        device_id: str,
        frame: AttachmentBeginCommand,
    ) -> CommandReply:
        session_id = self._normalize_session_id(frame.session_id)
        try:
            record = await asyncio.to_thread(
                self._require_attachments().begin_upload,
                device_id=device_id,
                attachment_id=frame.payload.attachment_id,
                session_id=session_id,
                filename=frame.payload.filename,
                content_type=frame.payload.content_type,
                size_bytes=frame.payload.size_bytes,
                sha256=frame.payload.sha256,
            )
        except (AttachmentRequestError, AttachmentStateError) as error:
            raise MobileCommandError("attachment_begin_rejected", str(error)) from error
        return CommandReply(
            type="attachment.begin.ok",
            session_id=session_id,
            payload={
                **attachment_descriptor(record),
                "next_offset": record.transferred_bytes,
                "chunk_size": MAX_ATTACHMENT_CHUNK_BYTES,
                "state": record.state,
            },
        )

    async def _finish_attachment(
        self,
        device_id: str,
        frame: AttachmentFinishCommand,
    ) -> CommandReply:
        session_id = self._normalize_session_id(frame.session_id)
        try:
            record = await asyncio.to_thread(
                self._require_attachments().finish_upload,
                device_id=device_id,
                session_id=session_id,
                attachment_id=frame.payload.attachment_id,
            )
        except (AttachmentRequestError, AttachmentStateError) as error:
            raise MobileCommandError(
                "attachment_finish_rejected", str(error)
            ) from error
        await self._runtime.publish_event(
            event_type="attachment.ready",
            device_id=device_id,
            session_id=session_id,
            payload=attachment_descriptor(record),
        )
        return CommandReply(
            type="attachment.finish.ok",
            session_id=session_id,
            payload={**attachment_descriptor(record), "state": "ready"},
        )

    def _download_attachment(
        self,
        frame: AttachmentDownloadCommand,
        stored: CommandReply | None = None,
    ) -> CommandReply:
        """读取一个出站附件分片，并让二进制帧先于确认回复发送。"""

        session_id = self._normalize_session_id(frame.session_id)
        try:
            outbound = self._require_attachments().read_outbound_chunk(
                session_id=session_id,
                attachment_id=frame.payload.attachment_id,
                offset=frame.payload.offset,
            )
        except (AttachmentRequestError, AttachmentStateError) as error:
            raise MobileCommandError(
                "attachment_download_rejected", str(error)
            ) from error
        next_offset = outbound.offset + len(outbound.data)
        payload: dict[str, object] = {
            **outbound.descriptor,
            "offset": outbound.offset,
            "next_offset": next_offset,
            "complete": outbound.eof,
        }
        if stored is not None and stored.payload != payload:
            raise RuntimeError("已完成的附件下载回复与当前文件状态不一致")
        return CommandReply(
            type="attachment.download.ok",
            session_id=session_id,
            payload=payload,
            binary=AttachmentChunk(
                attachment_id=frame.payload.attachment_id,
                offset=outbound.offset,
                data=outbound.data,
            ),
        )

    async def _list_sessions(
        self,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        """发布全部移动会话索引，供手机按需分页拉取缺失历史。"""

        # 1. 所有已认证手机共享 mobile 渠道的完整会话空间
        _expect_keys(frame.payload, set())
        ctx = self._require_ctx()
        session_rows = {
            item["key"]: item for item in ctx.session_manager.list_sessions()
        }
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
                raise RuntimeError(
                    f"已绑定移动会话在 session store 中不存在: {session_id}"
                )
            messages, total = (
                ctx.session_manager.control_store.list_messages_for_dashboard(
                    session_key=session_id,
                    page=1,
                    page_size=1,
                    sort_by="seq",
                    sort_order="asc",
                )
            )
            first_content = str(messages[0]["content"]).strip() if messages else ""
            items.append(
                {
                    "session_id": session_id,
                    "title": (
                        first_content.splitlines()[0][:32]
                        if first_content
                        else "新对话"
                    ),
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
        query = _history_query_payload(frame.payload)
        store = self._require_ctx().session_manager.control_store
        if query["content_ref_version"] == 1:
            snapshot_max_seq = query["snapshot_max_seq"]
            if snapshot_max_seq is None:
                total, snapshot_max_seq = store.mobile_history_snapshot(session_id)
            else:
                _, current_max_seq = store.mobile_history_snapshot(session_id)
                if snapshot_max_seq > current_max_seq:
                    raise MobileCommandError(
                        "invalid_snapshot",
                        "历史快照高水位超过服务端当前序列",
                    )
                total = store.mobile_history_count_through(session_id, snapshot_max_seq)
            after_seq = cast(int, query["after_seq"])
            snapshot_max_seq = cast(int, snapshot_max_seq)
            items = store.list_mobile_history_page(
                session_key=session_id,
                after_seq=after_seq,
                through_seq=snapshot_max_seq,
                page_size=cast(int, query["page_size"]),
            )
            next_after_seq = int(items[-1]["seq"]) if items else after_seq
            has_more = (
                len(items) == cast(int, query["page_size"])
                and next_after_seq < snapshot_max_seq
            )
        else:
            pagination = {
                "page": cast(int, query["page"]),
                "page_size": cast(int, query["page_size"]),
            }
            items, total = store.list_messages_for_dashboard(
                session_key=session_id,
                page=pagination["page"],
                page_size=pagination["page_size"],
                sort_by="seq",
                sort_order="asc",
            )
        mobile_items = [await self._mobile_history_item(item) for item in items]
        if query["content_ref_version"] == 1:
            page_payload: dict[str, object] = {
                "items": cast(list[object], mobile_items),
                "total": total,
                "page_size": query["page_size"],
                "content_ref_version": 1,
                "after_seq": query["after_seq"],
                "next_after_seq": next_after_seq,
                "snapshot_max_seq": snapshot_max_seq,
                "has_more": has_more,
            }
        else:
            page_payload = {
                "items": cast(list[object], mobile_items),
                "total": total,
                "page": query["page"],
                "page_size": query["page_size"],
            }
        _fit_mobile_history_payload(
            page_payload,
            allow_content_refs=query["content_ref_version"] == 1,
        )
        await self._runtime.publish_event(
            event_type="history.page",
            session_id=session_id,
            device_id=device_id,
            payload=page_payload,
        )
        return CommandReply(
            type="history.get.ok",
            session_id=session_id,
            payload={
                key: value for key, value in page_payload.items() if key != "items"
            },
        )

    async def _send_message(
        self,
        device_id: str,
        frame: MessageSendCommand,
    ) -> CommandReply:
        session_id = self._normalize_session_id(frame.session_id)
        if frame.id != frame.payload.client_message_id:
            raise MobileCommandError(
                "client_message_id_mismatch",
                "message.send 的命令 ID 必须与 client_message_id 一致",
            )
        if not frame.payload.text.strip() and not frame.payload.media_refs:
            raise MobileCommandError("empty_message", "文字和附件不能同时为空")
        # 1. 时间链起点：客户端 message.send 到达服务端
        turn_milestone(
            logger,
            "tl:send.received",
            session_id=session_id,
            client_message_id=frame.payload.client_message_id,
        )
        self._send_received_at[(session_id, frame.payload.client_message_id)] = (
            monotonic()
        )
        try:
            return await self._send_message_inner(device_id, frame, session_id)
        except asyncio.CancelledError:
            # 2a. turn 尚未 started 前被取消：显式删除自己的计时起点后原样重抛。
            _ = self._send_received_at.pop(
                (session_id, frame.payload.client_message_id),
                None,
            )
            raise
        except Exception:
            # 2b. 任何异常都只删自己的计时起点，绝不连带同 session 排队消息。
            _ = self._send_received_at.pop(
                (session_id, frame.payload.client_message_id),
                None,
            )
            raise

    async def _send_message_inner(
        self,
        device_id: str,
        frame: MessageSendCommand,
        session_id: str,
    ) -> CommandReply:
        ctx = self._require_ctx()
        claimed_session = self._runtime.storage.has_session_claim(session_id)
        if claimed_session and not ctx.session_manager.session_exists(session_id):
            raise MobileCommandError(
                "session_not_found",
                "会话已从电脑端删除，请在手机上新建会话后继续",
            )
        try:
            media = self._require_attachments().resolve_uploads(
                device_id=device_id,
                session_id=session_id,
                attachment_ids=list(frame.payload.media_refs),
            )
        except (AttachmentRequestError, AttachmentStateError) as error:
            raise MobileCommandError("attachment_not_ready", str(error)) from error
        reply = self._resolve_reply(session_id, frame.payload.reply_to)
        inbound_content = frame.payload.text
        metadata: dict[str, object] = {
            "client_request_id": frame.id,
            "client_message_id": frame.payload.client_message_id,
            "client_created_at": frame.payload.client_created_at,
            "device_id": device_id,
            "require_existing_session": True,
        }
        if frame.payload.model_runtime_id is not None:
            metadata["model_runtime_id"] = frame.payload.model_runtime_id
            metadata["model_reasoning_effort"] = (
                frame.payload.model_reasoning_effort or ""
            )
        if reply is not None:
            metadata.update(
                {
                    "display_content": frame.payload.text,
                    "reply_to_message_id": reply.message_id,
                    "reply_role": reply.role,
                    "reply_preview": reply.preview,
                }
            )
            inbound_content = build_reply_inbound_text(
                frame.payload.text,
                reply.content,
                sender_label="你" if reply.role == "user" else "Akashic",
            )
        self._runtime.storage.claim_session(
            device_id=device_id,
            session_id=session_id,
            created_at=_utc_now(),
        )
        if not claimed_session:
            _ = ctx.session_manager.get_or_create(session_id)
        try:
            _, admission_id = ctx.session_manager.admit_existing(session_id)
        except KeyError as error:
            raise MobileCommandError(
                "session_not_found",
                "会话已从电脑端删除，请在手机上新建会话后继续",
            ) from error
        inbound = InboundMessage(
            channel=self.name,
            sender=f"device:{device_id}",
            chat_id=self._chat_id(session_id),
            content=inbound_content,
            media=media,
            metadata=metadata,
            session_admission_id=admission_id,
        )
        try:
            await ctx.bus.publish_inbound(inbound)
        except BaseException:
            ctx.session_manager.release_admission(admission_id)
            raise
        # 3. 时间链：入站消息被总线接受并返回 ACK
        received_at = self._send_received_at.get(
            (session_id, frame.payload.client_message_id)
        )
        turn_milestone(
            logger,
            "tl:send.ack",
            session_id=session_id,
            client_message_id=frame.payload.client_message_id,
            duration_ms=(
                (monotonic() - received_at) * 1_000 if received_at is not None else None
            ),
        )
        return CommandReply(
            type="message.send.ok",
            session_id=session_id,
            payload={
                "accepted": True,
                "client_message_id": frame.payload.client_message_id,
            },
        )

    def _resolve_reply(
        self,
        session_id: str,
        reference: MessageReplyReference | None,
    ) -> _ResolvedReply | None:
        """把客户端引用解析为同会话的 canonical 消息摘要。"""

        if reference is None:
            return None
        store = self._require_ctx().session_manager.control_store
        if reference.message_id is not None:
            target = store.get_message(reference.message_id)
        elif reference.client_message_id is not None:
            target = store.get_message_by_client_id(
                session_id,
                reference.client_message_id,
            )
        else:
            target = store.get_message_by_delivery_id(
                session_id,
                cast(str, reference.delivery_id),
            )
        if target is None:
            raise MobileCommandError(
                "reply_target_missing", "被引用的消息不存在或尚未同步"
            )
        if target["session_key"] != session_id:
            raise MobileCommandError(
                "reply_target_session_mismatch", "不能引用其他会话的消息"
            )
        role = str(target["role"])
        if role not in {"user", "assistant"}:
            raise RuntimeError(f"被引用消息角色无效: {target['id']} {role}")
        content = _reply_source_text(target)
        return _ResolvedReply(
            message_id=str(target["id"]),
            role=role,
            content=content,
            preview=_reply_preview(content),
        )

    async def _stop_turn(
        self,
        device_id: str,
        frame: GenericCommand,
    ) -> CommandReply:
        _expect_keys(frame.payload, set())
        session_id = self._require_mobile_session(frame.session_id)
        turn_id = frame.turn_id
        if turn_id is None:
            raise MobileCommandError("turn_id_required", "停止生成必须携带 turn_id")
        turn = self._require_ctx().session_manager.control_store.read_turn(turn_id)
        if (
            turn is not None
            and turn.thread_id == session_id
            and turn.status.is_terminal
        ):
            # 已缓冲 delta 必须先于终态发布，保证 delta→terminal 顺序；
            # completed 由 durable message.final 恢复，其余终态经 barrier 收口。
            if turn.status is not TurnStatus.COMPLETED:
                _ = await self._publish_terminal(
                    session_id=session_id,
                    turn_id=turn_id,
                    event_type="turn.interrupted",
                    payload=self._interrupt_payload(
                        session_id,
                        turn_id,
                        status=turn.status.value,
                        message="服务端已确认本轮生成结束",
                        reason="stop_reconciliation",
                    ),
                    device_id=device_id,
                )
            else:
                _ = await self._flush_deltas(session_id, turn_id)
            self._clear_turn_maps(session_id, turn_id)
            return CommandReply(
                type="turn.stop.ok",
                session_id=session_id,
                turn_id=turn_id,
                payload={
                    "status": "already_terminal",
                    "terminal_status": turn.status.value,
                    "message": "目标 turn 已经结束",
                },
            )
        active_turn_id = self._active_turn_ids.get(session_id)
        if active_turn_id is None:
            raise MobileCommandError("turn_not_active", "当前会话没有正在生成的内容")
        if active_turn_id != turn_id:
            raise MobileCommandError("stale_turn", "目标 turn 已结束或已被新一轮替代")
        interrupt = self._require_ctx().interrupt_controller
        if interrupt is None:
            raise MobileCommandError("interrupt_unavailable", "当前未启用中断功能")
        result = interrupt.request_interrupt(
            session_key=session_id,
            sender=f"device:{device_id}",
            command="/stop",
        )
        if result.status not in {"interrupted", "idle"}:
            raise RuntimeError(f"中断控制器返回未知状态: {result.status}")
        _ = await self._publish_terminal(
            session_id=session_id,
            turn_id=turn_id,
            event_type="turn.interrupted",
            payload=self._interrupt_payload(
                session_id,
                turn_id,
                status=result.status,
                message=result.message,
            ),
        )
        self._clear_turn_maps(session_id, turn_id)
        return CommandReply(
            type="turn.stop.ok",
            session_id=session_id,
            turn_id=turn_id,
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
            answer_segments=[],
            control_turn_id=event.control_turn_id or turn_id,
            client_message_id=event.client_message_id,
        )
        self._turn_started_at[process_key] = monotonic()
        # 同一 key 的旧终态墓碑（同 turn_id 重试）不得压制新一轮增量。
        _ = self._turn_terminals.pop(process_key, None)
        await self._runtime.publish_event(
            event_type="turn.started",
            session_id=event.session_key,
            turn_id=turn_id,
            payload={
                "content": event.content,
                "client_message_id": event.client_message_id,
                "control_turn_id": event.control_turn_id or turn_id,
            },
        )
        # 3. 时间链：服务端接受 turn；duration 为 send.received → turn.started
        received_at = self._send_received_at.get(
            (event.session_key, event.client_message_id)
        )
        turn_milestone(
            logger,
            "tl:turn.started",
            session_id=event.session_key,
            turn_id=turn_id,
            client_message_id=event.client_message_id,
            duration_ms=(
                (monotonic() - received_at) * 1_000 if received_at is not None else None
            ),
        )

    async def _on_stream_delta(self, event: StreamDeltaReady) -> None:
        self._raise_delta_failure()
        if event.channel != self.name:
            return
        session_id = event.session_key
        turn_id = event.turn_id or self._current_turn_id(session_id)
        # 0. 终态已收口（墓碑在而锁已清理）的迟到事件先读 closed 直接丢弃，
        #    绝不等待锁、绝不重建 batch/timer/lock，也绝不让它滑向崩溃路径。
        if (session_id, turn_id) in self._turn_terminals:
            self._log_delta_dropped(session_id, turn_id, event)
            return
        # 1. 单 owner 原子接受：state mutation 与 bounded chunks 接受在同一把
        #    per-turn 锁内，terminal 无法在检查与提交之间插入。
        accepted, flush_now = await self._accept_delta(session_id, turn_id, event)
        if not accepted or not flush_now:
            return
        # 2. 首段即时 flush 必须在锁外（锁内不做网络 I/O）；若 terminal 已代为
        #    flush 并收口，不得错误再标 published。
        published = await self._flush_deltas(session_id, turn_id)
        if published and (session_id, turn_id) not in self._turn_terminals:
            self._mark_first_deltas_published(session_id, turn_id)

    async def _accept_delta(
        self,
        session_id: str,
        turn_id: str,
        event: StreamDeltaReady,
    ) -> tuple[bool, bool]:
        """锁内原子接受一个增量事件；返回 (是否接受, 是否需锁外 flush)。"""

        flush_now = False
        async with self._delta_locked(session_id, turn_id, require_state=True) as lock:
            if lock is None:
                self._log_delta_dropped(session_id, turn_id, event)
                return False, False
            state = self._require_process_state(session_id, turn_id)
            started_at = self._turn_started_at.get((session_id, turn_id))
            # 首段即时 flush 触发器：该类型首段尚未真实发布（失败可重试）。
            first_thinking = not state.first_thinking_published
            first_answer = not state.first_answer_published
            # thinking 与 answer 两个非空字段各自无条件调用一次对应 locked
            # helper；聚合 flush 决策不得参与 helper 是否执行——or 短路会让
            # thinking 首段吞掉同一事件里的 content_delta。顺序仍 thinking→answer。
            thinking_flush = False
            answer_flush = False
            if event.thinking_delta:
                thinking_flush = (
                    self._accept_thinking_delta_locked(
                        session_id,
                        turn_id,
                        state,
                        started_at,
                        event.thinking_delta,
                    )
                    or first_thinking
                )
            if event.content_delta:
                answer_flush = (
                    self._accept_answer_delta_locked(
                        session_id,
                        turn_id,
                        state,
                        started_at,
                        event.content_delta,
                    )
                    or first_answer
                )
            flush_now = thinking_flush or answer_flush
        return True, flush_now

    def _accept_thinking_delta_locked(
        self,
        session_id: str,
        turn_id: str,
        state: _ProcessTurnState,
        started_at: float | None,
        delta: str,
    ) -> bool:
        """锁内接受 thinking delta：块身份 → bounded chunks 入批 → 提交状态。"""

        # 1. 创建/选择 thinking 块身份；锁内先不提交 ordinal 递增。
        if state.thinking_block is None:
            block_id = f"thinking:{turn_id}:{state.next_ordinal}"
            ordinal = state.next_ordinal
        else:
            block_id, ordinal = state.thinking_block
        # 2. bounded chunks 全部入批（持锁期间 terminal 不可能插入）。
        flush_now = False
        for chunk in _utf8_chunks(delta, _DELTA_FLUSH_BYTES):
            if self._accept_segment_locked(
                session_id=session_id,
                turn_id=turn_id,
                event_type="react.thinking.delta",
                delta=chunk,
                block_id=block_id,
                ordinal=ordinal,
            ):
                flush_now = True
        # 3. 全部接受成功后才提交完整 state mutation。
        if state.thinking_block is None:
            state.next_ordinal += 1
            state.thinking_block = (block_id, ordinal)
        # 4. 时间链：首个 thinking delta 到达（每轮只打一次）。
        if not state.first_thinking_received:
            state.first_thinking_received = True
            turn_milestone(
                logger,
                "tl:delta.first_thinking_received",
                session_id=session_id,
                turn_id=turn_id,
                client_message_id=state.client_message_id,
                duration_ms=(
                    (monotonic() - started_at) * 1_000
                    if started_at is not None
                    else None
                ),
            )
        return flush_now

    def _accept_answer_delta_locked(
        self,
        session_id: str,
        turn_id: str,
        state: _ProcessTurnState,
        started_at: float | None,
        delta: str,
    ) -> bool:
        """锁内接受 answer delta：bounded chunks 全部入批后再追加正文。"""

        flush_now = False
        for chunk in _utf8_chunks(delta, _DELTA_FLUSH_BYTES):
            if self._accept_segment_locked(
                session_id=session_id,
                turn_id=turn_id,
                event_type="answer.delta",
                delta=chunk,
                block_id=None,
                ordinal=None,
            ):
                flush_now = True
        # 全部接受成功后才提交完整 state mutation：state 有正文则 wire 必有。
        state.answer_segments.append(delta)
        if not state.first_answer_received:
            state.first_answer_received = True
            turn_milestone(
                logger,
                "tl:delta.first_answer_received",
                session_id=session_id,
                turn_id=turn_id,
                client_message_id=state.client_message_id,
                duration_ms=(
                    (monotonic() - started_at) * 1_000
                    if started_at is not None
                    else None
                ),
            )
        return flush_now

    def _log_delta_dropped(
        self,
        session_id: str,
        turn_id: str,
        event: StreamDeltaReady,
    ) -> None:
        """整事件原子丢弃：终态已收口后绝不让任何分片滑入批或 state。"""

        if event.thinking_delta:
            self._log_late_event_dropped(session_id, turn_id, "react.thinking.delta")
        if event.content_delta:
            self._log_late_event_dropped(session_id, turn_id, "answer.delta")

    async def _on_tool_call_started(self, event: ToolCallStarted) -> None:
        self._raise_delta_failure()
        if event.channel != self.name:
            return
        session_id = event.session_key
        turn_id = event.turn_id or self._current_turn_id(session_id)
        # 0. 终态已收口的迟到事件先读 closed 直接丢弃，不触碰任何 per-turn 结构。
        if (session_id, turn_id) in self._turn_terminals:
            self._log_late_event_dropped(session_id, turn_id, "react.tool.started")
            return
        # 1. 与 terminal 同一 owner 锁内串行：flush 已接受 delta → mutate →
        #    publish；terminal 之后排队的 tool 事件拿锁后见 closed 被丢弃。
        async with self._delta_locked(session_id, turn_id, require_state=True) as lock:
            if lock is None:
                self._log_late_event_dropped(session_id, turn_id, "react.tool.started")
                return
            _ = await self._flush_batch_locked(session_id, turn_id)
            state = self._require_process_state(session_id, turn_id)
            state.thinking_block = None
            if event.call_id in state.tool_blocks:
                raise RuntimeError(f"mobile tool call_id 重复开始: {event.call_id}")
            ordinal = state.next_ordinal
            state.next_ordinal += 1
            block_id = f"tool:{event.call_id}"
            state.tool_blocks[event.call_id] = (block_id, ordinal, monotonic())
            await self._runtime.publish_event(
                event_type="react.tool.started",
                session_id=session_id,
                turn_id=turn_id,
                payload={
                    "call_id": event.call_id,
                    "block_id": block_id,
                    "ordinal": ordinal,
                    "tool_name": event.tool_name,
                    "arguments": _mobile_tool_arguments(event.arguments),
                    "control_turn_id": state.control_turn_id,
                },
            )

    async def _on_tool_call_completed(self, event: ToolCallCompleted) -> None:
        self._raise_delta_failure()
        if event.channel != self.name:
            return
        session_id = event.session_key
        turn_id = event.turn_id or self._current_turn_id(session_id)
        if (session_id, turn_id) in self._turn_terminals:
            self._log_late_event_dropped(session_id, turn_id, "react.tool.completed")
            return
        async with self._delta_locked(session_id, turn_id, require_state=True) as lock:
            if lock is None:
                self._log_late_event_dropped(
                    session_id, turn_id, "react.tool.completed"
                )
                return
            _ = await self._flush_batch_locked(session_id, turn_id)
            state = self._require_process_state(session_id, turn_id)
            block = state.tool_blocks.get(event.call_id)
            if block is None:
                raise RuntimeError(
                    f"mobile tool completed 缺少 started: {event.call_id}"
                )
            block_id, ordinal, started_at = block
            await self._runtime.publish_event(
                event_type="react.tool.completed",
                session_id=session_id,
                turn_id=turn_id,
                payload={
                    "call_id": event.call_id,
                    "block_id": block_id,
                    "ordinal": ordinal,
                    "tool_name": event.tool_name,
                    "status": event.status,
                    "arguments": _mobile_tool_arguments(
                        event.final_arguments,
                    ),
                    "result_preview": event.result_preview,
                    "duration_ms": max(0, round((monotonic() - started_at) * 1_000)),
                    "control_turn_id": state.control_turn_id,
                },
            )

    async def _on_output_completed(self, event: TurnOutputCompleted) -> None:
        self._raise_delta_failure()
        if event.channel != self.name:
            return
        session_id = event.session_key
        turn_id = event.turn_id or self._current_turn_id(session_id)
        # 终态已收口则丢弃迟到信号，绝不重建 per-turn 结构。
        if (session_id, turn_id) in self._turn_terminals:
            self._log_late_event_dropped(session_id, turn_id, "turn.output.completed")
            return
        # 与 terminal 同一 owner 锁内 flush + durable publish，保证 output.completed
        # 要么先于 terminal 发布，要么在 terminal 已收口后被丢弃，绝不排在 terminal 之后。
        async with self._delta_locked(session_id, turn_id, require_state=True) as lock:
            if lock is None:
                self._log_late_event_dropped(
                    session_id, turn_id, "turn.output.completed"
                )
                return
            _ = await self._flush_batch_locked(session_id, turn_id)
            await self._runtime.publish_event(
                event_type="turn.output.completed",
                session_id=session_id,
                turn_id=turn_id,
                payload={"client_message_id": event.client_message_id},
                required_capability=TURN_OUTPUT_COMPLETED_CAPABILITY,
            )

    async def _on_response(self, message: OutboundMessage) -> None:
        outbound = channel_message_from_outbound(message)
        outbound.metadata["_channel_commit_role"] = "passive"
        receipt = await self._deliver_message(outbound)
        if not receipt.succeeded:
            raise RuntimeError(receipt.detail or "Mobile 被动消息提交失败")

    async def _deliver_passive_message(
        self,
        message: ChannelMessage,
    ) -> DeliveryReceipt:
        """提交已持久化 Turn 的 Mobile final 投影。"""

        self._raise_delta_failure()
        session_id = self._session_id(message.chat_id)
        raw_attempt_id = message.execution_attempt_id
        if raw_attempt_id is not None and (
            not isinstance(raw_attempt_id, str) or not raw_attempt_id
        ):
            raise RuntimeError("mobile final execution attempt id 无效")
        turn_id = (
            cast(str | None, raw_attempt_id)
            or message.control_turn_id
            or self._current_turn_id(session_id)
        )
        key = (session_id, turn_id)
        message_id = message.session_message_id
        media = [attachment.source for attachment in message.attachments]
        if media and message_id is None:
            raise RuntimeError("出站媒体缺少已持久化的 assistant 消息")
        source_metadata = dict(message.metadata)
        client_message_id = source_metadata.get("client_message_id")
        if client_message_id is not None and (
            not isinstance(client_message_id, str) or not client_message_id
        ):
            raise RuntimeError("mobile final 缺少完整 client 消息标识")
        if message.terminal_status in (
            TurnTerminalStatus.INTERRUPTED,
            TurnTerminalStatus.CANCELLED,
        ):
            # 1. 中断/取消使用权威 typed terminal；与 /stop 已发布终态共用墓碑幂等。
            if message.control_turn_id is None:
                raise RuntimeError("mobile interrupted terminal 缺少权威 turn_id")
            payload: dict[str, object] = {
                "status": message.terminal_status.value,
                "message": message.content or "本轮已中断。",
                "control_turn_id": message.control_turn_id,
            }
            if client_message_id is not None:
                payload["client_message_id"] = client_message_id
            started_at = self._turn_started_at.get(key)
            published = await self._publish_terminal(
                session_id=session_id,
                turn_id=turn_id,
                event_type="turn.interrupted",
                payload=payload,
            )
            if published:
                turn_milestone(
                    logger,
                    "tl:final.published",
                    session_id=session_id,
                    turn_id=turn_id,
                    client_message_id=cast(str, client_message_id or ""),
                    duration_ms=(
                        (monotonic() - started_at) * 1_000
                        if started_at is not None
                        else None
                    ),
                    counts=f"terminal={message.terminal_status.value}",
                )
            return DeliveryReceipt(DeliveryStatus.SUCCESS)
        metadata = {
            key: source_metadata[key]
            for key in ("mobile_attention",)
            if key in source_metadata
        }
        try:
            attachments = await self._outbound_descriptors(
                session_id,
                media,
                message_id=message_id,
            )
        except (
            AttachmentRequestError,
            AttachmentStateError,
            RemoteMediaError,
            OSError,
        ) as error:
            logger.warning(
                "mobile 远程媒体快照失败，保留最终文字: session=%s error=%s",
                session_id,
                error,
            )
            attachments = []
            metadata["media_delivery"] = {
                "status": "failed",
                "code": "media_unavailable",
                "message": "附件源暂时不可用",
            }
        final_payload: dict[str, object] = {
            "thinking": message.thinking or "",
            "attachments": attachments,
            "metadata": metadata,
            "control_turn_id": message.control_turn_id or turn_id,
        }
        user_message_id = source_metadata.get("persisted_user_message_id")
        # client_message_id 可单独存在（failed/中断终态）；persisted_user_message_id
        # 若存在仍要求是非空字符串。逐项校验，绝不因单项缺失而整体失败或猜测。
        if user_message_id is not None and (
            not isinstance(user_message_id, str) or not user_message_id
        ):
            raise RuntimeError("mobile final 缺少完整 user 消息标识")
        if user_message_id is not None:
            final_payload["user_message_id"] = user_message_id
        if client_message_id is not None:
            final_payload["client_message_id"] = client_message_id
        if message_id is not None:
            final_payload["message_id"] = message_id
        # 1. 与 terminal 同一 owner 锁内完成 flush → suffix → 收口，杜绝多段锁
        #    之间的迟到 delta 插入造成 suffix 重复或正文丢失。
        async with self._delta_locked(session_id, turn_id) as lock:
            if lock is None:
                # 2. 其他 owner 已先收口：不重复发布，完整正文由 durable history 恢复。
                return DeliveryReceipt(
                    DeliveryStatus.SUCCESS,
                    canonical_media=tuple(media),
                )
            _ = await self._flush_batch_locked(session_id, turn_id)
            state = self._process_turns.get(key)
            emitted_content = (
                ""
                if state is None
                else "".join(state.answer_segments) + state.final_suffix_emitted
            )
            # 3. 缺失正文锁内入批并记入已发布 suffix；publish 失败重试只补缺失部分，
            #    绝不重复已 flush 的 delta。
            _, final_content = self._accept_final_suffix_locked(
                session_id,
                turn_id,
                message_content=message.content,
                emitted_content=emitted_content,
                control_turn_id=cast(str, final_payload["control_turn_id"]),
            )
            final_payload["content"] = final_content
            started_at = self._turn_started_at.get(key)
            published = await self._close_terminal_locked(
                session_id,
                turn_id,
                event_type="message.final",
                payload=final_payload,
            )
            if not published:
                # 4. 其他 owner 已先收口；终态不重复发布。
                return DeliveryReceipt(
                    DeliveryStatus.SUCCESS,
                    canonical_media=tuple(media),
                )
        # 5. 时间链：message.final 进入服务端发布路径（权威终态已落 SessionDB）
        self._clear_turn_maps(session_id, turn_id)
        # 进程内 state 缺失（恢复态）时用已验证 outbound client_message_id 贯通，
        # 不用 current turn 猜；state 存在时仍以 turn.started 接受的同源 id 为准。
        if state is not None and state.client_message_id:
            trace_client_message_id = state.client_message_id
        else:
            trace_client_message_id = cast(
                str, source_metadata.get("client_message_id") or ""
            )
        turn_milestone(
            logger,
            "tl:final.published",
            session_id=session_id,
            turn_id=turn_id,
            client_message_id=trace_client_message_id,
            duration_ms=(
                (monotonic() - started_at) * 1_000 if started_at is not None else None
            ),
            counts="terminal=completed",
        )
        return DeliveryReceipt(
            DeliveryStatus.SUCCESS,
            canonical_media=tuple(media),
        )

    async def _mobile_history_item(
        self,
        item: Mapping[str, object],
    ) -> dict[str, object]:
        """裁剪内部字段，并把服务端媒体路径转换为稳定描述符。"""

        result = _mobile_history_item(item)
        media = item.get("media")
        if media is None:
            result["attachments"] = []
            return result
        if not isinstance(media, list) or not all(
            isinstance(path, str) for path in cast(list[object], media)
        ):
            raise ValueError(f"历史消息 media 不是字符串数组: {item['id']}")
        try:
            result["attachments"] = await self._outbound_descriptors(
                str(item["session_key"]),
                cast(list[str], media),
                message_id=str(item["id"]),
            )
        except (
            AttachmentRequestError,
            AttachmentStateError,
            RemoteMediaError,
            OSError,
        ) as error:
            logger.warning(
                "mobile 历史媒体恢复失败，保留文字历史: message=%s error=%s",
                item["id"],
                error,
            )
            result["attachments"] = []
            result["attachment_error"] = {
                "code": "media_unavailable",
                "message": "附件源暂时不可用",
            }
        return result

    async def _outbound_descriptors(
        self,
        session_id: str,
        media_paths: list[str],
        *,
        message_id: str | None = None,
    ) -> list[dict[str, object]]:
        """把本地媒体物化为不暴露路径的稳定附件描述符。"""

        if len(media_paths) > 10:
            raise ValueError("单条消息最多包含 10 个出站附件")
        if not media_paths:
            return []
        if message_id is not None:
            bound = await asyncio.to_thread(
                self._require_attachments().read_message_outbound,
                session_id=session_id,
                message_id=message_id,
            )
            if bound:
                if len(bound) != len(media_paths):
                    raise RuntimeError("历史消息附件槽位数量与 Session 不一致")
                return [attachment_descriptor(record) for record in bound]
        # 1. URL 先经过 SSRF 防护下载为受限持久快照
        snapshots: list[RemoteMediaSnapshot] = []
        paths: list[str] = []
        metadata: list[tuple[str, str] | None] = []
        try:
            for media_path in media_paths:
                if media_path.startswith(("http://", "https://")):
                    snapshot = await snapshot_remote_media(
                        media_path,
                        self._require_ctx().attachment_store,
                        max_bytes=(
                            self._runtime.config.max_attachment_mb * 1024 * 1024
                        ),
                    )
                    snapshots.append(snapshot)
                    paths.append(str(snapshot.path))
                    metadata.append((snapshot.filename, snapshot.content_type))
                else:
                    paths.append(media_path)
                    metadata.append(None)

            # 2. 文件复制和摘要移出事件循环，整批在单事务中注册
            records = await asyncio.to_thread(
                self._require_attachments().register_outbound_batch,
                session_id=session_id,
                local_media_paths=tuple(paths),
                metadata_overrides=tuple(metadata),
                message_id=message_id,
            )
            return [attachment_descriptor(record) for record in records]
        finally:
            for snapshot in snapshots:
                snapshot.path.unlink(missing_ok=True)

    @asynccontextmanager
    async def _delta_locked(
        self,
        session_id: str,
        turn_id: str,
        *,
        may_create: bool = True,
        require_state: bool = False,
    ) -> AsyncGenerator[asyncio.Lock | None]:
        """per-turn 串行临界区；终态已收口 yield None，绝不重建 per-turn 结构。"""

        key = (session_id, turn_id)
        lock = self._delta_locks.get(key)
        if lock is None:
            # 1. 先读 closed：墓碑在而锁已清理的 turn 绝不重建。
            if key in self._turn_terminals:
                yield None
                return
            if not may_create:
                yield None
                return
            if require_state and key not in self._process_turns:
                raise RuntimeError(
                    f"mobile process turn 未开始: {session_id}/{turn_id}"
                )
            lock = asyncio.Lock()
            self._delta_locks[key] = lock
        async with lock:
            # 2. 排队期间 terminal 可能已收口：拿锁后必须复核 closed。
            if key in self._turn_terminals:
                yield None
                return
            yield lock

    async def _buffer_delta(
        self,
        *,
        session_id: str,
        turn_id: str,
        event_type: str,
        delta: str,
        block_id: str | None,
        ordinal: int | None,
    ) -> bool:
        """把连续 delta 聚合成 16ms/4KiB 批；已收口返回 False 且不重建任何结构。"""

        flush_now = False
        async with self._delta_locked(session_id, turn_id) as lock:
            if lock is None:
                self._log_late_event_dropped(session_id, turn_id, event_type)
                return False
            flush_now = self._accept_segment_locked(
                session_id=session_id,
                turn_id=turn_id,
                event_type=event_type,
                delta=delta,
                block_id=block_id,
                ordinal=ordinal,
            )
        if flush_now:
            _ = await self._flush_deltas(session_id, turn_id)
        return True

    def _accept_segment_locked(
        self,
        *,
        session_id: str,
        turn_id: str,
        event_type: str,
        delta: str,
        block_id: str | None,
        ordinal: int | None,
        merge: bool = True,
    ) -> bool:
        """锁内把一段 delta 聚合进批并返回是否需立即 flush；调用方必须持锁。"""

        key = (session_id, turn_id)
        batch = self._delta_batches.get(key)
        if batch is None:
            # 1. 有界批：新建批与 16ms 定时器，绝不越过 256 活跃 turn 上限。
            if len(self._delta_batches) >= _MAX_DELTA_BATCHES:
                raise RuntimeError("mobile delta batch 已达到 256 个活跃 turn 上限")
            timer = asyncio.create_task(
                self._flush_after_interval(key),
                name=f"mobile-delta-flush:{turn_id}",
            )
            timer.add_done_callback(self._on_delta_timer_done)
            batch = _DeltaBatch(segments=[], byte_count=0, timer=timer)
            self._delta_batches[key] = batch
        # 2. 连续同身份段原地合并，保持原始顺序；merge=False 时每段独立，
        #    保证 bounded 分片逐条发布（final suffix 路径）。
        if (
            merge
            and batch.segments
            and (
                batch.segments[-1][0],
                batch.segments[-1][2],
                batch.segments[-1][3],
            )
            == (event_type, block_id, ordinal)
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
        return batch.byte_count >= _DELTA_FLUSH_BYTES

    async def _buffer_bounded_delta(
        self,
        *,
        session_id: str,
        turn_id: str,
        event_type: str,
        delta: str,
        block_id: str | None,
        ordinal: int | None,
    ) -> bool:
        """按 UTF-8 字节边界分片入批；任一分片被收口拒绝即停止。"""

        for chunk in _utf8_chunks(delta, _DELTA_FLUSH_BYTES):
            if not await self._buffer_delta(
                session_id=session_id,
                turn_id=turn_id,
                event_type=event_type,
                delta=chunk,
                block_id=block_id,
                ordinal=ordinal,
            ):
                return False
        return True

    async def _flush_after_interval(self, key: tuple[str, str]) -> None:
        await asyncio.sleep(_DELTA_FLUSH_INTERVAL_SECONDS)
        _ = await self._flush_deltas(*key)

    async def _flush_deltas(self, session_id: str, turn_id: str) -> bool:
        """按原始顺序发布已聚合 delta；已收口或无批返回 False，绝不重建锁。"""

        async with self._delta_locked(session_id, turn_id, may_create=False) as lock:
            if lock is None:
                return False
            return await self._flush_batch_locked(session_id, turn_id)

    async def _flush_batch_locked(self, session_id: str, turn_id: str) -> bool:
        """锁内逐段发布当前批；每段成功后才消费，失败保留失败段及后续段。"""

        key = (session_id, turn_id)
        batch = self._delta_batches.get(key)
        if batch is None:
            return False
        published_any = False
        while batch.segments:
            event_type, delta, block_id, ordinal = batch.segments[0]
            state = self._require_process_state(session_id, turn_id)
            payload: dict[str, object] = {
                "delta": delta,
                "control_turn_id": state.control_turn_id,
            }
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
            # 1. publish 确认成功后才消费该段，并精确扣减 UTF-8 byte_count；
            #    失败段与后续段原样留在批里，成功段不回卷，重试不丢不重。
            _ = batch.segments.pop(0)
            batch.byte_count -= len(delta.encode("utf-8"))
            published_any = True
        # 2. 全部段发布成功后：从 map 移除批，并取消非当前任务的 timer。
        _ = self._delta_batches.pop(key, None)
        current = asyncio.current_task()
        if batch.timer is not current:
            _ = batch.timer.cancel()
        return published_any

    def _accept_final_suffix_locked(
        self,
        session_id: str,
        turn_id: str,
        *,
        message_content: str,
        emitted_content: str,
        control_turn_id: str,
    ) -> tuple[str, str]:
        """锁内把 final 缺失正文入批并记账；返回 (suffix, final_content)。"""

        suffix = ""
        final_content = message_content
        if emitted_content and message_content.startswith(emitted_content):
            suffix = message_content[len(emitted_content) :]
            final_content = ""
        elif (
            not emitted_content
            and len(message_content.encode("utf-8")) > _DELTA_FLUSH_BYTES
        ):
            suffix = message_content
            final_content = ""
        if not suffix:
            return suffix, final_content
        key = (session_id, turn_id)
        state = self._process_turns.get(key)
        if state is None:
            # 恢复态（本进程无流式 state）也记账，保证失败重试不重复发布。
            state = _ProcessTurnState(
                next_ordinal=0,
                thinking_block=None,
                tool_blocks={},
                answer_segments=[],
                control_turn_id=control_turn_id,
                client_message_id="",
            )
            self._process_turns[key] = state
        state.final_suffix_emitted += suffix
        for chunk in _utf8_chunks(suffix, _DELTA_FLUSH_BYTES):
            _ = self._accept_segment_locked(
                session_id=session_id,
                turn_id=turn_id,
                event_type="answer.delta",
                delta=chunk,
                block_id=None,
                ordinal=None,
                merge=False,
            )
        return suffix, final_content

    async def _publish_terminal(
        self,
        *,
        session_id: str,
        turn_id: str,
        event_type: str,
        payload: dict[str, object],
        device_id: str | None = None,
    ) -> bool:
        """每 turn 唯一 owner 的终态收口：flush → publish → 成功后 closed → cleanup。"""

        key = (session_id, turn_id)
        async with self._delta_locked(session_id, turn_id, require_state=False) as lock:
            if lock is None:
                # 1. 其他 owner 已收口：不重复发布，也不重建任何 per-turn 结构。
                state = self._process_turns.get(key)
                turn_milestone(
                    logger,
                    "tl:terminal.dropped",
                    session_id=session_id,
                    turn_id=turn_id,
                    client_message_id=(
                        state.client_message_id if state is not None else ""
                    ),
                    outcome="already_closed",
                    counts=f"event_type={event_type}",
                )
                return False
            published = await self._close_terminal_locked(
                session_id,
                turn_id,
                event_type=event_type,
                payload=payload,
                device_id=device_id,
            )
        if published:
            self._clear_turn_maps(session_id, turn_id)
        return published

    async def _close_terminal_locked(
        self,
        session_id: str,
        turn_id: str,
        *,
        event_type: str,
        payload: dict[str, object],
        device_id: str | None = None,
    ) -> bool:
        """锁内收口：flush 已接受 delta → durable publish → 成功后才提交墓碑。"""

        key = (session_id, turn_id)
        # 1. 锁内复核：其他 owner 已成功收口则不重复发布。
        if key in self._turn_terminals:
            return False
        # 2. 按 wire 顺序先 flush 已接受 delta（含 final suffix）；已发布不回卷
        #    不重复，失败重试时对应 batch 为空，process state 保留供 suffix 计算。
        _ = await self._flush_batch_locked(session_id, turn_id)
        # 3. durable 终态发布：await 成功返回前没有任何已提交 closed 墓碑。
        if device_id is None:
            await self._runtime.publish_event(
                event_type=event_type,
                session_id=session_id,
                turn_id=turn_id,
                payload=payload,
            )
        else:
            await self._runtime.publish_event(
                event_type=event_type,
                session_id=session_id,
                turn_id=turn_id,
                payload=payload,
                device_id=device_id,
            )
        # 4. publish 确认成功后才在同一锁内提交终态墓碑（有界 256）。
        if len(self._turn_terminals) >= _MAX_DELTA_BATCHES:
            _ = self._turn_terminals.pop(next(iter(self._turn_terminals)))
        self._turn_terminals[key] = event_type
        return True

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

    def _interrupt_payload(
        self,
        session_id: str,
        turn_id: str,
        *,
        status: str,
        message: str,
        reason: str | None = None,
    ) -> dict[str, object]:
        """构造 turn.interrupted 载荷；进程内已知 client_message_id 必须贯通。"""

        turn = self._require_ctx().session_manager.control_store.read_turn(turn_id)
        control_turn_id: object = (
            turn.metadata.get("interactionId", turn.id) if turn is not None else turn_id
        )
        if not isinstance(control_turn_id, str) or not control_turn_id:
            raise RuntimeError("mobile interrupted logical turn id 无效")
        payload: dict[str, object] = {
            "status": status,
            "message": message,
            "control_turn_id": control_turn_id,
        }
        if reason is not None:
            payload["reason"] = reason
        state = self._process_turns.get((session_id, turn_id))
        if state is not None and state.client_message_id:
            payload["client_message_id"] = state.client_message_id
        return payload

    def _mark_first_deltas_published(
        self,
        session_id: str,
        turn_id: str,
    ) -> None:
        """flush 真实发布成功后打 published 里程碑；terminal 收口后不得调用。"""

        state = self._process_turns.get((session_id, turn_id))
        if state is None:
            return
        started_at = self._turn_started_at.get((session_id, turn_id))
        if state.first_thinking_received and not state.first_thinking_published:
            state.first_thinking_published = True
            turn_milestone(
                logger,
                "tl:delta.first_thinking_published",
                session_id=session_id,
                turn_id=turn_id,
                client_message_id=state.client_message_id,
                duration_ms=(
                    (monotonic() - started_at) * 1_000
                    if started_at is not None
                    else None
                ),
            )
        if state.first_answer_received and not state.first_answer_published:
            state.first_answer_published = True
            turn_milestone(
                logger,
                "tl:delta.first_answer_published",
                session_id=session_id,
                turn_id=turn_id,
                client_message_id=state.client_message_id,
                duration_ms=(
                    (monotonic() - started_at) * 1_000
                    if started_at is not None
                    else None
                ),
            )

    def _log_late_event_dropped(
        self,
        session_id: str,
        turn_id: str,
        event_type: str,
    ) -> None:
        """终态已收口后到达的迟到事件：结构化记录后丢弃，绝不重建 batch/timer。"""

        state = self._process_turns.get((session_id, turn_id))
        turn_milestone(
            logger,
            "tl:turn.late.drop",
            session_id=session_id,
            turn_id=turn_id,
            client_message_id=state.client_message_id if state is not None else "",
            outcome="terminal_closed",
            counts=f"event_type={event_type}",
        )

    def _clear_turn_maps(self, session_id: str, turn_id: str) -> None:
        """终态后清理本 turn 状态；active 与计时起点只删仍指向本 turn 的。"""

        # 1. 先取出本 turn 绑定的身份，再移除 process 状态本身。
        state = self._process_turns.pop((session_id, turn_id), None)
        client_message_id = state.client_message_id if state is not None else ""
        # 2. 只有 active 仍指向本 turn 时才 compare-delete，旧 A 绝不清新 B。
        if self._active_turn_ids.get(session_id) == turn_id:
            _ = self._active_turn_ids.pop(session_id, None)
        # 3. 兜底清掉残留 delta 批并取消定时器，禁止终态后任何迟到发布。
        batch = self._delta_batches.pop((session_id, turn_id), None)
        if batch is not None:
            _ = batch.timer.cancel()
        _ = self._delta_locks.pop((session_id, turn_id), None)
        _ = self._turn_started_at.pop((session_id, turn_id), None)
        # 4. 只删本 turn 的 send 计时起点，绝不连带同 session 排队消息。
        if client_message_id:
            _ = self._send_received_at.pop((session_id, client_message_id), None)

    def _require_ctx(self) -> ChannelContext:
        if self._ctx is None:
            raise RuntimeError("MobileRealtimeChannel 尚未启动")
        return self._ctx

    def _require_attachments(self) -> AttachmentTransferService:
        if self._attachments is None:
            raise RuntimeError("MobileRealtimeChannel 附件服务尚未启动")
        return self._attachments


def _command_hash(frame: ClientCommand) -> str:
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
                "message": "该命令上次执行时中断，请使用原命令 ID 核对状态",
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


def _history_query_payload(payload: Mapping[str, object]) -> dict[str, int | None]:
    """解析旧分页或 v1 正文引用游标，两种模式不能混用。"""

    # 1. 旧客户端继续使用 page；新客户端显式声明正文引用版本
    _expect_keys(
        payload,
        {
            "page",
            "page_size",
            "content_ref_version",
            "after_seq",
            "snapshot_max_seq",
        },
    )
    version = payload.get("content_ref_version", 0)
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version not in {0, 1}
    ):
        raise MobileCommandError(
            "unsupported_content_ref_version",
            "content_ref_version 只支持 1",
        )
    page_size = payload.get("page_size", 50)
    if (
        not isinstance(page_size, int)
        or isinstance(page_size, bool)
        or not 1 <= page_size <= 200
    ):
        raise MobileCommandError("invalid_pagination", "page_size 必须在 1..200")
    if version == 0:
        if "after_seq" in payload or "snapshot_max_seq" in payload:
            raise MobileCommandError("invalid_pagination", "旧分页不能携带 seq 游标")
        page = payload.get("page", 1)
        if not isinstance(page, int) or isinstance(page, bool) or page < 1:
            raise MobileCommandError("invalid_pagination", "page 必须是正整数")
        return {
            "page": page,
            "page_size": page_size,
            "content_ref_version": 0,
            "after_seq": None,
            "snapshot_max_seq": None,
        }

    # 2. v1 以 seq 游标恢复；-1 表示尚未消费任何消息
    if "page" in payload:
        raise MobileCommandError("invalid_pagination", "正文引用分页不能携带 page")
    after_seq = payload.get("after_seq", -1)
    if not isinstance(after_seq, int) or isinstance(after_seq, bool) or after_seq < -1:
        raise MobileCommandError("invalid_pagination", "after_seq 必须大于等于 -1")
    snapshot_max_seq = payload.get("snapshot_max_seq")
    if snapshot_max_seq is not None and (
        not isinstance(snapshot_max_seq, int)
        or isinstance(snapshot_max_seq, bool)
        or snapshot_max_seq < -1
    ):
        raise MobileCommandError(
            "invalid_snapshot",
            "snapshot_max_seq 必须大于等于 -1",
        )
    if snapshot_max_seq is not None and after_seq > snapshot_max_seq:
        raise MobileCommandError("invalid_snapshot", "after_seq 超过历史快照高水位")
    return {
        "page": None,
        "page_size": page_size,
        "content_ref_version": 1,
        "after_seq": after_seq,
        "snapshot_max_seq": snapshot_max_seq,
    }


def _expect_keys(payload: Mapping[str, object], allowed: set[str]) -> None:
    unexpected = set(payload) - allowed
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise MobileCommandError("invalid_payload", f"payload 包含未知字段: {names}")


def _expect_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise MobileCommandError(
            "invalid_payload",
            f"{field} 必须是长度 1..512 的字符串",
        )
    return value


def _expect_nonnegative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise MobileCommandError("invalid_payload", f"{field} 必须是非负整数")
    return value


def _expect_sha256(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise MobileCommandError("invalid_payload", f"{field} 必须是 SHA-256")
    digest = value.lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise MobileCommandError("invalid_payload", f"{field} 必须是 SHA-256")
    return digest


def _validate_reply_frame_size(frame: ClientCommand, reply: CommandReply) -> None:
    """在持久化前按真实 JSON 编码校验回复可投递。"""

    wire: dict[str, object] = {
        "v": 1,
        "kind": "reply",
        "type": reply.type,
        "id": frame.id,
        "connection_epoch": frame.connection_epoch,
        "payload": reply.payload,
    }
    if reply.session_id is not None:
        wire["session_id"] = reply.session_id
    if reply.turn_id is not None:
        wire["turn_id"] = reply.turn_id
    encoded = json.dumps(
        wire,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > MAX_JSON_FRAME_BYTES:
        raise RuntimeError(
            f"mobile reply 超过 {MAX_JSON_FRAME_BYTES} bytes: {reply.type}"
        )


def _mobile_history_item(item: Mapping[str, object]) -> dict[str, object]:
    """裁剪服务端内部字段，只向手机同步可展示历史。"""

    mobile_extra: dict[str, object] = {}
    for field in ("reasoning_content", "turn_duration_ms", "proactive", "delivery_id"):
        value = item.get(field)
        if isinstance(value, (str, int, float, bool)):
            mobile_extra[field] = value

    result: dict[str, object] = {
        "id": str(item["id"]),
        "session_key": str(item["session_key"]),
        "seq": cast(int, item["seq"]),
        "role": str(item["role"]),
        "content": str(item["content"]),
        "tool_chain": _mobile_tool_chain(item.get("tool_chain")),
        "extra": mobile_extra,
        "ts": str(item["timestamp"]),
    }
    client_message_id = item.get("client_message_id")
    if isinstance(client_message_id, str) and client_message_id:
        result["client_message_id"] = client_message_id
    control_turn_id = item.get("control_turn_id")
    if (
        result["role"] == "assistant"
        and isinstance(control_turn_id, str)
        and control_turn_id
    ):
        mobile_extra["control_turn_id"] = control_turn_id
    for field in ("reply_to_message_id", "reply_role", "reply_preview"):
        value = item.get(field)
        if isinstance(value, str) and value:
            result[field] = value
    return result


def _reply_preview(content: str) -> str:
    return " ".join(content.split())[:512] or "[无文字消息]"


def _reply_source_text(target: Mapping[str, object]) -> str:
    content = str(target["content"])
    if content.strip():
        return content
    media = target.get("media")
    if isinstance(media, list) and media:
        return "[附件]"
    return "[无文字消息]"


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
                arguments = call_record.get(
                    "final_arguments", call_record.get("arguments")
                )
                arguments_record = (
                    cast(dict[str, object], arguments)
                    if isinstance(arguments, dict)
                    else None
                )
                call: dict[str, object] = {
                    "call_id": str(call_record.get("call_id") or ""),
                    "name": name,
                    "status": str(call_record.get("status") or "success"),
                }
                if arguments_record is not None:
                    projected_arguments = _mobile_tool_arguments(
                        arguments_record,
                        max_bytes=_MOBILE_HISTORY_TOOL_ARGUMENT_MAX_BYTES,
                    )
                    call["arguments"] = projected_arguments
                    description = projected_arguments.get("description")
                else:
                    description = None
                if isinstance(description, str) and description:
                    call["description"] = description
                result = call_record.get("result")
                if result is not None:
                    call["result_preview"] = str(result)[:2000]
                calls.append(call)
        group["calls"] = calls
        groups.append(group)
    return groups


def _fit_mobile_history_payload(
    payload: dict[str, object],
    *,
    allow_content_refs: bool = False,
) -> None:
    """在历史事件接近帧上限时回收工具预算并外置长正文。"""

    # 1. 正常页面直接保留全部安全参数
    if _mobile_tool_argument_encoded_size(payload) <= _MOBILE_HISTORY_PAYLOAD_MAX_BYTES:
        return

    # 2. v1 优先外置最大的正文，保留 thinking/tool 展示语义
    items = cast(list[dict[str, object]], payload["items"])
    if allow_content_refs:
        content_items = sorted(
            items,
            key=lambda item: len(str(item["content"]).encode("utf-8")),
            reverse=True,
        )
        for item in content_items:
            content = str(item["content"])
            encoded = content.encode("utf-8")
            if not encoded:
                continue
            item["content"] = None
            item["content_ref"] = {
                "version": 1,
                "encoding": "utf-8",
                "byte_length": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "preview": content[:512],
            }
            if (
                _mobile_tool_argument_encoded_size(payload)
                <= _MOBILE_HISTORY_PAYLOAD_MAX_BYTES
            ):
                return

    # 3. 游标协议按真实帧预算缩小页面，后续请求会从新高水位继续
    if allow_content_refs:
        while len(items) > 1:
            _ = items.pop()
            payload["next_after_seq"] = cast(int, items[-1]["seq"])
            payload["has_more"] = True
            if (
                _mobile_tool_argument_encoded_size(payload)
                <= _MOBILE_HISTORY_PAYLOAD_MAX_BYTES
            ):
                return

    # 4. 单条消息仍超限时，回收可重新生成的工具参数与派生描述
    for item in reversed(items):
        chain = cast(list[dict[str, object]] | None, item["tool_chain"])
        if chain is None:
            continue
        for group in reversed(chain):
            calls = cast(list[dict[str, object]], group["calls"])
            for call in reversed(calls):
                if "arguments" not in call:
                    continue
                del call["arguments"]
                if (
                    _mobile_tool_argument_encoded_size(payload)
                    <= _MOBILE_HISTORY_PAYLOAD_MAX_BYTES
                ):
                    return

    for item in reversed(items):
        chain = cast(list[dict[str, object]] | None, item["tool_chain"])
        if chain is None:
            continue
        for group in reversed(chain):
            calls = cast(list[dict[str, object]], group["calls"])
            for call in reversed(calls):
                if "description" not in call:
                    continue
                del call["description"]
                if (
                    _mobile_tool_argument_encoded_size(payload)
                    <= _MOBILE_HISTORY_PAYLOAD_MAX_BYTES
                ):
                    return

    # 5. 继续用显式占位符收缩结果预览，保留工具名称与执行状态
    for item in reversed(items):
        chain = cast(list[dict[str, object]] | None, item["tool_chain"])
        if chain is None:
            continue
        for group in reversed(chain):
            calls = cast(list[dict[str, object]], group["calls"])
            for call in reversed(calls):
                preview = call.get("result_preview")
                if (
                    not isinstance(preview, str)
                    or preview == _MOBILE_HISTORY_DETAIL_OMITTED
                ):
                    continue
                call["result_preview"] = _MOBILE_HISTORY_DETAIL_OMITTED
                if (
                    _mobile_tool_argument_encoded_size(payload)
                    <= _MOBILE_HISTORY_PAYLOAD_MAX_BYTES
                ):
                    return

    # 6. 极长 thinking 也显式标记收缩，不让一条历史拖垮整个同步连接
    for item in reversed(items):
        chain = cast(list[dict[str, object]] | None, item["tool_chain"])
        if chain is None:
            continue
        for group in reversed(chain):
            for field in ("reasoning_content", "text"):
                value = group.get(field)
                if (
                    not isinstance(value, str)
                    or value == _MOBILE_HISTORY_DETAIL_OMITTED
                ):
                    continue
                group[field] = _MOBILE_HISTORY_DETAIL_OMITTED
                if (
                    _mobile_tool_argument_encoded_size(payload)
                    <= _MOBILE_HISTORY_PAYLOAD_MAX_BYTES
                ):
                    return

    # 7. 固定结构仍超限时明确失败，不能发送违规帧或静默删除工具记录
    code = "history_item_too_large" if allow_content_refs else "upgrade_required"
    raise MobileCommandError(code, "历史消息超过当前客户端可恢复的帧预算")


def _mobile_tool_arguments(
    arguments: Mapping[str, object],
    *,
    max_bytes: int = _MOBILE_TOOL_ARGUMENT_MAX_BYTES,
) -> dict[str, object]:
    """生成可安全持久化到手机端的有界工具参数投影。"""

    # 1. 先按结构、字段和值建立安全投影
    remaining = [_MOBILE_TOOL_ARGUMENT_MAX_ITEMS]
    projected = _mobile_tool_argument_value(arguments, depth=0, remaining=remaining)
    projected_record = cast(dict[str, object], projected)

    # 2. 再按真实 UTF-8 JSON 字节预算保留前导参数
    bounded: dict[str, object] = {}
    for key, value in projected_record.items():
        candidate = {**bounded, key: value}
        if _mobile_tool_argument_encoded_size(candidate) <= max_bytes:
            bounded[key] = value
            continue
        while (
            bounded
            and _mobile_tool_argument_encoded_size(
                {**bounded, "…": _MOBILE_TOOL_ARGUMENT_TRUNCATED}
            )
            > max_bytes
        ):
            _ = bounded.popitem()
        bounded["…"] = _MOBILE_TOOL_ARGUMENT_TRUNCATED
        break
    return bounded


def _mobile_tool_argument_value(
    value: object,
    *,
    depth: int,
    remaining: list[int],
) -> object:
    """递归脱敏并裁剪单个 JSON 参数值。"""

    # 1. 在协议边界限制递归深度和总节点数
    if depth > _MOBILE_TOOL_ARGUMENT_MAX_DEPTH or remaining[0] <= 0:
        return _MOBILE_TOOL_ARGUMENT_TRUNCATED
    remaining[0] -= 1

    # 2. 统一隐藏字符串中的凭据，再限制可展示文本长度
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if _mobile_tool_argument_contains_secret(value):
            return _MOBILE_TOOL_ARGUMENT_REDACTED
        if len(value) <= _MOBILE_TOOL_ARGUMENT_MAX_STRING_CHARS:
            return value
        return value[:_MOBILE_TOOL_ARGUMENT_MAX_STRING_CHARS] + "…"

    # 3. 对容器递归投影，在键名所有权层隐藏凭据
    if isinstance(value, Mapping):
        mapping = cast(Mapping[object, object], value)
        result: dict[str, object] = {}
        for index, (key, item) in enumerate(mapping.items()):
            if not isinstance(key, str):
                raise TypeError("mobile 工具参数对象键必须是字符串")
            if index >= _MOBILE_TOOL_ARGUMENT_MAX_CONTAINER_ITEMS or remaining[0] <= 0:
                result["…"] = _MOBILE_TOOL_ARGUMENT_TRUNCATED
                break
            if _mobile_tool_argument_is_secret(key):
                result[key] = _MOBILE_TOOL_ARGUMENT_REDACTED
                continue
            result[key] = _mobile_tool_argument_value(
                item,
                depth=depth + 1,
                remaining=remaining,
            )
        return result
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        result_list: list[object] = []
        redact_next = False
        for index, item in enumerate(sequence):
            if index >= _MOBILE_TOOL_ARGUMENT_MAX_CONTAINER_ITEMS or remaining[0] <= 0:
                result_list.append(_MOBILE_TOOL_ARGUMENT_TRUNCATED)
                break
            if redact_next:
                result_list.append(_MOBILE_TOOL_ARGUMENT_REDACTED)
                redact_next = False
                continue
            result_list.append(
                _mobile_tool_argument_value(
                    item,
                    depth=depth + 1,
                    remaining=remaining,
                )
            )
            redact_next = isinstance(
                item, str
            ) and _mobile_tool_argument_is_secret_flag(item)
        return result_list
    raise TypeError(f"mobile 工具参数包含非 JSON 类型: {type(value).__name__}")


def _mobile_tool_argument_is_secret(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return normalized in _MOBILE_TOOL_SECRET_KEYS or normalized.endswith(
        (
            "secret",
            "password",
            "passwd",
            "cookie",
            "privatekey",
            "secretaccesskey",
            "token",
            "apikey",
            "credentialfile",
            "credentialsfile",
        )
    )


def _mobile_tool_argument_contains_secret(value: str) -> bool:
    return _MOBILE_TOOL_SECRET_TEXT_PATTERN.search(value) is not None


def _mobile_tool_argument_is_secret_flag(value: str) -> bool:
    normalized = value.strip().lstrip("-").lower()
    return _mobile_tool_argument_is_secret(normalized)


def _mobile_tool_argument_encoded_size(value: Mapping[str, object]) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    )


def _mobile_ui_catalog_identity(
    catalog: dict[str, object],
) -> str:
    revision = catalog.get("catalog_revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{64}", revision):
        raise RuntimeError("mobile UI catalog_revision 无效")
    return revision


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = ["CommandReply", "MobileRealtimeChannel"]
