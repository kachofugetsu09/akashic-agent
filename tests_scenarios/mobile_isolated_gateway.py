from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import secrets
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

from bus.events import (
    AttachmentKind,
    ChannelAttachment,
    ChannelMessage,
    InboundMessage,
    TurnTerminalStatus,
)
from bus.events_lifecycle import (
    StreamDeltaReady,
    ToolCallCompleted,
    ToolCallStarted,
    TurnStarted,
)
from agent.config_models import MobileKeyEncryptionConfig, MobileRealtimeConfig
from agent.plugin_composition.channels import (
    ChannelFactoryContext,
    ChannelInboundMessage,
    ChannelRuntimePorts,
    RawInbound,
)
from infra.channels.base import AttachmentStore
from infra.mobile_realtime.gateway import (
    MobileGatewayRuntime,
    build_mobile_gateway_runtime,
    build_mobile_gateway_server,
)
from infra.mobile_realtime.key_protection import KeyProtectionError
from plugins.akasha.plugin import _mobile_recall_lane
from session.manager import SessionManager

_FIXED_GIF = bytes.fromhex(
    "47494638396101000100800000000000ffffff21f90401000000002c00000000010001000002024401003b"
)
logger = logging.getLogger(__name__)
_HISTORY_SESSION_ID = "akashic:00000000000070008000000000000001"
_FAULT_MODES = ("none", "stall_before_challenge", "stall_after_auth")
_PILOT_REPLY_CHUNKS = (
    "## WebUI 试点\n\n",
    "这段内容由隔离 Gateway ",
    "逐段推送，",
    "用来观察网页端式的生长效果。\n\n",
    "- 共享浅蓝主题\n",
    "- 同一消息组件\n",
    "- 原生能力仍独立\n\n",
    "```text\nWeb + Android = one WebUI\n```\n",
)
_PILOT_THINKING_BEFORE_TOOL = (
    "先确认两端是否正在使用同一套主题 token。",
    "再检查流式消息组件和平台能力边界。",
)
_PILOT_THINKING_AFTER_TOOL = (
    "工具结果表明共享主题已经生效。",
    "现在整理最终结论。",
)


@dataclass(frozen=True)
class ReplayToolCall:
    call_id: str
    name: str
    status: str
    arguments: dict[str, Any]
    final_arguments: dict[str, Any]
    result: str


@dataclass(frozen=True)
class ReplayStage:
    text: str
    reasoning: str
    calls: tuple[ReplayToolCall, ...]


@dataclass(frozen=True)
class ReplayTurn:
    content: str
    stages: tuple[ReplayStage, ...]

    @property
    def reasoning(self) -> str:
        return "".join(stage.reasoning for stage in self.stages)

    @property
    def call_count(self) -> int:
        return sum(len(stage.calls) for stage in self.stages)

    def session_tool_chain(self) -> list[dict[str, object]]:
        """生成 SessionDB 使用的普通 JSON 结构。"""

        return [
            {
                "text": stage.text or None,
                "reasoning_content": stage.reasoning,
                "calls": [
                    {
                        "call_id": call.call_id,
                        "name": call.name,
                        "status": call.status,
                        "arguments": call.arguments,
                        "final_arguments": call.final_arguments,
                        "result": call.result,
                    }
                    for call in stage.calls
                ],
            }
            for stage in self.stages
        ]


def load_replay_turn(path: Path) -> ReplayTurn:
    """从只读 Session 导出中加载并校验一条 assistant Turn。"""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("replay-turn 顶层必须是消息数组")
    assistants = [
        message
        for message in payload
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if len(assistants) != 1:
        raise ValueError("replay-turn 必须恰好包含一条 assistant 消息")
    message = assistants[0]
    content = message.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("replay-turn assistant content 必须是非空字符串")
    raw_chain = message.get("tool_chain")
    if isinstance(raw_chain, str):
        raw_chain = json.loads(raw_chain)
    if not isinstance(raw_chain, list):
        raise ValueError("replay-turn tool_chain 必须是数组或 JSON 数组字符串")

    stages: list[ReplayStage] = []
    for stage_index, raw_stage in enumerate(raw_chain, start=1):
        if not isinstance(raw_stage, dict):
            raise ValueError(f"replay-turn stage {stage_index} 必须是对象")
        raw_text = raw_stage.get("text")
        if raw_text is not None and not isinstance(raw_text, str):
            raise ValueError(f"replay-turn stage {stage_index} text 类型无效")
        reasoning = raw_stage.get("reasoning_content", "")
        if not isinstance(reasoning, str):
            raise ValueError(f"replay-turn stage {stage_index} reasoning 类型无效")
        raw_calls = raw_stage.get("calls", [])
        if not isinstance(raw_calls, list):
            raise ValueError(f"replay-turn stage {stage_index} calls 必须是数组")
        calls = tuple(
            _load_replay_tool_call(raw_call, stage_index=stage_index, call_index=index)
            for index, raw_call in enumerate(raw_calls, start=1)
        )
        stages.append(
            ReplayStage(
                text=raw_text or "",
                reasoning=reasoning,
                calls=calls,
            )
        )
    return ReplayTurn(content=content, stages=tuple(stages))


def _load_replay_tool_call(
    raw_call: object,
    *,
    stage_index: int,
    call_index: int,
) -> ReplayToolCall:
    """在 JSON 信任边界校验一条工具调用。"""

    location = f"stage {stage_index} call {call_index}"
    if not isinstance(raw_call, dict):
        raise ValueError(f"replay-turn {location} 必须是对象")
    texts: dict[str, str] = {}
    for field in ("call_id", "name", "status", "result"):
        value = raw_call.get(field)
        if not isinstance(value, str) or (field != "result" and not value):
            raise ValueError(f"replay-turn {location} {field} 类型无效")
        texts[field] = value
    arguments = raw_call.get("arguments")
    final_arguments = raw_call.get("final_arguments")
    if not isinstance(arguments, dict) or not isinstance(final_arguments, dict):
        raise ValueError(f"replay-turn {location} arguments 必须是对象")
    return ReplayToolCall(
        call_id=texts["call_id"],
        name=texts["name"],
        status=texts["status"],
        arguments=dict(arguments),
        final_arguments=dict(final_arguments),
        result=texts["result"],
    )


class IsolatedModelRegistry:
    """为隔离 E2E 提供一个确定性的模型目录。"""

    async def refresh(self) -> SimpleNamespace:
        return SimpleNamespace(
            generation_id=1,
            role_runtime_ids={"default": "isolated-model"},
        )

    def list_runtimes(self) -> list[dict[str, object]]:
        return [
            {
                "id": "isolated-model",
                "provider": "openai",
                "catalogProvider": "openai",
                "model": "isolated-model",
                "reasoningEffort": "medium",
                "supportedReasoningEfforts": ["low", "medium", "high"],
                "sourceId": "isolated-source",
                "sourceName": "Isolated E2E",
                "contextWindow": 128_000,
                "maxOutputTokens": 8_192,
                "inputModalities": ["text"],
                "capabilitySource": "test",
                "capabilitySources": {},
                "roles": ["default", "agent"],
            }
        ]


class GatewayFaultController:
    """为一次真机重连注入一个确定性的单次停滞点。"""

    def __init__(self, mode: str) -> None:
        if mode not in _FAULT_MODES:
            raise ValueError(f"未知隔离 Gateway 故障模式: {mode}")
        self.mode = mode
        self.triggered = False

    def claim_before_challenge(self, *, has_paired_device: bool) -> bool:
        if (
            self.triggered
            or not has_paired_device
            or self.mode != "stall_before_challenge"
        ):
            return False
        self.triggered = True
        return True

    def claim_after_auth(self) -> bool:
        if self.triggered or self.mode != "stall_after_auth":
            return False
        self.triggered = True
        return True


async def _stall_websocket(websocket: Any, *, accept: bool) -> None:
    """保持连接但不发送协议进展，直到客户端主动关闭。"""

    if accept:
        await websocket.accept()
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


def install_fault_mode(
    runtime: MobileGatewayRuntime,
    mode: str,
) -> GatewayFaultController:
    """在隔离 runtime 上安装一次性握手或同步停滞。"""

    controller = GatewayFaultController(mode)
    if mode == "none":
        return controller

    # 1. challenge 前停滞只在首次配对完成后触发，不阻塞扫码流程
    original_handle = runtime.handle_websocket

    async def handle_websocket(websocket: Any) -> None:
        if controller.claim_before_challenge(
            has_paired_device=bool(runtime.storage.list_active_devices()),
        ):
            print("fault_triggered=stall_before_challenge", flush=True)
            await _stall_websocket(websocket, accept=True)
            return
        await original_handle(websocket)

    runtime.handle_websocket = handle_websocket  # type: ignore[method-assign]

    # 2. auth 后停滞保留真实 challenge/proof，只阻断 resume 后的同步进展
    original_authenticated_loop = (
        runtime._authenticated_loop
    )  # pyright: ignore[reportPrivateUsage]

    async def authenticated_loop(
        websocket: Any,
        *,
        device_id: str,
        connection_epoch: int,
    ) -> None:
        if controller.claim_after_auth():
            print("fault_triggered=stall_after_auth", flush=True)
            await _stall_websocket(websocket, accept=False)
            return
        await original_authenticated_loop(
            websocket,
            device_id=device_id,
            connection_epoch=connection_epoch,
        )

    runtime._authenticated_loop = authenticated_loop  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]
    return controller


class EphemeralMasterKeys:
    def __init__(self) -> None:
        self.keys: dict[str, bytes] = {}

    def create(self) -> tuple[str, bytes]:
        key_id = uuid4().hex
        key = secrets.token_bytes(32)
        self.keys[key_id] = key
        return key_id, key

    def load(self, master_key_id: str) -> bytes:
        try:
            return self.keys[master_key_id]
        except KeyError as error:
            raise KeyProtectionError("隔离 master key 不存在") from error


class EventBus:
    def on(self, event_type: type[object], callback: object) -> None:
        return None


class PushTool:
    def register_channel(self, channel: str, **senders: object) -> None:
        if channel != "mobile":
            raise RuntimeError(f"隔离 Gateway 收到未知渠道: {channel}")


class IsolatedAkashaMobileUiProvider:
    """通过真实 Akasha module 和有界投影提供隔离真机查询。"""

    plugin_id = "akasha@builtin"

    def __init__(self) -> None:
        plugin_root = Path(__file__).parents[1] / "plugins" / "akasha"
        self._module = (plugin_root / "mobile_ui.js").read_text(encoding="utf-8")
        self._stylesheet = (plugin_root / "mobile_ui.css").read_text(encoding="utf-8")
        self._module_sha256 = hashlib.sha256(self._module.encode()).hexdigest()
        self._stylesheet_sha256 = hashlib.sha256(self._stylesheet.encode()).hexdigest()
        self._revision = hashlib.sha256(
            f"{self._module_sha256}:{self._stylesheet_sha256}".encode()
        ).hexdigest()
        self._item = {
            "id": self.plugin_id,
            "revision": self._revision,
            "module_sha256": self._module_sha256,
            "module_bytes": len(self._module.encode()),
            "stylesheet_sha256": self._stylesheet_sha256,
            "stylesheet_bytes": len(self._stylesheet.encode()),
            "navigation": {
                "label": "Akasha Inspector",
                "description": "隔离真机召回卡片",
            },
            "slots": ["turn.before_reasoning"],
        }

    def catalog(self) -> dict[str, object]:
        items = [self._item]
        encoded = json.dumps(
            items,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return {
            "catalog_revision": hashlib.sha256(encoded).hexdigest(),
            "items": items,
        }

    def asset(
        self,
        plugin_id: str,
        plugin_revision: str,
        kind: str,
        sha256: str,
    ) -> dict[str, object]:
        if plugin_id != self.plugin_id or plugin_revision != self._revision:
            raise ValueError("隔离 Akasha asset revision 无效")
        if kind == "module":
            content, expected = self._module, self._module_sha256
        elif kind == "stylesheet":
            content, expected = self._stylesheet, self._stylesheet_sha256
        else:
            raise ValueError("隔离 Akasha asset kind 无效")
        if sha256 != expected:
            raise ValueError("隔离 Akasha asset digest 无效")
        return {
            "plugin_id": plugin_id,
            "plugin_revision": plugin_revision,
            "kind": kind,
            "sha256": expected,
            "content": content,
        }

    async def query(
        self,
        plugin_id: str,
        plugin_revision: str,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        """用真实 card-v1 投影构造接近最坏体积的隔离结果。"""

        if plugin_id != self.plugin_id or plugin_revision != self._revision:
            raise ValueError("隔离 Akasha query revision 无效")
        if method != "recall.current" or set(payload) != {"message_id"}:
            raise ValueError("隔离 Akasha query 参数无效")
        if session_id is None or not session_id.startswith("akashic:"):
            raise ValueError("隔离 Akasha query 会话无效")
        lane = _mobile_recall_lane(
            [
                {
                    "user_text": "🌙" * 1_000,
                    "assistant_preview": "🌙" * 1_000,
                    "ts": "2026-07-28T00:00:00Z",
                    "score": 0.5,
                }
                for _ in range(40)
            ]
        )
        result: dict[str, object] = {
            "schema": "akasha.recall-card.v1",
            "query_id": "isolated-pixel7-query",
            "recall_capture_available": True,
            "left": lane,
            "right": lane,
            "tool_left": lane,
            "tool_right": lane,
        }
        encoded = json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        if len(encoded) >= 192 * 1024:
            raise RuntimeError("隔离 Akasha card 超过 192 KiB")
        return result


class FixedReplyBus:
    """把真实手机入站写入隔离会话库，并返回固定文字和媒体。"""

    def __init__(
        self,
        manager: SessionManager,
        reply_media: Path,
        *,
        tokens_per_second: float = 0,
        stream_tokens: int = 1_200,
        stream_chunk_chars: int = 24,
        replay_turn: ReplayTurn | None = None,
    ) -> None:
        self._manager = manager
        self._reply_media = reply_media
        self._tokens_per_second = tokens_per_second
        self._stream_tokens = stream_tokens
        self._stream_chunk_chars = stream_chunk_chars
        self._replay_turn = replay_turn
        self._runtime: MobileGatewayRuntime | None = None
        self._reply_tasks: set[asyncio.Task[None]] = set()

    def bind(self, runtime: MobileGatewayRuntime) -> None:
        self._runtime = runtime

    def subscribe_outbound(self, channel: str, callback: object) -> None:
        if channel != "mobile":
            raise RuntimeError(f"隔离 Gateway 收到未知渠道订阅: {channel}")

    async def reserve_mobile_channel_handoff(self, raw: RawInbound) -> bool:
        return True

    async def defer_mobile_channel_handoff(self, handoff_id: str) -> None:
        raise RuntimeError(f"隔离 Gateway 不应延迟 handoff: {handoff_id}")

    def has_pending_mobile_handoff(
        self,
        *,
        session_key: str,
        client_message_id: str,
    ) -> bool:
        return False

    async def admit(self, raw: RawInbound) -> bool:
        """把当前 v3 ingress 投影到确定性性能回复。"""

        inbound = raw.message
        if not isinstance(inbound, ChannelInboundMessage):
            raise TypeError("隔离 Gateway v3 ingress 消息类型无效")
        projected = InboundMessage(
            channel=inbound.channel,
            sender=inbound.sender,
            chat_id=inbound.chat_id,
            content=inbound.content,
            timestamp=inbound.timestamp,
            metadata=dict(inbound.metadata),
        )
        assistant_message_id = self._persist_reply(projected)
        task = asyncio.create_task(
            self.publish_reply(projected, assistant_message_id),
            name=f"isolated-mobile-reply:{raw.message_id}",
        )
        self._reply_tasks.add(task)
        task.add_done_callback(self._reply_tasks.discard)
        return True

    def _persist_reply(self, inbound: InboundMessage) -> str:
        """在 ACK 前持久化确定性的用户消息与最终回复。"""

        if self._replay_turn is None:
            _, _, reply_chunks, _, _ = self._stream_payloads()
            reply = "".join(reply_chunks)
            tool_chain = None
        else:
            reply = self._replay_turn.content
            tool_chain = self._replay_turn.session_tool_chain()
        session = self._manager.get_or_create(inbound.session_key)
        user_kwargs: dict[str, str] = {
            "client_message_id": cast(str, inbound.metadata["client_message_id"]),
        }
        for field in ("reply_to_message_id", "reply_role", "reply_preview"):
            value = inbound.metadata.get(field)
            if isinstance(value, str) and value:
                user_kwargs[field] = value
        display_content = inbound.metadata.get("display_content")
        _ = session.add_message(
            "user",
            display_content if isinstance(display_content, str) else inbound.content,
            media=inbound.media,
            **user_kwargs,
        )
        assistant_kwargs: dict[str, object] = {}
        if tool_chain is not None:
            assistant_kwargs["tool_chain"] = tool_chain
        _ = session.add_message(
            "assistant",
            reply,
            media=[str(self._reply_media)],
            **assistant_kwargs,
        )
        self._manager.save(session)
        return str(session.messages[-1]["id"])

    async def publish_reply(
        self,
        inbound: InboundMessage,
        assistant_message_id: str,
    ) -> None:
        """从已持久化的 fixture 回复发布真实 MobileRealtimeChannel 事件。"""

        runtime = self._require_runtime()
        if self._replay_turn is not None:
            await self._publish_replay_reply(
                runtime,
                inbound,
                assistant_message_id,
                self._replay_turn,
            )
            return
        thinking_before, thinking_after, reply_chunks, thinking_delay, answer_delay = (
            self._stream_payloads()
        )
        reply = "".join(reply_chunks)
        turn_id = uuid4().hex

        # 1. 通过真实 durable inbox 发布可断线恢复事件
        await runtime.channel._on_turn_started(  # pyright: ignore[reportPrivateUsage]
            TurnStarted(
                session_key=inbound.session_key,
                channel=runtime.channel.name,
                chat_id=inbound.chat_id,
                content=inbound.content,
                timestamp=datetime.now(timezone.utc),
                turn_id=turn_id,
                control_turn_id=turn_id,
                client_message_id=cast(str, inbound.metadata["client_message_id"]),
            )
        )
        for chunk in thinking_before:
            await runtime.channel._on_stream_delta(  # pyright: ignore[reportPrivateUsage]
                StreamDeltaReady(
                    session_key=inbound.session_key,
                    channel=runtime.channel.name,
                    chat_id=inbound.chat_id,
                    turn_id=turn_id,
                    thinking_delta=chunk,
                )
            )
            await asyncio.sleep(thinking_delay)
        for iteration in range(1, 7):
            call_id = f"pilot-tool-{iteration}-{turn_id}"
            tool_arguments: dict[str, Any] = {
                "description": f"检查移动流式链路阶段 {iteration}",
                "source": "frontend/chat/src/mobile-native.tsx",
                "targets": ["room", "kotlin", "android-webview"],
            }
            await runtime.channel._on_tool_call_started(  # pyright: ignore[reportPrivateUsage]
                ToolCallStarted(
                    session_key=inbound.session_key,
                    channel=runtime.channel.name,
                    chat_id=inbound.chat_id,
                    iteration=iteration,
                    call_id=call_id,
                    tool_name="inspect_mobile_stream_stage",
                    arguments=tool_arguments,
                    turn_id=turn_id,
                )
            )
            await asyncio.sleep(0.05)
            await runtime.channel._on_tool_call_completed(  # pyright: ignore[reportPrivateUsage]
                ToolCallCompleted(
                    session_key=inbound.session_key,
                    channel=runtime.channel.name,
                    chat_id=inbound.chat_id,
                    iteration=iteration,
                    call_id=call_id,
                    tool_name="inspect_mobile_stream_stage",
                    arguments=tool_arguments,
                    final_arguments=tool_arguments,
                    status="success",
                    result_preview=f"阶段 {iteration} 已记录。",
                    turn_id=turn_id,
                )
            )
        for chunk in thinking_after:
            await runtime.channel._on_stream_delta(  # pyright: ignore[reportPrivateUsage]
                StreamDeltaReady(
                    session_key=inbound.session_key,
                    channel=runtime.channel.name,
                    chat_id=inbound.chat_id,
                    turn_id=turn_id,
                    thinking_delta=chunk,
                )
            )
            await asyncio.sleep(thinking_delay)
        for chunk in reply_chunks:
            await runtime.channel._on_stream_delta(  # pyright: ignore[reportPrivateUsage]
                StreamDeltaReady(
                    session_key=inbound.session_key,
                    channel=runtime.channel.name,
                    chat_id=inbound.chat_id,
                    turn_id=turn_id,
                    content_delta=chunk,
                )
            )
            await asyncio.sleep(answer_delay)
        await runtime.channel._deliver_message(  # pyright: ignore[reportPrivateUsage]
            ChannelMessage(
                channel=runtime.channel.name,
                chat_id=inbound.chat_id,
                content=reply,
                attachments=(
                    ChannelAttachment(
                        kind=AttachmentKind.IMAGE,
                        source=str(self._reply_media),
                        filename=self._reply_media.name,
                    ),
                ),
                thinking="".join((*thinking_before, *thinking_after)),
                metadata={
                    "_channel_commit_role": "passive",
                    "client_message_id": inbound.metadata["client_message_id"],
                },
                control_turn_id=turn_id,
                execution_attempt_id=turn_id,
                session_message_id=assistant_message_id,
                terminal_status=TurnTerminalStatus.COMPLETED,
            )
        )

    async def _publish_replay_reply(
        self,
        runtime: MobileGatewayRuntime,
        inbound: InboundMessage,
        assistant_message_id: str,
        replay: ReplayTurn,
    ) -> None:
        """按原始思考、工具和文本顺序回放一条已校验 Turn。"""

        turn_id = uuid4().hex
        delay = 0.0 if self._tokens_per_second == 0 else 1.0 / self._tokens_per_second
        await runtime.channel._on_turn_started(  # pyright: ignore[reportPrivateUsage]
            TurnStarted(
                session_key=inbound.session_key,
                channel=runtime.channel.name,
                chat_id=inbound.chat_id,
                content=inbound.content,
                timestamp=datetime.now(timezone.utc),
                turn_id=turn_id,
                control_turn_id=turn_id,
                client_message_id=cast(str, inbound.metadata["client_message_id"]),
            )
        )
        for iteration, stage in enumerate(replay.stages, start=1):
            await self._publish_replay_text(
                runtime,
                inbound,
                turn_id,
                thinking=stage.reasoning,
                content=stage.text,
                delay=delay,
            )
            for call in stage.calls:
                await runtime.channel._on_tool_call_started(  # pyright: ignore[reportPrivateUsage]
                    ToolCallStarted(
                        session_key=inbound.session_key,
                        channel=runtime.channel.name,
                        chat_id=inbound.chat_id,
                        iteration=iteration,
                        call_id=call.call_id,
                        tool_name=call.name,
                        arguments=call.arguments,
                        turn_id=turn_id,
                    )
                )
                await asyncio.sleep(0.05)
                await runtime.channel._on_tool_call_completed(  # pyright: ignore[reportPrivateUsage]
                    ToolCallCompleted(
                        session_key=inbound.session_key,
                        channel=runtime.channel.name,
                        chat_id=inbound.chat_id,
                        iteration=iteration,
                        call_id=call.call_id,
                        tool_name=call.name,
                        arguments=call.arguments,
                        final_arguments=call.final_arguments,
                        status=call.status,
                        result_preview=call.result[:2_000],
                        turn_id=turn_id,
                    )
                )
        await self._publish_replay_text(
            runtime,
            inbound,
            turn_id,
            thinking="",
            content=replay.content,
            delay=delay,
        )
        await runtime.channel._deliver_message(  # pyright: ignore[reportPrivateUsage]
            ChannelMessage(
                channel=runtime.channel.name,
                chat_id=inbound.chat_id,
                content=replay.content,
                attachments=(
                    ChannelAttachment(
                        kind=AttachmentKind.IMAGE,
                        source=str(self._reply_media),
                        filename=self._reply_media.name,
                    ),
                ),
                thinking=replay.reasoning,
                metadata={
                    "_channel_commit_role": "passive",
                    "client_message_id": inbound.metadata["client_message_id"],
                },
                control_turn_id=turn_id,
                execution_attempt_id=turn_id,
                session_message_id=assistant_message_id,
                terminal_status=TurnTerminalStatus.COMPLETED,
            )
        )

    async def _publish_replay_text(
        self,
        runtime: MobileGatewayRuntime,
        inbound: InboundMessage,
        turn_id: str,
        *,
        thinking: str,
        content: str,
        delay: float,
    ) -> None:
        """按配置的 provider delta 大小发布一个回放文本阶段。"""

        for value, is_thinking in ((thinking, True), (content, False)):
            for chunk in _text_chunks(value, self._stream_chunk_chars):
                await runtime.channel._on_stream_delta(  # pyright: ignore[reportPrivateUsage]
                    StreamDeltaReady(
                        session_key=inbound.session_key,
                        channel=runtime.channel.name,
                        chat_id=inbound.chat_id,
                        turn_id=turn_id,
                        thinking_delta=chunk if is_thinking else "",
                        content_delta="" if is_thinking else chunk,
                    )
                )
                if delay:
                    await asyncio.sleep(delay)

    async def aclose(self) -> None:
        """先取消并收割 fixture 回复，再关闭它依赖的 channel 与 SessionDB。"""

        tasks = tuple(self._reply_tasks)
        for task in tasks:
            _ = task.cancel()
        if tasks:
            _ = await asyncio.gather(*tasks, return_exceptions=True)

    def _stream_payloads(
        self,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], float, float]:
        """生成普通演示流或固定频率的性能 delta 流。"""

        # 1. 默认模式保持既有设备 Gate 的内容和节奏
        if self._tokens_per_second == 0:
            return (
                _PILOT_THINKING_BEFORE_TOOL,
                _PILOT_THINKING_AFTER_TOOL,
                _PILOT_REPLY_CHUNKS,
                0.42,
                0.18,
            )

        # 2. 性能模式按配置的字符块控制 provider delta 频率
        thinking_count = min(8_213, self._stream_tokens)
        answer_count = self._stream_tokens - thinking_count
        thinking = _repeat_to_length("分析移动端流式渲染与工具调用。", thinking_count)
        answer = _repeat_to_length(
            "## WebUI 试点\n\n验证一百 token 每秒时的消息生长、滚动和图片显示。",
            answer_count,
        )
        delay = 1.0 / self._tokens_per_second
        chunk_size = self._stream_chunk_chars
        thinking_chunks = tuple(
            thinking[index:index + chunk_size]
            for index in range(0, len(thinking), chunk_size)
        )
        answer_chunks = tuple(
            answer[index:index + chunk_size]
            for index in range(0, len(answer), chunk_size)
        )
        return thinking_chunks, (), answer_chunks, delay, delay

    def _require_runtime(self) -> MobileGatewayRuntime:
        if self._runtime is None:
            raise RuntimeError("固定回复 bus 尚未绑定 Gateway")
        return self._runtime


def build_config(root: Path, host: str, port: int) -> MobileRealtimeConfig:
    return MobileRealtimeConfig(
        enabled=True,
        host=host,
        port=port,
        database=root / "gateway" / "mobile.db",
        lan_hostname="localhost",
        public_url="",
        key_encryption=MobileKeyEncryptionConfig(
            keyset_manifest=root / "gateway" / "keys" / "current.json"
        ),
    )


def _repeat_to_length(seed: str, length: int) -> str:
    repeats = (length + len(seed) - 1) // len(seed)
    return (seed * repeats)[:length]


def _text_chunks(value: str, chunk_size: int = 24) -> tuple[str, ...]:
    return tuple(
        value[index:index + chunk_size]
        for index in range(0, len(value), chunk_size)
    )


async def attach_open_mobile_v3(channel: Any, ingress: Any) -> Any:
    """为真机 fixture 打开一个 exact v3 ingress。"""

    context = ChannelFactoryContext(
        snapshot_id="device-perf-snapshot",
        generation_id="device-perf-generation",
        binding_token="device-perf-binding",
        config={},
        credentials={},
        provider_client_factory=cast(Any, object()),
        ingress=ingress,
        identity=None,
    )
    adapter = channel.build_v3_adapter(context)
    adapter.attach_runtime(
        ChannelRuntimePorts(
            snapshot_id=context.snapshot_id,
            generation_id=context.generation_id,
            binding_token=context.binding_token,
            ingress=context.ingress,
            identity=context.identity,
            attachment_import=context.attachment_import,
        )
    )
    _ = await adapter.start()
    adapter.open_admission()
    return adapter


def write_pairing_artifacts(root: Path, offer: dict[str, object]) -> None:
    """写出二维码原文和 PNG，供 USB 设备或模拟器扫码。"""

    payload = json.dumps(offer, ensure_ascii=False, separators=(",", ":"))
    json_path = root / "pairing-offer.json"
    png_path = root / "pairing-offer.png"
    _ = json_path.write_text(payload, encoding="utf-8")
    _ = subprocess.run(
        ["qrencode", "-m", "4", "-s", "8", "-o", str(png_path)],
        input=payload,
        text=True,
        check=True,
    )
    print(f"pairing_json={json_path}", flush=True)
    print(f"pairing_qr={png_path}", flush=True)


async def approve_pairing(runtime: MobileGatewayRuntime, pairing_id: str) -> None:
    """只批准本次隔离进程创建的一次性 pairing。"""

    while True:
        claim = runtime.admin.pending_claim(pairing_id)
        if claim is not None:
            code = cast(str, claim["confirmation_code"])
            device = runtime.admin.approve(pairing_id, code)
            print(
                f"pairing_approved device_id={device['device_id']} code={code}",
                flush=True,
            )
            return
        await asyncio.sleep(0.2)


async def run_harness(args: argparse.Namespace) -> None:
    """启动临时 TLS Gateway，直到收到 SIGINT 或 SIGTERM。"""

    # 1. 构造与真实 runtime 完全分离的目录和确定性数据
    replay_turn = (
        load_replay_turn(args.replay_turn.resolve())
        if args.replay_turn is not None
        else None
    )
    generated_root = args.root is None
    root = (
        Path(tempfile.mkdtemp(prefix="akashic-mobile-e2e-"))
        if generated_root
        else args.root.resolve()
    )
    _ = root.mkdir(parents=True, exist_ok=True)
    manager = SessionManager(root / "workspace")
    reply_media = args.reply_media.resolve() if args.reply_media is not None else None
    if reply_media is not None and not reply_media.is_file():
        raise ValueError(f"reply-media 不是文件: {reply_media}")
    media_suffix = reply_media.suffix if reply_media is not None else ".gif"
    media = root / "fixtures" / f"fixed-reply{media_suffix}"
    _ = media.parent.mkdir(parents=True, exist_ok=True)
    if reply_media is None:
        _ = media.write_bytes(_FIXED_GIF)
    else:
        _ = shutil.copy2(reply_media, media)
    config = build_config(root, args.host, args.port)
    runtime, keyset = build_mobile_gateway_runtime(
        config,
        root,
        master_keys=EphemeralMasterKeys(),
    )
    runtime.channel.bind_mobile_ui_provider(IsolatedAkashaMobileUiProvider())
    runtime.channel.bind_model_registry(cast(Any, IsolatedModelRegistry()))
    fault_controller = install_fault_mode(runtime, args.fault_mode)
    if args.tokens_per_second < 0:
        raise ValueError("tokens-per-second 不能为负数")
    if args.stream_tokens <= 0:
        raise ValueError("stream-tokens 必须为正数")
    if not 1 <= args.stream_chunk_chars <= 4_096:
        raise ValueError("stream-chunk-chars 必须在 1 到 4096 之间")
    if args.history_messages < 2:
        raise ValueError("history-messages 必须至少为 2")
    bus = FixedReplyBus(
        manager,
        media,
        tokens_per_second=args.tokens_per_second,
        stream_tokens=args.stream_tokens,
        stream_chunk_chars=args.stream_chunk_chars,
        replay_turn=replay_turn,
    )
    bus.bind(runtime)
    await runtime.channel.start(
        cast(
            Any,
            SimpleNamespace(
                bus=bus,
                session_manager=manager,
                event_bus=EventBus(),
                push_tool=PushTool(),
                interrupt_controller=None,
                attachment_store=AttachmentStore(root / "attachments"),
                command_catalog_provider=lambda: (
                    ("memorystatus", "查看隔离命令入口"),
                ),
                log=logger,
            ),
        )
    )
    adapter = await attach_open_mobile_v3(runtime.channel, bus)
    if not args.empty_history:
        history = manager.get_or_create(_HISTORY_SESSION_ID)
        _ = history.add_message(
            "user",
            "这是隔离 Gateway 的历史消息",
            client_message_id="01J00000000000000000000000",
        )
        _ = history.add_message("assistant", "历史同步成功后应只出现一次。")
        for index in range(2, args.history_messages):
            role = "user" if index % 2 == 0 else "assistant"
            content = _repeat_to_length(
                f"第 {index + 1} 条隔离历史用于长会话投影测量。",
                320,
            )
            message_kwargs: dict[str, object] = {}
            if role == "assistant" and index % 4 == 3:
                message_kwargs["tool_chain"] = [
                    {
                        "text": "核对历史投影。",
                        "calls": [
                            {
                                "call_id": f"history-tool-{index}",
                                "name": "inspect_history",
                                "status": "success",
                                "arguments": {"index": index},
                                "result": "历史投影已核对。",
                            }
                        ],
                    }
                ]
            _ = history.add_message(role, content, **message_kwargs)
        manager.save(history)
    offer = runtime.admin.create_offer()
    write_pairing_artifacts(root, offer)
    print(f"isolated_root={root}", flush=True)
    print(f"history_session={_HISTORY_SESSION_ID}", flush=True)
    print(f"adb_reverse=adb reverse tcp:{args.port} tcp:{args.port}", flush=True)
    print(f"fault_mode={fault_controller.mode}", flush=True)
    print(f"tokens_per_second={args.tokens_per_second}", flush=True)
    print(f"stream_tokens={args.stream_tokens}", flush=True)
    print(f"stream_chunk_chars={args.stream_chunk_chars}", flush=True)
    print(f"history_messages={args.history_messages}", flush=True)
    if replay_turn is not None:
        print(f"replay_stages={len(replay_turn.stages)}", flush=True)
        print(f"replay_calls={replay_turn.call_count}", flush=True)
        print(f"replay_reasoning_chars={len(replay_turn.reasoning)}", flush=True)
        print(f"replay_content_chars={len(replay_turn.content)}", flush=True)

    # 2. 启动真实 TLS WebSocket，并自动批准唯一的隔离配对请求
    server = build_mobile_gateway_server(runtime, keyset)
    approval_task = asyncio.create_task(
        approve_pairing(runtime, cast(str, offer["pairing_id"]))
    )
    try:
        await server.serve()
    finally:
        _ = approval_task.cancel()
        try:
            _ = await asyncio.gather(approval_task, return_exceptions=True)
        finally:
            await bus.aclose()
            await adapter.stop()
            await runtime.channel.stop()
            manager.close()
            runtime.close()
            if generated_root and not args.keep:
                shutil.rmtree(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="启动不接触真实 workspace/DB 的 Android MobileRealtime Gateway",
    )
    _ = parser.add_argument(
        "--root", type=Path, help="显式隔离根目录；指定后不会自动删除"
    )
    _ = parser.add_argument("--host", default="127.0.0.1")
    _ = parser.add_argument("--port", type=int, default=16323)
    _ = parser.add_argument(
        "--fault-mode",
        choices=_FAULT_MODES,
        default="none",
        help="配对后仅注入一次指定阶段停滞，用于验证手机自动恢复",
    )
    _ = parser.add_argument(
        "--keep",
        action="store_true",
        help="保留自动创建的临时根目录",
    )
    _ = parser.add_argument(
        "--tokens-per-second",
        type=float,
        default=0,
        help="每秒发送的 provider delta 数；0 保持演示节奏",
    )
    _ = parser.add_argument(
        "--stream-tokens",
        type=int,
        default=1_200,
        help="性能流 thinking 与 answer 的总字符数",
    )
    _ = parser.add_argument(
        "--stream-chunk-chars",
        type=int,
        default=24,
        help="每个性能 provider delta 的字符数；真实高频场景使用 1",
    )
    _ = parser.add_argument(
        "--history-messages",
        type=int,
        default=2,
        help="预置历史消息数；性能场景可按真实会话规模放大",
    )
    _ = parser.add_argument(
        "--reply-media",
        type=Path,
        help="复制到隔离目录并随固定回复发送的真实尺寸媒体文件",
    )
    _ = parser.add_argument(
        "--replay-turn",
        type=Path,
        help="只读加载一份 Session 消息 JSON，并回放其中唯一的 assistant Turn",
    )
    _ = parser.add_argument(
        "--empty-history",
        action="store_true",
        help="不预置 Session，用于验证新安装创建第一条共享会话",
    )
    return parser.parse_args()


def main() -> None:
    try:
        asyncio.run(run_harness(parse_args()))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
