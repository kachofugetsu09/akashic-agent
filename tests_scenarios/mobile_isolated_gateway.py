from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import secrets
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

from bus.events import InboundMessage, OutboundMessage
from bus.events_lifecycle import TurnStarted
from agent.config_models import MobileKeyEncryptionConfig, MobileRealtimeConfig
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
_HISTORY_SESSION_ID = "mobile:00000000-0000-7000-8000-000000000001"
_FAULT_MODES = ("none", "stall_before_challenge", "stall_after_auth")


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
    original_authenticated_loop = runtime._authenticated_loop  # pyright: ignore[reportPrivateUsage]

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
        self._stylesheet_sha256 = hashlib.sha256(
            self._stylesheet.encode()
        ).hexdigest()
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
        if session_id != _HISTORY_SESSION_ID or turn_id is not None:
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

    def __init__(self, manager: SessionManager, reply_media: Path) -> None:
        self._manager = manager
        self._reply_media = reply_media
        self._runtime: MobileGatewayRuntime | None = None

    def bind(self, runtime: MobileGatewayRuntime) -> None:
        self._runtime = runtime

    def subscribe_outbound(self, channel: str, callback: object) -> None:
        if channel != "mobile":
            raise RuntimeError(f"隔离 Gateway 收到未知渠道订阅: {channel}")

    async def publish_inbound(self, message: object) -> None:
        """持久化一轮对话，并走真实 MobileRealtimeChannel 发布回复。"""

        # 1. 按真实生命周期语义持久化干净正文和引用投影
        inbound = cast(InboundMessage, message)
        runtime = self._require_runtime()
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
        _ = session.add_message(
            "assistant",
            "隔离 Gateway 已收到消息，这是固定媒体回复。",
            media=[str(self._reply_media)],
        )
        self._manager.save(session)
        assistant_message_id = str(session.messages[-1]["id"])
        turn_id = uuid4().hex

        # 2. 通过真实 durable inbox 发布可断线恢复事件
        await runtime.channel._on_turn_started(  # pyright: ignore[reportPrivateUsage]
            TurnStarted(
                session_key=inbound.session_key,
                channel="mobile",
                chat_id=inbound.chat_id,
                content=inbound.content,
                timestamp=datetime.now(timezone.utc),
                turn_id=turn_id,
            )
        )
        await runtime.channel._on_response(  # pyright: ignore[reportPrivateUsage]
            OutboundMessage(
                channel="mobile",
                chat_id=inbound.chat_id,
                content="隔离 Gateway 已收到消息，这是固定媒体回复。",
                media=[str(self._reply_media)],
                control_turn_id=turn_id,
                session_message_id=assistant_message_id,
            )
        )

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
    generated_root = args.root is None
    root = (
        Path(tempfile.mkdtemp(prefix="akashic-mobile-e2e-"))
        if generated_root
        else args.root.resolve()
    )
    _ = root.mkdir(parents=True, exist_ok=True)
    manager = SessionManager(root / "workspace")
    media = root / "fixtures" / "fixed-reply.gif"
    _ = media.parent.mkdir(parents=True, exist_ok=True)
    _ = media.write_bytes(_FIXED_GIF)
    config = build_config(root, args.host, args.port)
    runtime, keyset = build_mobile_gateway_runtime(
        config,
        root,
        master_keys=EphemeralMasterKeys(),
    )
    runtime.channel.bind_mobile_ui_provider(IsolatedAkashaMobileUiProvider())
    fault_controller = install_fault_mode(runtime, args.fault_mode)
    bus = FixedReplyBus(manager, media)
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
                mobile_bot_commands=(("memorystatus", "查看隔离命令入口"),),
            ),
        )
    )
    history = manager.get_or_create(_HISTORY_SESSION_ID)
    _ = history.add_message(
        "user",
        "这是隔离 Gateway 的历史消息",
        client_message_id="01J00000000000000000000000",
    )
    _ = history.add_message("assistant", "历史同步成功后应只出现一次。")
    manager.save(history)
    offer = runtime.admin.create_offer()
    write_pairing_artifacts(root, offer)
    print(f"isolated_root={root}", flush=True)
    print(f"history_session={_HISTORY_SESSION_ID}", flush=True)
    print(f"adb_reverse=adb reverse tcp:{args.port} tcp:{args.port}", flush=True)
    print(f"fault_mode={fault_controller.mode}", flush=True)

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
            await runtime.channel.stop()
            manager.close()
            runtime.close()
            if generated_root and not args.keep:
                shutil.rmtree(root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="启动不接触真实 workspace/DB 的 Android MobileRealtime Gateway",
    )
    _ = parser.add_argument("--root", type=Path, help="显式隔离根目录；指定后不会自动删除")
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
    return parser.parse_args()


def main() -> None:
    try:
        asyncio.run(run_harness(parse_args()))
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
