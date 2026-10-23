"""画面传输：流服务的会话令牌、输入权限，以及 WebRTC 与 WebSocket 之间的选择。"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import secrets
from typing import Literal, cast

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from agent.plugin_composition import ServiceKey
from plugins.plugin_settings.contract import SettingsField, SettingsForm

# WebRTC 媒体端口；镜像 start.sh 与 Workload 的 udp_ports 用同一个号。
MEDIA_PORT = 30000
StreamMode = Literal["webrtc", "websockets"]


class StreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lan_address: str | None = None

    @field_validator("lan_address")
    @classmethod
    def _address(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return str(ipaddress.ip_address(value.strip()))


STREAM_CONFIG = ServiceKey[StreamConfig]("computer.stream.v1")
STREAM_FORM = SettingsForm(
    title="浏览器画面",
    description="画面默认经网页连接传输；能直连时改走 WebRTC，帧率更高、远程不易卡。",
    fields=(
        SettingsField(
            "lan_address", "宿主局域网地址",
            hint=f"例如 192.168.0.107。填写后，同一局域网的浏览器经 UDP {MEDIA_PORT} 直连画面；留空只用网页连接。",
        ),
    ),
)


class StreamAccess:
    """每条画面连接持一个会话令牌；最新的连接可以操作时由它持有键鼠，否则由占位令牌持有，流服务据此丢弃输入。"""

    def __init__(self, gateway: str, stream: str) -> None:
        self._gateway = gateway.rstrip("/")
        self._stream = stream.rstrip("/")
        self._hold = secrets.token_urlsafe(18)
        self._live: dict[str, bool] = {}
        self._lock = asyncio.Lock()
        # 每切一次传输加一；连接记下建立时的值，断开时据此判断是否被别的窗口换了传输顶掉。
        self.switches = 0

    # 同步失败或被取消时撤回这个令牌，免得没有连接的条目一直占着“最新连接”。
    async def open(self, allowed: bool) -> str:
        token = secrets.token_urlsafe(18)
        self._live[token] = allowed
        try:
            await self._sync()
        except BaseException:
            _ = self._live.pop(token, None)
            raise
        return token

    async def allow(self, token: str, allowed: bool) -> None:
        if token in self._live and self._live[token] != allowed:
            self._live[token] = allowed
            await self._sync()

    async def close(self, token: str) -> None:
        if self._live.pop(token, None) is not None:
            await self._sync()

    async def mode(self) -> object:
        async with httpx.AsyncClient(timeout=10.0) as client:
            result = await client.get(self._stream + "/api/status")
            return cast(dict[str, object], result.raise_for_status().json()).get("current_mode")

    # 流服务同时只服务一种传输，切换约 15ms，并断开另一种传输上的连接。
    async def use(self, mode: StreamMode) -> None:
        if await self.mode() == mode:
            return
        async with self._lock, httpx.AsyncClient(timeout=10.0) as client:
            self.switches += 1
            result = await client.post(self._stream + "/api/switch", json={"mode": mode},
                                       headers=await self._master(client))
            _ = result.raise_for_status()

    async def _sync(self) -> None:
        async with self._lock, httpx.AsyncClient(timeout=10.0) as client:
            newest = next(reversed(self._live), None)
            holder = newest if newest is not None and self._live[newest] else self._hold
            table: dict[str, dict[str, str | bool]] = {token: {"role": "controller"} for token in self._live}
            table[holder] = {"role": "controller", "mk_control": True}
            result = await client.post(self._stream + "/api/tokens", json=table, headers=await self._master(client))
            _ = result.raise_for_status()

    # 主令牌随容器重建而变，每次向网关取，不缓存。
    async def _master(self, client: httpx.AsyncClient) -> dict[str, str]:
        result = await client.get(self._gateway + "/stream/token")
        _ = result.raise_for_status()
        token = cast(dict[str, object], result.json()).get("token")
        if not isinstance(token, str) or not token:
            raise httpx.HTTPError("Computer stream token is unavailable")
        return {"Authorization": "Bearer " + token}


def signaling_hello(text: str, token: str) -> str:
    """面板总以主控端身份登记，会话令牌由 Core 填入，浏览器拿不到。"""
    if not text.startswith("HELLO "):
        return text
    _, kind, raw = text.split(" ", 2)
    meta = cast(dict[str, object], json.loads(raw))
    meta.update(client_token=token, client_type="controller")
    return f"HELLO {kind} {json.dumps(meta)}"


def lan_candidate(text: str, address: str | None) -> str | None:
    """服务端发出 offer 后补一条宿主局域网地址的候选；容器自己报的是桥接网段地址，浏览器连不到。"""
    peer, _, raw = text.partition(" ")
    if address is None or not raw.startswith('{"sdp"'):
        return None
    sdp = cast(dict[str, dict[str, str]], json.loads(raw))["sdp"]
    if sdp.get("type") != "offer":
        return None
    mid = next((line[6:].strip() for line in sdp.get("sdp", "").splitlines() if line.startswith("a=mid:")), "0")
    candidate = f"candidate:akashic 1 udp 2130706431 {address} {MEDIA_PORT} typ host"
    return f"{peer} {json.dumps({'ice': {'candidate': candidate, 'sdpMid': mid, 'sdpMLineIndex': 0}})}"
