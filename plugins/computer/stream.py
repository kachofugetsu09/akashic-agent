"""画面传输：流服务的会话令牌、输入权限，以及 WebRTC 与 WebSocket 之间的选择。"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import secrets
import time
from collections.abc import Awaitable, Callable
from typing import Literal, cast

import httpx
from pydantic import BaseModel, ConfigDict, field_validator

from agent.plugin_composition import CredentialRef, ServiceKey
from plugins.plugin_settings.contract import SettingsField, SettingsForm

logger = logging.getLogger(__name__)

# WebRTC 媒体端口；镜像 start.sh 与 Workload 的 udp_ports 用同一个号。
MEDIA_PORT = 30000
StreamMode = Literal["webrtc", "websockets"]
IceServer = dict[str, object]
# Cloudflare Realtime TURN：按 key 申请短期凭据，过半有效期再换。
TURN_API = "https://rtc.live.cloudflare.com/v1/turn/keys/{}/credentials/generate"
TURN_TTL = 86400


class StreamConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lan_address: str | None = None
    turn_key_id: str | None = None
    turn_api_token: CredentialRef | None = None

    @field_validator("lan_address")
    @classmethod
    def _address(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return str(ipaddress.ip_address(value.strip()))


STREAM_FORM = SettingsForm(
    title="浏览器画面",
    description="画面默认经网页连接传输；能直连或配置了中继时改走 WebRTC，帧率更高、远程不易卡。",
    fields=(
        SettingsField(
            "lan_address", "宿主局域网地址",
            hint=f"例如 192.168.0.107。填写后，同一局域网的浏览器经 UDP {MEDIA_PORT} 直连画面；留空只用网页连接。",
        ),
        SettingsField("turn_key_id", "Cloudflare TURN Key ID",
                      hint="在 Cloudflare 后台 Realtime → TURN 创建；填写后不在局域网时画面经 Cloudflare 中继。"),
        SettingsField("turn_api_token", "Cloudflare TURN API Token", kind="secret"),
    ),
)


class StreamSettings:
    """设置里的直连地址与远程中继；中继凭据按需向 Cloudflare 申请，不常驻 API Token。"""

    def __init__(self, config: StreamConfig, read_token: Callable[[], Awaitable[str]]) -> None:
        self.config = config
        self._read_token = read_token
        self._ice: tuple[list[IceServer], float] | None = None
        self._lock = asyncio.Lock()

    @property
    def remote(self) -> bool:
        return self.config.turn_key_id is not None and self.config.turn_api_token is not None

    @property
    def webrtc(self) -> bool:
        return self.config.lan_address is not None or self.remote

    # 申请失败只记日志并返回空表：有局域网地址仍可直连，否则面板回退网页连接。
    async def ice_servers(self) -> list[IceServer]:
        if not self.remote:
            return []
        async with self._lock:
            if self._ice is None or time.monotonic() > self._ice[1]:
                try:
                    self._ice = (await self._generate(), time.monotonic() + TURN_TTL / 2)
                except httpx.HTTPError as error:
                    logger.warning("Cloudflare TURN credentials unavailable: %s", error)
                    return []
            return self._ice[0]

    async def _generate(self) -> list[IceServer]:
        token = await self._read_token()
        async with httpx.AsyncClient(timeout=10.0) as client:
            result = await client.post(TURN_API.format(self.config.turn_key_id), json={"ttl": TURN_TTL},
                                       headers={"Authorization": "Bearer " + token})
            servers = cast(dict[str, object], result.raise_for_status().json()).get("iceServers")
        # 接口可能给单个对象，也可能给列表。
        listed = cast(list[object], servers) if isinstance(servers, list) else [servers]
        return [cast(IceServer, server) for server in listed if isinstance(server, dict)]


STREAM_CONFIG = ServiceKey[StreamSettings]("computer.stream.v1")


def relay_only(servers: list[IceServer]) -> list[IceServer]:
    """流服务只用 TCP/TLS 中继：容器的 UDP 出口可能经代理，STUN 和 UDP 中继会让每次建连空等超时。"""
    kept: list[IceServer] = []
    for server in servers:
        urls = server.get("urls")
        listed = [urls] if isinstance(urls, str) else cast(list[object], urls) if isinstance(urls, list) else []
        tcp = [url for url in listed if isinstance(url, str) and (url.startswith("turns:") or url.endswith("transport=tcp"))]
        if tcp:
            kept.append({**server, "urls": tcp})
    return kept


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

    # 写入流服务的 ICE 配置文件；新连接即生效，内容不变时流服务不会重建什么。
    async def relay(self, servers: list[IceServer]) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            result = await client.put(self._gateway + "/stream/rtc", json={"iceServers": servers})
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
