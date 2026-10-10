from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import astuple, dataclass
from typing import Protocol

from pydantic import BaseModel
from agent.plugin_composition import ServiceKey


class RequestTransport(Protocol):
    """动态 RPC 可使用的当前连接窄传输端口。"""

    connection_id: str


TransportCall = Callable[[BaseModel, RequestTransport], Awaitable[object]]


@dataclass(frozen=True)
class RpcMethod:
    """一个固定协议入口的参数边界与处理函数。"""

    params: type[BaseModel]
    call: Callable[[BaseModel], Awaitable[object]]
    call_with_transport: TransportCall | None = None

    @staticmethod
    def key(name: str) -> ServiceKey[RpcMethod]:
        """扩展方法不能替换 Gateway 保留入口。"""
        if not name or name.strip() != name:
            raise ValueError("RPC 方法名称必须非空且无首尾空白")
        if name in astuple(NAMES):
            raise ValueError(f"控制方法已经存在: {name}")
        return ServiceKey[RpcMethod]("gateway.rpc:" + name)

    async def invoke(self, params: BaseModel, transport: RequestTransport | None) -> object:
        if self.call_with_transport is not None:
            if transport is None:
                raise RuntimeError("动态 RPC 缺少 RequestTransport")
            return await self.call_with_transport(params, transport)
        return await self.call(params)


@dataclass(frozen=True, slots=True)
class RpcNames:
    """Gateway 保留方法的固定协议名称。"""

    initialize: str = "initialize"
    server_status: str = "server/status"
    session_create: str = "session/create"
    session_list: str = "session/list"
    message_read: str = "message/read"
    message_send: str = "message/send"
    session_follow: str = "session/follow"
    session_unfollow: str = "session/unfollow"
    plugin_install: str = "plugin/install"
    plugin_status: str = "plugin/status"
    plugin_update: str = "plugin/update"
    plugin_drain: str = "plugin/disable-and-drain"
    plugin_uninstall: str = "plugin/uninstall"


NAMES = RpcNames()
