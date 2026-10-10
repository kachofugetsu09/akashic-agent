from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import astuple, dataclass
from typing import Protocol
from asyncio import Future
from agent.plugin_contracts import CallRef

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


class FrameResolver(Protocol):
    """读取一个 Input 对应的最终 Output 身份。"""

    def __call__(self) -> str | None: ...


class FrameReservation(Protocol):
    """等待已接纳的最终 Output 写出；route 结束时抛出 LookupError。"""

    async def wait_output(self, message_id: str) -> None: ...


class FrameClaim(Protocol):
    """持有一个 ToolCall 的最终写出回执，直到消费或放弃。"""

    @property
    def ending_message_id(self) -> str | None: ...
    async def wait_output(self) -> None: ...
    def consume(self) -> None: ...
    def abort(self) -> None: ...


class FrameRouteStage(Protocol):
    """操作成功才提交替代 route；放弃不改变当前 owner。"""

    @property
    def reservation(self) -> FrameReservation: ...
    def commit(self) -> FrameReservation: ...
    def abort(self) -> None: ...


class ControlFrames(Protocol):
    """绑定连接与 Input 的实际写出回执；缺失或结束的 route 抛出 LookupError。"""

    def route_input_with_owner(self, session_id: str, input_id: str, connection_id: str,
                               resolver: FrameResolver) -> tuple[FrameReservation, bool]: ...
    def stage_input(self, session_id: str, input_id: str, connection_id: str,
                    resolver: FrameResolver) -> FrameRouteStage: ...
    async def wait_input(self, session_id: str, input_id: str, ending: str) -> None: ...
    def release_input(self, session_id: str, input_id: str) -> None: ...
    def settle_input(self, session_id: str, input_id: str, error: BaseException | None = None) -> None: ...
    def active_session_ids(self) -> frozenset[str]: ...
    def active_input_ids(self, session_id: str) -> tuple[str, ...]: ...
    def arm_claim(self, session_id: str, input_id: str, call_ref: CallRef) -> FrameClaim: ...
    def claim_for(self, session_id: str, call_ref: CallRef) -> FrameClaim | None: ...
    def track_page(self, connection_id: str, page: Mapping[str, object], written: Future[None]) -> bool: ...
    def fail_connection(self, connection_id: str, error: BaseException) -> None: ...


CONTROL_FRAMES = ServiceKey[ControlFrames]("gateway.frames.v1")
