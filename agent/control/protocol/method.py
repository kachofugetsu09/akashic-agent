from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from .models import StrictModel


class OutputReservation(Protocol):
    """绑定一次 Input 的同连接最终 Output 观察。"""

    async def wait_output(self, message_id: str) -> None: ...


class RequestTransport(Protocol):
    """动态 RPC 可使用的当前连接窄传输端口。"""

    connection_id: str

    def reserve_input(self, session_id: str, input_id: str) -> OutputReservation: ...


TransportCall = Callable[[StrictModel, RequestTransport], Awaitable[object]]


@dataclass(frozen=True)
class RpcMethod:
    """一个固定协议入口的参数边界与处理函数。"""

    params: type[StrictModel]
    call: Callable[[StrictModel], Awaitable[object]]
    call_with_transport: TransportCall | None = None

    async def invoke(self, params: StrictModel, transport: RequestTransport | None) -> object:
        if self.call_with_transport is not None:
            if transport is None:
                raise RuntimeError("动态 RPC 缺少 RequestTransport")
            return await self.call_with_transport(params, transport)
        return await self.call(params)
