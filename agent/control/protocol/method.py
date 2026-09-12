from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel


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

    async def invoke(self, params: BaseModel, transport: RequestTransport | None) -> object:
        if self.call_with_transport is not None:
            if transport is None:
                raise RuntimeError("动态 RPC 缺少 RequestTransport")
            return await self.call_with_transport(params, transport)
        return await self.call(params)
