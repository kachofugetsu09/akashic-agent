from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .models import StrictModel


@dataclass(frozen=True)
class RpcMethod:
    """一个固定协议入口的参数边界与处理函数。"""

    params: type[StrictModel]
    call: Callable[[StrictModel], Awaitable[object]]
