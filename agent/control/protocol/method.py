"""兼容入口；控制协议 RPC 方法边界由 `agent.plugin_contracts.control_method` 拥有。"""

from agent.plugin_contracts.control_method import (
    RequestTransport,
    RpcMethod,
    TransportCall,
    StrictModel,
)

__all__ = ["RequestTransport", "RpcMethod", "TransportCall", "StrictModel"]
