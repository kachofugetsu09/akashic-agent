"""控制传输的插件入口；方法名称、参数和行为由插件声明。"""
from agent.control.protocol.method import RequestTransport, RpcMethod
from agent.control.protocol.models import METHOD_PARAMS, StrictModel
from agent.plugin_composition.model import ServiceKey


def rpc_method_key(name: str) -> ServiceKey[RpcMethod]:
    """每个方法使用既有 Service 生命周期，不能覆盖宿主保留入口。"""
    if not name or name.strip() != name:
        raise ValueError("RPC 方法名称必须是非空且没有首尾空白的字符串")
    if name in METHOD_PARAMS:
        raise ValueError(f"控制方法已经存在: {name}")
    return ServiceKey[RpcMethod]("control.rpc:" + name)


__all__ = ["RequestTransport", "RpcMethod", "StrictModel", "rpc_method_key"]
