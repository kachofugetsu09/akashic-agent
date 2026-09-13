"""Request-scoped model RPC projection for the ordinary client plugin."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from agent.plugin_composition.rpc import RpcMethod
from agent.plugin_composition.requests import RequestContext

from .capabilities import MODEL_CALL, MODEL_CATALOG_RPC, MODEL_COMMAND, MODEL_DISCOVER
from .services import ModelControlUnavailable


class ScopedModelRpcControl:
    """Resolve one declared model RPC only during its request scope."""

    _METHODS = {
        "models/call_stats": MODEL_CALL,
        "models/catalog": MODEL_CATALOG_RPC,
        "models/discover": MODEL_DISCOVER,
        "models/command": MODEL_COMMAND,
    }

    def __init__(
        self,
        open_scope: Callable[[], AbstractAsyncContextManager[RequestContext]],
    ) -> None:
        if not callable(open_scope):
            raise TypeError("model RPC scope 必须可调用")
        self._open_scope = open_scope

    async def invoke_rpc(self, method: str, params: Mapping[str, object]) -> object:
        """Invoke one model RPC with a fresh exact capability scope."""

        key = self._METHODS.get(method)
        if key is None:
            raise ModelControlUnavailable(f"不支持的模型 RPC: {method}")
        async with self._open_scope() as scope:
            provider = scope.require(key)
            if not isinstance(provider, RpcMethod):
                raise ModelControlUnavailable(f"模型 RPC provider 类型无效: {method}")
            request = provider.params.model_validate(dict(params))
            return await provider.invoke(request, None)


__all__ = ["ScopedModelRpcControl"]
