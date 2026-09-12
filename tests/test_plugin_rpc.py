"""外部 RPC 的方法、参数和生命周期由实际 provider 拥有。"""
import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import cast

import pytest

from agent.control.protocol.router import ConnectionRouter
from agent.control.service import ControlService
from agent.plugin_composition.rpc import RpcMethod, StrictModel, rpc_method_key


class FirstParams(StrictModel):
    first: str


class SecondParams(StrictModel):
    second: int


@pytest.mark.asyncio
async def test_connection_uses_current_plugin_schema_and_keeps_inflight_method():
    entered, release = asyncio.Event(), asyncio.Event()
    active = []
    frames = []

    async def first(params):
        entered.set()
        await release.wait()
        assert active == ["first"]
        return params.first

    async def second(params):
        return params.second

    current = RpcMethod(FirstParams, first)

    @asynccontextmanager
    async def resolve(name):
        selected = current if name == "example/inspect" else None
        marker = "first" if selected is not None and selected.params is FirstParams else "second"
        active.append(marker)
        try:
            yield selected
        finally:
            active.remove(marker)

    async def send(frame):
        frames.append(frame)

    service = cast(ControlService, SimpleNamespace(
        methods={}, resolve_method=resolve, initialize=lambda _: {},
    ))
    router = ConnectionRouter(service, send)

    async def request(identity, method, params):
        await router.handle_line(json.dumps({"jsonrpc": "2.0", "id": identity,
            "method": method, "params": params}).encode())

    await request(1, "initialize", {"protocolVersion": "2.0",
        "clientInfo": {"name": "test", "version": "1"}})
    await router.handle_line(b'{"jsonrpc":"2.0","method":"initialized"}')
    pending = asyncio.create_task(request(2, "example/inspect", {"first": "old"}))
    try:
        await entered.wait()
        current = RpcMethod(SecondParams, second)
        await request(3, "example/inspect", {"second": 42})
        await request(4, "example/inspect", {"first": "stale schema"})
        release.set()
        await pending
        current = None
        await request(5, "example/inspect", {})
    finally:
        release.set()
        await pending
        await router.close()
    responses = {frame["id"]: frame for frame in frames}
    assert responses[2]["result"] == "old"
    assert responses[3]["result"] == 42
    assert responses[4]["error"]["code"] == -32602
    assert responses[5]["error"]["code"] == -32601
    assert active == []


def test_plugin_method_cannot_replace_host_management():
    with pytest.raises(ValueError, match="已经存在"):
        rpc_method_key("plugin/install")
