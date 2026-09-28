"""一次性组合验证：真实 Web listener 与 Channel 请求的可选能力寿命。"""
from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from pydantic import BaseModel

from agent.plugin_composition import CompositionError, CompositionRoot
from agent.plugin_composition.channel_io import (
    CHANNEL_ATTACHMENT_IMPORT, CHANNEL_ATTACHMENT_READ, CHANNEL_IDENTITY, INPUT_CUSTODY,
    ChannelAttachmentImport, ChannelAttachmentRead, ChannelIdentity,
    unavailable, unavailable_input_custody,
)
from agent.plugin_composition.channels import CHANNEL_INPUT, CHANNELS
from agent.plugin_composition.host import HOST_INFO, HostInfo
from agent.plugin_composition.model import FiberState, PluginRuntime, ServiceKey
from agent.plugin_composition.rpc import RpcMethod
from plugins.akashic_clients import plugin as clients
from plugins.akashic_clients.capabilities import (
    CLIENT_CAPABILITIES, INSPECTION_DOCUMENTS_LIST, INSPECTION_RPC_KEYS, MODEL_RPC_KEYS,
)
from plugins.channels import plugin as channels
from plugins.runtime_inspection import plugin as inspection


class EmptyParams(BaseModel):
    pass


async def install_ports(root: CompositionRoot, workspace: Path):
    """为真实 Channel 场景安装端口；未使用的业务调用一律拒绝。"""
    def runtime(name: str, files: tuple[str, ...] = ()) -> PluginRuntime:
        return PluginRuntime(name, name + "-generation", workspace, workspace / name,
                             workspace, {}, workspace_files=files)

    # 1. 真实 Channel provider；未使用的外部能力是拒绝端口，不伪造回复成功。
    await root.context.provide(HOST_INFO, HostInfo("scenario", False))
    await root.context.provide(INPUT_CUSTODY, unavailable_input_custody())
    await root.context.provide(CHANNEL_IDENTITY, ChannelIdentity(unavailable, unavailable, unavailable))
    await root.context.provide(CHANNEL_ATTACHMENT_IMPORT, ChannelAttachmentImport(unavailable))
    await root.context.provide(CHANNEL_ATTACHMENT_READ, ChannelAttachmentRead(unavailable, unavailable))
    await root.context.provide(CHANNEL_INPUT, unavailable)
    for key in CLIENT_CAPABILITIES:
        if key not in (*INSPECTION_RPC_KEYS, *MODEL_RPC_KEYS):
            await root.context.provide(key, unavailable)
    return runtime


async def check(workspace: Path) -> None:
    """只调用健康和检查接口；模型、入站与外部发送均明确拒绝。"""
    root = CompositionRoot("optional-request-scenario")
    (workspace / "runtime").mkdir()

    runtime = await install_ports(root, workspace)
    try:
        await root.mount(channels.apply, name="channels", inject=channels.inject, runtime=runtime("channels"))
        chat = await root.mount(clients.apply, name="clients", inject=clients.inject, runtime=runtime("clients"))
        assert chat.state is FiberState.ACTIVE, chat.error
        host = root.context.require(CHANNELS)
        state = next(iter(host._bindings.values()))
        adapter = state.adapter
        token = chat.context.fiber.activation_token
        socket = workspace / "runtime/web-chat.sock"
        async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=str(socket)), base_url="http://local") as http:
            assert (await http.get("/api/chat/health")).status_code == 200
            missing = await http.get("/api/chat/runtime/documents")
            assert missing.status_code == 503, missing.text
            assert missing.json() == {"detail": "运行检查能力不可用"}

            # 2. 安装与卸载真实检查插件，不重建聊天 adapter 或 activation。
            provider = await root.mount(inspection.apply, name="inspection", inject=inspection.inject,
                                        runtime=runtime("inspection"))
            assert (await http.get("/api/chat/runtime/documents")).status_code == 200
            await provider.dispose()
            assert (await http.get("/api/chat/runtime/documents")).status_code == 503
            assert (await http.get("/api/chat/health")).status_code == 200
            assert state.adapter is adapter and chat.context.fiber.activation_token is token

            # 3. 已借用请求阻止 provider 释放，完成后继续保持聊天。
            entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()

            async def slow_provider(ctx):
                async def read(_params):
                    entered.set()
                    await release.wait()
                    assert not closed.is_set()
                    return {"items": []}
                await ctx.effect(lambda: closed.set)
                await ctx.provide(INSPECTION_DOCUMENTS_LIST, RpcMethod(EmptyParams, read))

            slow = await root.mount(slow_provider, name="slow-inspection")
            request = asyncio.create_task(http.get("/api/chat/runtime/documents"))
            await asyncio.wait_for(entered.wait(), 5)
            disposing = asyncio.create_task(slow.dispose())
            barrier = asyncio.Event()
            asyncio.get_running_loop().call_soon(barrier.set)
            await barrier.wait()
            assert not closed.is_set() and not disposing.done()
            release.set()
            assert (await request).status_code == 200
            await disposing
            assert closed.is_set()
            assert (await http.get("/api/chat/health")).status_code == 200

        # 4. 未声明、跨 Task 和 scope 退出后的借用必须失败。
        async with adapter._context.open_scope() as scope:
            try:
                with scope.borrow(ServiceKey("scenario.undeclared")):
                    raise AssertionError("未声明能力被允许")
            except CompositionError as error:
                assert error.code == "SERVICE_UNDECLARED"

            async def wrong_task():
                with scope.borrow(INSPECTION_DOCUMENTS_LIST):
                    raise AssertionError("跨 Task 借用被允许")
            try:
                await asyncio.create_task(wrong_task())
            except CompositionError as error:
                assert error.code == "REQUEST_SCOPE_MISSING"
        try:
            with scope.borrow(INSPECTION_DOCUMENTS_LIST):
                raise AssertionError("过期 scope 借用被允许")
        except CompositionError as error:
            assert error.code == "REQUEST_SCOPE_MISSING"
    finally:
        await root.dispose()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-optional-") as directory:
        asyncio.run(check(Path(directory)))
    print("PASS: missing/install/uninstall/in-flight drain/scope boundary; no model or external delivery")
