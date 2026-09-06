from pathlib import Path
from typing import cast

import pytest

from agent.plugin_composition import (
    MCP_SERVERS,
    TOOL_CATALOG,
    WORKLOADS,
    CompositionRoot,
    PluginRuntime,
    PluginTools,
    PluginWorkloads,
)
from agent.plugin_composition.mcp_slots import (
    PluginMcpServers,
    _freeze_plugin_mcp_servers,
)
from agent.plugin_composition.tool_catalog import _freeze_plugin_tools
from agent.plugins.static_manifest import (
    load_static_plugin_manifest,
    validate_module_exports,
)
from plugins.computer import plugin


@pytest.mark.asyncio
async def test_computer_plugin_mounts_with_static_manifest(tmp_path: Path) -> None:
    """真实 Root 验证静态身份、MCP 声明及 Emit listener，避免只测假 Context。"""
    path = Path(plugin.__file__).parent
    manifest = load_static_plugin_manifest(path)
    validate_module_exports(manifest, plugin, plugin_root=path)
    root = CompositionRoot("computer-test")
    mcp = PluginMcpServers(root.instance_token)
    tools = PluginTools(root.instance_token)
    for key, value in [
        (MCP_SERVERS, mcp),
        (TOOL_CATALOG, tools),
        (WORKLOADS, PluginWorkloads(root.instance_token)),
    ]:
        await root.context.provide(key, value)
    try:
        await root.mount(
            lambda ctx: plugin.apply(ctx, {}),
            name="computer",
            inject=plugin.inject,
            runtime=PluginRuntime(
                plugin_id="computer",
                generation_id="computer-test",
                plugin_dir=path,
                data_dir=tmp_path / "plugin-data",
                workspace=tmp_path,
                config={},
            ),
        )
        registry = _freeze_plugin_mcp_servers(mcp, root.instance_token)
        binding = next(iter(registry.values()))
        assert dict(binding.definition.env) == {}
        assert binding.definition.required_tools == ()
        assert binding.definition.candidate_read_only_tools == ()
        assert len(registry) == 1
        catalog = _freeze_plugin_tools(
            tools, root.instance_token, {"computer": "computer-test"}
        )
        assert len(catalog) == 1
        from types import SimpleNamespace

        from agent.plugin_composition.workload_slots import _freeze_plugin_workloads
        from agent.plugins.generation import PluginGeneration
        from agent.plugins.manager import _validate_static_manifest_runtime
        from agent.plugins.snapshot import RuntimeSnapshot

        workloads = _freeze_plugin_workloads(
            root.context.get(WORKLOADS), root.instance_token
        )
        snapshot = RuntimeSnapshot(
            "static",
            {},
            None,
            composition_active_plugin_ids=frozenset({"computer"}),
            mcp_server_registry=registry,
            workload_registry=workloads,
        )
        _validate_static_manifest_runtime(
            snapshot,
            {
                # 静态校验只使用这份 generation 的 manifest、目录和派生命令字段。
                "computer": cast(
                    PluginGeneration,
                    SimpleNamespace(static_manifest=manifest, plugin_dir=path),
                )
            },
        )
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_computer_control_against_container(tmp_path: Path) -> None:
    """真实 MCP 子进程、调用上下文、截图文件和取消回执；不启动 Agent Loop。"""
    import asyncio
    import json
    import os
    import sys

    import httpx

    from agent.tools.base import ToolExecutionContext
    from plugins.computer.control import endpoint_name

    gateway = os.environ.get("COMPUTER_TEST_GATEWAY")
    if not gateway:
        pytest.skip("requires a disposable Computer container")
    path = Path(plugin.__file__).parent
    data_root = tmp_path / "plugin-data"
    data_root.mkdir()
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        str(path / "mcp_server.py"),
        env={
            **os.environ,
            "COMPUTER_URL": gateway,
            "AKA_PLUGIN_DATA_DIR": str(data_root),
        },
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    root = CompositionRoot("computer-integration")
    tools = PluginTools(root.instance_token)
    for key, value in [
        (MCP_SERVERS, PluginMcpServers(root.instance_token)),
        (TOOL_CATALOG, tools),
        (WORKLOADS, PluginWorkloads(root.instance_token)),
    ]:
        await root.context.provide(key, value)
    try:
        child.stdin.write(
            b'{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}\n'
        )
        await child.stdin.drain()
        assert (
            json.loads(await child.stdout.readline())["result"]["serverInfo"]["name"]
            == "akashic-computer"
        )
        await root.mount(
            lambda ctx: plugin.apply(ctx, {}),
            name="computer",
            inject=plugin.inject,
            runtime=PluginRuntime(
                plugin_id="computer",
                generation_id="computer-integration",
                plugin_dir=path,
                data_dir=data_root,
                workspace=tmp_path,
                config={},
            ),
        )
        catalog = _freeze_plugin_tools(
            tools, root.instance_token, {"computer": "computer-integration"}
        )
        handler = next(iter(catalog.values())).handler

        def context(call: str) -> ToolExecutionContext:
            return ToolExecutionContext(
                origin_session_key="real-context",
                turn_id="real-turn",
                execution_id=call,
            )

        async def run(call: str, code: str) -> str:
            """确认通用工具接口返回 Computer 的 JSON 文本，并提供可取消的协程。"""
            value = await handler(context(call), {"code": code})
            assert isinstance(value, str)
            return value

        output = json.loads(
            await run(
                "first",
                "nodeRepl.write(41+1); await nodeRepl.emitImage((await sky.get_screenshot())[0].bytes);",
            )
        )
        assert "42" in json.dumps(output)
        assert list((data_root / "screenshots").glob("*"))

        async def check_screenshot_read(result):
            """用真实截图和既有模型夹具验证图片直传及文字模型提示。"""
            import base64
            from io import BytesIO
            from types import SimpleNamespace

            from PIL import Image

            from agent.tool_runtime import append_tool_result
            from agent.tools.base import ToolResult
            from agent.tools.filesystem import ReadFileTool
            from tests.model_plugin_fakes import bind_test_model_snapshot

            references = [
                json.loads(item["text"])
                for item in result["content"]
                if item["type"] == "text" and item["text"].startswith("{")
            ]
            reference = next(x for x in references if x.get("kind") == "screenshot_file")
            assert "read_file" in reference["next"]
            screenshot = Path(reference["path"])
            assert screenshot.is_relative_to(data_root)
            for modalities in [("text", "image"), ("text",)]:
                async with bind_test_model_snapshot(
                    SimpleNamespace(input_modalities=modalities)
                ):
                    read = await ReadFileTool(enable_bridge=False).execute(str(screenshot))
                messages = []
                append_tool_result(
                    messages, tool_call_id="read-screenshot", content=read,
                    tool_name="read_file",
                )
                if "image" in modalities:
                    assert isinstance(read, ToolResult)
                    block = messages[-1]["content"][1]
                    assert block["type"] == "image_url"
                    image = base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
                    with Image.open(BytesIO(image)) as viewed, Image.open(screenshot) as saved:
                        assert viewed.size == saved.size
                else:
                    assert isinstance(read, str)
                    assert "read_image_vision" in read
                    assert len(messages) == 1
                    assert isinstance(messages[0]["content"], str)

        # 1. 唯一 JS 入口保留读图合同；实际 MCP discovery 不再发布旧工具。
        await check_screenshot_read(output)
        child.stdin.write(b'{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n')
        await child.stdin.drain()
        reply = json.loads(await child.stdout.readline())
        assert reply["result"]["tools"] == []
        child.stdin.write(
            b'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"browser_action","arguments":{"action":"navigate","url":"about:blank"}}}\n'
        )
        await child.stdin.drain()
        reply = json.loads(await child.stdout.readline())
        assert reply["error"]["code"] == -32601

        # 2. 保留既有取消及 Turn 收尾验证。
        reader, writer = await asyncio.open_unix_connection(
            "\0" + endpoint_name(data_root)
        )
        writer.write(
            json.dumps(
                {
                    "op": "run",
                    "context": {
                        "session_id": "cancel",
                        "turn_id": "turn",
                        "call_id": "held",
                    },
                    "code": "var h=sky.drag_handle(); await h.start({x:300,y:300}); while(true){}",
                    "timeoutMs": 30000,
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()
        # 服务端在读取 run 后开始等待同一连接的 cancel；不依赖短 sleep 猜测输入状态。
        writer.write(b'{"cancel":true}\n')
        await writer.drain()
        receipt = json.loads(await asyncio.wait_for(reader.readline(), 20))
        assert receipt == {"cancelled": True, "released": True, "effects": "may_remain"}
        writer.close()
        await writer.wait_closed()
        assert "healthy" in await run(
            "after", "nodeRepl.write('healthy');"
        )
        async with httpx.AsyncClient(base_url=gateway) as client:
            running = asyncio.create_task(
                run("task-cancel", "await new Promise(()=>{});")
            )
            async with asyncio.timeout(10):
                while not (await client.get("/activity")).json()["active"]:
                    pass
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(running, 30)
        assert "after cancel" in await run(
            "post-cancel", "nodeRepl.write('after cancel');"
        )
        async with httpx.AsyncClient(base_url=gateway) as client:
            new = await client.post(
                "/browser/action", json={"action": "tab_new", "url": "about:blank"}
            )
            new.raise_for_status()
            closed = await client.post(
                "/browser/action",
                json={"action": "tab_close", "target_id": new.json()["target_id"]},
            )
            closed.raise_for_status()
        scratch = json.loads(
            await run(
                "scratch",
                "nodeRepl.write((await browser.tabs.new()).id);",
            )
        )["content"][0]["text"]
        from agent.plugin_composition import RUNTIME_STOPPING, RuntimeStopping
        from agent.plugins.snapshot import (
            RuntimeSnapshot,
            RuntimeSnapshotStore,
            bind_runtime_snapshot,
            reset_runtime_snapshot,
        )
        from agent.turn_events.after_turn import AFTER_TURN_COMMITTED
        from bus.events_lifecycle import TurnCommitted

        store = RuntimeSnapshotStore()
        store.install(
            RuntimeSnapshot(
                "computer-event",
                {},
                None,
                composition_root=root,
                composition_topology=root.topology_view(),
            )
        )
        lease = store.lease()
        token = bind_runtime_snapshot(lease)
        try:
            root.context.emit(
                AFTER_TURN_COMMITTED,
                TurnCommitted(
                    session_key="real-context",
                    turn_id="real-turn",
                    channel="test",
                    chat_id="test",
                    input_message="test",
                    persisted_user_message="test",
                    assistant_response="done",
                    tools_used=["computer"],
                ),
            )
            await root.context.serial(RUNTIME_STOPPING, RuntimeStopping())
        finally:
            reset_runtime_snapshot(token)
            await lease.release()
            await store.close()
        async with httpx.AsyncClient(base_url=gateway) as client:
            listing = await client.post(
                "/driver/run",
                json={
                    "context": {
                        "session_id": "inspect",
                        "turn_id": "inspect",
                        "call_id": "inspect",
                    },
                    "code": "nodeRepl.write(await browser.tabs.list());",
                },
            )
            listing.raise_for_status()
            assert scratch not in listing.text
    finally:
        await root.dispose()
        child.stdin.close()
        await asyncio.wait_for(child.wait(), 25)
        assert child.returncode == 0, (await child.stderr.read()).decode()
