"""正式安装的业务组合经真实 HTTP driver/sender 写入最终送达回执。"""

import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import shutil
import socket
import subprocess

from aiohttp import web
import pytest

from agent.config_models import Config
from agent.plugin_composition import (
    AddConnection, AddModel, CapabilitySources, ModelCapabilities, ModelKind,
    ServiceKey, SetDefaultModel,
)
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugin_contracts import Input, Output
from agent.plugins.install import install_git_plugin
from agent.plugins.model_control import RuntimeModelControl
from agent.plugins.snapshot import lease_runtime_snapshot
from bootstrap.init_workspace import init_workspace
from bootstrap.tools import build_core_runtime
from core.net.http import SharedHttpResources
from session.log import MessageLog


@pytest.mark.asyncio
async def test_installed_reply_reaches_sender_and_durable_receipt(tmp_path, monkeypatch):
    """原插件源码移走后，输入、模型、回复和发送只使用正式安装的实现。"""
    # 1. 外部边界使用 loopback 协议服务；业务插件全部使用真实发布实现。
    requests = []
    deliveries = []

    async def models(_request):
        return web.json_response({"data": [{"id": "fixture"}]})

    async def complete(request):
        requests.append(await request.json())
        chunks = [
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "installed reply"},
                          "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
        ]
        return web.Response(
            text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n",
            content_type="text/event-stream",
        )

    async def send(request):
        payload = await request.json()
        deliveries.append(payload)
        return web.json_response({"ok": True, "result": {"message_id": 731}})

    server = web.Application()
    server.router.add_get("/v1/models", models)
    server.router.add_post("/v1/chat/completions", complete)
    server.router.add_post("/botfixture-token/sendMessage", send)
    runner = web.AppRunner(server)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    endpoint = f"http://127.0.0.1:{sock.getsockname()[1]}"
    await web.SockSite(runner, sock).start()

    workspace = tmp_path / "workspace"
    home = tmp_path / "plugin-home"
    source_root = tmp_path / "package-sources"
    source_backup = tmp_path / "package-sources.before-acceptance"
    core = None
    http = SharedHttpResources()
    try:
        init_workspace(config_path=tmp_path / "config.toml", workspace=workspace)
        monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(home))
        names = (
            "sources", "content", "context", "tools", "conversation", "react",
            "turn_projection", "reply_program", "reply", "tool_search", "models",
            "openai_compatible", "standard_tools", "delivery", "delivery_policy", "telegram_sender",
        )
        installed = {}
        for name in names:
            source = source_root / name
            shutil.copytree(Path(__file__).parents[1] / "plugins" / name, source,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            subprocess.run(["git", "init", "--quiet", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "add", "."], check=True)
            subprocess.run([
                "git", "-C", str(source), "-c", "user.name=acceptance",
                "-c", "user.email=acceptance@example.invalid", "-c", "commit.gpgSign=false",
                "commit", "--quiet", "-m", "fixed acceptance source",
            ], check=True)
            installed[name] = install_git_plugin(
                workspace=workspace, source=str(source), marketplace="acceptance", plugins_home=home,
            )
        sender_config = installed["telegram_sender"].data_path / "config.local.toml"
        sender_config.write_text(
            f'enabled=true\ntoken="fixture-token"\napi_base="{endpoint}"\ntimeout_seconds=3\n',
        )
        (installed["context"].data_path / "config.local.toml").write_text(
            'summary_source=[]\nprompt_sources={skills="standard_tools@acceptance"}\n'
        )
        # 保留恢复点；运行不再访问生成这些安装的原始源码位置。
        source_root.rename(source_backup)
        core = build_core_runtime(Config(), workspace, http, plugin_dirs=[])
        await core.start()
        host = core.plugin_manager
        assert set(host.current_snapshot.generations) == {
            f"{item.plugin_name}@{item.marketplace}" for item in installed.values()
        }
        for name in names:
            item = installed[name]
            generation = host.generation(f"{item.plugin_name}@{item.marketplace}")
            assert generation is not None
            descriptor = host._archive.read_descriptor(generation.archive_ref)
            archive_root = host._archive.open(descriptor["code"])
            assert Path(generation.instance.module.__file__).is_relative_to(archive_root)

        # 2. 用实际模型配置入口绑定 HTTP driver；没有替换业务能力或实际执行函数。
        control = RuntimeModelControl(host.snapshot_store)
        await control.apply(AddConnection(
            0, "local", "Local", "openai-compatible", endpoint + "/v1",
            "fixture", {"api_key": "fixture"},
        ))
        await control.apply(AddModel(
            1, "fixture", "local", ModelKind.CHAT, "fixture",
            ModelCapabilities(context_window=32000, max_output_tokens=8192, supports_tool_calls=True),
            CapabilitySources(),
        ))
        await control.apply(SetDefaultModel(2, "default", "fixture"))
        await host.start_runtime()
        async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
            accepted = await snapshot.composition_root.context.require(CHANNEL_INPUT)(
                "telegram:123", "input-1", ChannelInboundMessage(
                    "telegram", "user", "123", "hello", datetime.now(UTC), {},
                ),
            )
        assert isinstance(accepted.body, Input)

        # 3. 跟随消息后等待实际 delivery owner 完成，不能把 HTTP 200 当持久回执。
        async def confirmed():
            async for _ in core.message_log.catalog().follow():
                rows = core.message_log.reader("telegram:123").snapshot()
                output = next((row for row in rows if isinstance(row.body, Output)
                               and row.body.finish == "complete"), None)
                if output is None:
                    continue
                async with lease_runtime_snapshot(host.snapshot_store) as snapshot:
                    ctx = snapshot.composition_root.context
                    turn = ctx.require(ServiceKey("turn.projection.v1")).project(rows, "conversation")[-1]
                    await ctx.require(ServiceKey("delivery.final_output.v1")).wait(
                        core.message_log.reader("telegram:123"), turn,
                    )
                    history = ctx.require(ServiceKey("delivery.read.v1"))
                    receipt = history.status(output.message_id, "telegram")
                    assert receipt is not None and receipt["status"] == "delivered"
                    return rows, receipt
            raise AssertionError("日志在送达之前关闭")

        rows, receipt = await asyncio.wait_for(confirmed(), 15)
        assert len(requests) == 1
        assert len(deliveries) == 1
        assert str(deliveries[0]["chat_id"]) == "123"
        assert deliveries[0]["text"] == "installed reply"
        assert receipt["recipient"] == "123"
        assert receipt["receipt"]["provider_ids"] == ["731"]
        assert [type(row.body) for row in rows] == [Input, Output]
        assert rows[0] == accepted
        assert any(part.kind == "text" and part.value == "installed reply" for part in rows[1].body.parts)
        await core.stop()
        core = None
        reopened = MessageLog(workspace / "sessions.db")
        try:
            assert reopened.reader("telegram:123").snapshot() == rows
            saved = reopened.owner("plugin:delivery@acceptance").scan(start="delivery:", stop="delivery;")
            assert len(saved) == 1
            assert saved[0][1].value["phase"] == "delivered"
            assert saved[0][1].value["receipt"]["provider_ids"] == ("731",)
        finally:
            reopened.close()
    finally:
        try:
            if core is not None:
                await core.stop()
        finally:
            await http.aclose()
            await runner.cleanup()
            if source_backup.exists():
                source_backup.rename(source_root)
