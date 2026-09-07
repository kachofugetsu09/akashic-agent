"""真实 ModelsState execution 的代际、重试和 task 绑定合同。"""

from __future__ import annotations

import asyncio
import json
import socket

import pytest
from aiohttp import web

from agent.config_models import Config
from agent.plugin_composition import (
    AddConnection,
    AddModel,
    CapabilitySources,
    CHAT_MODELS,
    ModelCapabilities,
    ModelKind,
    ModelRole,
    ModelRequest,
    SetDefaultModel,
)
from agent.plugins.model_control import RuntimeModelControl
from agent.plugins.snapshot import lease_runtime_snapshot
from bootstrap import tools as bootstrap
from bootstrap.init_workspace import init_workspace
from core.net.http import SharedHttpResources


@pytest.mark.asyncio
async def test_model_execution_pins_binding_and_rejects_child_task_inheritance(
    tmp_path, monkeypatch,
):
    """模型 execution 固定一次 descriptor，不能被默认切换或子 task 改写。"""

    calls: list[dict[str, object]] = []

    async def models(_request):
        return web.json_response({"data": [{"id": "first"}, {"id": "second"}]})

    async def completions(request):
        body = await request.json()
        calls.append(body)
        chunks = [
            {
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None},
                ],
            },
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ]
        payload = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        return web.Response(text=payload + "data: [DONE]\n\n", content_type="text/event-stream")

    provider = web.Application()
    provider.router.add_get("/v1/models", models)
    provider.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(provider)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()

    workspace = tmp_path / "workspace"
    init_workspace(config_path=tmp_path / "config.toml", workspace=workspace)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(), workspace, http)
    try:
        await core.start()
        await core.plugin_manager.start_runtime()
        control = RuntimeModelControl(core.plugin_manager.snapshot_store)
        await control.apply(
            AddConnection(
                0, "local", "Local", "openai-compatible",
                f"http://127.0.0.1:{port}/v1", "fixture", {"api_key": "fixture"},
            )
        )
        capabilities = ModelCapabilities(
            context_window=32000,
            max_output_tokens=1024,
            supports_tool_calls=True,
            supported_reasoning_efforts=("low", "high"),
        )
        await control.apply(
            AddModel(1, "first", "local", ModelKind.CHAT, "first", capabilities, CapabilitySources())
        )
        await control.apply(SetDefaultModel(2, ModelRole.DEFAULT, "first"))

        async with lease_runtime_snapshot(core.plugin_manager.snapshot_store) as snapshot:
            context = snapshot.composition_root.context
            models_service = context.require(CHAT_MODELS)
            async with models_service.execution() as first_execution:
                first_descriptor = first_execution.chat(ModelRole.AGENT).descriptor
                assert first_descriptor.model_id == "first"
                response = await first_execution.chat(ModelRole.AGENT).complete(
                    ModelRequest(
                        messages=({"role": "user", "content": "first"},),
                        on_delta=lambda _delta: _noop_delta(),
                    )
                )
                assert response.content == "ok"
                assert calls[-1]["model"] == "first"

                async with models_service.execution() as nested:
                    assert nested is first_execution
                    assert nested.chat(ModelRole.AGENT).descriptor == first_descriptor

                await control.apply(
                    AddModel(3, "second", "local", ModelKind.CHAT, "second", capabilities, CapabilitySources())
                )
                await control.apply(SetDefaultModel(4, ModelRole.DEFAULT, "second"))

                # 已打开的 execution 固定旧 descriptor；默认切换只影响下一次 execution。
                async with models_service.execution() as still_first:
                    assert still_first is first_execution
                    assert still_first.chat(ModelRole.AGENT).descriptor == first_descriptor
                await first_execution.chat(ModelRole.AGENT).complete(
                    ModelRequest(
                        messages=({"role": "user", "content": "retry"},),
                        on_delta=lambda _delta: _noop_delta(),
                    )
                )
                assert calls[-1]["model"] == "first"
                assert first_execution.chat(ModelRole.AGENT).descriptor == first_descriptor

                lease_count = snapshot.lease_count

                async def child_execution() -> None:
                    with pytest.raises(RuntimeError, match="不能由子 task 继承"):
                        async with models_service.execution():
                            raise AssertionError("子 task 不应取得 model execution")

                await asyncio.create_task(child_execution())
                assert snapshot.lease_count == lease_count
                assert len(calls) == 2

            async with models_service.execution() as second_execution:
                second_descriptor = second_execution.chat(ModelRole.AGENT).descriptor
                assert second_descriptor.model_id == "second"
                assert second_descriptor.binding_id != first_descriptor.binding_id
                await second_execution.chat(ModelRole.AGENT).complete(
                    ModelRequest(
                        messages=({"role": "user", "content": "second"},),
                        on_delta=lambda _delta: _noop_delta(),
                    )
                )

        assert [call["model"] for call in calls] == ["first", "first", "second"]
    finally:
        await core.bus.aclose()
        await core.stop()
        await http.aclose()
        await runner.cleanup()


async def _noop_delta() -> None:
    """为真实流式 provider 请求提供最小 delta consumer。"""
