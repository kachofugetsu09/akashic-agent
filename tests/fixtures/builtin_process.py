"""在独立进程启动真实内置 App；stdout 只额外发布测试连接地址。"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import signal
import sys

from agent.config_models import Config
from agent.plugin_composition import (
    AddConnection, AddModel, CapabilitySources, ModelCapabilities, ModelKind,
    ModelRole, SetDefaultModel,
)
from agent.plugins.model_control import RuntimeModelControl
from bootstrap.app import AppRuntime
from bootstrap.init_workspace import init_workspace
from bootstrap.runtime_readiness import RuntimeReadiness


async def serve(root: Path, endpoint: str) -> None:
    """使用正式启动和关闭路径，只在首次启动配置 loopback 模型。"""
    # 1. 所有运行数据和插件安装目录都在调用方的临时目录内。
    workspace = root / "workspace"
    if not (root / "config.toml").exists():
        init_workspace(config_path=root / "config.toml", workspace=workspace)
        settings = workspace / "plugin-data/reply-builtin/config.local.toml"
        settings.parent.mkdir(parents=True, exist_ok=True)
        settings.write_text('tools = ["write_file", "read_file", "message_push"]\n', encoding="utf-8")

    ready = asyncio.Event()
    stop = asyncio.Event()

    class Readiness(RuntimeReadiness):
        def mark_ready(self) -> None:
            super().mark_ready()
            ready.set()

    app = AppRuntime(Config(), workspace, readiness=Readiness(workspace, f"fixture-{os.getpid()}"))
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    loop.add_signal_handler(signal.SIGINT, stop.set)
    running = asyncio.create_task(app.run())
    waiting = asyncio.create_task(ready.wait())
    stopped = asyncio.create_task(stop.wait())
    try:
        # 2. readiness 必须来自真实 App，提前退出直接传播实际失败。
        done, _ = await asyncio.wait((running, waiting), timeout=30, return_when=asyncio.FIRST_COMPLETED)
        if running in done:
            await running
            raise RuntimeError("App 在 ready 前退出")
        if not ready.is_set():
            raise TimeoutError("App 未发布 readiness")
        assert app.core is not None and app.app_server is not None
        control = RuntimeModelControl(app.core.plugin_manager.snapshot_store)
        if not (await control.catalog()).role_bindings:
            await control.apply(AddConnection(0, "fixture", "Fixture", "openai-compatible",
                endpoint, "fixture", {"api_key": "fixture"}))
            await control.apply(AddModel(1, "fixture", "fixture", ModelKind.CHAT, "fixture",
                ModelCapabilities(context_window=64000, max_output_tokens=4096, supports_tool_calls=True),
                CapabilitySources()))
            await control.apply(SetDefaultModel(2, ModelRole.DEFAULT, "fixture"))
        print(json.dumps({"fixture_ready": True, "pid": os.getpid(),
                          "endpoint": str(app.app_server.endpoint)}), flush=True)
        done, _ = await asyncio.wait((running, stopped), return_when=asyncio.FIRST_COMPLETED)
        if running in done:
            await running
    finally:
        # 3. 正常结束通过 App.run 的 finally 清理；强杀由父测试显式执行。
        waiting.cancel()
        stopped.cancel()
        running.cancel()
        await asyncio.gather(waiting, stopped, return_exceptions=True)
        try:
            await running
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(serve(Path(sys.argv[1]), sys.argv[2]))
