"""在容器内启动真实宿主，模型配置走现有 RPC owner。"""
import asyncio
import json
from importlib.metadata import distributions
import os
from pathlib import Path
import signal
import sys
from typing import Any, cast

from runtime import ROOT
from agent.config import Config
from bootstrap.app import AppRuntime
from plugins.gateway.contract import RpcMethod


async def main(directory: Path, endpoint: str, model: str):
    """保存启动事实；SIGTERM 沿正式 shutdown 排空资源。"""
    state = directory / "state"
    workspace, config = state / "workspace", state / "config.toml"
    app = AppRuntime(Config.load(config, workspace=workspace), workspace)
    stopped = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stopped.set)
    try:
        # 1. 全量装配并检查真实 Fiber，失败不能冒充就绪。
        await app.start()
        root = app.core.plugin_manager.live_root
        fibers = [{"name": f.name, "state": str(f.state), "error": str(f.error)} for f in root.fibers()]
        (directory / "startup.json").write_text(json.dumps(fibers, indent=2))
        failed = [f for f in fibers if f["error"] != "None"]
        if failed:
            raise RuntimeError(f"插件启动失败: {failed}")
        method = root.service_value(RpcMethod.key("models/command"))

        async def command(values):
            result = cast(dict[str, Any], await method.call(method.params.model_validate(values)))
            if result["status"] != 200:
                raise RuntimeError(f"模型配置失败: {result}")
            return result["body"]["revision"]

        # 2. 凭据只进入本次私有状态，报告不记录配置正文。
        revision = await command({"type": "add_connection", "expected_revision": 0,
            "connection_id": "benchmark", "name": "MiniWoB", "driver_id": "openai-compatible",
            "endpoint": endpoint, "auth_identity": "miniwob",
            "credential": {"api_key": Path("/run/miniwob-key").read_text().strip()},
            "driver_config": {"allow_unverified_manual": True, "thinking_format": "deepseek"}})
        revision = await command({"type": "add_model", "expected_revision": revision,
            "model_id": "benchmark", "connection_id": "benchmark", "kind": "chat", "model": model,
            "capabilities": {"context_window": 1000000, "max_output_tokens": 8192,
                "input_modalities": ["text", "image"], "supports_tool_calls": True},
            "capability_sources": {}})
        await command({"type": "set_default", "expected_revision": revision,
                       "role": "default", "model_id": "benchmark"})
        (directory / "ready.json").write_text(json.dumps({"pid": os.getpid(), "python": sys.version,
            "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())}))
        await stopped.wait()
    finally:
        await app.shutdown()


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), sys.argv[2], sys.argv[3]))
