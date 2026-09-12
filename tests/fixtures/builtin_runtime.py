"""独立服务的启动合同；场景通过公开 SDK/HTTP 和磁盘观察运行结果。"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import sys

import httpx

from tests.fixtures.formal_plugins import FULL_RUNTIME_PLUGINS, install_formal_plugins

ROOT = Path(__file__).parents[2]


@asynccontextmanager
async def dashboard(root: Path, plugin: str) -> AsyncIterator[httpx.AsyncClient]:
    """从真实 Web bootstrap 取得当前 generation 身份，再访问该插件路由。"""
    address = json.loads((root / "fixture-connection.json").read_text())
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=address["chat"]), base_url="http://fixture") as chat:
        response = await chat.get("/api/chat/web-ui/bootstrap")
        response.raise_for_status()
        catalog = response.json()
    plugin_id = plugin if "@" in plugin else f"{plugin}@fixture"
    module = next(item for item in catalog["modules"] if item["pluginId"] == plugin_id)
    headers = {"x-akashic-web-snapshot": catalog["snapshotId"], "x-akashic-web-catalog": catalog["catalogId"],
               "x-akashic-web-module": module["pluginId"], "x-akashic-web-generation": module["generationId"]}
    async with httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=address["dashboard"]),
                                base_url="http://fixture", headers=headers) as web:
        yield web


async def configure_models(root: Path, endpoint: str, settings: dict) -> None:
    """从实际 HTTP 设置入口配置本地模型；重开已有 workspace 时保留用户设置。"""
    async with dashboard(root, "models") as web:
        catalog = await web.get("/api/dashboard/models/catalog")
        catalog.raise_for_status()
        if catalog.json()["roleBindings"]:
            return
        commands = [
            {"type": "add_connection", "connection_id": "fixture", "name": "Fixture",
             "driver_id": "openai-compatible", "endpoint": endpoint, "auth_identity": "fixture",
             "credential": {"api_key": "fixture"}},
            {"type": "add_model", "model_id": "fixture", "connection_id": "fixture", "kind": "chat", "model": "fixture",
             "capabilities": {"context_window": settings.get("context_window", 64000), "max_output_tokens": 4096,
                              "supports_tool_calls": True}, "capability_sources": {}},
            {"type": "set_default", "role": "default", "model_id": "fixture"},
        ]
        if settings.get("embedding"):
            commands.extend([
                {"type": "add_model", "model_id": "embedding", "connection_id": "fixture", "kind": "embedding", "model": "embedding",
                 "capabilities": {"embedding_dimensions": 8, "embedding_normalization": "l2"}, "capability_sources": {}},
                {"type": "set_default", "role": None, "model_id": "embedding"},
            ])
        for revision, command in enumerate(commands):
            response = await web.post("/api/dashboard/models/command", json={"expected_revision": revision, **command})
            assert response.status_code == 200, response.text


@asynccontextmanager
async def runtime(root: Path, model_endpoint: str, *, settings: dict | None = None) -> AsyncIterator[tuple[asyncio.subprocess.Process, str]]:
    """启动独立 App 进程，保留 stderr，并只清理本次进程。"""
    root.mkdir(parents=True, exist_ok=True)
    if settings is not None:
        (root / "fixture-config.json").write_text(json.dumps(settings), encoding="utf-8")
    plugin_home, _ = install_formal_plugins(
        root, FULL_RUNTIME_PLUGINS, configure_materials=True,
        initialize_persona=True,
    )
    launch = json.loads(os.environ.get("AKASHIC_FIXTURE_COMMAND", "null"))
    command = ([sys.executable, "-m", "tests.fixtures.builtin_process", str(root), model_endpoint]
               if launch is None else [arg.replace("{root}", str(root)).replace("{model_endpoint}", model_endpoint)
                                       for arg in launch])
    environment = {key: value for key, value in os.environ.items() if not key.startswith("AKASHIC_")}
    environment.update({
      "AKASHIC_PLUGIN_HOME": str(plugin_home),
        "HOME": str(root / "home"),
        "XDG_CONFIG_HOME": str(root / "home/config"),
        "XDG_CACHE_HOME": str(root / "home/cache"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(ROOT),
    })
    with (root / "process.log").open("ab") as errors:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=ROOT, env=environment, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=errors,
        )
        try:
            assert process.stdout is not None
            async with asyncio.timeout(40):
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        await process.wait()
                        raise AssertionError((root / "process.log").read_text(errors="replace"))
                    if line.startswith(b'{"fixture_ready":'):
                        ready = json.loads(line)
                        assert ready["pid"] == process.pid
                        break
            (root / "fixture-connection.json").write_text(json.dumps(ready), encoding="utf-8")
            await configure_models(root, model_endpoint, settings or {})
            yield process, ready["endpoint"]
        finally:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), 20)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                    raise AssertionError("App 未能正常结束，已保留 process.log")
