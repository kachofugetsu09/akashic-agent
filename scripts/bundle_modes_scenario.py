"""用真实发行制品验收三种组合、关闭和 headless 命令回复。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def headless_reply(app, workspace: Path, config: Path) -> None:
    """真实 OpenAI-compatible driver 访问隔离 HTTP 模型，回复走完整 CLI 链路。"""
    from plugins.gateway.contract import RpcMethod
    requests = []

    async def respond(reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            headers = dict(line.split(b":", 1) for line in head.split(b"\r\n")[1:] if b":" in line)
            raw = await reader.readexactly(int(headers.get(b"Content-Length", headers.get(b"content-length", b"0"))))
            payload = json.loads(raw) if raw else {}
            requests.append(payload)
            message = {"role": "assistant", "content": "bundle reply"}
            if head.startswith(b"GET /v1/models "):
                body = json.dumps({"object": "list", "data": [{"id": "scenario", "object": "model"}]}).encode()
                content_type = "application/json"
            elif payload.get("stream"):
                chunks = [
                    {"id": "scenario", "choices": [{"index": 0, "delta": message, "finish_reason": None}]},
                    {"id": "scenario", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                ]
                body = ("".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
                content_type = "text/event-stream"
            else:
                body = json.dumps({"id": "scenario", "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}).encode()
                content_type = "application/json"
            writer.write(f"HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(respond, "127.0.0.1", 0)
    async with server:
        port = server.sockets[0].getsockname()[1]
        root = app.core.plugin_manager.live_root
        method = root.service_value(RpcMethod.key("models/command"))
        async def command(values):
            result = await method.call(method.params.model_validate(values))
            assert result["status"] == 200, result
            return result["body"]["revision"]
        revision = await command({"type": "add_connection", "expected_revision": 0,
            "connection_id": "local", "name": "Local scenario", "driver_id": "openai-compatible",
            "endpoint": f"http://127.0.0.1:{port}/v1", "auth_identity": "scenario",
            "credential": {"api_key": "scenario"}, "driver_config": {"allow_unverified_manual": True}})
        revision = await command({"type": "add_model", "expected_revision": revision,
            "model_id": "local", "connection_id": "local", "kind": "chat", "model": "scenario",
            "capabilities": {"context_window": 131072, "max_output_tokens": 4096, "supports_tool_calls": True}, "capability_sources": {}})
        await command({"type": "set_default", "expected_revision": revision, "role": "default", "model_id": "local"})
        process = await asyncio.create_subprocess_exec(sys.executable, str(ROOT / "main.py"),
            "exec", "--new", "--session", "programmatic:bundle", "--final-only", "hello bundle",
            "--config", str(config), "--workspace", str(workspace),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            output, errors = await asyncio.wait_for(process.communicate(), 60)
            assert process.returncode == 0 and output.decode().strip() == "bundle reply", (output, errors)
            assert requests and any("hello bundle" in json.dumps(request) for request in requests)
        finally:
            if process.returncode is None:
                process.kill()
                await process.communicate()


async def run(distribution: Path, base: Path, mode: str) -> None:
    """从正式安装入口提交选择，再以 AppRuntime 启停完整宿主。"""
    from agent.config import Config
    from agent.plugins.bundles import distribution_bundle
    from agent.plugins.selection import PluginSelection
    from agent.plugin_composition import FiberState
    from bootstrap.app import AppRuntime
    from bootstrap.init_workspace import init_workspace
    from scripts.install_plugin_distribution import ensure_bundle

    base.mkdir()
    workspace, home, config = base / "workspace", base / "home", base / "config.toml"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home),
        AKASHIC_PLUGIN_DISTRIBUTION=str(distribution), AKASHIC_PLUGIN_BUNDLE=mode,
        AKASHIC_EXTRA_PLUGIN_DIRS="", AKASHIC_EXECUTION_MODE="local",
        PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "sdk/python/src"))))
    os.environ.pop("AKASHIC_WEB_PORT", None)
    config.write_text("[runtime]\n")
    init_workspace(config_path=config, workspace=workspace)
    ensure_bundle(distribution, distribution / "bundles" / f"{mode}.toml",
        workspace=workspace, plugins_home=home, config_path=config,
        receipt_path=workspace / "runtime/distribution-install.json")
    selection = PluginSelection(workspace)
    reference = selection.read()
    assert reference is not None, "空组合也需要明确的持久选择"
    chosen = {selection.read_input(item)["plugin_id"] for item in selection.components(reference)}
    expected = {row.plugin for row in distribution_bundle(distribution / "bundles") if not row.disabled}
    assert chosen == expected, (chosen - expected, expected - chosen)
    app = AppRuntime(Config.load(config, workspace=workspace), workspace)
    try:
        await app.start()
        manager = app.core.plugin_manager
        assert set(manager._active_generations) == expected
        assert manager.live_root is not None
        assert all(item.state is not FiberState.FAILED for item in manager.live_root.fibers())
        if mode == "minimal":
            assert not expected
            assert not json.loads((workspace / "runtime/endpoints.json").read_text())["endpoints"]
        if mode == "headless":
            assert manager.generation("ui@release") is None
            assert manager.generation("channels@release") is None
            await headless_reply(app, workspace, config)
    finally:
        await app.shutdown()
    assert selection.read() == reference
    assert not json.loads((workspace / "runtime/endpoints.json").read_text())["endpoints"]
    assert not manager._active_generations
    print(json.dumps({"mode": mode, "selected": len(expected), "clean_shutdown": True,
                      "cli_reply": mode == "headless"}), flush=True)


if __name__ == "__main__":
    distribution = Path(sys.argv[1]).resolve(strict=True)
    if len(sys.argv) == 4:
        asyncio.run(run(distribution, Path(sys.argv[2]), sys.argv[3]))
    else:
        base = Path(tempfile.mkdtemp(prefix="akashic-bundle-modes-"))
        for mode in ("minimal", "headless", "base"):
            subprocess.run([sys.executable, __file__, str(distribution), str(base / mode), mode], check=True)
        print(json.dumps({"evidence": str(base), "modes": ["minimal", "headless", "base"]}))
