"""在真实调试镜像中运行完整入口、RPC、重启、exec 与 stdio。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.plugin_composition.config_input import load_config


def docker(arguments: list[str]) -> subprocess.CompletedProcess[bytes]:
    """保留真实退出码和输出；失败不替换镜像或模拟成功。"""
    result = subprocess.run(["docker", *arguments], capture_output=True, timeout=60)
    try:
        result.check_returncode()
    except subprocess.CalledProcessError as error:
        error.add_note(result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace"))
        raise
    return result


async def exercise(image: str, sandbox: Path) -> dict[str, bool]:
    """一次性 sandbox 独占持久状态，代码挂载只读并核对既有消息。"""
    # 1. 使用实际 provider 源码；只安装公共 API 的实现不参与组合。
    names = ("gateway", "sources", "models", "content", "commands", "conversation", "programmatic",
             "turn_projection", "ui", "reply", "onboarding", "workloads", "delivery")
    for name in names:
        shutil.copytree(ROOT / "plugins" / name, sandbox / "sources" / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    config = sandbox / "config.toml"
    config.write_text('[agent.plugins]\ndisabled_builtin = ["reply", "onboarding", "workloads", "delivery"]\n')
    before = config.read_bytes()
    stable = sandbox / "workspace/runtime/plugin-stable.json"
    options = ["--network", "none", "--mount", f"type=bind,src={ROOT},dst=/app,readonly",
               "--mount", f"type=bind,src={sandbox},dst=/sandbox",
               "-e", "AKASHIC_EXTRA_PLUGIN_DIRS=/sandbox/sources",
               "-e", f"AKASHIC_HOST_UID={os.getuid()}", "-e", f"AKASHIC_HOST_GID={os.getgid()}",
               "-e", "AKASHIC_EXECUTION_MODE=local"]
    docker(["run", "--rm", *options, image, "init"])
    assert stable.is_file() and json.loads(stable.read_text())["root_ref"] is None
    gateway = sandbox / "workspace/plugin-data/gateway-builtin/config.input.json"
    saved_gateway = gateway.read_bytes()
    values, _ = load_config(gateway.parent)
    assert values["listen"] == "/sandbox/akashic.sock"
    assert config.read_bytes() == before
    name = "akashic-debug-entrypoint-" + uuid4().hex[:12]

    def start() -> None:
        docker(["run", "-d", "--name", name, *options, image, "run"])

    async def request(method: str, params: dict[str, object]) -> dict[str, object]:
        reader, writer = await asyncio.open_unix_connection(sandbox / "akashic.sock")
        try:
            writer.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                "protocolVersion": "2.0", "clientInfo": {"name": "entrypoint-scenario", "version": "1"}}}).encode() + b"\n")
            await writer.drain()
            initialized = json.loads(await reader.readline())
            assert "error" not in initialized, initialized
            writer.write(b'{"jsonrpc":"2.0","method":"initialized"}\n')
            writer.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": method, "params": params}).encode() + b"\n")
            await writer.drain()
            frame = json.loads(await reader.readline())
            assert "error" not in frame, frame
            return frame["result"]
        finally:
            writer.close()
            await writer.wait_closed()

    async def ready() -> None:
        async with asyncio.timeout(30):
            while True:
                state = json.loads(docker(["inspect", name]).stdout)[0]["State"]
                if not state["Running"]:
                    logs = docker(["logs", name])
                    raise AssertionError(logs.stdout.decode(errors="replace") + logs.stderr.decode(errors="replace"))
                try:
                    result = await request("server/status", {})
                except (FileNotFoundError, ConnectionRefusedError):
                    await asyncio.sleep(0.05)
                else:
                    assert result["bootId"] and result["protocolVersion"] == "2.0"
                    return

    # 2. 完整 run 启动真实 listener；exec 通过原 socket 追加输入，随后重启。
    start()
    try:
        await ready()
        docker(["run", "--rm", *options, image, "exec", "--new", "--detach",
                "--session", "programmatic:debug-entrypoint", "--message-id", "debug-input", "debug entrypoint"])
        with sqlite3.connect(sandbox / "workspace/sessions.db") as database:
            rows = database.execute("SELECT * FROM messages ORDER BY session_key,seq").fetchall()
        assert rows
        assert config.read_bytes() == before and gateway.read_bytes() == saved_gateway
        docker(["stop", "-t", "20", name])
        docker(["rm", name])
        start()
        await ready()
        result = await request("message/read", {"session_id": "programmatic:debug-entrypoint"})
        assert result["items"]
        with sqlite3.connect(sandbox / "workspace/sessions.db") as database:
            assert database.execute("SELECT * FROM messages ORDER BY session_key,seq").fetchall() == rows
        assert config.read_bytes() == before and gateway.read_bytes() == saved_gateway
    finally:
        docker(["stop", "-t", "20", name])
        docker(["rm", name])
    # 3. stdio 从同一已选输入冷启动；真实握手后 EOF 排空宿主。
    frames = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2.0", "clientInfo": {"name": "entrypoint-scenario", "version": "1"}}},
        {"jsonrpc": "2.0", "method": "initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "server/status", "params": {}},
    ]
    result = subprocess.run(["docker", "run", "--rm", "-i", *options, image, "app-server", "--stdio"],
        input=b"".join(json.dumps(frame).encode() + b"\n" for frame in frames), capture_output=True, timeout=60)
    assert result.returncode == 0, (result.stdout, result.stderr)
    output = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(output) == 2 and all("error" not in frame for frame in output), output
    assert output[1]["result"]["bootId"]
    assert config.read_bytes() == before and gateway.read_bytes() == saved_gateway
    with sqlite3.connect(sandbox / "workspace/sessions.db") as database:
        assert database.execute("SELECT * FROM messages ORDER BY session_key,seq").fetchall() == rows
    return {"debug_image_entrypoint": True, "init_owner_config": True, "run_rpc": True,
            "exec_input": True, "restart_preserves_messages": True, "stdio_eof": True,
            "core_config_not_rewritten": True, "gateway_config_not_overwritten": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="debug-entrypoint-") as directory:
        print(json.dumps(asyncio.run(exercise(args.image, Path(directory)))))
