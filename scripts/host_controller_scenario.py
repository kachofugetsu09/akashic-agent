"""真实插件安装与 Docker Controller 验证容器、持久挂载和 generation 清理。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROBE = '''from agent.plugin_composition import ServiceKey
from plugins.workloads.contract import WORKLOADS, Workload, WorkloadPort, WorkloadData, WorkloadHealth, WorkloadLimits
api_version = 3
name = "container_probe"
version = "1.0.0"
inject = (WORKLOADS,)
URL = ServiceKey[str]("scenario.container-url")
async def apply(ctx):
    resource = Workload("server", IMAGE,
        ("python", "-c", "import signal, sys; signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); from pathlib import Path; from http.server import HTTPServer, SimpleHTTPRequestHandler; from functools import partial; Path('/data/health').write_text('first'); HTTPServer(('0.0.0.0',8080),partial(SimpleHTTPRequestHandler,directory='/data')).serve_forever()"),
        (WorkloadPort("http", 8080),), (WorkloadData("data", "/data"),),
        WorkloadHealth("http", timeout_seconds=10), WorkloadLimits(64, 0.2, 32))
    handle = await ctx.require(WORKLOADS).register(ctx, resource)
    await ctx.provide(URL, handle.url(ctx, "http"))
'''


def commit(path: Path) -> None:
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Scenario", "-c",
                    "user.email=scenario@example.invalid", "commit", "-qm", "scenario"], check=True)


async def exercise(directory: Path, network: str, image: str) -> dict[str, object]:
    """在独立 Docker 网络内走真实安装、HTTP 健康与资源关闭路径。"""
    import httpx
    from agent.plugin_composition.model import ServiceKey
    from agent.plugins.install import install_git_plugin
    from agent.plugins.manager import PluginManager
    from agent.plugins.selection import PluginSelection

    # 1. Controller 与插件使用同一隔离 workspace；容器只挂载其自己的数据。
    workspace, home, sources = directory / "w", directory / "home", directory / "sources"
    for relative in ("plugin-data", "runtime/plugin-validation"):
        (workspace / relative).mkdir(parents=True)
    PluginSelection(workspace).initialize()
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION="",
                      AKASHIC_WORKLOAD_SOCKET=str(directory / "relay.sock"))
    for name in ("host_execution", "workloads"):
        shutil.copytree(ROOT / "plugins" / name, sources / name, ignore=shutil.ignore_patterns("__pycache__"))
    probe = sources / "container_probe"
    probe.mkdir()
    (probe / "plugin.py").write_text(PROBE.replace("IMAGE", repr(image)))
    for source in sources.iterdir():
        subprocess.run(["git", "init", "-q", "--initial-branch=source", str(source)], check=True)
        commit(source)
        install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    distribution = directory / "image"
    (distribution / "sources").mkdir(parents=True)
    shutil.copytree(sources / "host_execution", distribution / "sources/host_execution",
                    ignore=shutil.ignore_patterns(".git"))
    (distribution / "profiles").mkdir()
    (distribution / "profiles/default.json").write_text(json.dumps({"marketplace": "release"}))
    (distribution / "distribution.json").write_text(json.dumps({"source_commit": "a" * 40,
        "plugins": [{"name": "host_execution"}]}))
    state = directory / "leases.json"
    with (directory / "controller.log").open("wb") as output:
        controller = await asyncio.create_subprocess_exec(sys.executable, "-m", "agent.plugins.entrypoints",
            "--distribution", str(distribution), "--workspace", str(workspace), "--config", str(directory / "config.toml"),
            "workload-controller", "--workspace", str(workspace), "--socket", str(directory / "controller.sock"),
            "--state", str(state), "--network", network, "--allowed-uid", str(os.getuid()),
            "--socket-uid", str(os.getuid()), "--socket-gid", str(os.getgid()),
            "--workload-uid", str(os.getuid()), "--workload-gid", str(os.getgid()), stdout=output, stderr=output)
        actions: list[str] = []
        relays: set[asyncio.Task] = set()
        async def relay(reader, writer):
            task = asyncio.current_task()
            assert task is not None
            relays.add(task)
            upstream = None
            try:
                frame = await reader.readline()
                actions.append(json.loads(frame)["action"])
                peer, upstream = await asyncio.open_unix_connection(directory / "controller.sock")
                upstream.write(frame)
                await upstream.drain()
                writer.write(await peer.readline())
                await writer.drain()
            finally:
                if upstream is not None:
                    upstream.close()
                    await upstream.wait_closed()
                writer.close()
                await writer.wait_closed()
                relays.remove(task)
        relay_server = await asyncio.start_unix_server(relay, path=directory / "relay.sock")
        host = PluginManager([], workspace=workspace, installed_cache_root=home / "cache")
        try:
            async with asyncio.timeout(10):
                while not (directory / "controller.sock").exists():
                    if controller.returncode is not None:
                        raise RuntimeError((directory / "controller.log").read_text())
                    await asyncio.sleep(0.02)
            await host.load_all()
            root = host.live_root
            assert root is not None
            async def read():
                async with httpx.AsyncClient(trust_env=False) as client:
                    response = await client.get(root.context.require(ServiceKey("scenario.container-url")) + "/health")
                    response.raise_for_status()
                    return response.text
            assert await read() == "first"
            assert actions.count("cleanup_candidates") == 1
            first = next(iter(json.loads(state.read_text()).values()))["container_id"]
            # 2. 真正换代关闭旧 lease，再创建新的 HTTP 服务；挂载数据仍在原 owner。
            (probe / "plugin.py").write_text(PROBE.replace("IMAGE", repr(image)).replace("write_text('first')", "write_text('second')"))
            commit(probe)
            await host.install(source=str(probe), marketplace="lab", ref_name="", sparse_paths=[], update_id="container-update")
            await host.wait_idle()
            assert host.read_update("container-update").state == "active" and host.live_root is root
            assert await read() == "second"
            second = next(iter(json.loads(state.read_text()).values()))["container_id"]
            assert second != first
            execution_source = sources / "host_execution"
            entry = execution_source / "plugin.py"
            entry.write_text(entry.read_text() + "\n# new generation\n")
            commit(execution_source)
            await host.install(source=str(execution_source), marketplace="lab", ref_name="", sparse_paths=[], update_id="execution-update")
            await host.wait_idle()
            assert await read() == "second" and actions.count("cleanup_candidates") == 1
            await host.terminate_all()
            assert json.loads(state.read_text()) == {}
            # 3. 重启原选择并卸载执行 owner，只有硬依赖分支撤下，租约排空。
            host = PluginManager([], workspace=workspace, installed_cache_root=home / "cache")
            await host.load_all()
            root = host.live_root
            assert root is not None and await read() == "second"
            assert actions.count("cleanup_candidates") == 2
            await host.uninstall("host_execution@lab")
            await host.wait_idle()
            assert root.context.get(ServiceKey("scenario.container-url")) is None
            assert json.loads(state.read_text()) == {}
            assert (workspace / "plugin-data/container_probe-lab/data/health").read_text() == "second"
            assert not (workspace / "sessions.db").exists()
        finally:
            try:
                await host.terminate_all()
            finally:
                relay_server.close()
                await relay_server.wait_closed()
                await asyncio.gather(*relays)
                (directory / "relay.sock").unlink()
                if controller.returncode is None:
                    controller.terminate()
                await asyncio.wait_for(controller.wait(), 30)
                assert controller.returncode == 0, (directory / "controller.log").read_text()
    assert not (directory / "controller.sock").exists()
    closed = [json.loads(path.read_text()) for path in (directory / "closed").glob("*.json")]
    assert len(closed) == 1 and closed[0]["controller"]["leases_empty"]
    return {"real_docker_http": True, "generation": True, "restart": True,
            "disabled_owner_drained": True, "data_preserved": True, "controller_closed": True, "cleanup_once_per_boot": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="akashic-release-020:dependency-check")
    parser.add_argument("--child", type=Path)
    parser.add_argument("--network")
    parser.add_argument("--workload-image")
    args = parser.parse_args()
    if args.child is not None:
        print(json.dumps(asyncio.run(exercise(args.child, args.network, args.workload_image))))
        return
    network = "hc-" + uuid.uuid4().hex[:12]
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", "python:3.12-slim-bookworm", "--format", "{{json .RepoDigests}}"], text=True))[0]
    subprocess.run(["docker", "network", "create", "--internal", network], check=True, stdout=subprocess.PIPE)
    try:
        folder = tempfile.mkdtemp(prefix="hc-")
        print(f"evidence={folder}", flush=True)
        subprocess.run(["docker", "run", "--rm", "--network", network, "--user", f"{os.getuid()}:{os.getgid()}",
                "--group-add", str(Path("/var/run/docker.sock").stat().st_gid), "-v", "/var/run/docker.sock:/var/run/docker.sock",
                "-v", f"{ROOT}:/repo:ro", "-v", f"{folder}:{folder}", "-e", "PYTHONPATH=/repo", "-e", "NO_PROXY=*", "-e", "no_proxy=*", "-w", "/repo",
                "--entrypoint", "/opt/venv/bin/python", args.image, "/repo/scripts/host_controller_scenario.py",
                "--child", folder, "--network", network, "--workload-image", image], check=True)
    finally:
        subprocess.run(["docker", "network", "rm", network], check=True, stdout=subprocess.PIPE)


if __name__ == "__main__":
    main()
