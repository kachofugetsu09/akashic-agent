"""在独立目录启动完整 Akashic 宿主与真实 Workload Controller。"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "sdk/python/src"))


def docker(*args: str) -> str:
    return subprocess.check_output(["docker", *args], text=True, stderr=subprocess.STDOUT).strip()


@asynccontextmanager
async def runtime(directory: Path, *, max_steps: int, endpoint: str, model: str, key_file: Path):
    """独占新状态；真实配置、接纳、回复、工具和账本均走现有 owner。"""
    from agent.plugin_composition.config_input import save_config
    from agent.plugins.bundles import set_plugin_choice
    from bootstrap.init_workspace import init_workspace
    from plugins.host_execution.controller import WorkloadControllerServer

    # 1. 初始化新 workspace，不复用个人配置、记忆、登录或插件 cache。
    state = directory / "state"
    state.mkdir()
    workspace, config = state / "workspace", state / "config.toml"
    home = state / "home"
    home.mkdir()
    network = "ak-miniwob-" + os.urandom(5).hex()
    sockets = tempfile.TemporaryDirectory(prefix="ak-miniwob-")
    socket_dir = Path(sockets.name)
    socket = socket_dir / "controller.sock"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home / "plugins"),
        AKASHIC_PLUGIN_DISTRIBUTION="", AKASHIC_EXTRA_PLUGIN_DIRS=str(ROOT / "plugins"),
        AKASHIC_EXECUTION_MODE="local", AKASHIC_WORKLOAD_SOCKET=str(socket),
        PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "sdk/python/src"))))
    os.environ.pop("AKASHIC_WEB_PORT", None)
    init_workspace(config_path=config, workspace=workspace)
    set_plugin_choice(workspace, "subagent", enabled=False)
    data = workspace / "plugin-data"
    save_config(data / "reply-builtin", {"max_steps": max_steps})
    save_config(data / "akashic_clients-builtin", {"enabled": False, "web": {"enabled": False}})
    save_config(data / "gateway-builtin", {"enabled": True, "listen": str(socket_dir / "rpc.sock")})
    save_config(data / "context-builtin", {
        "prompt_sources": {"default_prompt": "prompt", "markdown_memory": "markdown_memory",
                           "skills": "standard_tools"}, "summary_source": ["compaction", "compaction"]})
    (workspace / "runtime/plugin-validation").mkdir(parents=True, exist_ok=True)
    docker("network", "create", network)
    controller = WorkloadControllerServer(workspace=workspace, socket_path=socket,
        docker_socket=Path("/var/run/docker.sock"), state_path=state / "controller.json",
        network=network, allowed_uid=os.getuid(), socket_uid=os.getuid(), socket_gid=os.getgid(),
        workload_uid=os.getuid(), workload_gid=os.getgid())
    serving = asyncio.create_task(controller.serve())
    app_name = network + "-app"
    venv, base = Path(sys.prefix).resolve(), Path(sys.base_prefix)
    image = "ghcr.io/kachofugetsu09/akashic-computer@sha256:f48cab96d491c4965a34877fac6d64adb3fcd2d94fff47cdcb83c1e8af1f6a45"
    started = False
    containers = []
    try:
        # 2. Core 与 Computer 位于同一 Docker 网络，使用正式容器 DNS。
        async with asyncio.timeout(15):
            while not socket.exists():
                if serving.done():
                    await serving
                await asyncio.sleep(.05)
        env = ["HOME", "AKASHIC_PLUGIN_HOME", "AKASHIC_PLUGIN_DISTRIBUTION",
               "AKASHIC_EXTRA_PLUGIN_DIRS", "AKASHIC_EXECUTION_MODE", "AKASHIC_WORKLOAD_SOCKET", "PYTHONPATH"]
        options = ["-e", f"PATH={venv}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"]
        options += [part for key in env for part in ("-e", key + "=" + os.environ[key])]
        mounts = [part for path in (ROOT, venv, base) for part in ("-v", f"{path}:{path}:ro")]
        docker("run", "-d", "--name", app_name, "--network", network,
            "--user", f"{os.getuid()}:{os.getgid()}", "--entrypoint", str(venv / "bin/python"),
            "-v", f"{directory}:{directory}", "-v", f"{socket_dir}:{socket_dir}", "-v", f"{key_file}:/run/miniwob-key:ro",
            *mounts, *options, image, str(Path(__file__).with_name("service.py")),
            str(directory), endpoint, model)
        started = True
        async with asyncio.timeout(180):
            while not (directory / "ready.json").exists():
                if docker("inspect", app_name, "--format", "{{.State.Running}}") != "true":
                    raise RuntimeError("Akashic 启动失败，见 app.log")
                await asyncio.sleep(.2)
        containers = docker("ps", "-q", "--filter", f"network={network}", "--filter", "label=com.akashic.workload=true").splitlines()
        if len(containers) != 1:
            raise RuntimeError(f"需要唯一 Computer 容器: {containers}")
        detail = json.loads(docker("inspect", containers[0]))[0]
        ip = detail["NetworkSettings"]["Networks"][network]["IPAddress"]
        net = json.loads(docker("network", "inspect", network))[0]
        identity = {"container": containers[0], "image_id": detail["Image"],
                    "network": network, "gateway": f"http://{ip}:8080",
                    "host": net["IPAM"]["Config"][0]["Gateway"],
                    "workspace": str(workspace), "config": str(config),
                    "rpc": str(socket_dir / "rpc.sock"), "app_container": app_name}
        (directory / "runtime.json").write_text(json.dumps(identity, indent=2))
        yield identity
    finally:
        # 4. 由宿主正常排空 Workload；保留所有实验消息和配置供复查。
        try:
            if containers:
                (directory / "computer.log").write_text(docker("logs", containers[0]))
                (directory / "computer.json").write_text(docker("inspect", containers[0]))
            if started:
                await asyncio.to_thread(docker, "stop", "-t", "60", app_name)
                (directory / "app.log").write_text(docker("logs", app_name))
                docker("rm", app_name)
        finally:
            serving.cancel()
            try:
                await serving
            except asyncio.CancelledError:
                pass
            docker("network", "rm", network)
            sockets.cleanup()
