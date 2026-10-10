"""真实进程在取消等待后可续接，并随 HostExecution 换代、停用和重启清理。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROBE = '''from agent.plugin_composition import ServiceKey
from plugins.host_execution.contract import PROCESSES
api_version = 3
name = "process_probe"
version = "1.0.0"
inject = (PROCESSES,)
async def apply(ctx):
    async def run(argv):
        return await ctx.require(PROCESSES).exec_command(ctx, "scenario", command="scenario",
            argv=argv, cwd=ctx.data_root, env={}, tty=False, yield_time_ms=250,
            max_output_tokens=100, hard_timeout_s=30)
    async def poll(identity):
        return await ctx.require(PROCESSES).write_stdin(ctx, "scenario", execution_id=identity,
            chars="", yield_time_ms=5000, max_output_tokens=100)
    await ctx.provide(ServiceKey("scenario.run"), ctx.entrypoint(run))
    await ctx.provide(ServiceKey("scenario.poll"), ctx.entrypoint(poll))
'''


def commit(path: Path) -> None:
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Scenario", "-c",
                    "user.email=scenario@example.invalid", "commit", "-qm", "scenario"], check=True)


async def run(directory: Path) -> dict[str, bool]:
    """通过真实安装 owner 换代，检查原进程 PID 和旁支 Fiber。"""
    from agent.plugin_composition import ServiceKey
    from agent.plugins.install import install_git_plugin
    from agent.plugins.manager import PluginManager
    from agent.plugins.selection import PluginSelection

    workspace, home = directory / "workspace", directory / "home"
    workspace.mkdir()
    PluginSelection(workspace).initialize()
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION="",
                      AKASHIC_EXECUTION_MODE="local", AKASHIC_WORKLOAD_SOCKET="")
    sources = directory / "sources"
    for name in ("host_execution", "ui", "timer"):
        source = sources / name
        shutil.copytree(ROOT / "plugins" / name, source, ignore=shutil.ignore_patterns("__pycache__"))
        subprocess.run(["git", "init", "-q", "--initial-branch=source", str(source)], check=True)
        commit(source)
        install_git_plugin(workspace=workspace, source=str(source), marketplace="lab", plugins_home=home)
    probe = directory / "probes/process_probe"
    probe.mkdir(parents=True)
    (probe / "plugin.py").write_text(PROBE)
    host = PluginManager(workspace=workspace, plugin_dirs=[probe.parent], installed_cache_root=home / "cache")
    pids: list[int] = []

    async def start_process():
        root = host.live_root
        assert root is not None
        result = await root.context.require(ServiceKey("scenario.run"))([
            sys.executable, "-c", "import os,time; print(os.getpid(),flush=True); time.sleep(25)",
        ])
        assert result.execution_id is not None
        pid = int(result.output)
        pids.append(pid)
        return result.execution_id, pid

    def absent(pid: int) -> None:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        raise AssertionError(f"旧 generation 仍有进程: {pid}")

    try:
        await host.load_all()
        await host.start_runtime()
        root = host.live_root
        assert root is not None
        identity, pid = await start_process()
        poll = asyncio.create_task(root.context.require(ServiceKey("scenario.poll"))(identity))
        await asyncio.sleep(0)
        poll.cancel()
        try:
            await poll
        except asyncio.CancelledError:
            pass
        os.kill(pid, 0)
        neighbor = host._active_generations["timer@lab"].fiber
        provider = sources / "host_execution/plugin.py"
        provider.write_text(provider.read_text().replace('version = "1.0.0"', 'version = "1.0.1"'))
        commit(provider.parent)
        await host.install(source=str(provider.parent), marketplace="lab", ref_name="", sparse_paths=[], update_id="replace-process-owner")
        assert host._operation is not None
        await host._operation.task
        assert host.read_update("replace-process-owner").state == "active"
        absent(pid)
        assert host._active_generations["timer@lab"].fiber is neighbor
        _, pid = await start_process()
        await host.uninstall("host_execution@lab")
        assert host._operation is not None
        await host._operation.task
        absent(pid)
        assert host._active_generations["process_probe"].fiber.state != "active"
        assert host._active_generations["timer@lab"].fiber is neighbor
    finally:
        await host.terminate_all()
    for pid in pids:
        absent(pid)
    # 同一选择重启：缺席 provider 不阻断宿主和无关插件。
    host = PluginManager(workspace=workspace, plugin_dirs=[probe.parent], installed_cache_root=home / "cache")
    try:
        await host.load_all()
        await host.start_runtime()
        assert host._active_generations["timer@lab"].fiber.state == "active"
        assert host._active_generations["process_probe"].fiber.state != "active"
    finally:
        await host.terminate_all()
    return {"cancel_keeps_process": True, "generation_cleanup": True,
            "uninstall_cleanup": True, "unrelated_stable": True, "restart": True}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="akashic-process-owner-") as directory:
        print(json.dumps(asyncio.run(run(Path(directory)))))
