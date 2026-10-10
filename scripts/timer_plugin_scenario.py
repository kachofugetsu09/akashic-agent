"""在真实组合中验证 Timer 换代、停用与 Scheduler 持久通知。"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 第二实现用事件循环 callback 登记；它与 asyncio task 实现共享公开合同。
ALTERNATIVE = '''import asyncio
import secrets
from datetime import UTC, datetime
from plugins.timer.contract import TimerReceipt, TimerStatus

class Handle:
    def __init__(self, deadline):
        self.id = "callback:" + secrets.token_hex(16)
        self.deadline = deadline
        self.done = asyncio.get_running_loop().create_future()
        delay = max(0.0, (deadline - datetime.now(UTC)).total_seconds())
        self.callback = asyncio.get_running_loop().call_later(delay, self.finish, TimerStatus.FIRED)
    def finish(self, status):
        if not self.done.done():
            self.done.set_result(TimerReceipt(self.id, self.deadline, datetime.now(UTC), status))
    async def result(self):
        return await asyncio.shield(self.done)
    async def cancel(self):
        self.callback.cancel()
        self.finish(TimerStatus.CANCELLED)
        return await self.result()
    async def cleanup(self):
        await self.cancel()

class AsyncioOneShotTimer:
    def __init__(self):
        self.handles = set()
        self.closed = False
    def schedule(self, deadline):
        if self.closed:
            raise RuntimeError("timer closed")
        if deadline.tzinfo is None:
            raise ValueError("deadline needs timezone")
        handle = Handle(deadline.astimezone(UTC))
        self.handles.add(handle)
        handle.done.add_done_callback(lambda _: self.handles.remove(handle))
        return handle
    async def close(self):
        self.closed = True
        await asyncio.gather(*(handle.cleanup() for handle in tuple(self.handles)))
'''


def commit(path: Path) -> None:
    subprocess.run(["git", "-C", str(path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Scenario",
                    "-c", "user.email=scenario@example.invalid", "commit", "-qm", "scenario"], check=True)


async def run(directory: Path) -> dict[str, object]:
    """实际插件执行 instant 调度，替换 timer 后继续提交同一通知合同。"""
    from agent.config_models import Config
    from agent.plugin_composition.config_input import save_config
    from agent.plugins.install import install_git_plugin
    from agent.plugins.selection import PluginSelection
    from bootstrap.tools import build_core_runtime
    from core.net.http import SharedHttpResources
    from plugins.scheduler.schedule import ScheduledJob
    from plugins.scheduler.store import JobStore, fire_key
    from session.message import Output

    workspace = directory / "workspace"
    workspace.mkdir()
    PluginSelection(workspace).initialize()
    home = directory / "home"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION="")
    sources = directory / "plugins"
    for name in ("commands", "sources", "content", "context", "tools", "react", "models",
                 "turn_projection", "reply_program", "conversation", "delivery", "akashic_sender",
                 "standard_tools", "assets", "scheduler", "tool_search"):
        shutil.copytree(ROOT / "plugins" / name, sources / name,
                        ignore=shutil.ignore_patterns("__pycache__"))
    provider = directory / "timer"
    shutil.copytree(ROOT / "plugins/timer", provider, ignore=shutil.ignore_patterns("__pycache__"))
    subprocess.run(["git", "init", "-q", "--initial-branch=source", str(provider)], check=True)
    commit(provider)
    install_git_plugin(workspace=workspace, source=str(provider), marketplace="lab", plugins_home=home)
    probe = sources / "probe"
    probe.mkdir()
    (probe / "plugin.py").write_text('''from agent.plugin_composition import ServiceKey
from plugins.timer.contract import TIMERS
api_version = 3
name = "probe"
version = "1.0.0"
inject = (TIMERS,)
async def apply(ctx):
    with (ctx.data_root / "applies").open("a") as file:
        file.write("apply\\n")
    await ctx.provide(ServiceKey("scenario.timer"), ctx.require(TIMERS))
''')
    # 所有插件使用实际实现；instant 任务不会调用未配置的模型。
    save_config(workspace / "plugin-data/context-builtin", {"prompt_sources": {"skills": "standard_tools"}})
    http = SharedHttpResources()
    config = Config(workspace_path=workspace)
    store = JobStore(workspace / "schedules.json")

    async def notice(runtime, identity: str) -> None:
        job = ScheduledJob("after", "instant", datetime.now(UTC), "akashic", "scenario",
                           message=identity, id=identity)
        await store.add(identity, job, identity)
        # 真实持久订阅是完成屏障；轮询只负责跨连接事实追赶。
        stream = runtime.message_log.catalog().follow(poll_interval=0.01)
        try:
            async with asyncio.timeout(10):
                async for _ in stream:
                    final = runtime.message_log.reader("akashic:scenario").get("scheduler-notification:" + fire_key(job))
                    if final is not None:
                        assert isinstance(final.body, Output)
                        assert final.body.parts[0].value == identity
                        break
            # 发送回执和任务结算晚于追加；只读实际文件等待它们完成。
            async with asyncio.timeout(10):
                while store.read().fires[fire_key(job)].status == "pending":
                    ready = asyncio.get_running_loop().create_future()
                    asyncio.get_running_loop().call_later(0.01, ready.set_result, None)
                    await ready
            assert store.read().fires[fire_key(job)].status == "delivered"
        finally:
            await stream.aclose()

    def build():
        return build_core_runtime(config, workspace, http, plugin_dirs=[sources])

    runtime = build()
    try:
        await runtime.start()
        host = runtime.plugin_manager
        await host.start_runtime()
        root = host.live_root
        assert root is not None
        from plugins.timer.contract import TIMERS, TimerStatus
        timer = root.context.require(TIMERS)
        # 1. 没有 yield 就取消，仍得到稳定 cancelled 回执。
        immediate = timer.schedule(datetime.now(UTC) + timedelta(hours=1))
        cancelled = await immediate.cancel()
        assert cancelled.status == TimerStatus.CANCELLED
        assert await immediate.result() is cancelled
        assert (await timer.schedule(datetime.now(UTC)).result()).status == TimerStatus.FIRED
        await notice(runtime, "first")
        peers = {name: item.fiber for name, item in host._active_generations.items()
                 if name not in {"timer@lab", "scheduler", "probe"}}
        pending = timer.schedule(datetime.now(UTC) + timedelta(hours=1))
        # 2. 安装第二实现，硬依赖消费者按原组合规则换代，旁支保持 Fiber。
        (provider / "timer.py").write_text(ALTERNATIVE)
        commit(provider)
        await host.install(source=str(provider), marketplace="lab", ref_name="", sparse_paths=[], update_id="callback-timer")
        assert host._operation is not None
        await host._operation.task
        assert host.read_update("callback-timer").state == "active"
        assert (await pending.result()).status == TimerStatus.CANCELLED
        assert all(host._active_generations[name].fiber is fiber for name, fiber in peers.items())
        timer = root.context.require(TIMERS)
        assert (await timer.schedule(datetime.now(UTC)).result()).status == TimerStatus.FIRED
        await notice(runtime, "second")
        applies = workspace / "plugin-data/probe-builtin/applies"
        assert applies.read_text().splitlines() == ["apply", "apply"]
    finally:
        await runtime.stop()
    # 3. 同一 workspace 重启，旧通知不重复，第二实现仍可正常发新通知。
    runtime = build()
    try:
        await runtime.start()
        host = runtime.plugin_manager
        await host.start_runtime()
        await notice(runtime, "third")
        root = host.live_root
        assert root is not None
        before = root.context.require(TIMERS).schedule(datetime.now(UTC) + timedelta(hours=1))
        await host.uninstall("timer@lab")
        assert host._operation is not None
        await host._operation.task
        assert (await before.result()).status == TimerStatus.CANCELLED
        assert root.context.get(TIMERS) is None
        assert host._active_generations["scheduler"].fiber.state != "active"
        assert host._active_generations["content"].fiber.state == "active"
        messages = runtime.message_log.reader("akashic:scenario").snapshot()
        assert len(messages) == 3
    finally:
        await runtime.stop()
        await http.aclose()
    return {"immediate_cancel": True, "generation_cleanup": True, "second_provider": True,
            "scheduler_delivered": 3, "restart": True, "disable": True, "unrelated_stable": True}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="akashic-timer-") as temporary:
        print(json.dumps(asyncio.run(run(Path(temporary)))))


if __name__ == "__main__":
    main()
