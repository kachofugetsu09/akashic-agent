"""本地真实回复、Bridge 探测及技能文件生命周期实验；只写新建目录。"""
from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import grpc
from agent.host_bridge.client import HostBridgeShellProcessManager
from agent.host_bridge.monitor import HostBridgeStatus, _monitor
from agent.plugin_composition.archive import PluginArchive
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.channels import CHANNEL_INPUT, ChannelInboundMessage
from agent.plugin_composition.config_input import save_config
from agent.plugin_contracts.tools import ALL_TOOLS, TOOLS
from session.message import Output
from tests.test_default_reply import application


def add_skills(sources: Path, *, files: int, always: bool) -> None:
    """复用真实技能注册；模型使用既有固定回复夹具。"""
    # 1. 保留完整消息与工具准备路径，只固定远端模型输出。
    module = sources / "test_provider/plugin.py"
    source = module.read_text()
    assert source.count("if len(calls) == 1:") == 1
    module.write_text(source.replace("if len(calls) == 1:", "if False:"))
    save_config(sources.parent / "workspace/plugin-data/context-builtin",
                {"prompt_sources": {"skills": "skill_probe"}})
    for source_name, target in (("assets", "assets"), ("standard_tools", "skill_probe")):
        shutil.copytree(ROOT / "plugins" / source_name, sources / target,
                        ignore=shutil.ignore_patterns("__pycache__"))
    (sources / "skill_probe/plugin.py").write_text('''from agent.plugin_composition.assets import INSTALLED_ASSETS
from ._materials_boundary import MATERIALS
from ._tool_boundary import TOOLS
from .skills import register_skills
api_version = 3
name = "skill_probe"
version = "1.0.0"
inject = (TOOLS, MATERIALS, INSTALLED_ASSETS)
async def apply(ctx):
    await ctx.require(TOOLS).declare_group(ctx, always_on=True)
    await register_skills(ctx)
''')
    # 2. 用合成资源测试规模，不读取正式 workspace 或用户技能目录。
    bundle = sources / "skill_bundle"
    skill = bundle / "skills/local"
    skill.mkdir(parents=True)
    (bundle / "plugin.py").write_text('''from agent.plugin_composition.assets import INSTALLED_ASSETS
api_version = 3
name = "skill_bundle"
version = "1.0.0"
inject = (INSTALLED_ASSETS,)
async def apply(ctx):
    await ctx.require(INSTALLED_ASSETS).register(ctx, "skills", "skills")
''')
    (skill / "SKILL.md").write_text(
        "---\nname: local\ndescription: Local experiment\n"
        f"always: {str(always).lower()}\n"
        "metadata: {requires: {bins: [sh]}}\n---\nLocal fixture.\n")
    (skill / "resource.txt").write_text("preserved")
    for index in range(files):
        folder = skill / f"d{index // 100}"
        folder.mkdir(exist_ok=True)
        (folder / f"f{index}.txt").write_bytes(f"{index} local data\n".encode() * 16)


async def check_lifecycle(base: Path, log, host) -> list[str]:
    """真实 owner 上验证并发、归档损坏和取消后的物理排空。"""
    root = host.live_root
    assert root is not None
    tools = root.context.require(TOOLS)
    bindings = root.context.require(BINDINGS)
    ref = next(item for item in root.context.require(ALL_TOOLS)().refs if item.name == "load_skill")
    # 1. 并发捕获保存同一内容身份，恢复工具可读取原正文。
    identities = await asyncio.gather(*(tools.bind(ref, bindings) for _ in range(3)))
    assert len(set(identities)) == 1
    metadata = bindings.describe(identities[0], TOOLS)
    async with tools.open(metadata) as tool:
        arguments = await tool.prepare({"skill": "local"})
        result = await tool.invoke("local-read", arguments)
        assert result.outcome == "success" and "Local fixture." in str(result.parts)
    results = ["concurrent_capture_same_binding_and_readable_body"]
    # 2. 故意损坏本次实验自己的归档；捕获与恢复均须明确失败。
    archive = next((base / "app/workspace/plugin-data").glob("skill_probe-*/skill-files"))
    tree_ref = metadata["state"]["skills"]["local"]["tree_ref"]
    resource = archive / tree_ref / "tree/resource.txt"
    original = resource.read_bytes()
    resource.chmod(0o644)
    resource.write_text("corrupt")
    try:
        for action in ("capture", "read"):
            try:
                if action == "capture":
                    await tools.bind(ref, bindings)
                else:
                    async with tools.open(metadata) as tool:
                        await tool.invoke("corrupt-read", {"skill": "local"})
            except RuntimeError as error:
                assert "归档文件树损坏" in str(error)
            else:
                raise AssertionError("损坏归档被静默接受")
    finally:
        resource.write_bytes(original)
        resource.chmod(0o444)
    results.append("corrupt_archive_capture_and_read_fail_loud")
    # 3. Event 固定正在运行的文件工作；取消后资产卸载必须等它真正结束。
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    save = PluginArchive.save

    def held_save(archive, source, **kwargs):
        loop.call_soon_threadsafe(started.set)
        if not release.wait(30):
            raise TimeoutError("实验文件线程未被释放")
        return save(archive, source, **kwargs)

    before = log.read_bindings()
    with patch.object(PluginArchive, "save", held_save), patch.object(bindings, "bind", wraps=bindings.bind) as commit:
        capture = asyncio.create_task(tools.bind(ref, bindings))
        disposal = None
        try:
            await asyncio.wait_for(started.wait(), 5)
            capture.cancel()
            await asyncio.sleep(0)
            assert not capture.done()
            owner = host._active_generations["skill_bundle"].fiber
            disposal = asyncio.create_task(owner.dispose())
            await asyncio.wait_for(owner._admission_closed.wait(), 5)
            assert not disposal.done() and not capture.done()
            release.set()
            try:
                await capture
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("取消后仍提交了工具绑定")
            await asyncio.wait_for(disposal, 30)
            assert log.read_bindings() == before and commit.call_count == 0
        finally:
            release.set()
            if not capture.done():
                capture.cancel()
            await asyncio.gather(capture, return_exceptions=True)
            if disposal is not None:
                await disposal
    results.append("cancel_drains_files_before_asset_disposal_without_binding_commit")
    return results


async def run(base: Path, *, files: int, always: bool, max_lag: float) -> None:
    """启动独立 Bridge，运行本地消息链，收集结果后结束全部自有进程。"""
    base.mkdir(parents=True, exist_ok=False)
    token = base / "token"
    token.write_text("local-only")
    socket = base / "bridge.sock"
    commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    digest = "b" * 64
    bridge_log = (base / "bridge.log").open("w")
    bridge = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "agent.host_bridge.server", "--socket", str(socket),
        "--token-file", str(token), "--lease-timeout", "60", "--artifact-root", str(base / "bridge-artifacts"),
        "--release-commit", commit, "--toolchain-digest", digest,
        "--runtime-checkout", str(ROOT), "--bridge-python", sys.executable,
        cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT), "AKASHIC_EXECUTION_MODE": "local"},
        stdout=bridge_log, stderr=bridge_log,
    )
    client = HostBridgeShellProcessManager(socket, "local-boot", "local-only", commit, digest)
    tasks = []
    changed = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-files", "--modified", "--others", "--exclude-standard", "-z"],
    ).decode().split("\0")
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                     for name in sorted(set(changed)) if name}
    report = {"files": files, "always": always, "source_commit": commit,
              "changed_source_sha256": source_hashes, "cases": [], "passed": False}
    try:
        # 1. 只向本次创建的 UDS 声明 boot；不读取正式 token 或服务配置。
        async with grpc.aio.insecure_channel(f"unix:{socket}") as channel:
            await asyncio.wait_for(channel.channel_ready(), 15)
        await client.claim_boot()
        environment = dict(AKASHIC_EXECUTION_MODE="host-bridge", AKASHIC_HOST_BRIDGE_SOCKET=str(socket),
                           AKASHIC_HOST_BRIDGE_TOKEN="local-only", AKASHIC_BOOT_ID="local-boot",
                           AKASHIC_RUNTIME_COMMIT=commit, AKASHIC_HOST_TOOLCHAIN_DIGEST=digest)
        with patch.dict(os.environ, environment):
            async with application(base / "app", replying=True,
                                   extra_sources=lambda path: add_skills(path, files=files, always=always)) as (log, host):
                assert all(item["fiber_state"] == "active" for item in host.plugin_status()["plugins"])
                status = HostBridgeStatus(state="checking")
                tasks.append(asyncio.create_task(_monitor(socket, "local-boot", "local-only", commit, digest, status=status)))
                lags = []
                states = []

                async def tick():
                    while True:
                        start = time.perf_counter()
                        await asyncio.sleep(0.02)
                        lags.append((time.perf_counter(), time.perf_counter() - start - 0.02))
                        states.append(status.snapshot())

                tasks.append(asyncio.create_task(tick()))
                async with asyncio.timeout(10):
                    while status.state != "healthy":
                        await asyncio.sleep(0.01)
                states.clear()
                # 2. 真实 Input → reply → tools → Output；探测与回复共用 Core 主循环。
                for turn in range(3):
                    start = time.perf_counter()
                    root = host.live_root
                    assert root is not None
                    await root.context.require(CHANNEL_INPUT)(
                        f"test:room{turn}", f"u{turn}", ChannelInboundMessage(
                            "test", "user", f"room{turn}", "local test", datetime.now(UTC), {}))
                    async with asyncio.timeout(90):
                        async for _ in log.catalog().follow():
                            if any(isinstance(row.body, Output) and row.body.finish == "complete"
                                   for row in log.reader(f"test:room{turn}").snapshot()):
                                break
                    elapsed = time.perf_counter() - start
                    await asyncio.sleep(0.03)
                    lag = max(delay for instant, delay in lags if instant >= start)
                    record = {"turn": turn, "elapsed_s": elapsed, "max_loop_lag_s": lag, "status": status.snapshot()}
                    report["cases"].append(record)
                    print(json.dumps(record), flush=True)
                    assert lag < max_lag, f"Core 事件循环停顿 {lag:.3f}s"
                    await client.probe()
                report["lifecycle"] = await check_lifecycle(base, log, host)
                assert all(item["state"] == "healthy" for item in states)
    finally:
        # 3. 无论断言是否成功，都排空本次创建的任务与 Bridge。
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await client.close_transport()
        finally:
            try:
                if bridge.returncode is None:
                    bridge.terminate()
                await asyncio.wait_for(bridge.wait(), 10)
            finally:
                bridge_log.close()
                (base / "report.json").write_text(json.dumps(report, indent=2))
    report["passed"] = True
    (base / "report.json").write_text(json.dumps(report, indent=2))
    print(f"report={base / 'report.json'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--files", type=int, default=500)
    parser.add_argument("--always", action="store_true")
    parser.add_argument("--max-lag", type=float, default=0.25)
    args = parser.parse_args()
    asyncio.run(run(args.root.resolve(), files=args.files, always=args.always, max_lag=args.max_lag))
