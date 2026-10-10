#!/usr/bin/env python3
"""真实 watcher 加载原位修改，拒绝坏源码并保留当前 owner 和业务数据。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sdk/python/src")]

from agent.plugin_composition import ServiceKey
from agent.plugins.manager import PluginManager
from agent.plugins.watcher import PluginWatcher

VALUE = ServiceKey[str]("reload.value")


def source(version: str) -> str:
    return ("from agent.plugin_composition import ServiceKey\napi_version=3\n"
            f"name='reload_probe'\nversion={version!r}\nasync def apply(ctx):\n"
            f"    await ctx.provide(ServiceKey('reload.value'), {version!r})\n")


async def run(output: Path) -> None:
    """只写新建的运行目录；通过 watcher 回调等待真实更新完成。"""
    # 1. 用真实 CLI 初始化，加载可观测服务，保留独立业务文件。
    output.mkdir()
    home = output / "home"
    home.mkdir()
    os.environ["HOME"] = str(home)
    os.environ["AKASHIC_PLUGIN_HOME"] = str(output / "plugin-home")
    os.environ.pop("AKASHIC_PLUGIN_DISTRIBUTION", None)
    workspace = output / "workspace"
    subprocess.run([sys.executable, str(ROOT / "main.py"), "init", "--config", str(output / "config.toml"),
                    "--workspace", str(workspace)], check=True, capture_output=True, text=True)
    plugin = output / "plugins/reload_probe"
    plugin.mkdir(parents=True)
    entry = plugin / "plugin.py"
    entry.write_text(source("a"))
    manager = PluginManager([plugin.parent], workspace=workspace,
                            installed_cache_root=output / "plugin-home/cache")
    watcher = None
    task = None
    changed = asyncio.Event()

    async def after_reconcile() -> None:
        changed.set()

    try:
        await manager.load_all()
        original = manager.generation("reload_probe")
        assert original is not None and manager.live_root is not None
        data = original.data_dir / "keep.bin"
        data.write_bytes(b"user-owned-data")
        assert manager.live_root.context.require(VALUE) == "a"
        watcher = PluginWatcher(manager, baseline_revision=manager.watch_revision(),
                                interval_seconds=60, after_reconcile=after_reconcile)
        task = asyncio.create_task(watcher.run())
        # 2. 同一路径修改必须换代；只改来源标签不能替换实际 owner。
        entry.write_text(source("b"))
        watcher.wake()
        await asyncio.wait_for(changed.wait(), 10)
        current = manager.generation("reload_probe")
        assert current is not None and current is not original and original.scope.closed
        assert manager.live_root.context.require(VALUE) == "b"
        assert current.code_dir == original.code_dir == plugin.resolve()
        assert current.runtime_revision != original.runtime_revision
        (plugin / ".akashic-source.json").write_text(json.dumps({"commit": "different-evidence"}))
        assert await manager.reconcile_changed() == []
        assert manager.generation("reload_probe") is current
        # 3. 坏源码保留当前服务；修复后同一 watcher 能继续换代。
        changed.clear()
        entry.write_text("def invalid(:\n")
        watcher.wake()
        await asyncio.wait_for(changed.wait(), 10)
        assert manager.generation("reload_probe") is current
        assert manager.live_root.context.require(VALUE) == "b"
        changed.clear()
        entry.write_text(source("c"))
        watcher.wake()
        await asyncio.wait_for(changed.wait(), 10)
        assert manager.live_root.context.require(VALUE) == "c"
        assert manager.generation("reload_probe") is not current and current.scope.closed
        assert data.read_bytes() == b"user-owned-data"
        (output / "report.json").write_text(json.dumps({"passed": True, "same_path_reload": True,
            "provenance_only_keeps_owner": True, "bad_source_keeps_owner": True,
            "source_repair_reloads": True, "business_data_preserved": True}))
    finally:
        if watcher is not None:
            watcher.stop()
            await watcher.wait_stopped()
        if task is not None:
            await task
        await manager.terminate_all()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.output.resolve()))
