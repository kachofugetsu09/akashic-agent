#!/usr/bin/env python3
"""用真实旧 Core 建状态，经离线 CLI 升级并验证当前文件、卸载及重装。"""
from __future__ import annotations

import argparse
from collections.abc import Awaitable, Callable, Mapping
from typing import cast
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sdk/python/src")]

from agent.migrations.runner import MigrationRunner
from agent.plugin_composition.model import ServiceKey
from agent.plugins.manager import PluginManager
from agent.plugins.python_environment import PythonEnvironments
from agent.plugins.selection import PluginSelection, SelectionFormatError
from session.log import MessageLog

SEED = '''import asyncio,sys
from pathlib import Path
from agent.plugins.manager import PluginManager
from session.log import MessageLog
from session.message import Input,ContentPart,ContentReferences
workspace,home,source=map(Path,sys.argv[1:])
async def run():
    log=MessageLog(workspace/'sessions.db')
    log.writer(session_id='preserved',author='user',source='archive-e2e',body_types=(Input,),content={'text':lambda part:ContentReferences()}).append('original',Input((ContentPart('text','完整保留的原消息'),)))
    host=PluginManager([],workspace=workspace,installed_cache_root=home/'cache',message_log=log)
    try:
        await host.load_all()
        await host.install(source=str(source),marketplace='lab',ref_name='',sparse_paths=[],update_id='old-install')
        await host._operation.task
        await host.apply_config_input('archive_probe@lab','old-config',host.read_config_input('archive_probe@lab')['input_ref'],{'message':'original'})
        await host._operation.task
    finally:
        await host.terminate_all()
        log.close()
asyncio.run(run())
'''


def command(args: list[str], *, env: dict[str, str], cwd: Path, log: Path) -> None:
    """每个真实 CLI 保留输出，失败或超时直接结束验收。"""
    result = subprocess.run(args, cwd=cwd, env=env, capture_output=True, text=True, timeout=60)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"命令失败 {args[0]}；见 {log}")


def files(directory: Path) -> dict[str, str]:
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file() and "__pycache__" not in path.parts}


def messages(workspace: Path) -> list[tuple[object, ...]]:
    with sqlite3.connect(workspace / "sessions.db") as db:
        return db.execute("SELECT * FROM messages ORDER BY rowid").fetchall()


async def verify(workspace: Path, home: Path, source: Path, installed: Path) -> dict[str, object]:
    """真实 runtime 加载当前资源，保留数据后卸载并重装。"""
    log = MessageLog(workspace / "sessions.db")
    host = PluginManager([], workspace=workspace,
                         installed_cache_root=home / "cache", message_log=log)
    data = workspace / "plugin-data/archive_probe-lab"
    try:
        await host.load_all()
        generation = host.generation("archive_probe@lab")
        assert generation is not None and generation.code_dir == installed
        assert generation.config_projection == {"message": "external-current"}
        assert host.live_root is not None
        reader = host.live_root.context.require(ServiceKey[Callable[[], Awaitable[str]]]("probe.read"))
        assert await reader() == "external-current-resource"
        assert generation.input_ref is not None
        record = host._selection.read_input(generation.input_ref)
        for runtime, ref in cast(Mapping[str, str], record["python_environments"]).items():
            root = PythonEnvironments(workspace).open(ref)
            assert subprocess.check_output([str(root / runtime / ".venv/bin/python"), "-I", "-c",
                                            "import sys; print(sys.version_info.major)"], text=True).strip() == "3"
        assert "config" not in record
        current = files(data)
        await host.uninstall("archive_probe@lab")
        await host._operation.task
        assert not installed.exists() and files(data) == current
        await host.install(source=str(source), marketplace="lab", ref_name="", sparse_paths=[], update_id="reinstall")
        await host._operation.task
        assert host.read_update("reinstall").state == "active"
        assert files(data) == current
        replacement = host.generation("archive_probe@lab")
        assert replacement is not None and replacement.config_projection == {"message": "external-current"}
        return {"current_resources": "passed", "current_config": "passed", "real_interpreter": "passed",
                "uninstall_reinstall": "passed", "private_data_preserved": True}
    finally:
        await host.terminate_all()
        log.close()


def main() -> None:
    """只接受独立证据目录和已 checkout 的旧 Core，不控制宿主服务。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-core", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output, old_core = args.output.resolve(), args.old_core.resolve()
    output.mkdir()
    env = os.environ.copy()
    home = output / "home"
    home.mkdir()
    env.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(output / "plugins"),
               PYTHONPATH=os.pathsep.join((str(old_core), str(old_core / "sdk/python/src"))))
    for key in ("AKASHIC_PLUGIN_DISTRIBUTION", "AKASHIC_WORKLOAD_SOCKET", "AKASHIC_HOST_BRIDGE_SOCKET", "AKASHIC_CALL_CONTEXT"):
        env.pop(key, None)
    workspace, plugins_home, config = output / "workspace", output / "plugins", output / "config.toml"
    command([sys.executable, str(old_core / "main.py"), "init", "--config", str(config), "--workspace", str(workspace)],
            env=env, cwd=old_core, log=output / "old-init.log")
    source = output / "source"
    source.mkdir()
    (source / "plugin.py").write_text("api_version=3\nname='archive_probe'\nversion='1'\n"
        "from agent.plugin_composition.model import ServiceKey\n"
        "async def apply(ctx):\n"
        "    @ctx.entrypoint\n"
        "    async def read():\n"
        "        return (ctx.runtime.plugin_dir / 'resource.txt').read_text()\n"
        "    await ctx.provide(ServiceKey('probe.read'), read)\n")
    (source / "requirements.txt").write_text("")
    (source / "resource.txt").write_text("original-resource")
    command(["git", "init", "-q", str(source)], env=env, cwd=output, log=output / "git-init.log")
    command(["git", "add", "."], env=env, cwd=source, log=output / "git-add.log")
    command(["git", "-c", "user.name=e2e", "-c", "user.email=e2e@example.invalid", "-c",
             "commit.gpgsign=false", "commit", "-qm", "fixture"], env=env, cwd=source, log=output / "git-commit.log")
    command([sys.executable, "-c", SEED, str(workspace), str(plugins_home), str(source)],
            env=env, cwd=old_core, log=output / "old-seed.log")
    # 1. 通过真实旧安装制品修改外部资源和配置，旧归档继续原样留在磁盘。
    base = plugins_home / "cache/lab/archive_probe"
    pointer = json.loads((base / ".pointers.json").read_text())["stable"]
    installed = base / pointer
    (installed / "resource.txt").write_text("external-current-resource")
    from agent.plugin_composition.config_input import save_config
    save_config(workspace / "plugin-data/archive_probe-lab", {"message": "external-current"})
    (workspace / "plugin-data/archive_probe-lab/business.bin").write_bytes(b"retained")
    before_messages, old_archives = messages(workspace), files(workspace / "runtime/plugin-archives")
    assert old_archives
    try:
        PluginSelection(workspace).read()
    except SelectionFormatError:
        pass
    else:
        raise AssertionError("旧指针被正常运行路径自动接纳")
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT), str(ROOT / "sdk/python/src")))
    # 2. 离线 CLI 生成恢复点；Core 迁移之后才启动新版 runtime。
    command([sys.executable, str(ROOT / "scripts/upgrade_plugin_selection.py"), "--workspace", str(workspace),
             "--plugins-home", str(plugins_home), "--backup-dir", str(output / "recovery"), "--from-archive"],
            env=env, cwd=ROOT, log=output / "upgrade.log")
    assert (output / "recovery/workspace/runtime/plugin-stable.json").exists()
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(plugins_home))
    MigrationRunner(repo_root=ROOT, config_path=config, workspace=workspace, fixed_sources=()).run()
    report = asyncio.run(verify(workspace, plugins_home, source, installed))
    assert messages(workspace) == before_messages
    assert files(workspace / "runtime/plugin-archives") == old_archives
    report.update(old_archives_unchanged=True, messages_preserved=True, recovery_point=True)
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
