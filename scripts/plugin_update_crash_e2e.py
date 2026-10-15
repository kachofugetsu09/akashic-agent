#!/usr/bin/env python3
"""在安装事务各阶段 SIGKILL 真实进程，再用新进程启动并核对磁盘结算结果。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

# 子进程只做一件事：安装、在线更新或启动；崩溃点在真实调用返回后立即 SIGKILL。
CHILD = '''import asyncio, json, os, signal, sys
from pathlib import Path
mode, workspace, home, source, update_id, kill_at = sys.argv[1:7]
workspace, home = Path(workspace), Path(home)

def killing(function):
    def wrapped(*args, **kwargs):
        result = function(*args, **kwargs)
        os.kill(os.getpid(), signal.SIGKILL)
        return result
    return wrapped

import agent.plugins.install as install
import agent.plugins.manager as manager_module
from agent.plugins.reload_journal import ReloadJournal
if kill_at == "after-build":
    install._sync_artifact_tree = killing(install._sync_artifact_tree)
elif kill_at == "after-arm":
    ReloadJournal.arm_update = killing(ReloadJournal.arm_update)
elif kill_at == "after-pointer":
    install.write_pointers = killing(install.write_pointers)
elif kill_at == "after-install":
    manager_module.install_git_plugin = killing(manager_module.install_git_plugin)
elif kill_at == "after-choice":
    manager_module.set_plugin_choice = killing(manager_module.set_plugin_choice)

from agent.plugin_composition.model import ServiceKey
from agent.plugins.manager import PluginManager

async def run():
    if mode == "offline":
        install.install_git_plugin(workspace=workspace, source=source, marketplace="lab",
                                   plugins_home=home, update_id=update_id)
        print(json.dumps({"installed": update_id}))
        return
    host = PluginManager([], workspace=workspace, installed_cache_root=home / "cache")
    report = {}
    try:
        await host.load_all()
        if mode == "online":
            await host.install(source=source, marketplace="lab", ref_name="", sparse_paths=[], update_id=update_id)
            await host._operation.task
        elif mode == "uninstall":
            await host.uninstall("probe@lab")
            await host._operation.task
        running = host.generation("probe@lab") is not None
        report["running_version"] = await host.live_root.context.require(ServiceKey("probe.version"))() if running else None
        for item in update_id.split(","):
            try:
                state = host.read_update(item)
            except KeyError:
                report[item] = "not_recorded"
                continue
            report[item] = {"phase": host._reload_journal.update(item).phase,
                            "state": state.state, "selection": state.selection}
    except Exception as error:
        report["boot_error"] = f"{type(error).__name__}: {error}"
    finally:
        await host.terminate_all()
    print(json.dumps(report, ensure_ascii=False))

asyncio.run(run())
'''

PROBE = '''api_version = 3
name = "probe"
version = "{version}"
from agent.plugin_composition.model import ServiceKey
async def apply(ctx):
    @ctx.entrypoint
    async def read():
        return version
    await ctx.provide(ServiceKey("probe.version"), read)
'''


def git(source: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(source), "-c", "user.name=e2e", "-c", "user.email=e2e@example.invalid",
                    "-c", "commit.gpgsign=false", *args], check=True, capture_output=True)


def child(case: Path, env: dict[str, str], mode: str, update_id: str = "", kill_at: str = "none") -> dict[str, object]:
    """运行一个真实子进程；被 SIGKILL 时返回退出码，正常时返回其报告。"""
    result = subprocess.run(
        [sys.executable, "-c", CHILD, mode, str(case / "workspace"), str(case / "plugins"),
         str(case / "source"), update_id, kill_at],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    log = case / f"{mode}-{update_id or 'boot'}-{kill_at}.log"
    log.write_text(result.stdout + result.stderr)
    if result.returncode == -9:
        return {"killed": kill_at}
    if result.returncode:
        return {"exit": result.returncode, "log": str(log)}
    return json.loads(result.stdout.strip().splitlines()[-1])


def prepare(case: Path) -> dict[str, str]:
    """隔离 HOME、插件目录与 workspace，安装 v1 并完成首次启动。"""
    case.mkdir(parents=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith("AKASHIC_")}
    env.update(HOME=str(case / "home"), AKASHIC_PLUGIN_HOME=str(case / "plugins"),
               PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "sdk/python/src"))))
    (case / "home").mkdir()
    subprocess.run([sys.executable, str(ROOT / "main.py"), "init", "--config", str(case / "config.toml"),
                    "--workspace", str(case / "workspace")], cwd=ROOT, env=env, check=True, capture_output=True)
    source = case / "source"
    source.mkdir()
    (source / "plugin.py").write_text(PROBE.format(version="1"))
    git(source, "init", "-q")
    git(source, "add", ".")
    git(source, "commit", "-qm", "v1")
    assert child(case, env, "offline", "install-v1") == {"installed": "install-v1"}
    assert child(case, env, "boot")["running_version"] == "1"
    (source / "plugin.py").write_text(PROBE.format(version="2"))
    git(source, "commit", "-qam", "v2")
    return env


def run_case(output: Path, name: str, mode: str, kill_at: str) -> dict[str, object]:
    """崩溃一次后连续启动两次：第二次核对结算结果不随重启漂移。"""
    case = output / name
    env = prepare(case)
    update_id = f"{name}-v2"
    report: dict[str, object] = {"crash": child(case, env, mode, update_id, kill_at)}
    if name == "offline-retry-after-arm":
        # 不经启动直接重试安装：新安装入口自己结算遗留记录。
        report["retry"] = child(case, env, "offline", "retry-v2")
        update_id += ",retry-v2"
    report["first_boot"] = child(case, env, "boot", update_id)
    report["second_boot"] = child(case, env, "boot", update_id)
    artifacts = case / "plugins/cache/lab/probe/.artifacts"
    report["artifact_dirs"] = sorted(path.name for path in artifacts.iterdir()) if artifacts.is_dir() else []
    return report


CASES = {
    # 提交前崩溃：只留下未引用的制品或恢复点，期望启动旧版本。
    "offline-after-build": ("offline", "after-build"),
    "offline-after-arm": ("offline", "after-arm"),
    # 指针已原子换到新制品：磁盘已提交，期望启动新版本。
    "offline-after-pointer": ("offline", "after-pointer"),
    "online-after-pointer": ("online", "after-pointer"),
    # 在线安装已提交指针与安装记录，但运行选择尚未换代。
    "online-after-install": ("online", "after-install"),
    "offline-retry-after-arm": ("offline", "after-arm"),
    # 卸载已写停用选择、selection 尚未移除：期望启动时不再运行该插件。
    "online-uninstall-after-choice": ("uninstall", "after-choice"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    report = {name: run_case(output, name, *CASES[name]) for name in args.case or sorted(CASES)}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
