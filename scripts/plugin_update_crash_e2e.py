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
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.plugins.artifacts import pointer_state_path  # noqa: E402

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
elif kill_at == "after-enable":
    install.set_plugin_choice = killing(install.set_plugin_choice)
elif kill_at == "before-bind":
    # selection CAS 已提交、更新记录尚未绑定时崩溃。
    def die(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGKILL)
    manager_module.PluginManager._bind_installed_inputs = die

from agent.plugin_composition.model import ServiceKey
from agent.plugins.manager import PluginManager

async def run():
    if mode == "offline":
        install.install_git_plugin(workspace=workspace, source=source, marketplace="lab",
                                   plugins_home=home, update_id=update_id)
        print(json.dumps({"installed": update_id}))
        return
    if mode == "disable":
        from agent.plugins.bundles import set_plugin_choice
        set_plugin_choice(workspace, "probe@lab", enabled=False)
        print(json.dumps({"disabled": True}))
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
    # 首次启动（selection 为空）也要把安装记录接到实际输入。
    first = child(case, env, "boot", "install-v1")
    assert first["running_version"] == "1" and cast(dict[str, object], first["install-v1"])["state"] == "active", first
    (source / "plugin.py").write_text(PROBE.format(version="2"))
    git(source, "commit", "-qam", "v2")
    return env


def run_case(output: Path, name: str, mode: str, kill_at: str) -> dict[str, object]:
    """崩溃一次后连续启动两次：第二次核对结算结果不随重启漂移。"""
    case = output / name
    env = prepare(case)
    update_id = f"{name}-v2"
    if name == "offline-reinstall-after-enable":
        # 停用后重装同一 commit：制品与指针都不变，只有启用写入。
        git(case / "source", "reset", "-q", "--hard", "HEAD~1")
        assert child(case, env, "disable") == {"disabled": True}
    if name in {"boot-before-bind", "bad-current-entry", "bad-current-module"}:
        assert child(case, env, "offline", update_id) == {"installed": update_id}
    if name in {"bad-current-entry", "bad-current-module"}:
        # 已提交的新制品内容损坏（入口或其他模块）：插件不可用，不退回旧版本，更新不得显示为 active。
        base = case / "plugins/cache/lab/probe"
        stable = json.loads(pointer_state_path(base).read_text())["stable"]
        broken = "plugin.py" if name == "bad-current-entry" else "helper.py"
        (base / stable / broken).write_text("def broken(:\n")
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


# 场景 -> (模式, 崩溃点, 期望运行版本, 期望 {更新: (phase, state)})；两次启动结果都必须等于期望。
CASES: dict[str, tuple[str, str, str | None, dict[str, tuple[str, str]]]] = {
    # 提交前崩溃：只留下未引用的制品或恢复点，期望启动旧版本。
    "offline-after-build": ("offline", "after-build", "1", {}),
    "offline-after-arm": ("offline", "after-arm", "1", {"v2": ("rolled_back", "failed")}),
    "online-after-build": ("online", "after-build", "1", {}),
    "online-after-arm": ("online", "after-arm", "1", {"v2": ("rolled_back", "failed")}),
    # 指针已原子换到新制品：磁盘已提交，期望启动新版本。
    "offline-after-pointer": ("offline", "after-pointer", "2", {"v2": ("committed", "active")}),
    "online-after-pointer": ("online", "after-pointer", "2", {"v2": ("committed", "active")}),
    # 在线安装已提交指针与安装记录，但运行选择尚未换代。
    "online-after-install": ("online", "after-install", "2", {"v2": ("committed", "active")}),
    "offline-retry-after-arm": ("offline", "after-arm", "2", {"v2": ("rolled_back", "failed"),
                                                             "retry-v2": ("committed", "active")}),
    # 同制品重装在启用写入后崩溃：启用已是提交事实，不得被撤销。
    "offline-reinstall-after-enable": ("offline", "after-enable", "1", {"v2": ("committed", "active")}),
    # selection 已换代、绑定前崩溃：下次启动补绑。
    "boot-before-bind": ("boot", "before-bind", "2", {"v2": ("committed", "active")}),
    # 已提交新制品入口损坏：插件不可用，不自动退回旧版本，更新不得显示为 active。
    "bad-current-entry": ("boot", "none", None, {"v2": ("committed", "unknown")}),
    "bad-current-module": ("boot", "none", None, {"v2": ("committed", "unknown")}),
    # 卸载已写停用选择、selection 尚未移除：期望启动时不再运行该插件。
    "online-uninstall-after-choice": ("uninstall", "after-choice", None, {}),
}


def verify(name: str, report: dict[str, object]) -> list[str]:
    """逐场景核对崩溃确实发生、两次启动的运行版本与更新状态都等于期望。"""
    mode, kill_at, version, updates = CASES[name]
    problems: list[str] = []
    if kill_at != "none" and report["crash"] != {"killed": kill_at}:
        problems.append(f"崩溃未按预期发生: {report['crash']}")
    for boot in ("first_boot", "second_boot"):
        result = cast(dict[str, object], report[boot])
        if "boot_error" in result or "exit" in result:
            problems.append(f"{boot} 失败: {result}")
            continue
        if result.get("running_version") != version:
            problems.append(f"{boot} 运行版本 {result.get('running_version')!r}，期望 {version!r}")
        for suffix, (phase, state) in updates.items():
            raw = result.get(f"{name}-{suffix}" if suffix == "v2" else suffix)
            item = cast(dict[str, object], raw) if isinstance(raw, dict) else {"phase": raw}
            if (item.get("phase"), item.get("state")) != (phase, state):
                problems.append(f"{boot} {suffix} 为 {item}，期望 {(phase, state)}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=sorted(CASES), action="append")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        shutil.rmtree(output)
    report = {name: run_case(output, name, *CASES[name][:2]) for name in args.case or sorted(CASES)}
    problems = {name: verify(name, item) for name, item in report.items()}
    (output / "report.json").write_text(json.dumps({"cases": report, "problems": problems}, ensure_ascii=False, indent=2))
    for name, items in problems.items():
        print(("PASS " if not items else "FAIL ") + name + "".join(f"\n  - {item}" for item in items))
    sys.exit(1 if any(problems.values()) else 0)


if __name__ == "__main__":
    main()
