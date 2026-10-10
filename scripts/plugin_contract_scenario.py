"""在无 checkout 的 Core 副本中加载插件公共类型并验证停用、换代和重启。"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

CONTRACT = '''from dataclasses import dataclass
from agent.plugin_composition.model import ServiceKey

@dataclass(frozen=True, slots=True)
class Value:
    text: str

VALUE = ServiceKey[Value]("scenario.public-value")
'''
PROVIDER = '''from plugins.public_values.contract import VALUE, Value
api_version = 3
name = "public_values"
version = "1.0.0"
entrypoints = {"public-probe": "cli.main"}
async def apply(ctx):
    await ctx.provide(VALUE, Value("first"))
'''
CONSUMER = '''from plugins.public_values.contract import VALUE, Value
from agent.plugin_composition import ServiceKey
api_version = 3
name = "public_reader"
version = "1.0.0"
inject = (VALUE,)
READ = ServiceKey[object]("scenario.public-read")
async def apply(ctx):
    value = ctx.require(VALUE)
    if not isinstance(value, Value):
        raise TypeError("provider and consumer use different public types")
    await ctx.provide(READ, value.text)
'''


MANAGEMENT = '''from agent.plugin_composition import ServiceKey
from agent.plugin_composition.plugin_updates import PLUGIN_UPDATES
api_version = 3
name = "public_control"
version = "1.0.0"
inject = (PLUGIN_UPDATES,)
MANAGE = ServiceKey[object]("scenario.public-control")
async def apply(ctx):
    async def manage(action, **values):
        updates = ctx.require(PLUGIN_UPDATES)
        if action == "install":
            return await updates.install(ctx, values.get("update_id", "public-types-update"), source=values["source"], marketplace="lab")
        if action == "uninstall":
            return await updates.uninstall(ctx, "public_values@lab")
        if action == "drain":
            await updates.drain(ctx, "public_values@lab")
        elif action == "idle":
            await updates.wait_idle(ctx)
        elif action == "status":
            return updates.status(ctx)
        elif action == "read":
            return updates.read(ctx, values.get("update_id", "public-types-update"))
        else:
            raise ValueError(action)
    await ctx.provide(MANAGE, ctx.entrypoint(manage))
'''


def prepare(directory: Path) -> None:
    """复制真实 Core 与最小插件源码，子进程没有原仓库的导入路径。"""
    # 1. Core 产物中不包含 plugins，也没有源码 checkout 的 symlink。
    core = directory / "core"
    core.mkdir()
    for name in ("agent", "bootstrap", "core", "infra", "utils"):
        shutil.copytree(ROOT / name, core / name, ignore=shutil.ignore_patterns("__pycache__"))
    # 2. 公共模块与实现仍处于提供方自己的安装目录。
    plugins = directory / "installed"
    provider = plugins / "public_values"
    consumer = plugins / "public_reader"
    provider.mkdir(parents=True)
    consumer.mkdir()
    control = plugins / "public_control"
    control.mkdir()
    (control / "plugin.py").write_text(MANAGEMENT)
    (provider / "contract.py").write_text(CONTRACT)
    (provider / "plugin.py").write_text(PROVIDER)
    (provider / "cli.py").write_text('from plugins.public_values.contract import Value\n'
        'async def main(arguments, *, workspace, config_path):\n'
        '    value = Value("command")\n    print(value.text)\n    return 0\n')
    (consumer / "plugin.py").write_text(CONSUMER)
    # 包入口不能是隐含的业务加载路径。
    builtin = directory / "builtin/public_values"
    builtin.mkdir(parents=True)
    (builtin / "plugin.py").write_text(PROVIDER)
    (builtin / "contract.py").write_text(CONTRACT.replace("text: str", "text: str\n    builtin_only: bool = True"))
    (provider / "__init__.py").write_text('raise RuntimeError("package entry ran")\n')


async def exercise(directory: Path) -> dict[str, object]:
    """通过真实 Manager 生命周期检查公共类型与原安装选择。"""
    from agent.plugin_composition.model import ServiceKey
    from agent.plugins.manager import PluginManager
    from agent.plugins.install import install_git_plugin
    from agent.plugins.selection import PluginSelection
    from agent.plugins.manifest import set_plugin_enabled
    from agent.plugins.reload_journal import ReloadJournal

    workspace = directory / "workspace"
    workspace.mkdir()
    PluginSelection(workspace).initialize()
    source = directory / "installed"
    home = directory / "home"

    def commit(path: Path) -> None:
        subprocess.run(["git", "-C", str(path), "add", "."], check=True)
        subprocess.run(["git", "-C", str(path), "-c", "user.name=Scenario",
                        "-c", "user.email=scenario@example.invalid", "commit", "-qm", "scenario"], check=True)

    for name in ("public_values", "public_reader", "public_control"):
        path = source / name
        subprocess.run(["git", "init", "-q", "--initial-branch=source", str(path)], check=True)
        commit(path)
        install_git_plugin(workspace=workspace, source=str(path), marketplace="lab", plugins_home=home)

    def manager() -> PluginManager:
        return PluginManager(
            [directory / "builtin"], workspace=workspace,
            installed_cache_root=directory / "home/cache",
        )

    host = manager()
    try:
        # 1. 停用的已安装 API 仍优先于同名内置合同，不执行包入口。
        set_plugin_enabled("public_values@lab", enabled=False, plugins_home=home)
        host.discover()
        disabled_api = importlib.import_module("plugins.public_values.contract")
        assert "builtin_only" not in disabled_api.Value.__dataclass_fields__
        set_plugin_enabled("public_values@lab", enabled=True, plugins_home=home)
        await host.load_all()
        root = host.live_root
        assert root is not None
        assert root.context.require(ServiceKey("scenario.public-read")) == "first"
        manage = root.context.require(ServiceKey("scenario.public-control"))
        status = await manage("status")
        assert isinstance(status, dict) and status["selection_ref"] is not None
        module = sys.modules["plugins.public_values.contract"]
        assert module.__file__ is not None
        assert Path(module.__file__).is_relative_to(home / "cache")
        assert not (directory / "core/plugins").exists()
        # 2. 换代只改变实现，公共值类型与合同模块身份不变。
        (source / "public_values/plugin.py").write_text(PROVIDER.replace('Value("first")', 'Value("second")'))
        commit(source / "public_values")
        accepted = await manage("install", source=str(source / "public_values"))
        assert accepted.selection == "selected"
        await manage("idle")
        assert (await manage("read")).state == "active"
        assert root.context.require(ServiceKey("scenario.public-read")) == "second"
        assert sys.modules["plugins.public_values.contract"] is module
    finally:
        await host.terminate_all()
    # 3. 合同变更保留旧实例；更新状态属于真实 journal，不是发现异常。
    host = manager()
    try:
        await host.load_all()
        root = host.live_root
        assert root is not None
        manage = root.context.require(ServiceKey("scenario.public-control"))
        (source / "public_values/contract.py").write_text(
            CONTRACT.replace("text: str", "text: str\n    revision: int = 2"))
        cli = source / "public_values/cli.py"
        cli.write_text(cli.read_text().replace("print(value.text)", "print(value.text, value.revision)"))
        commit(source / "public_values")
        pending = await manage("install", source=str(source / "public_values"), update_id="contract-change")
        await manage("idle")
        assert pending.state == "restart_required" and not pending.error
        assert pending.active_input_ref != pending.input_ref and pending.fiber_state == "active"
        assert ReloadJournal(workspace).update("contract-change").restart_required
        assert (await manage("read", update_id="contract-change")).state == "restart_required"
        assert root.context.require(ServiceKey("scenario.public-read")) == "second"
        assert sys.modules["plugins.public_values.contract"] is module
        assert isinstance(await manage("status"), dict)
        from agent.plugins.entrypoints import invoke_plugin_command
        try:
            await invoke_plugin_command("public-probe", (), workspace=workspace,
                                        config_path=workspace / "config.toml")
        except RuntimeError as error:
            assert "须重启进程" in str(error)
        else:
            raise AssertionError("command mixed old public types with new selected source")
        # 4. pending 更新不锁死禁用和卸载；普通卸载保留制品与 journal。
        set_plugin_enabled("public_values@lab", enabled=False, plugins_home=home)
        await manage("drain")
        assert root.context.get(ServiceKey("scenario.public-read")) is None
        removed = await manage("uninstall")
        assert removed["state"] == "accepted"
        await manage("idle")
        assert ReloadJournal(workspace).update("contract-change").restart_required
        # 重新安装同一新合同，固定下一次启动的实际输入。
        pending = await manage("install", source=str(source / "public_values"), update_id="contract-reinstall")
        await manage("idle")
        assert pending.state == "restart_required"
    finally:
        await host.terminate_all()
    # 新 OS 进程不能继承 sys.modules 或 PublicContracts 的旧类型身份。
    subprocess.run([sys.executable, __file__, "--restart", str(directory)], check=True)
    return {"source_less_core": True, "public_type_shared": True,
            "package_entry_not_run": True, "generation": True, "restart": True, "disable": True,
            "management_from_plugin": True, "accepted_then_drained": True,
            "installed_overrides_builtin": True, "disabled_installed_api": True,
            "restart_required_persisted": True,
            "pending_update_manageable": True, "new_process_contract": True, "same_process_command_requires_restart": True}


async def check_restart(directory: Path) -> None:
    """新进程加载已提交的新合同和输入，不重新安装或猜测旧记录。"""
    from agent.plugin_composition.model import ServiceKey
    from agent.plugins.manager import PluginManager
    host = PluginManager([directory / "builtin"], workspace=directory / "workspace",
                         installed_cache_root=directory / "home/cache")
    try:
        await host.load_all()
        module = sys.modules["plugins.public_values.contract"]
        assert module.Value("new").revision == 2
        root = host.live_root
        assert root is not None
        assert root.context.require(ServiceKey("scenario.public-read")) == "second"
        assert host.read_update("contract-reinstall").state == "active"
    finally:
        await host.terminate_all()



def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--restart":
        directory = Path(sys.argv[2])
        sys.path.insert(0, str(directory / "core"))
        asyncio.run(check_restart(directory))
        return
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        directory = Path(sys.argv[2])
        sys.path.insert(0, str(directory / "core"))
        print(json.dumps(asyncio.run(exercise(directory))))
        return
    with tempfile.TemporaryDirectory(prefix="akashic-public-contract-") as temporary:
        directory = Path(temporary)
        prepare(directory)
        script = directory / "scenario.py"
        shutil.copyfile(__file__, script)
        environment = {**os.environ, "HOME": str(directory / "home"),
                       "PYTHONPATH": str(directory / "core"),
                       "AKASHIC_PLUGIN_HOME": str(directory / "home"),
                       "AKASHIC_PLUGIN_DISTRIBUTION": ""}
        subprocess.run([sys.executable, str(script), "--child", str(directory)],
                       cwd=directory, env=environment, check=True)


if __name__ == "__main__":
    main()
