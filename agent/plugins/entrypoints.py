"""按当前选择调用插件命令，不启动 Root、读取业务状态或持有工作区锁。"""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import inspect
from pathlib import Path
import secrets
import sys
from typing import Literal, cast

from agent.plugins.importer import FreshPluginImporter
from agent.plugins.public_contracts import public_contracts
from agent.plugins.selection import PluginSelection
from agent.plugins.source_resolver import ResolvedPluginSource
from agent.plugins.static_manifest import load_static_plugin_manifest


async def invoke_plugin_command(
    command: str, arguments: tuple[str, ...], *, workspace: Path, config_path: Path,
) -> int:
    """执行唯一被选择的命令 provider；资源与失败回执由命令实现负责。"""
    # 1. 唯一选择只读；不从缺失、损坏或未提交输入猜测 provider。
    selection = PluginSelection(workspace)
    reference = selection.read()
    if reference is None:
        raise LookupError(f"当前空选择没有命令 provider: {command}")
    sources: list[ResolvedPluginSource] = []
    matches: list[tuple[Path, str]] = []
    for identity in selection.components(reference):
        record = selection.read_input(identity)
        code = Path(cast(str, record["code"]))
        name, _, marketplace = cast(str, record["plugin_id"]).partition("@")
        sources.append(ResolvedPluginSource(code, cast(Literal["builtin", "installed"], record["source_type"]),
                                            marketplace, name))
        if record["version"] == 6:
            declarations = cast(dict[str, str], record["entrypoints"])
            if command in declarations:
                manifest = load_static_plugin_manifest(code)
                if manifest.name != name or dict(manifest.entrypoints).get(command) != declarations[command]:
                    raise ValueError(f"所选插件命令源码声明已变化: {record['plugin_id']}")
                matches.append((code, declarations[command]))
    if len(matches) != 1:
        raise LookupError(f"命令 {command} 需要唯一 provider，当前有 {len(matches)} 个")
    public_contracts.register(sources)
    # 2. 独立模块作用域只导入所声明的命令；plugin.py 的 apply 和包入口不执行。
    code, target = matches[0]
    prefix = "_plugin_command_" + secrets.token_hex(16)
    importer = FreshPluginImporter()
    importer.register(prefix, code)
    spec = importlib.machinery.ModuleSpec(prefix, None, is_package=True)
    spec.submodule_search_locations = [str(code)]
    sys.modules[prefix] = importlib.util.module_from_spec(spec)
    try:
        module_name, _, function_name = target.rpartition(".")
        module = importlib.import_module(f"{prefix}.{module_name}")
        function = getattr(module, function_name)
        result = function(arguments, workspace=workspace, config_path=config_path)
        if inspect.isawaitable(result):
            result = await result
        if type(result) is not int:
            raise TypeError(f"插件命令 {command} 必须返回进程退出码")
        return result
    finally:
        # 3. 命令实现退出并清理后释放其独立导入作用域，不撤下公共类型身份。
        importer.unregister(prefix)
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(prefix + "."):
                del sys.modules[name]
