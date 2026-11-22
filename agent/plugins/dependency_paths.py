"""已安装外置插件的依赖环境挂进主进程：site-packages 追加到 sys.path 末尾，Core 自带的包永远优先。"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

from agent.plugins.python_environment import PythonEnvironments, inprocess_runtime

_ROOT_RUNTIME = ".inprocess"


class DependencyEnvironmentError(RuntimeError):
    """依赖环境无法挂进当前进程；只让该插件加载失败，错误信息带可执行的修复动作。"""


# 插件根显式声明 requirements.inprocess.txt 才有进程内环境；声明了但环境缺失（未安装或 Python 小版本变化后失效）报错。
def root_environment(code: Path, environments: PythonEnvironments) -> Path | None:
    if inprocess_runtime(code) is None:
        return None
    environment = environments.inprocess_environment(code)
    if environment is None:
        raise DependencyEnvironmentError(
            "插件声明了 requirements.inprocess.txt，但对应的依赖环境不存在或已失效（Python 版本变化？）；"
            "请重新安装该插件（pkg upgrade 或 plugin-install）以重建依赖环境")
    return environment


# 环境创建时的 Python 主.次版本，取自环境元数据里已有的 input.base.version。
def _environment_python(root: Path) -> tuple[int, int]:
    record = json.loads((root / "environment.json").read_text())
    version = str(record["input"]["base"]["version"]).split(" ", 1)[0]
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


# 环境的 site-packages；Python 主.次版本与当前解释器不同则拒绝挂载（原生扩展 ABI 不兼容）。
def site_packages(root: Path) -> Path:
    created = _environment_python(root)
    current = (sys.version_info.major, sys.version_info.minor)
    if created != current:
        raise DependencyEnvironmentError(
            f"插件依赖环境由 Python {created[0]}.{created[1]} 创建，当前解释器是 {current[0]}.{current[1]}；"
            "请重新安装该插件（pkg upgrade 或 plugin-install）以重建依赖环境")
    found = sorted((root / _ROOT_RUNTIME / ".venv" / "lib").glob("python*/site-packages"))
    if len(found) != 1:
        raise DependencyEnvironmentError(f"插件依赖环境缺少 site-packages: {root}")
    return found[0]


class DependencyPaths:
    """进程内登记表：每个插件至多一个已挂载的依赖目录；条目只在被替换时变动，卸载后留到进程重启。"""

    def __init__(self) -> None:
        self._attached: dict[str, Path] = {}

    # 旧目录里已有模块被导入时，换成另一个环境只能重启进程。
    def restart_needed(self, plugin_name: str, site: Path) -> bool:
        previous = self._attached.get(plugin_name)
        if previous is None or previous == site:
            return False
        prefix = str(previous) + "/"
        return any(str(getattr(module, "__file__", None) or "").startswith(prefix)
                   for module in tuple(sys.modules.values()))

    # 挂载到 sys.path 末尾；依赖环境没变不重复追加，旧环境无已导入模块时原位替换。
    def attach(self, plugin_name: str, site: Path) -> None:
        previous = self._attached.get(plugin_name)
        if previous == site:
            return
        if self.restart_needed(plugin_name, site):
            raise DependencyEnvironmentError(f"插件 {plugin_name} 的依赖环境已变化且旧环境的模块已导入，需要重启进程")
        if previous is not None and str(previous) in sys.path:
            sys.path.remove(str(previous))
        sys.path.append(str(site))
        self._attached[plugin_name] = site
        importlib.invalidate_caches()


dependency_paths = DependencyPaths()
