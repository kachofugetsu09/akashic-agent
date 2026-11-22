"""已安装外置插件的依赖环境挂进主进程：site-packages 追加到 sys.path 末尾，Core 自带的包永远优先。"""
from __future__ import annotations

import importlib
import importlib.metadata
import sys
from pathlib import Path

from agent.plugins.python_environment import (
    PythonEnvironments, environment_python, inprocess_ref, inprocess_runtime, inprocess_site_dirs,
)


class DependencyEnvironmentError(RuntimeError):
    """依赖环境无法挂进当前进程；只让该插件加载失败，错误信息带可执行的修复动作。"""


# 进程内环境读安装时固定的引用（artifact 里的环境记录），不查任何可变指针；
# 缺记录、缺环境目录、元数据损坏、版本不匹配都转成带区分信息的 DependencyEnvironmentError。
def root_environment(code: Path, environments: PythonEnvironments) -> Path | None:
    # 1. 声明文件与安装记录的读取错误（坏 JSON、结构不对、不可读）也属于依赖环境边界，点名文件。
    try:
        if inprocess_runtime(code) is None:
            return None
        ref = inprocess_ref(code)
    except (OSError, ValueError) as error:
        raise DependencyEnvironmentError(
            f"无法读取插件的依赖环境记录（requirements.inprocess.txt 或 .akashic-python-environment）：{error}；"
            "请重新安装该插件以重建") from error
    if ref is None:
        raise DependencyEnvironmentError(
            "插件声明了 requirements.inprocess.txt，但安装记录里没有对应的依赖环境引用；请重新安装该插件")
    # 2. 引用指向的环境目录与元数据。
    try:
        return environments.open(ref)
    except FileNotFoundError as error:
        raise DependencyEnvironmentError(f"插件的依赖环境目录缺失（{ref[:12]}…）；请重新安装该插件以重建") from error
    except (OSError, ValueError) as error:
        raise DependencyEnvironmentError(f"插件的依赖环境元数据损坏：{error}；请重新安装该插件以重建") from error


# 环境的 site-packages；Python 主.次版本与当前解释器不同则拒绝挂载（原生扩展 ABI 不兼容）。
def site_packages(root: Path) -> Path:
    try:
        created = environment_python(root)
    except ValueError as error:
        raise DependencyEnvironmentError(f"插件的依赖环境元数据损坏：{error}；请重新安装该插件以重建") from error
    current = (sys.version_info.major, sys.version_info.minor)
    if created != current:
        raise DependencyEnvironmentError(
            f"插件依赖环境由 Python {created[0]}.{created[1]} 创建，当前解释器是 {current[0]}.{current[1]}；"
            "请重新安装该插件（pkg upgrade 或 plugin-install）以重建依赖环境")
    found = inprocess_site_dirs(root)
    if len(found) != 1:
        raise DependencyEnvironmentError(f"插件依赖环境缺少 site-packages: {root}")
    return found[0]


class DependencyPaths:
    """进程内登记表：每个插件至多一个已挂载的依赖目录；条目只在被替换时变动，卸载后留到进程重启。"""

    def __init__(self) -> None:
        self._attached: dict[str, Path] = {}
        self._restart_pending: set[str] = set()

    # 模块实际所在的位置：__file__、__spec__.submodule_search_locations 与 __path__（隐式命名空间包没有 __file__，
    # 位置在 __path__ 里）；归属必须按这些位置落在哪个已挂载环境下判断，不能只看顶层名字。
    @staticmethod
    def _locations(module: object) -> list[str]:
        found: list[str] = []
        file = getattr(module, "__file__", None)
        if file:
            found.append(str(file))
        spec = getattr(module, "__spec__", None)
        for item in getattr(spec, "submodule_search_locations", None) or ():
            found.append(str(item))
        try:
            found.extend(str(item) for item in getattr(module, "__path__", None) or ())
        except (TypeError, ImportError):
            pass
        return found

    def _imported_from(self, directory: Path) -> bool:
        prefix = str(directory) + "/"
        return any(location.startswith(prefix)
                   for module in tuple(sys.modules.values()) for location in self._locations(module))

    # 该分发包提供的模块名（点号形式）：取自它的文件清单；读不到时退回顶层名。
    @staticmethod
    def _provided(dist: importlib.metadata.Distribution) -> set[str]:
        names: set[str] = set()
        for file in dist.files or ():
            parts = file.parts
            if parts[0].endswith((".dist-info", ".data")) or not parts[-1].endswith((".py", ".so")):
                continue
            stem = parts[-1].split(".", 1)[0]
            names.add(".".join((*parts[:-1], "" if stem == "__init__" else stem)).strip("."))
        if not names:
            names = set((dist.read_text("top_level.txt") or "").split() or [dist.metadata["Name"].replace("-", "_")])
        return names

    # 该环境里的某个模块已经从另一个已挂载的插件环境（例如已卸载插件留下的）以不同版本导入。
    # Core 自带包的版本一致性由安装期的约束保证，这里只看插件环境之间的残留；命名空间子模块按位置归属判断。
    def _shadowed(self, site: Path) -> bool:
        others = tuple(str(path) + "/" for path in self._attached.values() if path != site)
        for dist in importlib.metadata.distributions(path=[str(site)]):
            for name in self._provided(dist):
                module = sys.modules.get(name)
                for location in () if module is None else self._locations(module):
                    if location.startswith(others) and self._loaded_version(location, dist.metadata["Name"]) != dist.version:
                        return True
        return False

    # 已导入的那份包的版本：读它所在环境里的 dist-info。
    @staticmethod
    def _loaded_version(origin: str, name: str) -> str | None:
        site = origin.split("/site-packages/", 1)[0] + "/site-packages"
        for dist in importlib.metadata.distributions(path=[site]):
            if dist.metadata["Name"].lower().replace("_", "-") == name.lower().replace("_", "-"):
                return dist.version
        return None

    # 已挂载的其他插件环境现在已不可用（目录被删、元数据坏）：它的模块可能仍驻留在本进程，版本事实读不到，
    # 约束构造也只能跳过它，所以不依赖逐模块检查，一律保守要求重启。
    def _attached_broken(self, plugin_name: str) -> bool:
        for name, site in self._attached.items():
            if name == plugin_name:
                continue
            try:
                environment_python(site.parents[4])
            except (ValueError, IndexError):
                return True
            if not site.is_dir():
                return True
        return False

    # 需要重启才能让该插件拿到自己的依赖：同插件旧环境的模块已导入；或（给出当前选择时）进程里还挂着
    # 不属于当前选择的环境；或它环境里的包已从别处以不同版本导入。
    def restart_needed(self, plugin_name: str, site: Path, selected: frozenset[str] | None = None) -> bool:
        previous = self._attached.get(plugin_name)
        if previous == site:
            return False
        if previous is not None and self._imported_from(previous):
            return True
        if self._attached_broken(plugin_name):
            return True
        if selected is not None and any(name not in selected for name in self._attached if name != plugin_name):
            return True
        return self._shadowed(site)

    # 因残留依赖需要重启的插件名；只在进程内存里，重启即清空，对账据此不再重试加载它们。
    def mark_restart_pending(self, plugin_name: str) -> None:
        self._restart_pending.add(plugin_name)

    def restart_pending(self, plugin_name: str) -> bool:
        return plugin_name in self._restart_pending

    # 挂载到 sys.path 末尾；依赖环境没变不重复追加，旧环境无已导入模块时原位替换。
    def attach(self, plugin_name: str, site: Path) -> None:
        previous = self._attached.get(plugin_name)
        if previous == site:
            return
        if self.restart_needed(plugin_name, site):
            raise DependencyEnvironmentError(f"插件 {plugin_name} 的依赖环境与进程里已导入的版本冲突，需要重启进程")
        if previous is not None and str(previous) in sys.path:
            sys.path.remove(str(previous))
        sys.path.append(str(site))
        self._attached[plugin_name] = site
        importlib.invalidate_caches()


dependency_paths = DependencyPaths()
