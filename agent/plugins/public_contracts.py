"""从已解析插件源码发布公共合同，类型身份在本进程内保持不变。"""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable

from agent.plugins.importer import FreshSourceLoader
from agent.plugins.source_resolver import ResolvedPluginSource


@dataclass(frozen=True)
class _Contract:
    owner: str
    path: Path
    digest: bytes


class PublicContracts(importlib.abc.MetaPathFinder):
    """只加载 contract.py，不执行插件包入口或 generation 的业务模块。"""

    def __init__(self) -> None:
        self._files: dict[str, _Contract] = {}

    def register(self, sources: Iterable[ResolvedPluginSource]) -> frozenset[str]:
        """登记实际源码中的公共模块；同进程改动合同必须明确重启。"""
        pending: dict[str, _Contract] = {}
        changed: set[str] = set()
        # 1. 发行版与外置安装均以已校验的 plugin identity 确定模块名。
        for source in sources:
            name = source.plugin_name.replace("-", "_")
            path = source.plugin_root / "contract.py"
            if not path.exists():
                if f"plugins.{name}" in self._files:
                    changed.add(source.plugin_name)
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"插件合同必须是普通文件: {path}")
            if not name.isidentifier():
                raise ValueError(f"插件名不能作为公共模块: {source.plugin_name}")
            module = f"plugins.{name}"
            digest = hashlib.sha256(path.read_bytes()).digest()
            candidate = pending.get(module)
            if candidate is not None and (candidate.owner != source.plugin_name or candidate.digest != digest):
                raise ValueError(f"同名安装源码的公共合同不同: {module}.contract")
            previous = self._files.get(module)
            if previous is not None:
                if previous.owner != source.plugin_name:
                    raise ValueError(f"公共模块名冲突: {module}")
                if previous.digest != digest:
                    changed.add(source.plugin_name)
                    continue
            loaded = sys.modules.get(f"{module}.contract")
            if previous is None and loaded is not None:
                loaded_path = loaded.__file__
                if loaded_path is None or hashlib.sha256(Path(loaded_path).read_bytes()).digest() != digest:
                    raise RuntimeError(f"已导入的公共合同与安装源码不同: {module}.contract")
            pending[module] = _Contract(source.plugin_name, path, digest)
        self._files.update(pending)
        # 2. namespace 包只服务于合同导入，不执行 __init__.py。
        if self._files and self not in sys.meta_path:
            sys.meta_path.insert(0, self)
        return frozenset(changed)

    def find_spec(
        self, fullname: str, path: object = None, target: object = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname == "plugins" or fullname in self._files:
            spec = importlib.machinery.ModuleSpec(fullname, None, is_package=True)
            if fullname == "plugins":
                paths = {str(item.path.parent.parent) for item in self._files.values()}
                spec.submodule_search_locations = sorted(paths)
            else:
                spec.submodule_search_locations = [str(self._files[fullname].path.parent)]
            return spec
        package, dot, leaf = fullname.rpartition(".")
        if dot and leaf == "contract" and package in self._files:
            item = self._files[package]
            if hashlib.sha256(item.path.read_bytes()).digest() != item.digest:
                raise RuntimeError(f"公共合同在登记后变化，须重启进程: {fullname}")
            return importlib.util.spec_from_file_location(
                fullname, item.path, loader=FreshSourceLoader(item.path),
            )
        return None


# 公共值类型属于进程使用的已安装 API；插件实例仍由各自 Fiber/generation 拥有。
public_contracts = PublicContracts()
