from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


class FreshSourceLoader(importlib.abc.Loader):
    def __init__(self, path: Path) -> None:
        self._path = path

    def create_module(self, spec: object) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        source = self._path.read_bytes()
        code = compile(source, str(self._path), "exec")
        exec(code, module.__dict__)


class FreshPluginImporter(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self._roots: dict[str, Path] = {}

    def register(self, module_name: str, plugin_root: Path) -> None:
        if not self._roots:
            sys.meta_path.insert(0, self)
        self._roots[module_name] = plugin_root.resolve(strict=False)

    def unregister(self, module_name: str) -> None:
        _ = self._roots.pop(module_name, None)
        if not self._roots and self in sys.meta_path:
            sys.meta_path.remove(self)

    def root_spec(self, module_name: str, path: Path):
        return importlib.util.spec_from_file_location(
            module_name,
            path,
            loader=FreshSourceLoader(path),
            submodule_search_locations=[str(path.parent)],
        )

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ):
        for module_name, plugin_root in self._roots.items():
            prefix = f"{module_name}."
            if not fullname.startswith(prefix):
                continue
            relative = fullname.removeprefix(prefix).split(".")
            module_path = plugin_root.joinpath(*relative).with_suffix(".py")
            if module_path.is_file():
                self._require_inside(plugin_root, module_path)
                return importlib.util.spec_from_file_location(
                    fullname,
                    module_path,
                    loader=FreshSourceLoader(module_path),
                )
            package_dir = plugin_root.joinpath(*relative)
            package_init = package_dir / "__init__.py"
            if package_init.is_file():
                self._require_inside(plugin_root, package_init)
                return importlib.util.spec_from_file_location(
                    fullname,
                    package_init,
                    loader=FreshSourceLoader(package_init),
                    submodule_search_locations=[str(package_dir)],
                )
        return None

    @staticmethod
    def _require_inside(plugin_root: Path, path: Path) -> None:
        try:
            _ = path.resolve(strict=False).relative_to(plugin_root)
        except ValueError as error:
            raise ImportError(f"插件模块路径越界: {path}") from error
