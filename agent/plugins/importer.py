from __future__ import annotations

import importlib.abc
import importlib.util
import hashlib
import marshal
import os
import sys
from pathlib import Path
from types import CodeType, ModuleType
from functools import lru_cache

from agent.plugins.manifest import plugins_root

# PEP 552 checked hash pyc：标志位 0b11，源码哈希一致才复用，不依赖 mtime。
_HASH_PYC_FLAGS = (0b11).to_bytes(4, "little")
_PYC_HEADER_SIZE = 16


@lru_cache(maxsize=128)
def _compile_source(source: bytes, filename: str) -> CodeType:
    """只复用相同路径和源码的字节码；模块字典与执行副作用仍每次独立。"""
    source_hash = importlib.util.source_hash(source)
    cache_path = _cache_path(filename)
    cached = _read_cached_code(cache_path, source_hash)
    if cached is not None:
        return cached
    code = compile(source, filename, "exec")
    _write_cached_code(cache_path, source_hash, code)
    return code


def ensure_source_compiles(source: bytes, filename: str) -> None:
    """校验源码可编译；PEP 552 缓存命中即证明同一内容此前编译通过。

    与插件导入共用同一份字节码缓存：安装校验期写下的 pyc 在真正导入时
    直接复用，换代重载也不再为未变化的源码重复编译。
    """
    _compile_source(source, filename)


# 编译校验事实按内容寻址：同一源码只需真正编译一次，命中只付一次哈希。
def _compile_fact_path(source: bytes) -> Path | None:
    tag = sys.implementation.cache_tag
    if tag is None:
        return None
    try:
        root = Path(sys.pycache_prefix) if sys.pycache_prefix else plugins_root() / "bytecode"
    except ValueError:
        return None
    digest = hashlib.sha256(source).hexdigest()
    return root / "compile-facts" / f"{digest}.{tag}.json"


def source_compile_verified(source: bytes) -> bool:
    """同一 Python 实现是否已编译通过这份源码；只读事实，不验证文件路径。"""
    path = _compile_fact_path(source)
    return path is not None and path.is_file()


def record_source_compile(source: bytes) -> None:
    """记录编译通过事实；只读安装目录或并发写失败时静默放弃，下次重新编译。"""
    path = _compile_fact_path(source)
    if path is None or sys.dont_write_bytecode:
        return
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = temporary.write_bytes(b"1")
        os.replace(temporary, path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


# 字节码属于宿主缓存，镜像源码绝对路径放在插件 home 下，不写入插件制品目录。
def _cache_path(filename: str) -> Path | None:
    tag = sys.implementation.cache_tag
    if tag is None:
        return None
    try:
        root = Path(sys.pycache_prefix) if sys.pycache_prefix else plugins_root() / "bytecode"
    except ValueError:
        return None
    source = Path(filename).resolve(strict=False)
    # 与 CPython 相同，优化级别进入文件名；-O 与非 -O 的字节码不能互相复用。
    level = sys.flags.optimize
    suffix = f"{tag}.opt-{level}" if level else tag
    return root.joinpath(*source.parent.parts[1:], f"{source.stem}.{suffix}.pyc")


def _read_cached_code(cache_path: Path | None, source_hash: bytes) -> CodeType | None:
    if cache_path is None:
        return None
    try:
        data = cache_path.read_bytes()
    except OSError:
        return None
    header = importlib.util.MAGIC_NUMBER + _HASH_PYC_FLAGS + source_hash
    if data[:_PYC_HEADER_SIZE] != header:
        return None
    try:
        code = marshal.loads(data[_PYC_HEADER_SIZE:])
    except (EOFError, ValueError, TypeError):
        return None
    return code if isinstance(code, CodeType) else None


# 只读安装目录或并发写失败时静默放弃缓存，编译结果仍然可用。
def _write_cached_code(cache_path: Path | None, source_hash: bytes, code: CodeType) -> None:
    if cache_path is None or sys.dont_write_bytecode:
        return
    data = importlib.util.MAGIC_NUMBER + _HASH_PYC_FLAGS + source_hash + marshal.dumps(code)
    temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        _ = temporary.write_bytes(data)
        os.replace(temporary, cache_path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


class FreshSourceLoader(importlib.abc.Loader):
    def __init__(self, path: Path) -> None:
        self._path = path

    def create_module(self, spec: object) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        source = self._path.read_bytes()
        code = _compile_source(source, str(self._path))
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
