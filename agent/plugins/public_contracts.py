"""从已解析插件源码发布公共合同，类型身份在本进程内保持不变。"""
from __future__ import annotations

import ast
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from collections.abc import Iterable

from agent.plugins.importer import FreshSourceLoader
from agent.plugins.manifest import plugins_root
from agent.plugins.source_resolver import ResolvedPluginSource


# 摘要算法指纹：同一源码只在算法与 Python 实现不变时复用缓存摘要。
@lru_cache(maxsize=1)
def _digest_fingerprint() -> str | None:
    try:
        parser = Path(__file__).read_bytes()
    except OSError:
        return None
    return hashlib.sha256(sys.implementation.cache_tag.encode() + b"\0" + parser).hexdigest()[:16]


def _digest_fact_path(source: bytes) -> Path | None:
    fingerprint = _digest_fingerprint()
    if fingerprint is None:
        return None
    try:
        root = plugins_root()
    except ValueError:
        return None
    digest = hashlib.sha256(source).hexdigest()
    return root / "contract-digests" / fingerprint / f"{digest}.json"


def _read_digest_fact(source: bytes) -> bytes | None:
    """读回的内容必须是合法摘要；任何损坏都当作未命中重新计算。"""
    path = _digest_fact_path(source)
    if path is None:
        return None
    try:
        value = path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    if len(value) != 64:
        return None
    try:
        digest = bytes.fromhex(value)
    except ValueError:
        return None
    return digest if len(digest) == 32 else None


# 只读安装目录或并发写失败时静默放弃缓存，摘要仍然每次现算。
def _write_digest_fact(source: bytes, digest: bytes) -> None:
    path = _digest_fact_path(source)
    if path is None or sys.dont_write_bytecode:
        return
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = temporary.write_text(digest.hex(), encoding="ascii")
        os.replace(temporary, path)
    except OSError:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


# 只对会影响类型身份的结构取摘要；注释、空白和 docstring 改动不要求重启进程。
def _contract_digest(path: Path) -> bytes:
    source = path.read_bytes()
    cached = _read_digest_fact(source)
    if cached is not None:
        return cached
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        # 语法错误留给该插件自己的导入失败，不在登记阶段拖垮其他插件。
        return hashlib.sha256(source).digest()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    digest = hashlib.sha256(ast.dump(tree).encode()).digest()
    _write_digest_fact(source, digest)
    return digest


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
            digest = _contract_digest(path)
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
                if loaded_path is None or _contract_digest(Path(loaded_path)) != digest:
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
            return importlib.util.spec_from_file_location(
                fullname, item.path, loader=FreshSourceLoader(item.path),
            )
        return None


# 公共值类型属于进程使用的已安装 API；插件实例仍由各自 Fiber/generation 拥有。
public_contracts = PublicContracts()
