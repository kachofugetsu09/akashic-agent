from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast
from uuid import uuid4

from agent.plugins.archive import (
    PluginArchive,
    encode_tree,
    tree_entries,
    sync_directory,
)
from agent.plugins.static_manifest import StaticPluginManifest, StaticPythonRuntime

ENVIRONMENT_FILE = ".akashic-python-environment"


class PythonEnvironments:
    """在最终路径创建固定 Python 环境；恢复只验证，不安装或修复依赖。"""

    def __init__(self, workspace: Path) -> None:
        self.path = workspace / "runtime" / "plugin-python-environments"
        if self.path.is_symlink():
            raise ValueError("Python 环境根不能是符号链接")
        self._archive_path = workspace / "runtime" / "plugin-archives"

    @property
    def archive(self) -> PluginArchive:
        if not self._archive_path.is_dir():
            raise FileNotFoundError("Python 环境归档缺失")
        return PluginArchive(self._archive_path)

    def prepare(self, code: Path, runtime: StaticPythonRuntime) -> str:
        """安装 owner 首次解析依赖；环境始终留在创建时的最终目录。"""
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._archive_path.mkdir(mode=0o700, parents=True, exist_ok=True)
        # 1. pip 可引用同包内文件，因此固定整个安装输入，不能只 hash 顶层 requirements。
        code_id = self.archive.save(
            code, exclude=frozenset({".venv", "node_modules", ENVIRONMENT_FILE})
        )
        source = self.archive.open(code_id)
        base = _base_profile(
            Path(sys.base_prefix)
            / "bin"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
        )
        requirements = _requirements(source, runtime)
        input_value: dict[str, object] = {
            "code": code_id,
            "base": base,
            "requirements": requirements,
        }
        input_id = hashlib.sha256(
            json.dumps(input_value, sort_keys=True).encode()
        ).hexdigest()
        pointer = self.path / (input_id + ".ref")
        if pointer.exists():
            if pointer.is_symlink():
                raise ValueError("Python 环境引用不能是符号链接")
            ref = pointer.read_text()
            _ = self.open(ref, source, runtime)
            return ref

        # 2. 不 rename venv；脚本 shebang 与 .pth 中的绝对路径从创建起就有效。
        location = uuid4().hex
        root = self.path / location
        root.mkdir(mode=0o700)
        published = False
        try:
            venv = root / runtime.runtime_root / ".venv"
            requirements_path = source / runtime.requirements
            has_requirements = bool(requirements_path.read_text().strip())
            options = [] if has_requirements else ["--without-pip"]
            _run(
                [
                    cast(str, base["executable"]),
                    "-I",
                    "-m",
                    "venv",
                    "--copies",
                    *options,
                    str(venv),
                ],
                source,
            )
            if has_requirements:
                # 本地 wheel/build/editable 的输入也留在最终路径；pip 无权写代码归档。
                build_source = root / "source"
                _ = shutil.copytree(source, build_source, symlinks=True)
                requirements_path = build_source / runtime.requirements
                _run(
                    [
                        str(venv / "bin/python"),
                        "-E",
                        "-s",
                        "-m",
                        "pip",
                        "--disable-pip-version-check",
                        "install",
                        "-r",
                        str(requirements_path),
                    ],
                    build_source,
                )
            if _base_profile(Path(cast(str, base["executable"]))) != base:
                raise RuntimeError("创建环境期间 base Python 发生变化")
            entries = tree_entries(root)
            for relative, kind, _ in entries:
                path = root / relative
                if kind == "file":
                    path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
                    with path.open("rb") as stream:
                        os.fsync(stream.fileno())
            for current, _, _ in os.walk(root, topdown=False):
                sync_directory(Path(current))
            sync_directory(self.path)
            ref = self.archive.save_descriptor(
                {
                    "version": 1,
                    "location": location,
                    "input": input_value,
                    "tree": hashlib.sha256(encode_tree(entries)).hexdigest(),
                }
            )
            published = True
            # 3. 并发首次解析保留各自材料；后续 admission 使用第一个完整引用。
            fd, name = tempfile.mkstemp(prefix=".pending-", dir=self.path)
            pending = Path(name)
            try:
                with os.fdopen(fd, "w") as stream:
                    _ = stream.write(ref)
                    stream.flush()
                    os.fsync(stream.fileno())
                try:
                    os.link(pending, pointer)
                except FileExistsError:
                    if pointer.is_symlink():
                        raise ValueError("Python 环境引用不能是符号链接")
                    ref = pointer.read_text()
                sync_directory(self.path)
            finally:
                pending.unlink()
            _ = self.open(ref, source, runtime)
            return ref
        finally:
            if not published:
                shutil.rmtree(root)

    def open(self, ref: str, code: Path, runtime: StaticPythonRuntime) -> Path:
        """验证完整恢复材料；缺失、损坏或解释器变化都不能借用当前环境。"""
        if self.path.is_symlink() or not self.path.is_dir():
            raise FileNotFoundError("Python 环境根缺失或是符号链接")
        record = self.archive.read_descriptor(ref)
        if record["version"] != 1:
            raise ValueError("Python 环境版本不兼容")
        location = cast(str, record["location"])
        if len(location) != 32 or any(
            char not in "0123456789abcdef" for char in location
        ):
            raise ValueError("Python 环境路径身份无效")
        root = self.path / location
        input_value = cast(Mapping[str, object], record["input"])
        expected_code = self.archive.open(cast(str, input_value["code"]))
        if tree_entries(code) != tree_entries(expected_code):
            raise RuntimeError("Python 环境与固定安装输入不一致")
        if input_value["requirements"] != _requirements(code, runtime):
            raise RuntimeError("Python 环境 requirements 不一致")
        base = cast(Mapping[str, object], input_value["base"])
        if _base_profile(Path(cast(str, base["executable"]))) != base:
            raise RuntimeError("Python 环境 base interpreter 已变化")
        if (
            hashlib.sha256(encode_tree(tree_entries(root))).hexdigest()
            != record["tree"]
        ):
            raise RuntimeError("Python 环境内容缺失或损坏")
        return root


def _base_profile(executable: Path) -> dict[str, object]:
    """固定机器上的 base Python；不声称环境可跨机器或系统升级搬运。"""
    path = executable.resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = subprocess.run(
        [
            str(path),
            "-I",
            "-S",
            "-c",
            "import json,sys; print(json.dumps([sys.version,sys.implementation.cache_tag,sys.base_prefix]))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "executable": str(path),
        "sha256": digest,
        "profile": tuple(json.loads(result.stdout)),
    }


def _requirements(code: Path, runtime: StaticPythonRuntime) -> dict[str, str]:
    return {
        runtime.requirements: hashlib.sha256(
            (code / runtime.requirements).read_bytes()
        ).hexdigest()
    }


def read_environment_refs(code: Path, manifest: StaticPluginManifest) -> dict[str, str]:
    """只读取安装 owner 已发布的环境选择。"""
    path = code / ENVIRONMENT_FILE
    if path.is_symlink():
        raise ValueError("Python 环境引用不能是符号链接")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Python 环境引用必须是映射")
    refs = cast(dict[str, object], value)
    if set(refs) != {item.runtime_root for item in manifest.python}:
        raise ValueError("Python 环境引用与 manifest 不一致")
    if any(
        not isinstance(ref, str)
        or len(ref) != 64
        or any(char not in "0123456789abcdef" for char in ref)
        for ref in refs.values()
    ):
        raise ValueError("Python 环境引用身份无效")
    return cast(dict[str, str], value)


def _run(command: list[str], cwd: Path) -> None:
    _ = subprocess.run(command, cwd=cwd, check=True, stdout=subprocess.DEVNULL)
