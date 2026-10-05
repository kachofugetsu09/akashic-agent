from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from uuid import uuid4
from utils.timing import measure

from agent.plugins.files import encode_tree, sync_directory, tree_entries
from agent.plugins.static_manifest import StaticPluginManifest, StaticPythonRuntime

ENVIRONMENT_FILE = ".akashic-python-environment"


@dataclass(frozen=True)
class OfflineWheels:
    """A wheel directory and the digest expected by its caller."""

    directory: Path
    tree_sha256: str


def wheel_tree_sha256(directory: Path) -> str:
    """Hash sorted wheel names and bytes without following links."""

    # 1. Check every path segment, then lock each file to the inode being hashed.
    if not isinstance(directory, Path) or not directory.is_absolute():
        raise ValueError("离线 wheel 目录必须是绝对路径")
    current = Path(directory.anchor)
    for part in directory.parts[1:]:
        current /= part
        info = current.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError(f"离线 wheel 路径不是普通目录: {current}")
    entries = sorted(directory.iterdir(), key=lambda item: item.name.encode("utf-8"))
    if not entries:
        raise ValueError("离线 wheel 目录不能为空")
    # 2. The exact serialization is UTF-8 filename, NUL, lowercase hex SHA-256, LF.
    digest = hashlib.sha256()
    from pip._vendor.packaging.utils import InvalidWheelFilename, parse_wheel_filename

    for path in entries:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or path.suffix != ".whl":
            raise ValueError(f"离线输入只能包含普通 .whl 文件: {path}")
        try:
            _ = parse_wheel_filename(path.name)
        except InvalidWheelFilename as error:
            raise ValueError(f"离线 wheel 文件名无效: {path.name}") from error
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            opened = os.fstat(fd)
            if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
                raise RuntimeError(f"离线 wheel 在读取前变化: {path}")
            with os.fdopen(fd, "rb", closefd=False) as stream:
                file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            after = os.fstat(fd)
            if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
                before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
            ):
                raise RuntimeError(f"离线 wheel 在读取中变化: {path}")
        finally:
            os.close(fd)
        digest.update(path.name.encode("utf-8") + b"\0" + file_hash.encode("ascii") + b"\n")
    return digest.hexdigest()


def _verify_offline_wheels(wheels: OfflineWheels) -> str:
    if not isinstance(wheels, OfflineWheels) or not isinstance(wheels.tree_sha256, str):
        raise ValueError("离线 wheel 输入无效")
    expected = wheels.tree_sha256
    if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
        raise ValueError("离线 wheel 摘要必须是小写 SHA-256")
    actual = wheel_tree_sha256(wheels.directory)
    if actual != expected:
        raise ValueError("离线 wheel 目录摘要不匹配")
    return actual


def _check_offline_requirements(path: Path) -> None:
    """Accept only named PEP 508 packages, leaving the source file unchanged."""

    from pip._vendor.packaging.requirements import InvalidRequirement, Requirement

    lines = path.read_text(encoding="utf-8").splitlines()
    found = False
    for number, raw in enumerate(lines, 1):
        line = raw.split(" #", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        if "\\" in line:
            raise ValueError(f"离线 requirements 第 {number} 行不支持续行或路径")
        if re.search(r"\.(?:tar(?:\.(?:gz|bz2|xz|zst))?|tgz|zip|whl)(?:\s|$)", line, re.IGNORECASE):
            raise ValueError(f"离线 requirements 第 {number} 行不支持本地包或源码包")
        if (path.parent / line).exists():
            raise ValueError(f"离线 requirements 第 {number} 行不支持本地路径")
        try:
            requirement = Requirement(line)
        except InvalidRequirement as error:
            raise ValueError(f"离线 requirements 第 {number} 行不是普通包要求: {line}") from error
        if requirement.url is not None:
            raise ValueError(f"离线 requirements 第 {number} 行不支持 URL/VCS: {line}")
        found = True
    if not found:
        raise ValueError("离线 wheel 输入只用于非空普通 requirements")


def preflight_offline_runtime(
    code: Path, runtime: StaticPythonRuntime, wheels: OfflineWheels, destination: Path,
) -> None:
    """Check the target interpreter's wheel closure before state migration."""

    requirements = code / runtime.requirements
    _check_offline_requirements(requirements)
    _ = _verify_offline_wheels(wheels)
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError("离线依赖检查目标必须是已有普通目录")
    _ = subprocess.run(
        [sys.executable, "-I", "-m", "pip", "--isolated", "download", "--no-index",
         f"--find-links={wheels.directory}", "--only-binary=:all:",
         "--no-cache-dir", "--no-input", "--dest", str(destination),
         "-r", str(requirements)],
        check=True, capture_output=True, text=True,
    )
    _ = _verify_offline_wheels(wheels)


class PythonEnvironments:
    """安装时准备最终路径，运行时直接读取已选环境。"""

    def __init__(self, workspace: Path) -> None:
        self.path = workspace / "runtime" / "plugin-python-environments"
        if self.path.is_symlink():
            raise ValueError("Python 环境根不能是符号链接")

    def prepared(self, code: Path, runtime: StaticPythonRuntime, *, wheel_digest: str = "") -> str:
        """读取安装阶段准备的环境，加载阶段不安装依赖。"""
        value = _environment_input(code, runtime, wheel_digest)
        pointer = self.path / (hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest() + ".ref")
        if pointer.is_symlink():
            raise ValueError("Python 环境引用不能是符号链接")
        ref = pointer.read_text()
        self.open(ref)
        return ref

    def prepare(
        self, code: Path, runtime: StaticPythonRuntime, *,
        offline_wheels: OfflineWheels | None = None,
    ) -> str:
        """Measure dependency preparation, including cache reuse."""
        with measure("plugin.environment", plugin=code.name, runtime=runtime.runtime_root) as timing:
            ref, reused = self._prepare(code, runtime, offline_wheels=offline_wheels)
            timing.update(ref=ref, reused=reused)
            return ref

    def _prepare(
        self, code: Path, runtime: StaticPythonRuntime, *,
        offline_wheels: OfflineWheels | None = None,
    ) -> tuple[str, bool]:
        """安装 owner 首次解析依赖；环境始终留在创建时的最终目录。"""
        wheel_digest = None
        if offline_wheels is not None:
            _check_offline_requirements(code / runtime.requirements)
            wheel_digest = _verify_offline_wheels(offline_wheels)
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        # 1. 环境复用只比较安装依赖输入，不保存插件运行代码快照。
        source = code.resolve(strict=True)
        executable = (
            Path(sys.base_prefix)
            / "bin"
            / f"python{sys.version_info.major}.{sys.version_info.minor}"
        ).resolve(strict=True)
        input_value = _environment_input(source, runtime, wheel_digest or "")
        input_id = hashlib.sha256(
            json.dumps(input_value, sort_keys=True).encode()
        ).hexdigest()
        pointer = self.path / (input_id + ".ref")
        if pointer.exists():
            if pointer.is_symlink():
                raise ValueError("Python 环境引用不能是符号链接")
            ref = pointer.read_text()
            _ = self.open(ref)
            return ref, True

        # 2. 不 rename venv；脚本 shebang 与 .pth 中的绝对路径从创建起就有效。
        location = uuid4().hex + uuid4().hex
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
                    str(executable),
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
                # 构建副本用于本地包和 editable 的绝对路径，运行代码仍来自安装目录。
                build_source = source
                if offline_wheels is None:
                    build_source = root / "source"
                    _ = shutil.copytree(source, build_source, symlinks=True,
                                        ignore=shutil.ignore_patterns(".git", ".venv", "node_modules", "__pycache__", ENVIRONMENT_FILE))
                    requirements_path = build_source / runtime.requirements
                    command = [
                        str(venv / "bin/python"), "-E", "-s", "-m", "pip",
                        "--disable-pip-version-check", "install", "-r", str(requirements_path),
                    ]
                else:
                    _ = _verify_offline_wheels(offline_wheels)
                    command = [
                        str(venv / "bin/python"), "-I", "-m", "pip", "--isolated",
                        "--disable-pip-version-check", "install", "--no-index",
                        f"--find-links={offline_wheels.directory}", "--only-binary=:all:",
                        "--no-cache-dir", "--no-input", "-r", str(requirements_path),
                    ]
                with measure("plugin.pip", plugin=code.name, runtime=runtime.runtime_root):
                    _run(command, build_source)
            with measure("plugin.environment.sync", plugin=code.name):
                for current, _, files in os.walk(root, topdown=False):
                    for name in files:
                        path = Path(current) / name
                        if path.is_symlink():
                            continue
                        path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
                        with path.open("rb") as stream:
                            os.fsync(stream.fileno())
                    sync_directory(Path(current))
                sync_directory(self.path)
            if offline_wheels is not None:
                _ = _verify_offline_wheels(offline_wheels)
            if _environment_input(source, runtime, wheel_digest or "") != input_value:
                raise RuntimeError("插件依赖输入在准备环境期间发生变化")
            # 环境元数据随实际目录保存；引用就是目录身份，无需另一个归档 owner。
            metadata = root / "environment.json"
            with metadata.open("x", encoding="utf-8") as stream:
                json.dump({"version": 2, "input": input_value}, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            sync_directory(root)
            sync_directory(self.path)
            ref = location
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
            _ = self.open(ref)
            return ref, False
        finally:
            if not published:
                shutil.rmtree(root)

    def open(self, ref: str) -> Path:
        """按安装引用读取环境目录，不扫描内容或探测解释器。"""
        if self.path.is_symlink() or not self.path.is_dir():
            raise FileNotFoundError("Python 环境根缺失或是符号链接")
        if re.fullmatch(r"(?:[0-9a-f]{32}|[0-9a-f]{64})", ref) is None:
            raise ValueError("Python 环境路径身份无效")
        location = ref
        root = self.path / location
        if root.is_symlink() or not root.is_dir():
            raise FileNotFoundError(f"Python 环境目录缺失或是符号链接: {root}")
        metadata = root / "environment.json"
        if metadata.is_symlink():
            raise ValueError("Python 环境元数据不能是符号链接")
        record = json.loads(metadata.read_text())
        if not isinstance(record, dict) or set(record) != {"version", "input"} or record["version"] != 2:
            raise ValueError("Python 环境格式不兼容；请显式升级或重新安装")
        return root


def _environment_input(code: Path, runtime: StaticPythonRuntime, wheel_digest: str) -> dict[str, object]:
    """Key named wheels by dependencies; retain source identity for local builds."""
    requirements = (code / runtime.requirements).read_bytes()
    value: dict[str, object] = {
        "base": {"executable": str((Path(sys.base_prefix) / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}").resolve(strict=True)),
                 "version": sys.version},
        "runtime_root": runtime.runtime_root,
    }
    if wheel_digest or not requirements.strip():
        value["requirements_sha256"] = hashlib.sha256(requirements).hexdigest()
    else:
        value["code"] = hashlib.sha256(encode_tree(tree_entries(
            code, exclude=frozenset({".venv", "node_modules", ENVIRONMENT_FILE}),
        ))).hexdigest()
    if wheel_digest:
        value["wheel_tree_sha256"] = wheel_digest
    return value


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
        or len(ref) not in {32, 64}
        or any(char not in "0123456789abcdef" for char in ref)
        for ref in refs.values()
    ):
        raise ValueError("Python 环境引用身份无效")
    return cast(dict[str, str], value)


def _run(command: list[str], cwd: Path) -> None:
    _ = subprocess.run(command, cwd=cwd, check=True, stdout=subprocess.DEVNULL)
