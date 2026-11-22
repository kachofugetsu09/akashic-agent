from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, cast
from uuid import uuid4
from utils.timing import measure

from agent.plugins.files import encode_tree, sync_directory, tree_entries
from agent.plugins.requirements_check import BUNDLED_WHEELS_DIR, check_bundled_wheels
from agent.plugins.resources import declared_refs
from agent.plugins.static_manifest import StaticPluginManifest, StaticPythonRuntime

if TYPE_CHECKING:
    from agent.plugins.dependency_constraints import Constraints

ENVIRONMENT_FILE = ".akashic-python-environment"
# 显式声明"这些依赖要进主进程"的文件；存在且非空才有进程内环境，根 requirements.txt 仍只是子进程 runtime。
INPROCESS_REQUIREMENTS = "requirements.inprocess.txt"
# 进程内环境在 refs 映射里的键；磁盘发现的 runtime_root 不会带尾随斜杠，所以不可能与任何子进程 runtime 冲突。
INPROCESS_KEY = "inprocess/"
_INPROCESS_DIR = ".inprocess"


@dataclass(frozen=True, slots=True)
class InprocessRuntime(StaticPythonRuntime):
    """显式声明的进程内依赖 runtime；类型本身就是标记，磁盘发现的 runtime 永远是基类。"""


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

    from agent.plugins.requirements_check import check_plain_requirements

    # 先过统一的纯文本校验（与联网求 wheel 之前的检查是同一份），再逐行用 packaging 解析。
    _ = check_plain_requirements(path.read_text(encoding="utf-8"), str(path.name))
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


# 把 path 所在文件系统的脏数据整体刷盘（libc syncfs），比逐文件 fsync 快两个数量级。
# 返回 False 表示本机没有 syncfs，调用方改走逐文件 fsync；真实写回失败（EIO、ENOSPC 等）直接抛出，不能发布环境。
def _sync_filesystem(path: Path) -> bool:
    try:
        syncfs = ctypes.CDLL(None, use_errno=True).syncfs
    except (AttributeError, OSError):
        return False
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        if syncfs(fd) != 0:
            number = ctypes.get_errno()
            raise OSError(number, f"syncfs 失败: {os.strerror(number)}", str(path))
    finally:
        os.close(fd)
    return True


# 没有 syncfs 时的等价做法：逐个文件与目录 fsync。
def _sync_tree(root: Path, top: Path) -> None:
    for current, _, files in os.walk(root, topdown=False):
        for name in files:
            path = Path(current) / name
            if not path.is_symlink():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        sync_directory(Path(current))
    sync_directory(top)


def preflight_offline_runtime(
    code: Path, runtime: StaticPythonRuntime, wheels: OfflineWheels, destination: Path,
    constraints: Constraints | None = None,
) -> None:
    """Check the target interpreter's wheel closure before state migration.

    进程内 runtime 传入与安装期相同的约束：预检通过就意味着停机窗口内带约束的离线安装也能通过。
    """

    requirements = code / runtime.requirements
    _check_offline_requirements(requirements)
    _ = _verify_offline_wheels(wheels)
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError("离线依赖检查目标必须是已有普通目录")
    command = [sys.executable, "-I", "-m", "pip", "--isolated", "download", "--no-index",
               f"--find-links={wheels.directory}", "--only-binary=:all:",
               "--no-cache-dir", "--no-input", "--dest", str(destination),
               "-r", str(requirements), *_constraint_options(constraints, constraints is not None)]
    if constraints is not None:
        _run_pip(command, code, constraints)
    else:
        _ = subprocess.run(command, check=True, capture_output=True, text=True)
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
        offline_wheels: OfflineWheels | None = None, constraints: Constraints | None = None,
    ) -> str:
        """Measure dependency preparation, including cache reuse."""
        with measure("plugin.environment", plugin=code.name, runtime=runtime.runtime_root) as timing:
            ref, reused = self._prepare(code, runtime, offline_wheels=offline_wheels, constraints=constraints)
            timing.update(ref=ref, reused=reused)
            return ref

    def _prepare(
        self, code: Path, runtime: StaticPythonRuntime, *,
        offline_wheels: OfflineWheels | None = None, constraints: Constraints | None = None,
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
        # 只有显式声明的进程内环境带约束；其余 runtime 的路径与改动前完全相同。
        constrained = constraints is not None and isinstance(runtime, InprocessRuntime)
        if constrained:
            _check_offline_requirements(source / runtime.requirements)
        base_pointer = self.path / (_input_key(input_value, runtime) + ".ref")
        # 指针只是"可复用候选"的查找优化：候选必须仍满足当前约束才复用，不满足就新建另一个环境，
        # 绝不改写已有指针（别的安装正在引用它）；按约束摘要另存一个候选指针。
        alt_pointer = (self.path / f"{base_pointer.stem}.{constraints.digest}.ref"
                       if constraints is not None and constrained else base_pointer)
        stale: list[Path] = []
        running_python: Callable[[], tuple[int, int]] = lambda: (sys.version_info.major, sys.version_info.minor)
        for candidate in dict.fromkeys((alt_pointer, base_pointer)):
            if not candidate.exists():
                continue
            if candidate.is_symlink():
                raise ValueError("Python 环境引用不能是符号链接")
            ref = candidate.read_text()
            # 不可用的候选（目录缺失、元数据坏、缺 site-packages、Python 版本不符）不能复用，也不能让安装失败：跳过并新建。
            if environment_unusable_reason(self, ref, running_python, inprocess=isinstance(runtime, InprocessRuntime)) is not None:
                stale.append(candidate)
                continue
            if constraints is None or not constrained or _matches(self.open(ref), constraints):
                return ref, True
        # 候选指针只是查找优化，已装 artifact 按自己记录里的引用直接找环境，不经过它；所以指向不可用环境的指针可以被替换
        # （没有任何已装版本能靠它加载），之后的安装不会再先撞到它。指向可用环境但约束不符的指针仍只新增不改写。
        pointer = stale[0] if stale else base_pointer if not base_pointer.exists() else alt_pointer

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
                    if isinstance(runtime, InprocessRuntime):
                        # 进程内依赖与部署路径一致：只装 wheel（不现编源码包），pip 选项与环境变量不能改写行为，
                        # 插件自带的纯 Python wheel 由 Core 拼成 --find-links。
                        command = [
                            str(venv / "bin/python"), "-I", "-m", "pip", "--isolated", "--disable-pip-version-check",
                            "install", "--only-binary=:all:", "--no-input", "-r", str(requirements_path),
                            *_constraint_options(constraints, constrained), *_bundled_options(build_source),
                        ]
                    else:
                        command = [
                            str(venv / "bin/python"), "-E", "-s", "-m", "pip",
                            "--disable-pip-version-check", "install", "-r", str(requirements_path),
                            *_constraint_options(constraints, constrained),
                        ]
                else:
                    _ = _verify_offline_wheels(offline_wheels)
                    command = [
                        str(venv / "bin/python"), "-I", "-m", "pip", "--isolated",
                        "--disable-pip-version-check", "install", "--no-index",
                        f"--find-links={offline_wheels.directory}", "--only-binary=:all:",
                        "--no-cache-dir", "--no-input", "-r", str(requirements_path),
                        *_constraint_options(constraints, constrained),
                    ]
                with measure("plugin.pip", plugin=code.name, runtime=runtime.runtime_root):
                    if constrained:
                        _run_pip(command, build_source, constraints, isolated=isinstance(runtime, InprocessRuntime))
                    else:
                        _run(command, build_source)
            with measure("plugin.environment.sync", plugin=code.name):
                # 1. 先逐个设只读权限（权限是元数据，随后整体落盘），2. 再对环境所在文件系统做一次整体刷盘，替代逐文件 fsync。
                for current, _, files in os.walk(root, topdown=False):
                    for name in files:
                        path = Path(current) / name
                        if not path.is_symlink():
                            path.chmod(0o555 if path.stat().st_mode & 0o111 else 0o444)
                if not _sync_filesystem(self.path):
                    _sync_tree(root, self.path)
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
                    if pointer in stale:
                        os.replace(pending, pointer)
                    else:
                        os.link(pending, pointer)
                except FileExistsError:
                    if pointer.is_symlink():
                        raise ValueError("Python 环境引用不能是符号链接")
                    other = pointer.read_text()
                    # 并发安装先发布了指针：只有它可用才采用，否则保留自己新建的环境并替换这个不可用的指针。
                    if environment_unusable_reason(self, other, running_python, inprocess=isinstance(runtime, InprocessRuntime)) is None:
                        ref = other
                    else:
                        os.replace(pending, pointer)
                sync_directory(self.path)
            finally:
                pending.unlink(missing_ok=True)
            if (problem := environment_unusable_reason(self, ref, running_python, inprocess=isinstance(runtime, InprocessRuntime))) is not None:
                raise RuntimeError(f"新建的依赖环境不可用: {problem}")
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
    if wheel_digest or isinstance(runtime, InprocessRuntime) or not requirements.strip():
        value["requirements_sha256"] = hashlib.sha256(requirements).hexdigest()
    else:
        value["code"] = hashlib.sha256(encode_tree(tree_entries(
            code, exclude=frozenset({".venv", "node_modules", ENVIRONMENT_FILE}),
        ))).hexdigest()
    if wheel_digest:
        value["wheel_tree_sha256"] = wheel_digest
    if isinstance(runtime, InprocessRuntime):
        # 自带 wheel 的内容进入环境键：同样的 requirements 配不同（或没有）自带 wheel，不能互相复用环境；
        # 同时每次准备都校验 wheels/（含复用路径），不合格的目录一律拒绝。
        bundled = check_bundled_wheels(code / BUNDLED_WHEELS_DIR)
        if bundled:
            value["bundled_wheels_sha256"] = hashlib.sha256("".join(
                f"{wheel.name}\0{hashlib.sha256(wheel.read_bytes()).hexdigest()}\n" for wheel in bundled).encode()).hexdigest()
    return value


# 环境引用键；进程内环境只看 requirements 与 Python 主.次版本，补丁升级和只改代码的升级都复用同一个环境。
def _input_key(value: dict[str, object], runtime: StaticPythonRuntime) -> str:
    keyed = dict(value)
    if isinstance(runtime, InprocessRuntime):
        keyed["base"] = {"python": f"{sys.version_info.major}.{sys.version_info.minor}"}
    return hashlib.sha256(json.dumps(keyed, sort_keys=True).encode()).hexdigest()


# 插件根声明了 requirements.inprocess.txt 才有进程内环境；用一个合成 runtime，不进 manifest。
def inprocess_runtime(code: Path) -> InprocessRuntime | None:
    path = code / INPROCESS_REQUIREMENTS
    if path.is_symlink() or not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return None
    return InprocessRuntime(requirements=INPROCESS_REQUIREMENTS, runtime_root=_INPROCESS_DIR)


# 随安装固定下来的进程内环境引用（与子进程 runtime 同一份记录）；记录缺失或没有该键返回 None，
# 内容不是合法 JSON 映射、引用格式无效时抛 ValueError（读取错误由调用方转换）。
def inprocess_ref(code: Path) -> str | None:
    path = code / ENVIRONMENT_FILE
    if path.is_symlink() or not path.is_file():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{ENVIRONMENT_FILE} 必须是映射")
    ref = cast(dict[str, object], value).get(INPROCESS_KEY)
    if ref is None:
        return None
    if not isinstance(ref, str) or len(ref) not in {32, 64} or any(char not in "0123456789abcdef" for char in ref):
        raise ValueError(f"{ENVIRONMENT_FILE} 里的进程内环境引用无效")
    return ref


# 环境创建时的 Python 主.次版本，取自环境元数据里已有的 input.base.version；元数据读不出来抛 ValueError。
def environment_python(root: Path) -> tuple[int, int]:
    try:
        record = json.loads((root / "environment.json").read_text())
        major, minor = str(record["input"]["base"]["version"]).split(" ", 1)[0].split(".")[:2]
        return int(major), int(minor)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise ValueError(f"环境元数据无法读取: {error}") from error


# 进程内环境的 site-packages 候选（可能为空）。
def inprocess_site_dirs(root: Path) -> list[Path]:
    return sorted((root / _INPROCESS_DIR / ".venv" / "lib").glob("python*/site-packages"))


# 环境引用能否使用的唯一判定（repair 判定与 _prepare 的候选复用共用）：返回不可用原因，可用返回 None。
# 目录缺失、元数据损坏、（进程内环境）缺 site-packages、或创建环境的 Python 主.次版本与目标解释器不同都不可用。
# python 是函数：只有走到版本比较才求值（目标在镜像里时求值要起容器）。
def environment_unusable_reason(
    environments: PythonEnvironments, ref: str, python: Callable[[], tuple[int, int]], *, inprocess: bool,
) -> str | None:
    try:
        root = environments.open(ref)
        created = environment_python(root)
    except FileNotFoundError:
        return f"依赖环境目录缺失（{ref[:12]}）"
    except (OSError, ValueError) as error:
        return f"依赖环境元数据损坏: {error}"
    if inprocess and len(inprocess_site_dirs(root)) != 1:
        return f"依赖环境缺少 site-packages（{ref[:12]}）"
    target = python()
    if created != target:
        return f"依赖环境由 Python {created[0]}.{created[1]} 创建，目标解释器是 {target[0]}.{target[1]}"
    return None


# 已装 artifact 的进程内环境是否需要重建，部署、启动补装和 pkg 共用的唯一判定；返回原因，不需要返回 None。
# 声明了非空 requirements.inprocess.txt，但：安装记录没有 "inprocess/" 引用、引用指向的环境目录缺失或元数据损坏、
# 环境缺 site-packages，或创建环境的 Python 主.次版本与 python() 给出的目标解释器不同。
# python 是函数：只有走到版本比较才求值（目标在镜像里时求值要起容器）。
def environment_repair_reason(
    code: Path, environments: PythonEnvironments, python: Callable[[], tuple[int, int]],
) -> str | None:
    # 1. 没声明进程内依赖就没有这类环境。
    try:
        if inprocess_runtime(code) is None:
            return None
        ref = inprocess_ref(code)
    except (OSError, ValueError) as error:
        return f"安装记录无法读取: {error}"
    if ref is None:
        return "安装记录里没有进程内依赖环境引用"
    # 2. 引用必须指向可用的环境。
    return environment_unusable_reason(environments, ref, python, inprocess=True)


def read_environment_refs(code: Path, manifest: StaticPluginManifest) -> dict[str, str]:
    """只读取安装 owner 已发布的环境选择。"""
    path = code / ENVIRONMENT_FILE
    if path.is_symlink():
        raise ValueError("Python 环境引用不能是符号链接")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Python 环境引用必须是映射")
    refs = cast(dict[str, object], value)
    expected = {item.runtime_root for item in manifest.python}
    if inprocess_runtime(code) is not None:
        expected.add(INPROCESS_KEY)
    expected.update(declared_refs(code))
    if set(refs) != expected:
        raise ValueError("Python 环境引用与 manifest 不一致")
    if any(
        not isinstance(ref, str)
        or len(ref) not in {32, 64}
        or any(char not in "0123456789abcdef" for char in ref)
        for ref in refs.values()
    ):
        raise ValueError("Python 环境引用身份无效")
    return cast(dict[str, str], value)


# 插件根的 wheels/（校验后）作为 --find-links；没有则不加。
def _bundled_options(source: Path) -> list[str]:
    return ["--find-links", str(source / BUNDLED_WHEELS_DIR)] if check_bundled_wheels(source / BUNDLED_WHEELS_DIR) else []


def _constraint_options(constraints: Constraints | None, constrained: bool) -> list[str]:
    if constraints is None or not constrained:
        return []
    return [item for file in constraints.files for item in ("-c", str(file))]


# 复用前的核对；环境无法挂载（如 Python 版本已变）按"不一致"处理，走重建而不是报错。
def _matches(environment: Path, constraints: Constraints) -> bool:
    from agent.plugins.dependency_constraints import environment_matches
    from agent.plugins.dependency_paths import DependencyEnvironmentError, site_packages

    try:
        return environment_matches(site_packages(environment), constraints)
    except DependencyEnvironmentError:
        return False


# pip 失败时保留输出：版本冲突翻译成可操作的说明，其他失败带上 pip 输出末尾（均为 ValueError，网关会传回原文）。
def _run_pip(command: list[str], cwd: Path, constraints: Constraints | None, *, isolated: bool = False) -> None:
    from agent.plugins.dependency_constraints import explain_failure

    # 隔离时清掉继承来的 PIP_*（PIP_NO_BINARY、PIP_INDEX_URL 等不能改写安装行为）；代理变量保留。
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PIP_")} if isolated else None
    result = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True, env=environment)
    if result.returncode == 0:
        return
    output = result.stdout + result.stderr
    conflict = None if constraints is None else explain_failure(output, constraints)
    if conflict is not None:
        raise conflict
    raise ValueError("pip 安装依赖失败: " + " ".join(output.strip().splitlines()[-6:]))


def _run(command: list[str], cwd: Path) -> None:
    _ = subprocess.run(command, cwd=cwd, check=True, stdout=subprocess.DEVNULL)
