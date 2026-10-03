from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Mapping
from datetime import date, datetime, time
from pathlib import Path
from typing import cast
from utils.timing import measure

from session.message import freeze_json
from session.message_codec import json_value
from agent.plugin_composition.channels import CredentialRef

_CACHE_NAMES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


class PluginArchive:
    """按内容保存插件文件树；不拥有安装指针、业务状态或自动回收。"""

    def __init__(self, path: Path, *, create: bool = True):
        if path.is_symlink():
            raise ValueError("插件归档目录不能是符号链接")
        if create:
            path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = path.resolve()

    def save(self, source: Path, *, exclude: frozenset[str] = frozenset()) -> str:
        """Measure code archive work and report actual cache reuse."""
        with measure("plugin.archive", plugin=source.name) as timing:
            identity, reused = self._save(source, exclude=exclude)
            timing.update(ref=identity, reused=reused)
            return identity

    def _save(self, source: Path, *, exclude: frozenset[str]) -> tuple[str, bool]:
        """按复制后的内容命名并原子发布；返回前同步磁盘。"""
        # 1. 文件树是完整输入；运行环境等边界由调用者明确选定。
        if self.path.is_relative_to(source.resolve()):
            raise ValueError("插件归档不能写入自身输入目录")
        with measure("plugin.archive.hash", plugin=source.name):
            source_entries = tree_entries(source, exclude=exclude)
        identity = hashlib.sha256(encode_tree(source_entries)).hexdigest()
        if (self.path / identity).exists() or (self.path / identity).is_symlink():
            _ = self.open(identity)
            sync_directory(self.path)
            return identity, True
        pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.path))
        try:
            tree = pending / "tree"
            with measure("plugin.archive.copy", plugin=source.name):
                _ = shutil.copytree(
                    source,
                    tree,
                    symlinks=True,
                    ignore=shutil.ignore_patterns(*(_CACHE_NAMES | exclude)),
                )
                actual = tree_entries(tree)
                payload = encode_tree(actual)
                archive_id = hashlib.sha256(payload).hexdigest()
            # 2. 归档文件先落盘，再让内容身份可见。
            with measure("plugin.archive.sync", plugin=source.name, files=len(actual)):
                for relative, kind, _ in actual:
                    item = tree / relative
                    if kind == "file":
                        item.chmod(0o555 if item.stat().st_mode & 0o111 else 0o444)
                        with item.open("rb") as stream:
                            os.fsync(stream.fileno())
                for current, _, _ in os.walk(tree, topdown=False, followlinks=False):
                    sync_directory(Path(current))
                sync_directory(pending)
            # 3. 同内容复用已发布对象，不覆盖已有目录。
            target = self.path / archive_id
            if target.exists() or target.is_symlink():
                _ = self.open(archive_id)
            else:
                try:
                    _ = pending.rename(target)
                except OSError as error:
                    if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                        raise
                    _ = self.open(archive_id)
            sync_directory(self.path)
            return archive_id, False
        finally:
            # 只清理本次尚未发布的临时副本，已发布归档没有减少路径。
            if pending.exists():
                shutil.rmtree(pending)

    def open(self, archive_id: str) -> Path:
        """读取安装固定的目录，不重新计算内容摘要。"""
        if re.fullmatch(r"[0-9a-f]{64}", archive_id) is None:
            raise ValueError("插件归档身份必须是 SHA-256")
        root = self.path / archive_id
        tree = root / "tree"
        if root.is_symlink() or tree.is_symlink():
            raise ValueError("插件归档对象不能是符号链接")
        if not tree.is_dir():
            raise FileNotFoundError(f"插件归档目录缺失: {tree}")
        return tree

    def save_descriptor(self, value: Mapping[str, object]) -> str:
        """以内容身份发布不可变配置闭包，不覆盖已有恢复证据。"""
        payload = json.dumps(
            json_value(freeze_json(value)),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        identity = hashlib.sha256(payload).hexdigest()
        target = self.path / f"{identity}.json"
        if target.exists() or target.is_symlink():
            if target.is_symlink() or target.read_bytes() != payload:
                raise RuntimeError("插件归档 descriptor 损坏")
            sync_directory(self.path)
            return identity
        fd, name = tempfile.mkstemp(prefix=".pending-", dir=self.path)
        pending = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                _ = stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(pending, target)
            except FileExistsError:
                if target.is_symlink() or target.read_bytes() != payload:
                    raise RuntimeError("插件归档 descriptor 损坏")
            sync_directory(self.path)
            return identity
        finally:
            pending.unlink()

    def read_descriptor(self, identity: str) -> Mapping[str, object]:
        """读取已发布的 descriptor，保留结构和路径检查。"""
        if re.fullmatch(r"[0-9a-f]{64}", identity) is None:
            raise ValueError("插件归档身份必须是 SHA-256")
        target = self.path / f"{identity}.json"
        if target.is_symlink():
            raise ValueError("插件归档 descriptor 不能是符号链接")
        value = json.loads(target.read_text())
        if not isinstance(value, dict):
            raise ValueError("插件归档 descriptor 必须是对象")
        return cast(Mapping[str, object], freeze_json(cast(dict[str, object], value)))


def tree_entries(
    root: Path, *, exclude: frozenset[str] = frozenset()
) -> list[tuple[str, str, str]]:
    """枚举完整文件树，拒绝外部链接和无法归档的特殊文件。"""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("插件归档输入必须是实际目录")
    resolved_root = root.resolve()
    entries: list[tuple[str, str, str]] = []
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = [
            name for name in directories if name not in _CACHE_NAMES | exclude
        ]
        files = [name for name in files if name not in _CACHE_NAMES | exclude]
        for name in sorted([*directories, *files]):
            item = Path(current) / name
            relative = item.relative_to(root).as_posix()
            mode = item.lstat().st_mode
            if stat.S_ISLNK(mode):
                target = os.readlink(item)
                # 相对内部链接移动后仍指向同一归档；绝对链接不是可搬运闭包。
                if os.path.isabs(target) or not item.resolve().is_relative_to(
                    resolved_root
                ):
                    raise ValueError(f"插件归档链接越界: {relative}")
                entries.append((relative, "link", target))
            elif stat.S_ISDIR(mode):
                entries.append((relative, "directory", ""))
            elif stat.S_ISREG(mode):
                with item.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                entries.append((relative, "file", f"{bool(mode & 0o111)}:{digest}"))
            else:
                raise ValueError(f"插件归档不接受特殊文件: {relative}")
    return sorted(entries)


def encode_tree(entries: list[tuple[str, str, str]]) -> bytes:
    return json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode()


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def encode_config(value: object) -> object:
    """保存 TOML 值与不含密钥的凭据引用，不借助 pickle 或可执行对象。"""
    if isinstance(value, CredentialRef):
        return ["credential", list(value.path)]
    if isinstance(value, datetime):
        return ["datetime", value.isoformat()]
    if isinstance(value, date):
        return ["date", value.isoformat()]
    if isinstance(value, time):
        return ["time", value.isoformat()]
    if isinstance(value, Mapping):
        mapping = cast(Mapping[str, object], value)
        return ["map", {key: encode_config(item) for key, item in mapping.items()}]
    if isinstance(value, (list, tuple)):
        return [
            "list",
            [
                encode_config(item)
                for item in cast(list[object] | tuple[object, ...], value)
            ],
        ]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"插件归档不接受配置类型: {type(value).__name__}")


def decode_config(value: object) -> object:
    """按固定标签还原配置；配置字典和列表不会与类型标签碰撞。"""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if not isinstance(value, (tuple, list)):
        raise ValueError("插件归档配置结构无效")
    items = cast(list[object] | tuple[object, ...], value)
    if len(items) != 2:
        raise ValueError("插件归档配置结构无效")
    kind, payload = items
    if kind == "map" and isinstance(payload, Mapping):
        return {
            key: decode_config(item)
            for key, item in cast(Mapping[str, object], payload).items()
        }
    if kind == "list" and isinstance(payload, (tuple, list)):
        return [
            decode_config(item)
            for item in cast(list[object] | tuple[object, ...], payload)
        ]
    if kind == "credential" and isinstance(payload, (tuple, list)):
        return CredentialRef(tuple(cast(list[str] | tuple[str, ...], payload)))
    if isinstance(payload, str):
        if kind == "date":
            return date.fromisoformat(payload)
        if kind == "datetime":
            return datetime.fromisoformat(payload)
        if kind == "time":
            return time.fromisoformat(payload)
    raise ValueError("插件归档配置标签无效")
