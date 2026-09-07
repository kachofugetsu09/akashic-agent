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

from session.message import freeze_json
from session.message_codec import json_value
from agent.plugin_composition.channels import CredentialRef

_CACHE_NAMES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


class PluginArchive:
    """按内容保存插件文件树；不拥有安装指针、业务状态或自动回收。"""

    def __init__(self, path: Path):
        if path.is_symlink():
            raise ValueError("插件归档目录不能是符号链接")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = path.resolve()

    def save(self, source: Path, *, exclude: frozenset[str] = frozenset()) -> str:
        """先固定并验证文件树，再原子发布；返回前同步磁盘。"""
        # 1. 文件树是完整输入；运行环境等边界由调用者明确选定。
        if self.path.is_relative_to(source.resolve()):
            raise ValueError("插件归档不能写入自身输入目录")
        expected = tree_entries(source, exclude=exclude)
        identity = hashlib.sha256(encode_tree(expected)).hexdigest()
        if (self.path / identity).exists() or (self.path / identity).is_symlink():
            _ = self.open(identity)
            sync_directory(self.path)
            return identity
        pending = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.path))
        try:
            tree = pending / "tree"
            _ = shutil.copytree(
                source,
                tree,
                symlinks=True,
                ignore=shutil.ignore_patterns(*(_CACHE_NAMES | exclude)),
            )
            actual = tree_entries(tree)
            if actual != expected:
                raise RuntimeError("归档期间插件文件树发生变化")
            payload = encode_tree(actual)
            archive_id = hashlib.sha256(payload).hexdigest()

            # 2. 归档文件及索引先落盘，再让内容身份可见。
            for relative, kind, _ in actual:
                item = tree / relative
                if kind == "file":
                    item.chmod(0o555 if item.stat().st_mode & 0o111 else 0o444)
                    with item.open("rb") as stream:
                        os.fsync(stream.fileno())
            with (pending / "index.json").open("xb") as stream:
                _ = stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            for current, _, _ in os.walk(tree, topdown=False, followlinks=False):
                sync_directory(Path(current))
            sync_directory(pending)

            # 3. 同内容可重复发布；已有对象必须完整，不能覆盖修复损坏。
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
            return archive_id
        finally:
            # 只清理本次尚未发布的临时副本，已发布归档没有减少路径。
            if pending.exists():
                shutil.rmtree(pending)

    def open(self, archive_id: str) -> Path:
        """验证完整归档后返回精确目录；不读取 installed/stable/latest。"""
        if re.fullmatch(r"[0-9a-f]{64}", archive_id) is None:
            raise ValueError("插件归档身份必须是 SHA-256")
        root = self.path / archive_id
        if root.is_symlink() or (root / "index.json").is_symlink():
            raise ValueError("插件归档对象不能是符号链接")
        payload = (root / "index.json").read_bytes()
        if hashlib.sha256(payload).hexdigest() != archive_id:
            raise RuntimeError("插件归档索引损坏")
        tree = root / "tree"
        if encode_tree(tree_entries(tree)) != payload:
            raise RuntimeError("插件归档文件树损坏")
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
        """只读取由给定 hash 固定的 descriptor。"""
        if re.fullmatch(r"[0-9a-f]{64}", identity) is None:
            raise ValueError("插件归档身份必须是 SHA-256")
        target = self.path / f"{identity}.json"
        if target.is_symlink():
            raise ValueError("插件归档 descriptor 不能是符号链接")
        payload = target.read_bytes()
        if hashlib.sha256(payload).hexdigest() != identity:
            raise RuntimeError("插件归档 descriptor 损坏")
        value = json.loads(payload)
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
