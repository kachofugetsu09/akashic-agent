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
from pathlib import Path
from typing import cast
from utils.timing import measure
from agent.plugins.files import tree_entries, encode_tree, sync_directory

from session.message import freeze_json
from session.message_codec import json_value

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
