"""安装文件树的校验、摘要和目录刷盘；不复制或保存运行快照。"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

_CACHE_NAMES = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def tree_entries(
    root: Path, *, exclude: frozenset[str] = frozenset()
) -> list[tuple[str, str, str]]:
    """枚举完整文件树，拒绝外部链接和安装不支持的特殊文件。"""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("插件文件树输入必须是实际目录")
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
                # 相对内部链接移动后仍指向同一文件树；绝对链接不能随安装目录移动。
                if os.path.isabs(target) or not item.resolve().is_relative_to(
                    resolved_root
                ):
                    raise ValueError(f"插件文件树链接越界: {relative}")
                entries.append((relative, "link", target))
            elif stat.S_ISDIR(mode):
                entries.append((relative, "directory", ""))
            elif stat.S_ISREG(mode):
                with item.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                entries.append((relative, "file", f"{bool(mode & 0o111)}:{digest}"))
            else:
                raise ValueError(f"插件文件树不接受特殊文件: {relative}")
    return sorted(entries)


def encode_tree(entries: list[tuple[str, str, str]]) -> bytes:
    return json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode()


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


