"""Linux inotify 递归树监听：插件热重载的外部变化事件来源。

不引入第三方依赖；仅在 Linux 上可用，其他平台或初始化失败时由调用方回退轮询。
事件不携带路径语义，只表示"监听目标内可能有变化"，实际变更判定仍由
PluginManager.watch_revision 的元数据指纹完成，因此误报只会多一次廉价探测。
"""
from __future__ import annotations

import asyncio
import ctypes
import errno
import os
import struct
from collections.abc import Callable, Iterable
from pathlib import Path

_IN_MODIFY = 0x00000002
_IN_ATTRIB = 0x00000004
_IN_MOVED_FROM = 0x00000040
_IN_MOVED_TO = 0x00000080
_IN_CREATE = 0x00000100
_IN_DELETE = 0x00000200
_IN_DELETE_SELF = 0x00000400
_IN_MOVE_SELF = 0x00000800
_IN_ONLYDIR = 0x01000000
_IN_ISDIR = 0x40000000
_IN_Q_OVERFLOW = 0x00004000
_IN_IGNORED = 0x00008000
_WATCH_EVENTS = (
    _IN_MODIFY | _IN_ATTRIB | _IN_CREATE | _IN_DELETE
    | _IN_MOVED_FROM | _IN_MOVED_TO | _IN_DELETE_SELF | _IN_MOVE_SELF
)
_IN_NONBLOCK = 0x00000800
_IN_CLOEXEC = 0x00080000

_EVENT_HEADER = struct.Struct("iIII")
# 单进程监听预算；超出即认为事件来源不可用，避免挤占全局 inotify 配额。
_MAX_WATCHES = 8192


class InotifyTreeWatcher:
    """监听一组递归树根与指定文件；任何相关事件都触发一次回调。

    - 树根：递归监听整棵目录树，排除规则与全树指纹使用的排除规则一致，
      符号链接目录不跟随；新建子目录自动纳入监听。
    - 文件目标：监听其父目录，但只有目标文件名的事件才计为变化
      （数据目录可能有高频运行时写入，不能整目录计入）。
    - 暂不存在的根：退化为监听其父目录（不递归），等待其出现。

    回调在创建 watcher 的事件循环上执行；队列溢出同样只表现为一次回调。
    """

    def __init__(
        self,
        on_change: Callable[[], None],
        *,
        exclude: Callable[[str], bool],
    ) -> None:
        self._on_change = on_change
        self._exclude = exclude
        self._libc: ctypes.CDLL | None = None
        self._fd: int | None = None
        # wd -> 绝对路径；三种目标各一张表区分事件语义。
        self._tree_dirs: dict[int, str] = {}
        self._flat_dirs: dict[int, str] = {}
        self._file_dirs: dict[int, tuple[str, frozenset[str]]] = {}
        self._covered: set[str] = set()
        self._trees: tuple[Path, ...] = ()
        self._files: tuple[Path, ...] = ()
        self._missing_trees: set[Path] = set()
        self._saturated = False

    @property
    def active(self) -> bool:
        """初始化成功且未因监听预算耗尽而失效。"""
        return self._fd is not None and not self._saturated

    def start(self) -> bool:
        """初始化 inotify；平台不支持或 fd 失败时返回 False 由调用方回退轮询。"""
        if self._fd is not None:
            return not self._saturated
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            fd = libc.inotify_init1(_IN_NONBLOCK | _IN_CLOEXEC)
            if fd < 0:
                return False
        except (AttributeError, OSError):
            return False
        try:
            asyncio.get_running_loop().add_reader(fd, self._drain)
        except NotImplementedError:
            # Windows 的 ProactorEventLoop 不支持 add_reader。
            os.close(fd)
            return False
        self._libc = libc
        self._fd = fd
        self._rebuild()
        return not self._saturated

    def set_targets(
        self,
        trees: Iterable[Path],
        files: Iterable[Path],
    ) -> None:
        """重建全部监听；目标未变时不重建，重建成本远低于一次全树指纹。"""
        trees = tuple(trees)
        files = tuple(files)
        if trees == self._trees and files == self._files:
            # 上次缺失的根现已出现时必须重建，把父目录扁平监听升级为递归树。
            if not any(os.path.isdir(root) for root in self._missing_trees):
                return
        self._trees = trees
        self._files = files
        if self._fd is not None:
            self._rebuild()

    def close(self) -> None:
        if self._fd is None:
            return
        _ = asyncio.get_running_loop().remove_reader(self._fd)
        os.close(self._fd)
        self._fd = None
        self._tree_dirs = {}
        self._flat_dirs = {}
        self._file_dirs = {}
        self._covered = set()

    def _rebuild(self) -> None:
        assert self._fd is not None and self._libc is not None
        for wd in [*self._tree_dirs, *self._flat_dirs, *self._file_dirs]:
            self._libc.inotify_rm_watch(self._fd, wd)
        self._tree_dirs = {}
        self._flat_dirs = {}
        self._file_dirs = {}
        self._covered = set()
        self._missing_trees: set[Path] = set()
        self._saturated = False
        for root in self._trees:
            if self._saturated:
                return
            if os.path.isdir(root):
                self._add_tree(root)
            else:
                # 根尚未出现：监听父目录等待创建，不做递归。
                self._missing_trees.add(root)
                self._add_flat(os.path.dirname(root) or os.sep)
        file_targets: dict[str, set[str]] = {}
        for target in self._files:
            parent = os.path.dirname(target) or os.sep
            file_targets.setdefault(parent, set()).add(os.path.basename(target))
        for parent, names in file_targets.items():
            if self._saturated:
                return
            if parent in self._covered:
                # 已在递归树内：文件事件已被树监听覆盖。
                continue
            self._add_file_dir(parent, frozenset(names))

    def _add_watch(self, directory: str, *, only_dir: bool) -> int | None:
        assert self._fd is not None and self._libc is not None
        if len(self._covered) >= _MAX_WATCHES:
            self._saturated = True
            return None
        mask = _WATCH_EVENTS | (_IN_ONLYDIR if only_dir else 0)
        wd = self._libc.inotify_add_watch(self._fd, os.fsencode(directory), mask)
        if wd < 0:
            # 单个目录不可读或已消失不视为整体失效。
            if ctypes.get_errno() in (errno.EACCES, errno.ENOENT, errno.ENOTDIR):
                return None
            self._saturated = True
            return None
        self._covered.add(directory)
        return wd

    def _add_flat(self, directory: str) -> None:
        if directory in self._covered:
            return
        wd = self._add_watch(directory, only_dir=False)
        if wd is not None:
            self._flat_dirs[wd] = directory

    def _add_file_dir(self, directory: str, names: frozenset[str]) -> None:
        wd = self._add_watch(directory, only_dir=False)
        if wd is not None:
            self._file_dirs[wd] = (directory, names)

    def _add_tree(self, root: Path) -> None:
        pending = [os.fspath(root)]
        while pending:
            if self._saturated:
                return
            current = pending.pop()
            try:
                real = os.path.realpath(current)
            except OSError:
                continue
            if real in self._covered:
                continue
            wd = self._add_watch(real, only_dir=True)
            if wd is None:
                continue
            self._tree_dirs[wd] = real
            try:
                with os.scandir(real) as entries:
                    children = [
                        entry.path
                        for entry in entries
                        if entry.is_dir(follow_symlinks=False)
                        and not self._exclude(entry.name)
                    ]
            except OSError:
                continue
            pending.extend(children)

    def _drain(self) -> None:
        assert self._fd is not None
        try:
            data = os.read(self._fd, 65536)
        except BlockingIOError:
            return
        except OSError:
            self._on_change()
            return
        changed = False
        offset = 0
        while offset + _EVENT_HEADER.size <= len(data):
            wd, mask, _cookie, name_len = _EVENT_HEADER.unpack_from(data, offset)
            offset += _EVENT_HEADER.size
            name = os.fsdecode(data[offset : offset + name_len].rstrip(b"\0"))
            offset += name_len
            if mask & _IN_Q_OVERFLOW:
                changed = True
                continue
            if mask & _IN_IGNORED:
                directory = self._tree_dirs.pop(wd, None)
                if directory is not None:
                    self._covered.discard(directory)
                directory = self._flat_dirs.pop(wd, None)
                if directory is not None:
                    self._covered.discard(directory)
                file_entry = self._file_dirs.pop(wd, None)
                if file_entry is not None:
                    self._covered.discard(file_entry[0])
                changed = True
                continue
            if wd in self._file_dirs:
                if name in self._file_dirs[wd][1]:
                    changed = True
                continue
            if wd in self._flat_dirs:
                changed = True
                continue
            directory = self._tree_dirs.get(wd)
            if directory is None:
                continue
            if self._exclude(name):
                continue
            if mask & (_IN_CREATE | _IN_MOVED_TO) and mask & _IN_ISDIR:
                # 新建与移入的目录都要递归纳入监听，否则目录内部后续修改静默丢失
                # （评审 #1119）。
                self._add_tree(Path(directory) / name)
            changed = True
        if changed:
            self._on_change()
