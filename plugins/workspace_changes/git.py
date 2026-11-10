"""经执行主机运行固定 argv 的 git：用私有 index 拍工作树快照，不动用户的 index、HEAD 与 refs。"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import os
import secrets
from typing import Literal

from agent.plugin_composition import Context
from plugins.host_execution.contract import PROCESSES

HARD_TIMEOUT_S = 20
MAX_FILES = 2000
LARGE_UNTRACKED_KIB = 1024  # 更大的未跟踪文件不进快照：每次变化都会往对象库写一份完整 blob
MAX_PATCH_BYTES = 256 * 1024
_LIST_TOKENS = 256 * 1024  # 首段 512 KiB 稳定保留，足够 2000 个文件的 numstat
_ENV_KEYS = ("PATH", "HOME", "XDG_CONFIG_HOME", "SYSTEMROOT")
_INDEX = "akashic-changes.index"
_DIFF = ("diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--no-color")

Status = Literal["A", "M", "D", "T"]


class GitUnavailable(Exception):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason

    def view(self) -> dict[str, object]:
        return {"reason": self.reason, "message": str(self)}


@dataclass(frozen=True, slots=True)
class Repo:
    toplevel: str
    git_dir: str
    submodules: bool


# base 是拍快照时 HEAD 的树（unborn 时为空树），tree 是同一时刻的工作树；skipped 是没计入的大文件。
@dataclass(frozen=True, slots=True)
class Snapshot:
    base: str
    tree: str
    skipped: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FileChange:
    path: str
    added: int | None  # 二进制文件没有行数
    removed: int | None


@dataclass(frozen=True, slots=True)
class FileStatus:
    path: str
    status: Status
    added: int | None
    removed: int | None


@dataclass(frozen=True, slots=True)
class FileList:
    files: tuple[FileStatus, ...]
    truncated: bool


@dataclass(frozen=True, slots=True)
class FilePatch:
    patch: str
    truncated: bool


@dataclass(frozen=True, slots=True)
class _Run:
    code: int | None
    output: bytes
    omitted: int
    finish: str


class Git:
    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self._locks: dict[str, asyncio.Lock] = {}
        self._initialized: set[str] = set()
        # 树 id 是内容地址：同一对树的文件列表与 patch 不变，命中缓存总是正确的。
        self._lists: _Lru[tuple[str, str, str], FileList] = _Lru(64)
        self._patches: _Lru[tuple[str, str, str, str], FilePatch] = _Lru(256)

    # 在起始目录解析仓库；目录、仓库与 filter 的问题都给出原因，不当作“没有改动”。
    async def resolve(self, path: str | None) -> Repo:
        if path is None:
            raise GitUnavailable("no_directory", "会话还没有设置工作目录")
        run = await self._git(path, "rev-parse", "--show-toplevel", "--absolute-git-dir")
        if run.code != 0:
            text = _text(run.output)
            if "cannot change to" in text:
                raise GitUnavailable("missing", f"工作目录不存在或无法进入：{path}")
            if "not a git repository" in text:
                raise GitUnavailable("not_repository", "不是 git 仓库，暂不统计改动")
            raise _failed(run)
        lines = _text(run.output).splitlines()
        if len(lines) != 2:
            raise GitUnavailable("git_failed", "无法解析仓库位置（可能是裸仓库）")
        toplevel, git_dir = lines
        await self._refuse_filters(toplevel)
        gitmodules = await self._git(toplevel, "ls-files", "-z", "--", ".gitmodules")
        return Repo(toplevel, git_dir, gitmodules.code == 0 and bool(gitmodules.output))

    # 每个仓库一个私有 index，跨调用复用：stat 没变的文件 add 时不再重新哈希；同一仓库同一时刻只跑一个。
    # index 由 git 原子替换，超时或取消只会留下锁文件，只删锁；git 报错时才删 index 重建一次。
    async def snapshot(self, repo: Repo) -> Snapshot:
        lock = self._locks.setdefault(repo.git_dir, asyncio.Lock())
        index = os.path.join(repo.git_dir, _INDEX)
        extra = {"GIT_INDEX_FILE": index}
        async with lock:
            try:
                try:
                    snapshot = await self._snapshot(repo.toplevel, extra, keep_stat=repo.git_dir in self._initialized)
                except GitUnavailable as error:
                    if error.reason != "git_failed":
                        raise
                    # 私有 index 可能已坏（如 gc 清掉了只被它引用的对象）：删掉重建一次。
                    await self._remove(index, index + ".lock")
                    snapshot = await self._snapshot(repo.toplevel, extra, keep_stat=False)
                self._initialized.add(repo.git_dir)
                return snapshot
            except BaseException:
                await self._remove(index + ".lock")
                raise

    async def _snapshot(self, toplevel: str, extra: dict[str, str], *, keep_stat: bool) -> Snapshot:
        """从 HEAD 构建工作树快照，只有本进程已初始化的私有 index 才复用 stat。"""
        # 1. 首次完整读入，清掉旧缓存可能继承的 assume-unchanged 等标志；后续单树合并保留 stat。
        #    不复制用户 index；之前加进来、后来被忽略的新文件随 HEAD 重置去掉；unborn 时清空。
        head = await self._git(toplevel, "rev-parse", "-q", "--verify", "HEAD^{tree}")
        if head.code == 0:
            base = _text(head.output).strip()
            _ = await self._must(toplevel, "read-tree", *(["-m"] if keep_stat else []), base, extra=extra)
        else:
            _ = await self._must(toplevel, "read-tree", "--empty", extra=extra)
            base = await self._must(toplevel, "write-tree", extra=extra)
        # 2. add -A 覆盖整个工作树，只重新哈希 stat 变了的文件；过大的未跟踪文件排除在外。
        skipped = await self._large_untracked(toplevel, extra)
        excludes = [f":(exclude,literal){path}" for path in skipped]
        _ = await self._must(toplevel, "add", "-A", "--", ".", *excludes, extra=extra)
        return Snapshot(base, await self._must(toplevel, "write-tree", extra=extra), skipped)

    # 超过 LARGE_UNTRACKED_KIB 的未跟踪文件（没进 .gitignore 的数据库、日志）：轮询时它们一变就是一份新 blob。
    # 扫描范围与 add -A 一致；MAX_FILES 只限制展示，每批 200 个交给 find 判断大小。
    async def _large_untracked(self, toplevel: str, extra: dict[str, str]) -> tuple[str, ...]:
        run = await self._git(toplevel, "ls-files", "-o", "--exclude-standard", "-z", extra=extra, tokens=_LIST_TOKENS)
        if run.code != 0:
            raise _failed(run)
        if run.omitted:
            raise GitUnavailable("too_many_files", "未跟踪文件列表过长，无法完整检查大小，未拍快照")
        paths = [os.path.join(toplevel, _text(item)) for item in run.output.split(b"\0") if item]
        large: list[str] = []
        for start in range(0, len(paths), 200):
            argv = ["find", *paths[start:start + 200], "-maxdepth", "0", "-type", "f", "-size", f"+{LARGE_UNTRACKED_KIB}k", "-print0"]
            found = await self._exec(argv, _environment(), _LIST_TOKENS)
            if found.code != 0:
                raise _failed(found)
            if found.omitted:
                raise GitUnavailable("too_many_files", "大文件列表过长，无法完整排除，未拍快照")
            large.extend(os.path.relpath(_text(item), toplevel) for item in found.output.split(b"\0") if item)
        return tuple(large)

    # 两棵树之间的文件列表：numstat 给行数，name-status 给增删改类型。按轮记录时要在结束当下算出：
    # 快照的裸树没有引用，gc 后不可读。
    async def files(self, repo: Repo, before: str, after: str) -> FileList:
        key = (repo.git_dir, before, after)
        if (cached := self._lists.get(key)) is not None:
            return cached
        counted = await self._git(repo.toplevel, *_DIFF, "-z", "--numstat", before, after, tokens=_LIST_TOKENS)
        named = await self._git(repo.toplevel, *_DIFF, "-z", "--name-status", before, after, tokens=_LIST_TOKENS)
        for run in (counted, named):
            if run.code != 0:
                raise _failed(run)
        numbers, more = _numstat(counted)
        statuses = _statuses(named)
        listed = FileList(tuple(FileStatus(item.path, statuses.get(item.path, "M"), item.added, item.removed) for item in numbers), more)
        self._lists.put(key, listed)
        return listed

    # 单个文件的 patch，超过 MAX_PATCH_BYTES 时截在行边界。
    async def patch(self, repo: Repo, before: str, after: str, path: str) -> FilePatch:
        key = (repo.git_dir, before, after, path)
        if (cached := self._patches.get(key)) is not None:
            return cached
        run = await self._git(repo.toplevel, *_DIFF, "--patch", before, after, "--", f":(literal){path}", tokens=MAX_PATCH_BYTES // 2)
        if run.code != 0:
            raise _failed(run)
        data = _head(run, b"\n")
        cut = len(data) > MAX_PATCH_BYTES
        if cut:
            data = data[: data.rfind(b"\n", 0, MAX_PATCH_BYTES) + 1]
        found = FilePatch(_text(data), cut or bool(run.omitted))
        self._patches.put(key, found)
        return found

    # 只读查询：成功时返回去掉首尾空白的输出，失败返回 None。
    async def query(self, repo: Repo, *args: str, tokens: int = 4096) -> str | None:
        run = await self._git(repo.toplevel, *args, tokens=tokens)
        return _text(run.output).strip() if run.code == 0 else None

    # 声明了 filter 的仓库（含 LFS）不拍快照：add 会执行外部程序。只配了驱动、没有属性用到时照常。
    async def _refuse_filters(self, toplevel: str) -> None:
        drivers = await self._git(toplevel, "config", "--get-regexp", r"^filter\..*\.(clean|process)$")
        if drivers.code != 0:
            return
        tracked = await self._git(toplevel, "grep", "--untracked", "-l", "-I", "-e", "filter=", "--", ":(glob)**/.gitattributes")
        if tracked.code == 0:
            raise GitUnavailable("filter", "不支持：仓库使用 filter（如 LFS）")
        for path in await self._attribute_files(toplevel):
            # 仓库外的文件 grep 不接受，进到所在目录按文件名查；文件或目录不存在时不算用到。
            found = await self._exec(["git", "-C", os.path.dirname(path), "grep", "--no-index", "-l", "-e", "filter=", "--", os.path.basename(path)], _environment(), 256)
            if found.code == 0:
                raise GitUnavailable("filter", "不支持：仓库使用 filter（如 LFS）")

    # 工作树之外的属性来源：仓库的 info/attributes 与全局 core.attributesFile（未配置时取 XDG 默认位置）。
    async def _attribute_files(self, toplevel: str) -> list[str]:
        info = await self._git(toplevel, "rev-parse", "--path-format=absolute", "--git-path", "info/attributes")
        configured = await self._git(toplevel, "config", "--path", "--get", "core.attributesFile")
        env = _environment()
        default = os.path.join(env.get("XDG_CONFIG_HOME") or os.path.join(env.get("HOME", "/"), ".config"), "git", "attributes")
        files = [_text(info.output).strip() if info.code == 0 else ""]
        files.append(_text(configured.output).strip() if configured.code == 0 else default)
        return [os.path.join(toplevel, path) for path in files if path]

    async def _must(self, cwd: str, *args: str, extra: dict[str, str]) -> str:
        run = await self._git(cwd, *args, extra=extra)
        if run.code != 0:
            raise _failed(run)
        return _text(run.output).strip()

    # 每次调用一个独立 owner；取消或超时后先等进程组结束再返回。
    async def _git(self, cwd: str, *args: str, extra: dict[str, str] | None = None, tokens: int = 4096) -> _Run:
        argv = ["git", "-C", cwd, "-c", "core.quotePath=false", *args]
        return await self._exec(argv, {**_environment(), **(extra or {})}, tokens)

    async def _remove(self, *paths: str) -> None:
        _ = await self._exec(["rm", "-f", "--", *paths], _environment(), 256)

    async def _exec(self, argv: list[str], env: dict[str, str], tokens: int) -> _Run:
        processes = self._ctx.require(PROCESSES)
        owner = f"git:{secrets.token_hex(8)}"
        try:
            result = await processes.exec_command(
                self._ctx, owner, command=" ".join(argv), argv=argv, cwd=None, env=env, tty=False,
                yield_time_ms=(HARD_TIMEOUT_S + 5) * 1000, max_output_tokens=tokens, hard_timeout_s=HARD_TIMEOUT_S,
            )
        finally:
            _ = await processes.terminate_owner(self._ctx, owner)
        if result.finish_reason == "timeout":
            raise GitUnavailable("timeout", f"git 超过 {HARD_TIMEOUT_S} 秒未完成，仓库可能过大")
        return _Run(result.exit_code, _strip_notes(result.output), result.output_omitted_bytes, result.finish_reason)


def _environment() -> dict[str, str]:
    env = {key: os.environ[key] for key in _ENV_KEYS if key in os.environ}
    _ = env.setdefault("PATH", os.defpath)
    # 非交互、不刷新用户 index 的 stat 缓存、不经分页器、错误信息用英文便于识别。
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", GIT_PAGER="cat", LC_ALL="C")
    return env


# stderr 与 stdout 合在一起：开头的 warning:/hint: 行（如属性文件不可读、引用名有歧义）不是数据。
def _strip_notes(output: bytes) -> bytes:
    while output.startswith((b"warning: ", b"hint: ")):
        end = output.find(b"\n")
        if end < 0 or b"\0" in output[:end]:
            break
        output = output[end + 1:]
    return output


# 输出过长时首段稳定保留，丢掉省略标记之后的部分和最后一条不完整记录。
def _head(run: _Run, separator: bytes) -> bytes:
    data = run.output
    if run.omitted:
        data = data[: data.rfind(b"\n... ")]
        data = data[: data.rfind(separator) + 1]
    return data


def _numstat(run: _Run) -> tuple[tuple[FileChange, ...], bool]:
    files: list[FileChange] = []
    for record in _head(run, b"\0").split(b"\0"):
        if not record:
            continue
        added, removed, path = record.split(b"\t", 2)
        files.append(FileChange(
            path.decode("utf-8", "replace"),
            None if added == b"-" else int(added), None if removed == b"-" else int(removed),
        ))
    return tuple(files[:MAX_FILES]), bool(run.omitted) or len(files) > MAX_FILES


# name-status -z 是“类型\0路径\0”成对出现；不认识的类型按修改算。
def _statuses(run: _Run) -> dict[str, Status]:
    parts = _head(run, b"\0").split(b"\0")
    known: dict[bytes, Status] = {b"A": "A", b"M": "M", b"D": "D", b"T": "T"}
    return {parts[i + 1].decode("utf-8", "replace"): known.get(parts[i], "M") for i in range(0, len(parts) - 1, 2)}


class _Lru[K, V]:
    def __init__(self, size: int) -> None:
        self._size = size
        self._items: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        value = self._items.get(key)
        if value is not None:
            self._items.move_to_end(key)
        return value

    def put(self, key: K, value: V) -> None:
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._size:
            _ = self._items.popitem(last=False)


def _failed(run: _Run) -> GitUnavailable:
    lines = _text(run.output).strip().splitlines()
    return GitUnavailable("git_failed", f"git 执行失败：{lines[-1] if lines else run.finish}")


def _text(output: bytes) -> str:
    return output.decode("utf-8", "replace")
