"""经执行主机运行固定 argv 的 git：用临时 index 拍工作树快照，不动用户的 index、HEAD 与 refs。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import secrets

from agent.plugin_composition import Context
from plugins.host_execution.contract import PROCESSES

HARD_TIMEOUT_S = 20
MAX_FILES = 2000
MAX_PATCH_BYTES = 256 * 1024
_LIST_TOKENS = 256 * 1024  # 首段 512 KiB 稳定保留，足够 2000 个文件的 numstat
_ENV_KEYS = ("PATH", "HOME", "XDG_CONFIG_HOME", "SYSTEMROOT")


class GitUnavailable(Exception):
    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class Repo:
    toplevel: str
    git_dir: str
    submodules: bool


# base 是拍快照时 HEAD 的树（unborn 时为空树），tree 是同一时刻的工作树。
@dataclass(frozen=True, slots=True)
class Snapshot:
    base: str
    tree: str


@dataclass(frozen=True, slots=True)
class FileChange:
    path: str
    added: int | None  # 二进制文件没有行数
    removed: int | None


@dataclass(frozen=True, slots=True)
class Changes:
    files: tuple[FileChange, ...]
    patch: str | None
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

    # 临时 index 从 HEAD 的树开始，add -A 覆盖整个工作树；同一仓库同一时刻只跑一个。
    async def snapshot(self, repo: Repo) -> Snapshot:
        lock = self._locks.setdefault(repo.git_dir, asyncio.Lock())
        index = os.path.join(repo.git_dir, f"akashic-changes-{secrets.token_hex(8)}")
        extra = {"GIT_INDEX_FILE": index}
        async with lock:
            try:
                head = await self._git(repo.toplevel, "rev-parse", "-q", "--verify", "HEAD^{tree}")
                start = ("read-tree", _text(head.output).strip()) if head.code == 0 else ("read-tree", "--empty")
                _ = await self._must(repo.toplevel, *start, extra=extra)
                base = await self._must(repo.toplevel, "write-tree", extra=extra)
                _ = await self._must(repo.toplevel, "add", "-A", "--", ".", extra=extra)
                tree = await self._must(repo.toplevel, "write-tree", extra=extra)
                return Snapshot(base, tree)
            finally:
                await self._remove(index)

    # 立刻算出文件列表与 patch 保存下来：裸树没有引用，gc 后不可读。
    async def diff(self, repo: Repo, before: str, after: str) -> Changes:
        args = ("diff", "--no-ext-diff", "--no-textconv", "--no-renames", "--no-color")
        listed = await self._git(repo.toplevel, *args, "-z", "--numstat", before, after, tokens=_LIST_TOKENS)
        if listed.code != 0:
            raise _failed(listed)
        files, more = _numstat(listed)
        patch = await self._git(repo.toplevel, *args, "--patch", before, after, tokens=MAX_PATCH_BYTES // 4)
        if patch.code != 0:
            raise _failed(patch)
        whole = patch.omitted == 0
        return Changes(files, _text(patch.output) if whole else None, more or not whole)

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
        return [path for path in files if os.path.isabs(path)]

    async def _must(self, cwd: str, *args: str, extra: dict[str, str]) -> str:
        run = await self._git(cwd, *args, extra=extra)
        if run.code != 0:
            raise _failed(run)
        return _text(run.output).strip()

    # 每次调用一个独立 owner；取消或超时后先等进程组结束再返回。
    async def _git(self, cwd: str, *args: str, extra: dict[str, str] | None = None, tokens: int = 4096) -> _Run:
        argv = ["git", "-C", cwd, "-c", "core.quotePath=false", *args]
        return await self._exec(argv, {**_environment(), **(extra or {})}, tokens)

    async def _remove(self, index: str) -> None:
        _ = await self._exec(["rm", "-f", "--", index, index + ".lock"], _environment(), 256)

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
        return _Run(result.exit_code, result.output, result.output_omitted_bytes, result.finish_reason)


def _environment() -> dict[str, str]:
    env = {key: os.environ[key] for key in _ENV_KEYS if key in os.environ}
    _ = env.setdefault("PATH", os.defpath)
    # 非交互、不刷新用户 index 的 stat 缓存、不经分页器、错误信息用英文便于识别。
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0", GIT_PAGER="cat", LC_ALL="C")
    return env


def _numstat(run: _Run) -> tuple[tuple[FileChange, ...], bool]:
    data = run.output
    if run.omitted:
        # 输出过长时首段稳定保留，丢掉省略标记之后的部分和最后一条不完整记录。
        data = data[: data.rfind(b"\n... ")]
        data = data[: data.rfind(b"\0") + 1]
    files: list[FileChange] = []
    for record in data.split(b"\0"):
        if not record:
            continue
        added, removed, path = record.split(b"\t", 2)
        files.append(FileChange(
            path.decode("utf-8", "replace"),
            None if added == b"-" else int(added), None if removed == b"-" else int(removed),
        ))
    return tuple(files[:MAX_FILES]), bool(run.omitted) or len(files) > MAX_FILES


def _failed(run: _Run) -> GitUnavailable:
    lines = _text(run.output).strip().splitlines()
    return GitUnavailable("git_failed", f"git 执行失败：{lines[-1] if lines else run.finish}")


def _text(output: bytes) -> str:
    return output.decode("utf-8", "replace")
