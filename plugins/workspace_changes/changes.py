"""会话工作目录的改动统计：未提交、本分支与单个提交三种范围按需对比；每轮的窗口见 turns.py。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from agent.plugin_composition import Context, ServiceKey
from core.common.file_io import run_file_io
from plugins.working_directory.contract import WORKING_DIRECTORY

from .git import Git, GitUnavailable, Repo
from .turns import TurnWindows

MAX_COMMITS = 30
SCOPE = re.compile(r"wt|base|[0-9a-f]{7,64}")
OID = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_BASE_CANDIDATES = ("refs/remotes/origin/HEAD", "refs/remotes/origin/main", "refs/remotes/origin/master", "refs/heads/main", "refs/heads/master")
_EMPTY_TREE = {40: "4b825dc642cb6eb9a060e54bf8d69288fbee4904", 64: "6ef19b41225c5369f1c104d45d8d85efa9b057b53b14b4b9b939dd74decc5321"}


@dataclass(frozen=True, slots=True)
class Commit:
    sha: str
    short: str
    subject: str
    time: str


# 当前分支、默认分支与两者的 merge-base；找不到基线时 merge_base 为 None。
@dataclass(frozen=True, slots=True)
class _Branch:
    name: str | None
    base_ref: str | None
    merge_base: str | None
    commits: tuple[Commit, ...]


class WorkspaceChanges:
    def __init__(self, ctx: Context) -> None:
        self._ctx = ctx
        self.git = Git(ctx)
        self.windows = TurnWindows(ctx, self.git, self.repo)

    # 工作目录在调用时读取；仓库按这一刻的目录解析。
    async def repo(self, session: str) -> Repo:
        directories = self._ctx.require(WORKING_DIRECTORY)
        snapshot = await run_file_io(lambda: directories.snapshot(session))
        return await self.git.resolve(snapshot.path)

    # 一个范围的文件列表（不含 patch）：wt 未提交，base 本分支全部改动，十六进制为单个提交。
    async def scope(self, session: str, scope: str) -> dict[str, object]:
        try:
            if not SCOPE.fullmatch(scope):
                raise GitUnavailable("invalid", "不认识的范围")
            async with self._ctx.runtime_scope():
                repo = await self.repo(session)
                return await self._scope(repo, scope)
        except GitUnavailable as error:
            return {"status": "unavailable", **error.view()}

    # 单个文件在两棵树之间的 patch，路径按字面匹配。
    async def patch(self, session: str, before: str, after: str, path: str) -> dict[str, object]:
        try:
            if not (OID.fullmatch(before) and OID.fullmatch(after)) or not path or "\0" in path or len(path) > 4096:
                raise GitUnavailable("invalid", "参数不合法")
            async with self._ctx.runtime_scope():
                found = await self.git.patch(await self.repo(session), before, after, path)
                return {"status": "ok", **asdict(found)}
        except GitUnavailable as error:
            return {"status": "unavailable", **error.view()}

    async def _scope(self, repo: Repo, scope: str) -> dict[str, object]:
        # 1. 分支信息；找不到基线时 base 退回未提交的改动。
        branch = await self._branch(repo)
        snapshot = None
        if scope in ("wt", "base"):
            snapshot = await self.git.snapshot(repo)
            if scope == "base" and branch.merge_base is not None:
                before = await self._must(repo, f"{branch.merge_base}^{{tree}}")
            else:
                before, scope = snapshot.base, "wt"
            after = snapshot.tree
        else:
            scope, before, after = await self._commit(repo, scope)
        # 2. 文件列表按两棵树缓存，没有变化的轮询不再跑 diff。
        listed = await self.git.files(repo, before, after)
        return {
            "status": "ok", "repo": repo.toplevel, "branch": branch.name, "base_ref": branch.base_ref,
            "base_unknown": branch.merge_base is None, "commits": [asdict(item) for item in branch.commits],
            "scope": scope, "before": before, "after": after, "files": [asdict(item) for item in listed.files],
            "truncated": listed.truncated, "submodules": repo.submodules,
            "skipped": list(snapshot.skipped) if snapshot is not None else [],
        }

    async def _branch(self, repo: Repo) -> _Branch:
        name = await self.git.query(repo, "symbolic-ref", "-q", "--short", "HEAD")
        base = await self._default_branch(repo)
        # 用完整引用名求 merge-base：同名的 tag 与分支不会有歧义。
        merge_base = None if base is None else await self.git.query(repo, "merge-base", base, "HEAD")
        commits = () if merge_base is None else await self._commits(repo, merge_base)
        return _Branch(name, None if base is None else _short(base), merge_base, commits)

    # 默认分支（完整引用名）：origin/HEAD 指向且仍存在的远端分支，否则 origin/main、origin/master、本地 main、master。
    # 远端优先：本地 main 常常落后，拿它当基线会混进远端已有的提交。一次 for-each-ref 列出全部候选。
    async def _default_branch(self, repo: Repo) -> str | None:
        listed = await self.git.query(repo, "for-each-ref", "--format=%(refname)\t%(symref)", *_BASE_CANDIDATES)
        refs = {ref: symref for ref, _, symref in (line.partition("\t") for line in (listed or "").splitlines())}
        target = refs.get("refs/remotes/origin/HEAD")
        if target and target in refs:
            return target
        return next((ref for ref in _BASE_CANDIDATES[1:] if ref in refs), None)

    # merge_base..HEAD 的提交，新的在前；记录以 NUL 分隔，字段以 0x1f 分隔，标题放最后。
    async def _commits(self, repo: Repo, merge_base: str) -> tuple[Commit, ...]:
        args = ("log", "-z", "--no-show-signature", f"--max-count={MAX_COMMITS}", "--format=%H%x1f%h%x1f%cI%x1f%s")
        output = await self.git.query(repo, *args, f"{merge_base}..HEAD", tokens=16 * 1024)
        if output is None:
            return ()
        records = [record.split("\x1f", 3) for record in output.split("\0") if record]
        return tuple(Commit(sha, short, subject, time) for sha, short, time, subject in records)

    # 单个提交：第一个父提交的树（根提交为空树）对比该提交的树。
    async def _commit(self, repo: Repo, sha: str) -> tuple[str, str, str]:
        full = await self.git.query(repo, "rev-parse", "-q", "--verify", f"{sha}^{{commit}}")
        if full is None:
            raise GitUnavailable("commit_missing", "找不到这个提交")
        after = await self._must(repo, f"{full}^{{tree}}")
        parent = await self.git.query(repo, "rev-parse", "-q", "--verify", f"{full}^1^{{tree}}")
        return full, parent or _EMPTY_TREE[len(full)], after

    async def _must(self, repo: Repo, revision: str) -> str:
        found = await self.git.query(repo, "rev-parse", "-q", "--verify", revision)
        if found is None:
            raise GitUnavailable("git_failed", f"git 无法解析 {revision}")
        return found


def _short(ref: str) -> str:
    return ref.removeprefix("refs/remotes/").removeprefix("refs/heads/")


WORKSPACE_CHANGES = ServiceKey[WorkspaceChanges]("workspace_changes.service.v1")
