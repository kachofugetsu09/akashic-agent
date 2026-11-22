"""extra 仓库 index 分支：组、说明、每包大小、包间依赖、前置条件。只供 pkg groups / pkg install @组 读取，别的命令不依赖它。"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from agent.plugins.extra_repo import git_env, list_heads

SCHEMA_VERSION = 1
_TIMEOUT_S = 60


@dataclass(frozen=True)
class IndexGroup:
    id: str
    title: str
    description: str
    required: bool
    members: tuple[str, ...]


@dataclass(frozen=True)
class IndexPackage:
    name: str
    desc: str
    group: str
    requires: tuple[str, ...]
    needs: tuple[str, ...]
    images: tuple[str, ...]
    size_bytes: int


@dataclass(frozen=True)
class PackageIndex:
    sha: str
    groups: dict[str, IndexGroup]
    packages: dict[str, IndexPackage]
    # 空串表示取自远端当前版本；否则是离线时回落到缓存的说明。
    note: str = ""


def _strings(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in cast(list[object], value)):
        raise RuntimeError(f"index 格式无效: {where}")
    return tuple(cast(list[str], value))


# 解析 index.json 文本；结构不对一律报错，不猜。
def parse_index(text: str, sha: str) -> PackageIndex:
    try:
        document = cast(dict[str, object], json.loads(text))
        if document.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"index schema_version 不是 {SCHEMA_VERSION}，请升级 Core")
        groups = {gid: IndexGroup(gid, str(row["title"]), str(row["description"]), bool(row["required"]),
                                  _strings(row["members"], f"groups.{gid}.members"))
                  for gid, row in cast(dict[str, dict[str, object]], document["groups"]).items()}
        packages = {name: IndexPackage(name, str(row["desc"]), str(row["group"]), _strings(row["requires"], f"{name}.requires"),
                                       _strings(row["needs"], f"{name}.needs"), _strings(row["images"], f"{name}.images"),
                                       int(cast(int, row["size_bytes"])))
                    for name, row in cast(dict[str, dict[str, object]], document["packages"]).items()}
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"index 格式无效: {error}") from error
    return PackageIndex(sha, groups, packages)


# 浅取 index 提交并读出 index.json。
def _fetch_text(repo: str, sha: str) -> str:
    with tempfile.TemporaryDirectory(prefix="akashic-index-") as directory:
        work = Path(directory)
        for command in (["git", "init", "--quiet"], ["git", "fetch", "--quiet", "--depth=1", "--no-tags", "--", repo, sha],
                        ["git", "show", f"{sha}:index.json"]):
            result = subprocess.run(command, cwd=work, capture_output=True, text=True, check=False, timeout=_TIMEOUT_S, env=git_env())
            if result.returncode != 0:
                raise RuntimeError(f"取 index 失败（{command[1]}）: {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else result.returncode}")
        return result.stdout


def _cache_path(home: Path, repo: str) -> Path:
    return home / "index-cache" / (hashlib.sha256(repo.strip().rstrip("/").encode()).hexdigest()[:16] + ".json")


# 读取 index：远端 SHA 与缓存相同直接用缓存；不同则取新的并写缓存；联网失败回落到缓存并带说明；都没有则报错。
def load_index(repo: str, home: Path) -> PackageIndex:
    path = _cache_path(home, repo)
    cached: tuple[str, str] | None = None
    if path.is_file():
        record = cast(dict[str, str], json.loads(path.read_text(encoding="utf-8")))
        cached = (record["sha"], record["text"])
    try:
        _, sha = list_heads(repo)
        if not sha:
            raise RuntimeError(f"{repo} 没有 index 分支")
        if cached is not None and cached[0] == sha:
            return parse_index(cached[1], sha)
        text = _fetch_text(repo, sha)
        index = parse_index(text, sha)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        if cached is None:
            raise RuntimeError(f"无法读取包组信息（{str(error).splitlines()[0]}），本机也没有缓存") from error
        return replace(parse_index(cached[1], cached[0]),
                       note=f"联网失败（{str(error).splitlines()[0]}），使用本机缓存的包组信息")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    _ = temporary.write_text(json.dumps({"sha": sha, "text": text}), encoding="utf-8")
    temporary.replace(path)
    return index


# 依赖闭包：给定包名加上它们（传递地）requires 的包，按"依赖在前"排序；未知名字抛 KeyError 由调用方转成明确报错。
def closure(index: PackageIndex, names: list[str]) -> list[str]:
    ordered: list[str] = []
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in ordered:
            return
        if name in visiting:
            raise RuntimeError(f"index 里的包依赖成环: {name}")
        visiting.add(name)
        for dependency in index.packages[name].requires:
            visit(dependency)
        visiting.discard(name)
        ordered.append(name)

    for name in names:
        visit(name)
    return ordered


# 把命令行参数里的 @组 展开成包名（保持其余名字原样），再补上依赖闭包；返回 (按依赖排序的名字, 自动带上的依赖名)。
def expand(index: PackageIndex, tokens: list[str]) -> tuple[list[str], list[str]]:
    requested: list[str] = []
    for token in tokens:
        if token.startswith("@"):
            group = index.groups.get(token[1:])
            if group is None:
                raise RuntimeError(f"没有这个包组: {token}（pkg groups 列出全部组）")
            requested.extend(group.members)
        else:
            requested.append(token)
    unknown = [name for name in requested if name not in index.packages]
    if unknown:
        raise RuntimeError("index 里没有这些包: " + ", ".join(unknown))
    full = closure(index, list(dict.fromkeys(requested)))
    return full, [name for name in full if name not in requested]
