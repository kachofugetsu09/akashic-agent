"""extra 仓库 index 分支：组、说明、每包大小、包间依赖、前置条件。只供 pkg groups / pkg install @组 读取，别的命令不依赖它。"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

from agent.plugins.extra_repo import git_env, list_heads
from agent.plugins.prerequisites import parse_needs
from agent.plugins.static_manifest import is_plugin_name

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


class IndexFormatError(RuntimeError):
    """远端 index 内容不合格（不是网络问题）：不回落旧缓存，也不当作联网失败报告。"""


# 对 index 一律按不受信输入处理：先限大小再解析，字段类型、名字字符集、引用完整性、依赖环都在边界检查。
MAX_INDEX_BYTES = 1 << 20
_MAX_GROUPS = 100
_MAX_PACKAGES = 500
_MAX_TEXT = 500


def _bad(where: str, why: str) -> IndexFormatError:
    return IndexFormatError(f"远端 index 格式无效（{where}）：{why}")


def _mapping(value: object, where: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _bad(where, "必须是对象")
    return cast(dict[str, object], value)


def _text(value: object, where: str) -> str:
    if not isinstance(value, str) or len(value) > _MAX_TEXT:
        raise _bad(where, f"必须是不超过 {_MAX_TEXT} 字符的字符串")
    return value


def _names(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(cast(list[object], value)) > _MAX_PACKAGES:
        raise _bad(where, "必须是名字数组")
    items = tuple(_text(item, where) for item in cast(list[object], value))
    if not all(is_plugin_name(item) for item in items):
        raise _bad(where, "名字必须符合插件名规则（小写字母开头，只含小写字母、数字、_、-）")
    return items


def _strings(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(cast(list[object], value)) > 50:
        raise _bad(where, "必须是字符串数组")
    return tuple(_text(item, where) for item in cast(list[object], value))


# 前置条件声明与读端同一条规则（parse_needs）：非法项在读 index 时就拒绝，点名哪个包哪一项，不让它写进声明后阻塞全局启用。
def _needs(value: object, name: str) -> tuple[str, ...]:
    try:
        return tuple(parse_needs(value, f"packages.{name}.needs"))
    except ValueError as error:
        raise _bad(f"packages.{name}.needs", str(error)) from error


def _group(group_id: str, value: object) -> IndexGroup:
    row = _mapping(value, f"groups.{group_id}")
    required = row.get("required")
    if not isinstance(required, bool):
        raise _bad(f"groups.{group_id}.required", "必须是布尔值")
    members = _names(row.get("members"), f"groups.{group_id}.members")
    if len(set(members)) != len(members):
        raise _bad(f"groups.{group_id}.members", "有重复成员")
    return IndexGroup(group_id, _text(row.get("title"), f"groups.{group_id}.title"),
                      _text(row.get("description"), f"groups.{group_id}.description"), required, members)


def _package(name: str, value: object) -> IndexPackage:
    row = _mapping(value, f"packages.{name}")
    size = row.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise _bad(f"packages.{name}.size_bytes", "必须是非负整数")
    group = _text(row.get("group"), f"packages.{name}.group")
    return IndexPackage(name, _text(row.get("desc"), f"packages.{name}.desc"), group, _names(row.get("requires"), f"packages.{name}.requires"),
                        _needs(row.get("needs"), name), _strings(row.get("images"), f"packages.{name}.images"), size)


# 引用完整性：成员与 requires 指向存在的包，包的 group 与组成员互相一致，依赖无环。
def _check_links(groups: dict[str, IndexGroup], packages: dict[str, IndexPackage]) -> None:
    owner: dict[str, str] = {}
    for group in groups.values():
        for member in group.members:
            if member not in packages:
                raise _bad(f"groups.{group.id}.members", f"成员 {member} 不在 packages 里")
            if member in owner:
                raise _bad(f"groups.{group.id}.members", f"{member} 同时属于组 {owner[member]}")
            owner[member] = group.id
    for package in packages.values():
        if owner.get(package.name) != package.group:
            raise _bad(f"packages.{package.name}.group", "与组的成员列表不一致")
        for dependency in package.requires:
            if dependency not in packages or dependency == package.name:
                raise _bad(f"packages.{package.name}.requires", f"依赖 {dependency} 不存在或指向自己")
    for name in packages:
        try:
            _ = closure_of(packages, [name])
        except RuntimeError as error:
            raise _bad("packages", str(error)) from error


# 解析 index.json 文本；任何不合格都抛 IndexFormatError，不让别的异常逃出。
def parse_index(text: str, sha: str) -> PackageIndex:
    if len(text.encode("utf-8")) > MAX_INDEX_BYTES:
        raise _bad("文件", f"超过 {MAX_INDEX_BYTES} 字节")
    try:
        document = _mapping(json.loads(text), "顶层")
    except ValueError as error:
        raise _bad("顶层", f"不是合法 JSON: {error}") from error
    if document.get("schema_version") != SCHEMA_VERSION or isinstance(document.get("schema_version"), bool):
        raise IndexFormatError(f"远端 index 的 schema_version 不是 {SCHEMA_VERSION}，请升级 Core")
    raw_groups, raw_packages = _mapping(document.get("groups"), "groups"), _mapping(document.get("packages"), "packages")
    if len(raw_groups) > _MAX_GROUPS or len(raw_packages) > _MAX_PACKAGES:
        raise _bad("groups/packages", "条目过多")
    for key in (*raw_groups, *raw_packages):
        if not is_plugin_name(key):
            raise _bad("名字", f"{key!r} 含不允许的字符")
    groups = {key: _group(key, value) for key, value in raw_groups.items()}
    packages = {key: _package(key, value) for key, value in raw_packages.items()}
    _check_links(groups, packages)
    return PackageIndex(sha, groups, packages)


# 浅取 index 提交并读出 index.json：提交的树只能有这一个文件，先看大小再读内容；按字节取回再显式按 UTF-8 解码。
# 强度边界：1 MiB 上限只限解析与缓存；浅 fetch 本身仍会先把对象落到临时对象库（远端发布超大对象时临时占用磁盘，随后即删）。
def _fetch_text(repo: str, sha: str) -> str:
    with tempfile.TemporaryDirectory(prefix="akashic-index-") as directory:
        work = Path(directory)

        def git(*command: str) -> bytes:
            result = subprocess.run(["git", *command], cwd=work, capture_output=True, check=False, timeout=_TIMEOUT_S, env=git_env())
            if result.returncode != 0:
                tail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()
                raise RuntimeError(f"取 index 失败（{command[0]}）: {tail[-1] if tail else result.returncode}")
            return result.stdout

        _ = git("init", "--quiet")
        _ = git("fetch", "--quiet", "--depth=1", "--no-tags", "--", repo, sha)
        if git("ls-tree", "--name-only", sha).decode("utf-8", errors="replace").split() != ["index.json"]:
            raise _bad("提交", "树里不是只有 index.json")
        if int(git("cat-file", "-s", f"{sha}:index.json").strip()) > MAX_INDEX_BYTES:
            raise _bad("文件", f"超过 {MAX_INDEX_BYTES} 字节")
        try:
            return git("show", f"{sha}:index.json").decode("utf-8")
        except UnicodeDecodeError as error:
            raise _bad("文件", "不是合法的 UTF-8") from error


def _cache_path(home: Path, repo: str) -> Path:
    return home / "index-cache" / (hashlib.sha256(repo.strip().rstrip("/").encode()).hexdigest()[:16] + ".json")


# 读缓存：超限、损坏、字段类型不对或内容不合格都当没有缓存（继续联网取），不抛出。
def _read_cache(path: Path) -> tuple[str, str] | None:
    try:
        if not path.is_file() or path.stat().st_size > 2 * MAX_INDEX_BYTES:
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            return None
        sha, text = cast(dict[str, object], record).get("sha"), cast(dict[str, object], record).get("text")
        if not isinstance(sha, str) or not isinstance(text, str):
            return None
        _ = parse_index(text, sha)
        return sha, text
    except (OSError, ValueError, IndexFormatError):
        return None


# 唯一临时文件加原子替换：并行的 pkg 命令互不踩。
def _write_cache(path: Path, sha: str, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            _ = stream.write(json.dumps({"sha": sha, "text": text}))
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


# 读取 index：远端 SHA 与缓存相同直接用缓存；不同则取新的并写缓存；联网失败回落到缓存并带说明；
# 远端 index 内容不合格是另一类错误：直接报，不回落旧缓存。
def load_index(repo: str, home: Path) -> PackageIndex:
    path = _cache_path(home, repo)
    cached = _read_cache(path)
    try:
        _, sha = list_heads(repo)
        if not sha:
            raise RuntimeError(f"{repo} 没有 index 分支")
        if cached is not None and cached[0] == sha:
            return parse_index(cached[1], sha)
        text = _fetch_text(repo, sha)
    except IndexFormatError:
        raise
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        if cached is None:
            raise RuntimeError(f"无法读取包组信息（{str(error).splitlines()[0]}），本机也没有缓存") from error
        return replace(parse_index(cached[1], cached[0]),
                       note=f"联网失败（{str(error).splitlines()[0]}），使用本机缓存的包组信息")
    index = parse_index(text, sha)
    _write_cache(path, sha, text)
    return index


# 依赖闭包：给定包名加上它们（传递地）requires 的包，按"依赖在前"排序；未知名字抛 KeyError 由调用方转成明确报错。
def closure(index: PackageIndex, names: list[str]) -> list[str]:
    return closure_of(index.packages, names)


def closure_of(packages: dict[str, IndexPackage], names: list[str]) -> list[str]:
    ordered: list[str] = []
    visiting: set[str] = set()

    def visit(name: str) -> None:
        if name in ordered:
            return
        if name in visiting:
            raise RuntimeError(f"index 里的包依赖成环: {name}")
        visiting.add(name)
        for dependency in packages[name].requires:
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
