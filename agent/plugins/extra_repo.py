"""extra 仓库的来源解析与缺口分类；只依赖标准库，宿主部署脚本与包管理命令共用。"""
from __future__ import annotations

import base64
import fcntl
import json
import os
import re
import subprocess
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from agent.plugins.manifest import plugins_root

DEFAULT_EXTRA_REPO = "https://github.com/kachofugetsu09/akashic-plugins.git"
BRANCH_PREFIX = "refs/heads/release/"
INDEX_REF = "refs/heads/index"
_GIT_TIMEOUT_S = 60

# 容器里取不到宿主环境，求 wheel 的 docker run 按名字透传这些代理变量（证书文件路径在容器里没有意义，不传）。
PROXY_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy")
# 会改变 git 定位仓库的变量（见 git(1) Environment Variables）：宿主脚本在临时仓库里 init/fetch/bundle，
# 继承它们会把操作指到别处；其余变量（代理、证书、认证、GIT_CONFIG_*、LANG 等）原样继承，与 pkg 在线发现一致。
_GIT_REPOSITORY_LOCATION = frozenset({
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR", "GIT_NAMESPACE", "GIT_PREFIX", "GIT_IMPLICIT_WORK_TREE", "GIT_CEILING_DIRECTORIES",
    "GIT_GRAFT_FILE", "GIT_SHALLOW_FILE", "GIT_REPLACE_REF_BASE", "GIT_NO_REPLACE_OBJECTS", "GIT_QUARANTINE_PATH",
    "GIT_INTERNAL_SUPER_PREFIX", "GIT_DISCOVERY_ACROSS_FILESYSTEM",
})

GapAction = Literal["install", "switch", "retire", "optional", "retired"]


@dataclass(frozen=True)
class SelectedPackage:
    plugin_id: str
    builtin: bool
    origin: str
    head: str
    # 进程内依赖环境需要重建的原因（environment_repair_reason）；不需要为空。
    repair: str = ""


@dataclass(frozen=True)
class Gap:
    plugin_id: str
    upstream: str
    action: GapAction


@dataclass(frozen=True)
class ExtraAction:
    kind: Literal["install", "switch", "upgrade"]
    plugin_id: str
    commit: str


# 拉取私有源时按地址附上只读凭据：经 GIT_CONFIG_* 注入，只活在这次 git 进程里，不写进任何仓库的配置。
def git_env() -> dict[str, str]:
    environment = {key: value for key, value in os.environ.items() if key not in _GIT_REPOSITORY_LOCATION}
    environment["GIT_TERMINAL_PROMPT"] = "0"
    count = int(environment.get("GIT_CONFIG_COUNT", "0") or "0")
    for source in private_sources():
        token = source_token(source.name)
        if not token:
            continue
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        environment[f"GIT_CONFIG_KEY_{count}"] = f"http.{source.url}.extraHeader"
        environment[f"GIT_CONFIG_VALUE_{count}"] = f"Authorization: Basic {basic}"
        count += 1
    if count:
        environment["GIT_CONFIG_COUNT"] = str(count)
    return environment


# ---- 来源清单：像 pacman.conf 的仓库段，官方源总在最前，私有源按登记顺序排在后面 ----

SOURCES_FILE = "sources.toml"
_TOKENS_DIR = "source-tokens"
_SOURCE_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,39}")
# local 留给本机目录安装；release 是官方源的身份后缀。
_RESERVED_NAMES = frozenset({"local", "release", "core", "extra"})


@dataclass(frozen=True)
class Source:
    # 名称同时是这个来源装出的插件 ID 后缀（calendar@<名称>），装过包后不能再改。
    name: str
    url: str
    official: bool = False


def _sources_path() -> Path:
    return plugins_root() / SOURCES_FILE


# 私有源按文件里的顺序；文件不存在等于没有私有源，格式坏了直接报错，不静默忽略。
def private_sources() -> tuple[Source, ...]:
    path = _sources_path()
    if not path.is_file():
        return ()
    document = tomllib.loads(path.read_text(encoding="utf-8"))
    rows = document.get("source", [])
    if not isinstance(rows, list):
        raise RuntimeError(f"{path} 格式错误：应为 [[source]] 列表")
    sources: list[Source] = []
    for row in cast(list[object], rows):
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or not isinstance(row.get("url"), str):
            raise RuntimeError(f"{path} 格式错误：每个 [[source]] 需要 name 与 url")
        item = cast(dict[str, str], row)
        sources.append(Source(item["name"], item["url"]))
    return tuple(sources)


# 全部来源：官方在前；官方身份后缀取自发行版，没有发行版时（开发环境）仍叫 release。
def all_sources(official_repo: str | None = None) -> tuple[Source, ...]:
    official = Source(official_marketplace() or "release", resolve_repo(official_repo), official=True)
    return (official, *private_sources())


def source_token(name: str) -> str:
    path = plugins_root() / _TOKENS_DIR / name
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def _write_sources(sources: Iterable[Source]) -> None:
    lines = ["# 私有插件源，按顺序查找：同名包取第一个有它的来源。由 pkg source add/remove 维护。", ""]
    for source in sources:
        lines += ["[[source]]", f"name = {json.dumps(source.name)}", f"url = {json.dumps(source.url)}", ""]
    path = _sources_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    _ = temporary.write_text("\n".join(lines), encoding="utf-8")
    _ = temporary.replace(path)


# 来源清单的读改写互斥：网页、CLI、Agent 可能同时增删来源。
@contextmanager
def _sources_locked() -> Iterator[None]:
    path = plugins_root() / "sources.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


# 登记私有源：先确认地址真的是插件源（有 index 分支），再落盘；凭据单独存放、只给本用户读。
def add_source(name: str, url: str, token: str = "") -> Source:
    url = url.strip()
    official = all_sources()[0]
    # 1. 名称与地址的基本校验；官方源的实际名称（发行版声明的 marketplace）也不能占用。
    if _SOURCE_NAME.fullmatch(name) is None or name in _RESERVED_NAMES or name == official.name:
        reserved = ", ".join(sorted({*_RESERVED_NAMES, official.name}))
        raise ValueError(f"来源名称 {name!r} 不可用：小写字母、数字、点、横线，且不能是 {reserved}")
    with _sources_locked():
        current = private_sources()
        if any(item.name == name for item in current):
            raise ValueError(f"来源 {name} 已存在；要换地址先 pkg source remove {name}")
        if any(same_origin(item.url, url) for item in (*current, official)):
            raise ValueError(f"{url} 已经登记过")
        # 2. 凭据先写，下面的探测就能带上它。
        tokens = plugins_root() / _TOKENS_DIR
        if token:
            tokens.mkdir(mode=0o700, parents=True, exist_ok=True)
            (tokens / name).touch(mode=0o600)
            _ = (tokens / name).write_text(token.strip(), encoding="utf-8")
        source = Source(name, url)
        # 3. 先登记再探测，探测失败只撤掉这一条；地址读不到或不是插件源都当场说明。
        _write_sources((*current, source))
        try:
            _probe(url)
        except BaseException:
            _write_sources(item for item in private_sources() if item.name != name)
            (tokens / name).unlink(missing_ok=True)
            raise
    return source


def _probe(url: str) -> None:
    heads, index = list_heads(url)
    if not index:
        raise RuntimeError(f"{url} 不是插件源：没有 index 分支（需要按 akashic-plugins 的发布方式生成）")
    if not heads:
        raise RuntimeError(f"{url} 还没有发布任何插件（没有 release/* 分支）")


def remove_source(name: str) -> None:
    with _sources_locked():
        current = private_sources()
        if not any(item.name == name for item in current):
            raise ValueError(f"没有名为 {name} 的私有源")
        _write_sources(item for item in current if item.name != name)
        (plugins_root() / _TOKENS_DIR / name).unlink(missing_ok=True)


# 来源仓库：显式参数优先，其次环境变量，最后是默认常量。
def resolve_repo(option: str | None) -> str:
    return option or os.environ.get("AKASHIC_EXTRA_REPO", "").strip() or DEFAULT_EXTRA_REPO


# 官方 marketplace：目标发行版 distribution.json 声明的那个；没有发行版时无法确定，返回 None。
def official_marketplace() -> str | None:
    configured = os.environ.get("AKASHIC_PLUGIN_DISTRIBUTION", "").strip()
    if not configured:
        return None
    report = json.loads((Path(configured) / "distribution.json").read_text(encoding="utf-8"))
    marketplace = report.get("marketplace")
    return marketplace if isinstance(marketplace, str) and marketplace else None


# 来源是否就是配置的官方 extra 仓库（环境变量或默认常量解析出的那个）。
def is_official_repo(repo: str) -> bool:
    return same_origin(repo, resolve_repo(None))


# 两个仓库地址是否同一来源：只去首尾空白与末尾 /（pkg、安装器守卫与部署清单共用这一口径）。
def same_origin(left: str, right: str) -> bool:
    return left.strip().rstrip("/") == right.strip().rstrip("/")


# 一次 ls-remote 取全部 release/<name> 分支与 index 分支：返回 (name 到提交 SHA 的映射, index 的 SHA，没有则空串)。
def list_heads(repo: str) -> tuple[dict[str, str], str]:
    result = subprocess.run(
        ["git", "ls-remote", "--heads", "--", repo, BRANCH_PREFIX + "*", INDEX_REF],
        capture_output=True, text=True, check=False, timeout=_GIT_TIMEOUT_S,
        env=git_env(),
    )
    if result.returncode != 0:
        raise RuntimeError(f"git ls-remote 失败: {repo}\n{result.stderr.strip()}")
    heads: dict[str, str] = {}
    index = ""
    for line in result.stdout.splitlines():
        sha, _, ref = line.partition("\t")
        if len(sha) != 40:
            continue
        if ref.startswith(BRANCH_PREFIX):
            heads[ref.removeprefix(BRANCH_PREFIX)] = sha
        elif ref == INDEX_REF:
            index = sha
    return heads, index


# 只要 release/<name> 分支的映射（pkg 的既有调用方）。
def list_release_heads(repo: str) -> dict[str, str]:
    return list_heads(repo)[0]


# 缺口分类，pkg sync 与部署清单共用。判据是"现在是否在用"，不只看 preset：
#   preset 启用的行            -> 换来源（旧选择里是内置）或首装；
#   当前选择里是内置输入（在用）-> 上游存在则换来源，上游不存在是真正的退役（switch / retire），退役会让在运行的能力消失；
#   其余只在用户 patch 里启用的行 -> 上游存在只提示、不自动装（optional），不存在记为已退役并跳过（retired）。
# heads 是惰性的：没有缺口时不联网。
def classify_gaps(
    *, enabled: Iterable[str], present: frozenset[str], preset: frozenset[str],
    heads: Callable[[], Mapping[str, str]], needs_switch: Callable[[str], bool],
    in_use: frozenset[str] = frozenset(),
) -> list[Gap]:
    missing = [plugin_id for plugin_id in enabled if plugin_id not in present]
    if not missing:
        return []
    remote = heads()
    gaps: list[Gap] = []
    for plugin_id in missing:
        upstream = plugin_id.partition("@")[0] in remote
        if plugin_id in preset:
            action: GapAction = "switch" if needs_switch(plugin_id) else "install"
        elif plugin_id in in_use:
            action = "switch" if upstream else "retire"
        else:
            action = "optional" if upstream else "retired"
        gaps.append(Gap(plugin_id, "available" if upstream else "missing-upstream", action))
    return gaps


# 已装且来源就是 extra 仓库、远端有新提交的包（升级）。
def upgrade_actions(
    selected: Mapping[str, SelectedPackage], heads: Callable[[], Mapping[str, str]], repo: str, *, gaps_only: bool = False,
) -> list[ExtraAction]:
    actions: list[ExtraAction] = []
    for plugin_id, package in sorted(selected.items()):
        if package.builtin or not same_origin(package.origin, repo):
            continue
        # 1. 提交落后：升级到上游新提交（新提交自带环境）；启动补装不升级。
        remote = "" if gaps_only else heads().get(plugin_id.partition("@")[0], "")
        if remote and remote != package.head:
            actions.append(ExtraAction("upgrade", plugin_id, remote))
        # 2. 提交没变但进程内环境需要重建（旧 Core 装的、Python 小版本变了等）：同一提交重装，kind 仍是 upgrade。
        elif package.repair and package.head:
            actions.append(ExtraAction("upgrade", plugin_id, package.head))
    return actions
