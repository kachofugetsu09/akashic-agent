from __future__ import annotations

import re
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_LOCK = ROOT / "docker" / "debug" / "plugin-passive-composition-v3.lock.json"
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")
EXPECTED_PLUGIN_IDS = ("citation", "meme")


@dataclass(frozen=True)
class SourceLock:
    id: str
    repository: str
    requested_ref: str
    resolved_sha: str
    change_source_pr_head: str


@dataclass(frozen=True)
class SourceEvidence:
    id: str
    repository: str
    requested_ref: str
    resolved_sha: str
    change_source_pr_head: str
    tree: str


@dataclass(frozen=True)
class GateLock:
    contract: SourceLock
    plugins: tuple[SourceLock, ...]


@dataclass(frozen=True)
class ContractEvidence:
    contract: str
    plugin_ids: tuple[str, ...]
    source_sha256: tuple[str, ...]
    plugin_classes: tuple[tuple[str, ...], ...]


def _load_lock(path: Path) -> GateLock:
    """Strictly load the immutable protocol and plugin source set."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "contract",
        "plugins",
    }:
        raise ValueError("passive v3 组合锁根结构无效")
    if raw["schema_version"] != 1:
        raise ValueError(f"不支持的 passive v3 组合锁版本: {raw['schema_version']}")
    contract = _parse_source_lock(raw["contract"])
    raw_plugins = raw["plugins"]
    if not isinstance(raw_plugins, list):
        raise ValueError("passive v3 组合锁 plugins 必须是列表")
    plugins = tuple(_parse_source_lock(item) for item in raw_plugins)
    if tuple(item.id for item in plugins) != EXPECTED_PLUGIN_IDS:
        raise ValueError("passive v3 组合锁的插件集合或顺序错误")
    if contract.id != "plugin_contracts":
        raise ValueError("passive v3 组合锁缺少 plugin_contracts owner")
    return GateLock(contract=contract, plugins=plugins)


def _parse_source_lock(raw: object) -> SourceLock:
    expected = {
        "id",
        "repository",
        "requested_ref",
        "resolved_sha",
        "change_source_pr_head",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"passive v3 组合锁字段无效: {raw}")
    item = cast(dict[str, object], raw)
    values = {name: _required_string(item, name) for name in expected}
    repository = values["repository"]
    if not repository.startswith("https://github.com/") or not repository.endswith(
        ".git"
    ):
        raise ValueError(f"源仓库必须是公开 GitHub HTTPS Git 地址: {repository}")
    revisions = ("requested_ref", "resolved_sha", "change_source_pr_head")
    for field in revisions:
        if COMMIT_PATTERN.fullmatch(values[field]) is None:
            raise ValueError(f"{field} 必须是完整 SHA: {values[field]}")
    if len({values[field] for field in revisions}) != 1:
        raise ValueError(f"三个 revision 必须固定到同一提交: {values['id']}")
    return SourceLock(**values)


def _required_string(item: dict[str, object], name: str) -> str:
    value = item[name]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"passive v3 组合锁字段必须是非空字符串: {name}")
    return value


def _checkout_locked_source(lock: SourceLock, checkout: Path) -> SourceEvidence:
    """Fetch one declared public object into a fresh, detached repository."""

    _run(("git", "init", "--quiet", str(checkout)), cwd=ROOT)
    _run(("git", "remote", "add", "origin", lock.repository), cwd=checkout)
    _run(
        ("git", "fetch", "--quiet", "--depth=1", "origin", lock.resolved_sha),
        cwd=checkout,
    )
    _run(("git", "checkout", "--quiet", "--detach", "FETCH_HEAD"), cwd=checkout)
    if _git_output(checkout, "rev-parse", "HEAD") != lock.resolved_sha:
        raise RuntimeError(f"检出提交与锁不一致: {lock.id}")
    if _git_output(checkout, "status", "--porcelain"):
        raise RuntimeError(f"检出后工作树不干净: {lock.id}")
    return SourceEvidence(
        id=lock.id,
        repository=lock.repository,
        requested_ref=lock.requested_ref,
        resolved_sha=lock.resolved_sha,
        change_source_pr_head=lock.change_source_pr_head,
        tree=_git_output(checkout, "rev-parse", "HEAD^{tree}"),
    )


def _verify_static_contract(
    contract_checkout: Path,
    plugin_paths: tuple[Path, ...],
) -> ContractEvidence:
    """Run the exact public contract checker and require pure v3 entrypoints."""

    command = (
        sys.executable,
        "-m",
        "akashic_plugin_contracts",
        "check",
        *(str(path) for path in plugin_paths),
    )
    completed = _run(command, cwd=contract_checkout)
    raw = json.loads(completed.stdout)
    reports = raw.get("reports")
    if raw.get("passed") is not True or raw.get("contract") != "akashic-plugin-api-v3":
        raise RuntimeError(f"插件静态合同失败: {raw}")
    if not isinstance(reports, list) or len(reports) != len(plugin_paths):
        raise RuntimeError(f"插件静态合同报告数量错误: {raw}")
    plugin_classes = tuple(tuple(item["plugin_classes"]) for item in reports)
    if any(plugin_classes):
        raise RuntimeError(f"纯 v3 Gate 发现 v2 Plugin 类: {plugin_classes}")
    return ContractEvidence(
        contract=str(raw["contract"]),
        plugin_ids=EXPECTED_PLUGIN_IDS,
        source_sha256=tuple(str(item["sha256"]) for item in reports),
        plugin_classes=plugin_classes,
    )


def _git_output(cwd: Path, *args: str) -> str:
    return _run(("git", *args), cwd=cwd).stdout.strip()


def _run(command: tuple[str, ...], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
