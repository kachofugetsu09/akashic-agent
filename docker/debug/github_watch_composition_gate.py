from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCK = ROOT / "docker/debug/github-watch-composition.lock.json"
DEFAULT_REPORT = ROOT / "docker/debug/reports/github-watch-composition/gate.json"
_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_PROTOCOL_PATHS = (
    "docs/design/plugin-timer-snapshot-scheduling-task-contract.md",
    "docs/design/plugin-agent-input-stable-admission-task-contract.md",
    "docs/design/plugin-turn-committed-event-task-contract.md",
)


@dataclass(frozen=True, slots=True)
class ProtocolSource:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ProviderSource:
    id: str
    repository: str
    requested_ref: str
    resolved_sha: str
    change_source_pr_head: str
    pull_request: str
    tree: str
    plugin_sha256: str
    gate_path: str
    gate_sha256: str
    source_digest: str


@dataclass(frozen=True, slots=True)
class CompositionLock:
    profile: str
    protocol_sources: tuple[ProtocolSource, ...]
    provider: ProviderSource


@dataclass(frozen=True, slots=True)
class GateArguments:
    lock: Path
    report: Path
    require_clean_core: bool


def main() -> None:
    """验证固定 GitHub Watch v3 候选与当前 Core 的组合行为。"""

    # 1. 固定 Core、合同和 provider 身份
    args = _parse_args()
    core_status = _git(ROOT, "status", "--porcelain").splitlines()
    if args.require_clean_core and core_status:
        raise RuntimeError(f"Core Git worktree 不干净: {core_status}")
    lock_path = args.lock.resolve(strict=True)
    lock = _load_lock(lock_path)
    core_head = _git(ROOT, "rev-parse", "HEAD")
    core_tree = _git(ROOT, "rev-parse", "HEAD^{tree}")
    protocol_evidence = _verify_protocol_sources(
        lock.protocol_sources,
        core_head=core_head,
    )

    # 2. 只在一次性 checkout 和 workspace 运行 provider 自带场景
    with tempfile.TemporaryDirectory(prefix="akashic-github-watch-gate-") as raw:
        checkout = Path(raw) / lock.provider.id
        provider_evidence = _checkout_provider(lock.provider, checkout)
        provider_report = _run_provider_gate(
            lock.provider,
            checkout,
            core_head=core_head,
        )
    _validate_provider_report(
        provider_report,
        lock.provider,
        core_head=core_head,
        core_tree=core_tree,
    )

    # 3. 发布可重建身份和 Core-owned oracle 结果
    report = {
        "status": "passed",
        "checked_at": datetime.now(UTC).isoformat(),
        "consumer": {
            "repository": "https://github.com/kachofugetsu09/akashic-agent.git",
            "commit": core_head,
            "tree": core_tree,
            "dirty_status": core_status,
        },
        "protocol_sources": protocol_evidence,
        "provider": provider_evidence,
        "scenario": {
            "profile": lock.profile,
            "lock": str(lock_path.relative_to(ROOT)),
            "lock_sha256": _sha256(lock_path),
            "gate_sha256": _sha256(Path(__file__)),
        },
        "observations": provider_report["observations"],
    }
    report_path = args.report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    _ = report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"github-watch composition gate passed: {report_path}")


def _parse_args() -> GateArguments:
    parser = argparse.ArgumentParser(description="验证 GitHub Watch v3 组合候选")
    _ = parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    _ = parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    _ = parser.add_argument("--require-clean-core", action="store_true")
    raw = parser.parse_args()
    return GateArguments(
        lock=cast(Path, raw.lock),
        report=cast(Path, raw.report),
        require_clean_core=cast(bool, raw.require_clean_core),
    )


def _load_lock(path: Path) -> CompositionLock:
    """严格解析一个不可变的 GitHub Watch 组合锁。"""

    # 1. 根字段不接受缺失、额外字段或版本漂移
    decoded = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(decoded, dict):
        raise ValueError("GitHub Watch 组合锁根结构无效")
    raw = cast(dict[str, object], decoded)
    if set(raw) != {"schema_version", "profile", "protocol_sources", "provider"}:
        raise ValueError("GitHub Watch 组合锁根结构无效")
    if raw["schema_version"] != 1:
        raise ValueError(f"不支持的 GitHub Watch 组合锁版本: {raw['schema_version']}")
    profile = _required_string(raw, "profile")
    if profile != "github-watch-v3-composition-v1":
        raise ValueError(f"GitHub Watch 组合锁 profile 无效: {profile}")

    # 2. 合同路径和 provider 身份必须完整且按固定顺序出现
    sources = raw["protocol_sources"]
    if not isinstance(sources, list):
        raise ValueError("GitHub Watch 组合锁 protocol_sources 必须是数组")
    protocols = tuple(_parse_protocol(item) for item in cast(list[object], sources))
    if tuple(item.path for item in protocols) != _PROTOCOL_PATHS:
        raise ValueError("GitHub Watch 组合锁 protocol_sources 集合无效")
    provider = _parse_provider(raw["provider"])
    return CompositionLock(
        profile=profile,
        protocol_sources=protocols,
        provider=provider,
    )


def _parse_protocol(raw: object) -> ProtocolSource:
    if not isinstance(raw, dict) or set(raw) != {"path", "sha256"}:
        raise ValueError(f"GitHub Watch 协议 source 字段无效: {raw}")
    item = cast(dict[str, object], raw)
    path = _required_string(item, "path")
    digest = _required_digest(item, "sha256")
    return ProtocolSource(path=path, sha256=digest)


def _parse_provider(raw: object) -> ProviderSource:
    fields = {
        "id",
        "repository",
        "requested_ref",
        "resolved_sha",
        "change_source_pr_head",
        "pull_request",
        "tree",
        "plugin_sha256",
        "gate_path",
        "gate_sha256",
        "source_digest",
    }
    if not isinstance(raw, dict) or set(raw) != fields:
        raise ValueError(f"GitHub Watch provider source 字段无效: {raw}")
    item = cast(dict[str, object], raw)
    values = {name: _required_string(item, name) for name in fields}
    if values["id"] != "github-watch":
        raise ValueError("GitHub Watch provider id 无效")
    if not values["repository"].startswith("https://github.com/"):
        raise ValueError("GitHub Watch provider repository 必须是 GitHub HTTPS")
    if not values["pull_request"].startswith("https://github.com/"):
        raise ValueError("GitHub Watch provider pull_request 必须是 GitHub HTTPS")
    for name in ("resolved_sha", "change_source_pr_head", "tree"):
        if _SHA_PATTERN.fullmatch(values[name]) is None:
            raise ValueError(f"GitHub Watch provider {name} 必须是完整 SHA")
    if values["resolved_sha"] != values["change_source_pr_head"]:
        raise ValueError("GitHub Watch provider PR head 与 resolved SHA 不一致")
    for name in ("plugin_sha256", "gate_sha256", "source_digest"):
        if _DIGEST_PATTERN.fullmatch(values[name]) is None:
            raise ValueError(f"GitHub Watch provider {name} 必须是 SHA-256")
    if values["gate_path"] != "scripts/core_v3_gate.py":
        raise ValueError("GitHub Watch provider gate_path 无效")
    return ProviderSource(**values)


def _required_string(item: dict[str, object], name: str) -> str:
    value = item[name]
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"GitHub Watch 组合锁 {name} 必须是非空字符串")
    return value


def _required_digest(item: dict[str, object], name: str) -> str:
    value = _required_string(item, name)
    if _DIGEST_PATTERN.fullmatch(value) is None:
        raise ValueError(f"GitHub Watch 组合锁 {name} 必须是 SHA-256")
    return value


def _verify_protocol_sources(
    sources: tuple[ProtocolSource, ...],
    *,
    core_head: str,
) -> list[dict[str, str]]:
    evidence: list[dict[str, str]] = []
    for source in sources:
        path = ROOT / source.path
        if _sha256(path) != source.sha256:
            raise RuntimeError(f"GitHub Watch 协议 source 摘要漂移: {source.path}")
        evidence.append(
            {
                "repository": "https://github.com/kachofugetsu09/akashic-agent.git",
                "commit": core_head,
                "path": source.path,
                "sha256": source.sha256,
            }
        )
    return evidence


def _checkout_provider(source: ProviderSource, checkout: Path) -> dict[str, str]:
    """从公开仓库只检出锁定 provider commit。"""

    # 1. 在本次 Gate 开始时把 requested ref 解析为完整 commit
    resolved = _resolve_remote_ref(source.repository, source.requested_ref)
    if resolved != source.resolved_sha:
        raise RuntimeError(
            "GitHub Watch requested ref 已移动，必须更新不可变组合锁: "
            f"expected={source.resolved_sha} actual={resolved}"
        )

    # 2. 精确 fetch，不复用宿主插件 cache 或正式安装
    _run(("git", "init", "--quiet", str(checkout)), cwd=ROOT)
    _run(("git", "remote", "add", "origin", source.repository), cwd=checkout)
    _run(
        ("git", "fetch", "--quiet", "--depth=1", "origin", source.resolved_sha),
        cwd=checkout,
    )
    _run(("git", "checkout", "--quiet", "--detach", "FETCH_HEAD"), cwd=checkout)
    if _git(checkout, "rev-parse", "HEAD") != source.resolved_sha:
        raise RuntimeError("GitHub Watch provider commit 身份漂移")
    if _git(checkout, "status", "--porcelain"):
        raise RuntimeError("GitHub Watch provider checkout 不干净")

    # 3. tree、入口和场景脚本都必须与 lock 一致
    if _git(checkout, "rev-parse", "HEAD^{tree}") != source.tree:
        raise RuntimeError("GitHub Watch provider tree 身份漂移")
    if _sha256(checkout / "plugin.py") != source.plugin_sha256:
        raise RuntimeError("GitHub Watch plugin.py 摘要漂移")
    if _sha256(checkout / source.gate_path) != source.gate_sha256:
        raise RuntimeError("GitHub Watch provider Gate 摘要漂移")
    return asdict(source)


def _run_provider_gate(
    source: ProviderSource,
    checkout: Path,
    *,
    core_head: str,
) -> dict[str, object]:
    command = (
        sys.executable,
        str(checkout / source.gate_path),
        "--core",
        str(ROOT),
        "--expected-core",
        core_head,
        "--require-clean-plugin",
    )
    result = subprocess.run(command, cwd=checkout, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "GitHub Watch provider Gate 失败:\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    decoded = cast(object, json.loads(result.stdout))
    if not isinstance(decoded, dict):
        raise RuntimeError("GitHub Watch provider Gate 报告根结构无效")
    return cast(dict[str, object], decoded)


def _validate_provider_report(
    report: dict[str, object],
    source: ProviderSource,
    *,
    core_head: str,
    core_tree: str,
) -> None:
    """用 Core-owned oracle 核对 provider 场景的完整观察值。"""

    # 1. 组合身份必须与当前 Core 和固定 provider 完全一致
    expected_root = {
        "status",
        "core_head",
        "core_tree",
        "plugin_head",
        "plugin_tree",
        "plugin_dirty",
        "plugin_source_digest",
        "observations",
    }
    if set(report) != expected_root:
        raise RuntimeError("GitHub Watch provider Gate 报告字段漂移")
    expected_identity = {
        "status": "passed",
        "core_head": core_head,
        "core_tree": core_tree,
        "plugin_head": source.resolved_sha,
        "plugin_tree": source.tree,
        "plugin_dirty": [],
        "plugin_source_digest": source.source_digest,
    }
    for name, expected in expected_identity.items():
        if report[name] != expected:
            raise RuntimeError(f"GitHub Watch provider Gate identity 漂移: {name}")

    # 2. 已知 mutant 必须改变精确 catalog、effect 或清理观察值
    observations = report["observations"]
    expected_observations = _expected_observations()
    if observations != expected_observations:
        raise RuntimeError(
            "GitHub Watch provider Gate 行为漂移:\n"
            f"expected={json.dumps(expected_observations, ensure_ascii=False, sort_keys=True)}\n"
            f"actual={json.dumps(observations, ensure_ascii=False, sort_keys=True)}"
        )


def _expected_observations() -> dict[str, object]:
    return {
        "services": [
            "core.agent_input",
            "core.commands",
            "core.skills",
            "core.timer",
            "core.tools",
            "core.ui_slots",
        ],
        "listeners": ["serial:turn.after_turn.committed:github-watch"],
        "timers": ["github-watch:poll"],
        "tools": [
            "github_watch_runtime_info",
            "github_watch_post_comment",
            "github_watch_submit_review",
            "github_watch_push_branch",
            "github_watch_create_pr",
        ],
        "agent_input_created": [["github-watch", {"source": "github-watch"}]],
        "agent_input_submitted": [["github-watch", "session-1", "gate prompt"]],
        "runtime_data_paths": [
            "workspace/plugin-data/github-watch-builtin/events.sqlite3"
        ],
        "fake_watch_count": 1,
        "cleanup_operations": ["a" * 32],
        "old_root_effects_after_drain": [],
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_remote_ref(repository: str, requested_ref: str) -> str:
    result = subprocess.run(
        ("git", "ls-remote", repository, requested_ref),
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line.split() for line in result.stdout.splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 2 or rows[0][1] != requested_ref:
        raise RuntimeError(f"GitHub Watch requested ref 无法唯一解析: {requested_ref}")
    resolved = rows[0][0]
    if _SHA_PATTERN.fullmatch(resolved) is None:
        raise RuntimeError(f"GitHub Watch requested ref 返回无效 SHA: {resolved}")
    return resolved


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _run(command: tuple[str, ...], *, cwd: Path) -> None:
    _ = subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    main()
