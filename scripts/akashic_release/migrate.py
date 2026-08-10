from __future__ import annotations

from pathlib import Path

from scripts.akashic_release.manifest import read_json

_PHASES = (
    "停止旧 ingress 与后台写 owner",
    "复核 SQLite integrity 与普通文件 snapshot identity",
    "恢复到候选 state 并重装 plugin generation",
    "运行 legacy skill-link adoption",
    "以禁外发配置完成正式数据隔离验收",
    "维护者批准后切换唯一 ingress",
    "观察并保留旧端、快照和全部 generation",
)


def migration_plan(snapshot_manifest: Path) -> dict[str, object]:
    """Validate a rehearsal snapshot and return a no-write migration plan."""

    document = read_json(snapshot_manifest.resolve(strict=True))
    consistency = document.get("consistency")
    databases = document.get("databases")
    cleanup = document.get("cleanup")
    if not isinstance(consistency, dict) or not isinstance(databases, list):
        raise RuntimeError("snapshot manifest 缺少 consistency/databases")
    if not isinstance(cleanup, dict) or not isinstance(
        cleanup.get("exact_paths"), list
    ):
        raise RuntimeError("snapshot manifest 缺少 cleanup ownership")
    failed = [
        item
        for item in databases
        if not isinstance(item, dict)
        or item.get("source_integrity_check") != "ok"
        or item.get("target_integrity_check") != "ok"
    ]
    if failed:
        raise RuntimeError("snapshot manifest 含未通过 integrity 的数据库")
    return {
        "mode": "plan_only",
        "snapshotManifest": str(snapshot_manifest.resolve()),
        "phases": list(_PHASES),
        "automaticDataWrites": False,
        "stopBefore": "正式 ingress 或 workspace mutation",
    }
