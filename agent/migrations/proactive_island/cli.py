"""Plan or apply proactive-island handoff in an explicit workspace."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Sequence

from agent.migrations.proactive_island.handoff import (
    HandoffAdapter,
    HandoffReport,
    HandoffStatus,
    apply_handoff,
    preflight_handoff,
)
from agent.migrations.proactive_island.inventory import inventory_workspace
from agent.migrations.session_db_backup import backup_sqlite_database
from plugins.wake.migration import WakeRulesArchiveAdapter


def plan(workspace: Path, adapters: Sequence[HandoffAdapter] = ()) -> HandoffReport:
    """Inventory and preflight one workspace without creating any state."""

    inventory = inventory_workspace(workspace)
    return preflight_handoff(
        workspace,
        inventory,
        (*adapters, WakeRulesArchiveAdapter(workspace)),
    )


def apply(
    workspace: Path,
    backup_root: Path,
    adapters: Sequence[HandoffAdapter] = (),
) -> HandoffReport:
    """Back up active legacy sources, then apply target-first handoffs."""

    _require_absolute(workspace)
    inventory = inventory_workspace(workspace)
    selected = (*adapters, WakeRulesArchiveAdapter(workspace))
    report = preflight_handoff(workspace, inventory, selected)
    if report.status is not HandoffStatus.PLAN:
        return report

    # 1. Capture every legacy source before the first target owner write.
    backup_sources(workspace, backup_root)

    # 2. Each adapter commits its own target; Core appends lineage afterwards.
    return apply_handoff(workspace, inventory, selected, planned=report)


def backup_sources(workspace: Path, backup_root: Path) -> None:
    """Back up existing legacy SQLite and Markdown sources with full digests."""

    if backup_root.exists():
        raise FileExistsError(f"handoff backup root already exists: {backup_root}")
    backup_root.mkdir(parents=True, mode=0o700)
    sqlite_paths = (
        workspace / "wake_proactive.db",
        workspace / "drift" / "drift.db",
        workspace / "runtime" / "plugin-jobs" / "outcomes.sqlite",
    )
    sqlite_entries: list[dict[str, object]] = []
    for index, source in enumerate(sqlite_paths):
        if not source.is_file():
            continue
        target = backup_sqlite_database(
            source,
            backup_root / f"sqlite-{index}",
            migration="proactive-island-handoff-v1",
        )
        sqlite_entries.append(
            {
                "source": str(source),
                "backup": str(target.relative_to(backup_root)),
                "sha256": _digest(target),
            }
        )

    file_root = backup_root / "files"
    file_entries: list[dict[str, object]] = []
    for name in ("PROACTIVE_CONTEXT.md", "proactive_pending.md"):
        source = workspace / name
        if not source.is_file():
            continue
        file_root.mkdir(parents=True, exist_ok=True)
        target = file_root / name
        _ = shutil.copy2(source, target)
        file_entries.append(
            {
                "source": str(source),
                "backup": str(target.relative_to(backup_root)),
                "sha256": _digest(target),
                "size": target.stat().st_size,
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "migration": "proactive-island-handoff-v1",
        "sqlite": sqlite_entries,
        "files": file_entries,
    }
    manifest_path = backup_root / "manifest.json"
    _ = manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with manifest_path.open("rb") as stream:
        _ = os.fsync(stream.fileno())


def report_payload(report: HandoffReport) -> dict[str, object]:
    return {
        "status": report.status.value,
        "items": [
            {
                "locator": item.locator,
                "source_digest": item.source_digest,
                "target_identity": item.target_identity,
                "receipt_id": item.receipt_id,
                "state": item.state,
                "reason": item.reason,
            }
            for item in report.items
        ],
    }


def _require_absolute(workspace: Path) -> None:
    if not workspace.is_absolute():
        raise ValueError("apply workspace must be an absolute path")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["apply", "backup_sources", "plan", "report_payload"]
