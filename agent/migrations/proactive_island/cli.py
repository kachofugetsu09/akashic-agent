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
    HandoffItem,
    HandoffReport,
    HandoffStatus,
    apply_handoff,
    preflight_handoff,
)
from agent.migrations.proactive_island.inventory import (
    Inventory,
    inventory_digest,
    inventory_workspace,
)
from agent.migrations.proactive_island.retirement import (
    validate_retirement_blocks,
    without_retired_blocks,
    write_retirement_receipt,
)
from agent.migrations.session_db_backup import backup_sqlite_database
from agent.migrations.proactive_island.wake_rules import WakeRulesArchiveAdapter


def plan(workspace: Path, adapters: Sequence[HandoffAdapter] = ()) -> HandoffReport:
    """Inventory and preflight one workspace without creating any state."""

    inventory = without_retired_blocks(workspace, inventory_workspace(workspace))
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

    _require_backup_boundary(workspace, backup_root)
    inventory = without_retired_blocks(workspace, inventory_workspace(workspace))
    selected = (*adapters, WakeRulesArchiveAdapter(workspace))
    report = preflight_handoff(workspace, inventory, selected)
    if report.status is not HandoffStatus.PLAN:
        return report

    # 1. Capture every legacy source before the first target owner write.
    backup_sources(workspace, backup_root)

    # 2. Refuse stale active facts or blocks; retained history is not handoff input.
    current = without_retired_blocks(workspace, inventory_workspace(workspace))
    if inventory_digest(current) != inventory_digest(inventory):
        return HandoffReport(
            HandoffStatus.BLOCK,
            (
                HandoffItem(
                    locator="workspace:proactive-island-inventory",
                    source_digest=(
                        f"before={inventory_digest(inventory)};"
                        f"after={inventory_digest(current)}"
                    ),
                    target_identity=None,
                    receipt_id=None,
                    state="blocked",
                    reason="source_inventory_drift_after_backup",
                ),
            ),
        )

    # 3. Each adapter commits its own target; Core appends lineage afterwards.
    return apply_handoff(workspace, current, selected, planned=report)


def retire(
    workspace: Path,
    backup_root: Path,
    expected_inventory_digest: str,
    adapters: Sequence[HandoffAdapter] = (),
) -> HandoffReport:
    """Supersede exact blocked legacy state after target facts are durable."""

    _require_backup_boundary(workspace, backup_root)
    inventory = inventory_workspace(workspace)
    if inventory_digest(inventory) != expected_inventory_digest:
        return _inventory_drift_report(expected_inventory_digest, inventory)
    effective = without_retired_blocks(workspace, inventory)
    if not effective.blocks:
        completed = preflight_handoff(
            workspace,
            effective,
            (*adapters, WakeRulesArchiveAdapter(workspace)),
        )
        if completed.status in {HandoffStatus.APPLIED, HandoffStatus.READY}:
            return completed
    if not inventory.blocks:
        return apply(workspace, backup_root, adapters)
    validate_retirement_blocks(inventory)
    selected = (*adapters, WakeRulesArchiveAdapter(workspace))
    facts_only = Inventory(inventory.facts, ())
    report = preflight_handoff(workspace, facts_only, selected)
    if report.status is HandoffStatus.BLOCK:
        return report

    # 1. Capture and verify every legacy source before target writes.
    backup_sources(workspace, backup_root)
    current = inventory_workspace(workspace)
    if inventory_digest(current) != expected_inventory_digest:
        return _inventory_drift_report(expected_inventory_digest, current)

    # 2. Commit source-owned targets before approving any legacy block.
    applied = apply_handoff(
        workspace,
        Inventory(current.facts, ()),
        selected,
        planned=report,
    )
    if applied.status not in {HandoffStatus.APPLIED, HandoffStatus.READY}:
        return applied

    # 3. Publish one exact receipt; future plans suppress only matching blocks.
    _ = write_retirement_receipt(workspace, current, backup_root)
    return plan(workspace, adapters)


def _inventory_drift_report(
    expected_inventory_digest: str,
    current: Inventory,
) -> HandoffReport:
    return HandoffReport(
        HandoffStatus.BLOCK,
        (
            HandoffItem(
                locator="workspace:proactive-island-inventory",
                source_digest=(
                    f"expected={expected_inventory_digest};"
                    f"actual={inventory_digest(current)}"
                ),
                target_identity=None,
                receipt_id=None,
                state="blocked",
                reason="source_inventory_digest_mismatch",
            ),
        ),
    )


def backup_sources(workspace: Path, backup_root: Path) -> None:
    """Back up existing legacy SQLite and Markdown sources with full digests."""

    _require_backup_boundary(workspace, backup_root)
    if backup_root.exists():
        raise FileExistsError(f"handoff backup root already exists: {backup_root}")
    backup_root.mkdir(parents=True, mode=0o700)
    sqlite_paths = (
        workspace / "proactive.db",
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
    for name in (
        "PROACTIVE_CONTEXT.md",
        "proactive_pending.md",
        "proactive_quota.json",
    ):
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


def _require_backup_boundary(workspace: Path, backup_root: Path) -> None:
    """Require two disjoint absolute trees before creating a recovery point."""

    # 1. Relative paths make the recovery point depend on process cwd.
    if not workspace.is_absolute():
        raise ValueError("apply workspace must be an absolute path")
    if not backup_root.is_absolute():
        raise ValueError("apply backup root must be an absolute path")

    # 2. A backup inside its source, or vice versa, cannot be an independent copy.
    source = workspace.resolve(strict=False)
    backup = backup_root.resolve(strict=False)
    if source == backup or source in backup.parents or backup in source.parents:
        raise ValueError("apply backup root must be disjoint from workspace")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = ["apply", "backup_sources", "plan", "report_payload", "retire"]
