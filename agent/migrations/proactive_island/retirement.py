"""Bind operator-approved legacy blocks to one verified recovery point."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import cast
from uuid import uuid4

from agent.migrations.proactive_island.inventory import Inventory

RETIREMENT_REASON = "operator_approved_pre_cutover_supersession"
_APPROVABLE_REASONS = frozenset(
    {
        "proactive_continuity_owner_unavailable",
        "wake_continuity_owner_unavailable",
        "proposal_payload_unrecoverable",
        "proactive_quota_owner_unavailable",
    }
)


def without_retired_blocks(workspace: Path, inventory: Inventory) -> Inventory:
    """Remove only blocks exactly covered by the durable retirement receipt."""

    path = retirement_receipt_path(workspace)
    if not path.is_file():
        return inventory

    # 1. Validate the receipt and its independent recovery point.
    receipt = _read_receipt(path)
    _verify_backup(receipt)
    retired = {
        (str(item["locator"]), str(item["reason"]), str(item["source_digest"]))
        for item in _receipt_blocks(receipt)
    }

    # 2. Suppress exact matches only; changed or new blocks remain active.
    active = tuple(
        block
        for block in inventory.blocks
        if (block.locator, block.reason, block.source_digest) not in retired
    )
    return Inventory(inventory.facts, active)


def write_retirement_receipt(
    workspace: Path,
    inventory: Inventory,
    backup_root: Path,
) -> Path:
    """Atomically record exact approved blocks after target handoffs succeed."""

    validate_retirement_blocks(inventory)
    path = retirement_receipt_path(workspace)
    manifest = backup_root / "manifest.json"
    _verify_backup_manifest(backup_root, manifest)
    payload: dict[str, object] = {
        "schema_version": 1,
        "decision": RETIREMENT_REASON,
        "backup_root": str(backup_root),
        "backup_manifest_sha256": _digest(manifest),
        "blocks": [
            {
                "locator": block.locator,
                "reason": block.reason,
                "source_digest": block.source_digest,
            }
            for block in inventory.blocks
        ],
    }
    content = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError("proactive retirement receipt conflict")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        _ = temporary.write_bytes(content)
        with temporary.open("rb") as stream:
            _ = os.fsync(stream.fileno())
        _ = os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            _ = os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def retirement_receipt_path(workspace: Path) -> Path:
    return workspace / "runtime" / "proactive-island-handoff" / "retirement.json"


def validate_retirement_blocks(inventory: Inventory) -> None:
    """Reject blocks whose semantics were not approved for supersession."""

    unsupported = sorted(
        {block.reason for block in inventory.blocks} - _APPROVABLE_REASONS
    )
    if unsupported:
        raise RuntimeError(
            "proactive retirement contains unsupported blocks: "
            + ", ".join(unsupported)
        )


def _read_receipt(path: Path) -> dict[str, object]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("proactive retirement receipt must be an object")
    receipt = cast(dict[str, object], decoded)
    if (
        receipt.get("schema_version") != 1
        or receipt.get("decision") != RETIREMENT_REASON
    ):
        raise RuntimeError("unsupported proactive retirement receipt")
    _ = _receipt_blocks(receipt)
    return receipt


def _receipt_blocks(receipt: dict[str, object]) -> list[dict[str, object]]:
    raw = receipt.get("blocks")
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("proactive retirement receipt has no blocks")
    blocks: list[dict[str, object]] = []
    for item in cast(list[object], raw):
        if not isinstance(item, dict):
            raise RuntimeError("proactive retirement block must be an object")
        block = cast(dict[str, object], item)
        if any(
            not isinstance(block.get(field), str)
            for field in ("locator", "reason", "source_digest")
        ):
            raise RuntimeError("proactive retirement block identity is invalid")
        blocks.append(block)
    return blocks


def _verify_backup(receipt: dict[str, object]) -> None:
    raw_root = receipt.get("backup_root")
    expected = receipt.get("backup_manifest_sha256")
    if not isinstance(raw_root, str) or not Path(raw_root).is_absolute():
        raise RuntimeError("proactive retirement backup root is invalid")
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError("proactive retirement backup digest is invalid")
    root = Path(raw_root)
    manifest = root / "manifest.json"
    if _digest(manifest) != expected:
        raise RuntimeError("proactive retirement backup manifest changed")
    _verify_backup_manifest(root, manifest)


def _verify_backup_manifest(root: Path, manifest: Path) -> None:
    decoded = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("proactive handoff backup manifest must be an object")
    payload = cast(dict[str, object], decoded)
    for section in ("sqlite", "files"):
        entries = payload.get(section)
        if not isinstance(entries, list):
            raise RuntimeError(f"proactive handoff backup {section} is invalid")
        for raw in cast(list[object], entries):
            if not isinstance(raw, dict):
                raise RuntimeError("proactive handoff backup entry must be an object")
            entry = cast(dict[str, object], raw)
            relative = entry.get("backup")
            expected = entry.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise RuntimeError("proactive handoff backup entry identity is invalid")
            target = root / relative
            if (
                root.resolve() not in target.resolve().parents
                or _digest(target) != expected
            ):
                raise RuntimeError(
                    f"proactive handoff backup verification failed: {relative}"
                )


def _digest(path: Path) -> str:
    if not path.is_file():
        raise RuntimeError(f"proactive recovery artifact missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


__all__ = [
    "RETIREMENT_REASON",
    "retirement_receipt_path",
    "validate_retirement_blocks",
    "without_retired_blocks",
    "write_retirement_receipt",
]
