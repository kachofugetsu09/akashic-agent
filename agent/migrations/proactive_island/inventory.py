"""Inventory active legacy proactive facts without opening a writer."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from agent.migrations.proactive_island.reader import open_legacy_sqlite


class LegacyFactKind(StrEnum):
    WAKE_SOURCE_ITEM = "wake_source_item"
    WAKE_ACK = "wake_ack"
    WAKE_RULES = "wake_rules"


@dataclass(frozen=True, slots=True)
class LegacyFact:
    """Carry one exact source fact in memory while lineage stores only its digest."""

    kind: LegacyFactKind
    locator: str
    source_digest: str
    source_identity: str
    opaque: bytes


@dataclass(frozen=True, slots=True)
class InventoryBlock:
    """Describe one exact legacy fact that cannot be handed off."""

    locator: str
    reason: str
    source_digest: str = ""


@dataclass(frozen=True, slots=True)
class Inventory:
    facts: tuple[LegacyFact, ...]
    blocks: tuple[InventoryBlock, ...]


_WAKE_TERMINAL_STATUSES = frozenset({"expired", "quarantined"})
_WAKE_TABLES = frozenset(
    {
        "wake_runs",
        "wake_observations",
        "reservoir_events",
        "reservoir_quarantine",
        "reservoir_tombstones",
        "hazard_state",
        "hazard_monitor",
        "context_state",
        "context_reevaluate_state",
        "drift_state",
        "pending_acknowledgements",
    }
)
# runtime.py reloads the six tables below. wake_runs/wake_observations are history,
# and dashboard.py is the only production reader of hazard_monitor.
_WAKE_CONTINUITY_TABLES = frozenset(
    {
        "reservoir_quarantine",
        "reservoir_tombstones",
        "hazard_state",
        "context_state",
        "context_reevaluate_state",
        "drift_state",
    }
)
_PROACTIVE_TABLES = frozenset(
    {
        "deliveries",
        "session_state",
        "context_only_timestamps",
        "tick_log",
        "tick_step_log",
        "rejection_cooldown",
        "seen_items",
        "semantic_items",
        "kv_state",
    }
)
_PROACTIVE_CONTINUITY_TABLES = frozenset(
    {
        "deliveries",
        "session_state",
        "context_only_timestamps",
        "rejection_cooldown",
        "seen_items",
        "kv_state",
    }
)


def inventory_workspace(workspace: Path) -> Inventory:
    """Read every active legacy source and return deterministic facts and blocks."""

    root = workspace.resolve(strict=False)
    facts: list[LegacyFact] = []
    blocks: list[InventoryBlock] = []

    # 1. Read legacy databases through read-only SQLite connections only.
    _inventory_proactive(root / "proactive.db", blocks)
    _inventory_wake(root / "wake_proactive.db", facts, blocks)
    _inventory_drift(root / "drift" / "drift.db", blocks)
    _inventory_document_intents(root / "runtime" / "proactive-documents", blocks)

    # 2. Classify the two legacy Markdown facts without interpreting their text.
    _inventory_pending(root / "proactive_pending.md", blocks)
    _inventory_rules(root / "PROACTIVE_CONTEXT.md", facts)
    _inventory_quota(root / "proactive_quota.json", blocks)

    facts.sort(key=lambda item: (item.kind.value, item.locator))
    blocks.sort(key=lambda item: (item.locator, item.reason))
    return Inventory(tuple(facts), tuple(blocks))


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }


def _inventory_proactive(path: Path, blocks: list[InventoryBlock]) -> None:
    """Block each retained continuity table without copying its row history."""

    if not path.is_file():
        return
    with open_legacy_sqlite(path) as connection:
        tables = _tables(connection)
        for table in sorted(tables - _PROACTIVE_TABLES):
            digest = _table_digest(connection, table)
            blocks.append(
                InventoryBlock(f"proactive:{table}", "unknown_proactive_table", digest)
            )
        for table in sorted(tables & _PROACTIVE_CONTINUITY_TABLES):
            count = int(
                connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
            )
            if count:
                table_digest = _table_digest(connection, table)
                blocks.append(
                    InventoryBlock(
                        f"proactive:{table}",
                        "proactive_continuity_owner_unavailable",
                        f"rows={count};sha256={table_digest}",
                    )
                )


def _inventory_wake(
    path: Path,
    facts: list[LegacyFact],
    blocks: list[InventoryBlock],
) -> None:
    if not path.is_file():
        return
    with open_legacy_sqlite(path) as connection:
        tables = _tables(connection)
        _inventory_wake_continuity(connection, tables, blocks)
        if "reservoir_events" not in tables:
            blocks.append(InventoryBlock("wake:reservoir_events", "schema_missing"))
            return
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(reservoir_events)")
        }
        required = {
            "item_id",
            "kind",
            "source_id",
            "ack_source_id",
            "source_event_id",
            "status",
        }
        if not required.issubset(columns):
            blocks.append(
                InventoryBlock("wake:reservoir_events", "schema_columns_missing")
            )
            return
        rows = connection.execute(
            "SELECT * FROM reservoir_events ORDER BY item_id"
        ).fetchall()
        pending_rows = (
            connection.execute(
                "SELECT * FROM pending_acknowledgements "
                "ORDER BY source_id, source_event_id, item_id"
            ).fetchall()
            if "pending_acknowledgements" in tables
            else []
        )
    # 1. External source identity is the only cross-row uniqueness boundary.
    external: dict[tuple[str, str], tuple[str, str]] = {}
    reservoir_by_item = {str(row["item_id"]): row for row in rows}
    for row in rows:
        payload = _row_bytes(row)
        item_id = str(row["item_id"])
        locator = f"wake:reservoir_events:{item_id}"
        status = str(row["status"])
        kind = str(row["kind"])
        source_id = _required_owner(row["ack_source_id"], row["source_id"])
        event_id = _required_identity(row["source_event_id"])
        if source_id is None or event_id is None:
            blocks.append(
                InventoryBlock(locator, "source_identity_missing", _digest(payload))
            )
            continue
        digest = _digest(payload)
        previous = external.get((source_id, event_id))
        if previous is not None and previous != (item_id, digest):
            blocks.append(InventoryBlock(locator, "source_identity_conflict", digest))
            continue
        external[(source_id, event_id)] = (item_id, digest)
        if status == "unread":
            if kind not in {"alert", "content"}:
                blocks.append(
                    InventoryBlock(locator, f"unknown_wake_kind:{kind}", digest)
                )
                continue
            facts.append(
                LegacyFact(
                    kind=LegacyFactKind.WAKE_SOURCE_ITEM,
                    locator=locator,
                    source_digest=digest,
                    source_identity=source_id,
                    opaque=payload,
                )
            )
        elif status in {"consumed", "pending_expiry"}:
            continue
        elif status not in _WAKE_TERMINAL_STATUSES:
            blocks.append(
                InventoryBlock(locator, f"unknown_wake_status:{status}", digest)
            )

    # 2. An ACK row is active only with its exact reservoir owner still present.
    for row in pending_rows:
        item_id = str(row["item_id"])
        locator = (
            "wake:pending_acknowledgements:"
            f"{row['source_id']}:{row['source_event_id']}:{item_id}"
        )
        reservoir = reservoir_by_item.get(item_id)
        action = str(row["action"])
        payload = _row_bytes(row)
        digest = _digest(payload)
        if reservoir is None:
            blocks.append(InventoryBlock(locator, "orphan_pending_ack", digest))
            continue
        if action not in {"consume", "expire"}:
            blocks.append(
                InventoryBlock(locator, f"unknown_ack_action:{action}", digest)
            )
            continue
        source_identity = _required_identity(row["source_id"])
        if source_identity is None:
            blocks.append(InventoryBlock(locator, "source_identity_missing", digest))
            continue
        reservoir_owner = _required_owner(
            reservoir["ack_source_id"], reservoir["source_id"]
        )
        reservoir_event = _required_identity(reservoir["source_event_id"])
        if (
            source_identity != reservoir_owner
            or _required_identity(row["source_event_id"]) != reservoir_event
        ):
            blocks.append(
                InventoryBlock(locator, "ack_source_identity_conflict", digest)
            )
            continue
        facts.append(
            LegacyFact(
                kind=LegacyFactKind.WAKE_ACK,
                locator=locator,
                source_digest=digest,
                source_identity=source_identity,
                opaque=payload,
            )
        )


def _inventory_wake_continuity(
    connection: sqlite3.Connection,
    tables: set[str],
    blocks: list[InventoryBlock],
) -> None:
    """Block table-owned Wake continuity without decoding its domain fields."""

    # 1. A new table has no reviewed owner or history classification.
    for table in sorted(tables - _WAKE_TABLES):
        blocks.append(
            InventoryBlock(
                f"wake:{table}",
                "unknown_wake_table",
                _table_digest(connection, table),
            )
        )

    # 2. These stores are read by later ingress or Wake decisions.
    for table in sorted(tables & _WAKE_CONTINUITY_TABLES):
        count = int(connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
        if count:
            blocks.append(
                InventoryBlock(
                    f"wake:{table}",
                    "wake_continuity_owner_unavailable",
                    f"rows={count};sha256={_table_digest(connection, table)}",
                )
            )


def _inventory_drift(path: Path, blocks: list[InventoryBlock]) -> None:
    if not path.is_file():
        return
    with open_legacy_sqlite(path) as connection:
        tables = _tables(connection)
        if "skill_continuum" in tables:
            for row in connection.execute(
                "SELECT * FROM skill_continuum WHERE last_status='paused' "
                "ORDER BY skill_name"
            ):
                payload = _row_bytes(row)
                blocks.append(
                    InventoryBlock(
                        f"drift:skill_continuum:{row['skill_name']}",
                        "proposal_payload_unrecoverable",
                        _digest(payload),
                    )
                )
        if "runs" in tables:
            for row in connection.execute(
                "SELECT * FROM runs WHERE message_result='staged' ORDER BY id"
            ):
                identity = row["event_id"] if row["event_id"] is not None else row["id"]
                payload = _row_bytes(row)
                blocks.append(
                    InventoryBlock(
                        f"drift:runs:{identity}",
                        "proposal_payload_unrecoverable",
                        _digest(payload),
                    )
                )


def _inventory_document_intents(path: Path, blocks: list[InventoryBlock]) -> None:
    intents = path / "intents"
    if not intents.is_dir():
        return
    for entry in sorted(intents.iterdir(), key=lambda item: item.name):
        digest = _path_digest(entry)
        if entry.name.startswith("."):
            blocks.append(
                InventoryBlock(
                    f"documents:intents:{entry.name}",
                    "incomplete_intent_entry",
                    digest,
                )
            )
            continue
        blocks.append(
            InventoryBlock(
                f"documents:intents:{entry.name}",
                "paired_target_handoff_unavailable",
                digest,
            )
        )


def _inventory_pending(path: Path, blocks: list[InventoryBlock]) -> None:
    if path.is_file() and path.stat().st_size > 0:
        content = path.read_bytes()
        blocks.append(
            InventoryBlock(
                "documents:proactive_pending.md",
                "pending_document_owner_unavailable",
                _digest(content),
            )
        )


def _inventory_rules(path: Path, facts: list[LegacyFact]) -> None:
    if not path.is_file():
        return
    content = path.read_bytes()
    facts.append(
        LegacyFact(
            kind=LegacyFactKind.WAKE_RULES,
            locator="documents:PROACTIVE_CONTEXT.md",
            source_digest=_digest(content),
            source_identity="wake",
            opaque=content,
        )
    )


def _inventory_quota(path: Path, blocks: list[InventoryBlock]) -> None:
    """Preserve the exact current quota until a target owner can accept it."""

    if not path.is_file():
        return
    content = path.read_bytes()
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise RuntimeError(f"legacy proactive quota is not an object: {path}")
    blocks.append(
        InventoryBlock(
            "proactive:quota",
            "proactive_quota_owner_unavailable",
            _digest(content),
        )
    )


def inventory_digest(inventory: Inventory) -> str:
    """Digest source identities and blocks without retaining opaque payloads."""

    payload = {
        "facts": [
            {
                "kind": fact.kind.value,
                "locator": fact.locator,
                "source_digest": fact.source_digest,
                "source_identity": fact.source_identity,
            }
            for fact in inventory.facts
        ],
        "blocks": [
            {
                "locator": block.locator,
                "reason": block.reason,
                "source_digest": block.source_digest,
            }
            for block in inventory.blocks
        ],
    }
    return _digest(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _table_digest(connection: sqlite3.Connection, table: str) -> str:
    """Digest a table independently of SQLite's incidental row order."""

    rows = sorted(
        _row_bytes(row) for row in connection.execute(f'SELECT * FROM "{table}"')
    )
    digest = hashlib.sha256()
    for row in rows:
        digest.update(len(row).to_bytes(8, "big"))
        digest.update(row)
    return digest.hexdigest()


def _path_digest(path: Path) -> str:
    """Digest one legacy file tree without creating a parallel representation."""

    if path.is_file():
        return _digest(path.read_bytes())
    digest = hashlib.sha256()
    for entry in sorted(path.rglob("*"), key=lambda item: str(item.relative_to(path))):
        relative = str(entry.relative_to(path)).encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        if entry.is_file():
            content = entry.read_bytes()
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
        else:
            digest.update(b"directory")
    return digest.hexdigest()


def _row_bytes(row: sqlite3.Row) -> bytes:
    payload: dict[str, Any] = {key: _json_value(row[key]) for key in row.keys()}
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_value(value: object) -> object:
    if isinstance(value, bytes):
        return {"sqlite_blob_hex": value.hex()}
    return value


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _required_owner(primary: object, fallback: object) -> str | None:
    return _required_identity(primary) or _required_identity(fallback)


def _required_identity(value: object) -> str | None:
    return (
        value if isinstance(value, str) and value and value.strip() == value else None
    )
