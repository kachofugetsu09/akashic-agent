"""Inventory active legacy proactive facts without opening a writer."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


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


@dataclass(frozen=True, slots=True)
class Inventory:
    facts: tuple[LegacyFact, ...]
    blocks: tuple[InventoryBlock, ...]


_PROACTIVE_JOB_OWNERS = frozenset({"emotion", "emotion@github"})
_PROACTIVE_JOB_NAMES = frozenset({"merge_pending", "merge_proactive_pending"})
_TERMINAL_JOB_STATES = frozenset({"cancelled", "succeeded", "failed"})
_WAKE_TERMINAL_STATUSES = frozenset({"expired", "quarantined"})


def inventory_workspace(workspace: Path) -> Inventory:
    """Read every active legacy source and return deterministic facts and blocks."""

    root = workspace.resolve(strict=False)
    facts: list[LegacyFact] = []
    blocks: list[InventoryBlock] = []

    # 1. Read legacy databases through read-only SQLite connections only.
    _inventory_wake(root / "wake_proactive.db", facts, blocks)
    _inventory_drift(root / "drift" / "drift.db", blocks)
    _inventory_jobs(root / "runtime" / "plugin-jobs" / "outcomes.sqlite", blocks)
    _inventory_document_intents(root / "runtime" / "proactive-documents", blocks)

    # 2. Classify the two legacy Markdown facts without interpreting their text.
    _inventory_pending(root / "proactive_pending.md", blocks)
    _inventory_rules(root / "PROACTIVE_CONTEXT.md", facts)

    facts.sort(key=lambda item: (item.kind.value, item.locator))
    blocks.sort(key=lambda item: (item.locator, item.reason))
    return Inventory(tuple(facts), tuple(blocks))


def _connect_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    _ = connection.execute("PRAGMA query_only = ON")
    check = [tuple(row) for row in connection.execute("PRAGMA quick_check")]
    if check != [("ok",)]:
        connection.close()
        raise RuntimeError(f"legacy SQLite quick_check failed: {path}")
    return connection


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }


def _inventory_wake(
    path: Path,
    facts: list[LegacyFact],
    blocks: list[InventoryBlock],
) -> None:
    if not path.is_file():
        return
    with closing(_connect_read_only(path)) as connection:
        tables = _tables(connection)
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
        item_id = str(row["item_id"])
        locator = f"wake:reservoir_events:{item_id}"
        status = str(row["status"])
        kind = str(row["kind"])
        source_id = _required_owner(row["ack_source_id"], row["source_id"])
        event_id = _required_identity(row["source_event_id"])
        if source_id is None or event_id is None:
            blocks.append(InventoryBlock(locator, "source_identity_missing"))
            continue
        payload = _row_bytes(row)
        digest = _digest(payload)
        previous = external.get((source_id, event_id))
        if previous is not None and previous != (item_id, digest):
            blocks.append(InventoryBlock(locator, "source_identity_conflict"))
            continue
        external[(source_id, event_id)] = (item_id, digest)
        if status == "unread":
            if kind not in {"alert", "content"}:
                blocks.append(InventoryBlock(locator, f"unknown_wake_kind:{kind}"))
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
            blocks.append(InventoryBlock(locator, f"unknown_wake_status:{status}"))

    # 2. An ACK row is active only with its exact reservoir owner still present.
    for row in pending_rows:
        item_id = str(row["item_id"])
        locator = (
            "wake:pending_acknowledgements:"
            f"{row['source_id']}:{row['source_event_id']}:{item_id}"
        )
        reservoir = reservoir_by_item.get(item_id)
        action = str(row["action"])
        if reservoir is None:
            blocks.append(InventoryBlock(locator, "orphan_pending_ack"))
            continue
        if action not in {"consume", "expire"}:
            blocks.append(InventoryBlock(locator, f"unknown_ack_action:{action}"))
            continue
        source_identity = _required_identity(row["source_id"])
        if source_identity is None:
            blocks.append(InventoryBlock(locator, "source_identity_missing"))
            continue
        reservoir_owner = _required_owner(
            reservoir["ack_source_id"], reservoir["source_id"]
        )
        reservoir_event = _required_identity(reservoir["source_event_id"])
        if (
            source_identity != reservoir_owner
            or _required_identity(row["source_event_id"]) != reservoir_event
        ):
            blocks.append(InventoryBlock(locator, "ack_source_identity_conflict"))
            continue
        payload = _row_bytes(row)
        facts.append(
            LegacyFact(
                kind=LegacyFactKind.WAKE_ACK,
                locator=locator,
                source_digest=_digest(payload),
                source_identity=source_identity,
                opaque=payload,
            )
        )


def _inventory_drift(path: Path, blocks: list[InventoryBlock]) -> None:
    if not path.is_file():
        return
    with closing(_connect_read_only(path)) as connection:
        tables = _tables(connection)
        if "skill_continuum" in tables:
            for row in connection.execute(
                "SELECT skill_name FROM skill_continuum WHERE last_status='paused' "
                "ORDER BY skill_name"
            ):
                blocks.append(
                    InventoryBlock(
                        f"drift:skill_continuum:{row['skill_name']}",
                        "proposal_payload_unrecoverable",
                    )
                )
        if "runs" in tables:
            for row in connection.execute(
                "SELECT id, event_id FROM runs WHERE message_result='staged' ORDER BY id"
            ):
                identity = row["event_id"] if row["event_id"] is not None else row["id"]
                blocks.append(
                    InventoryBlock(
                        f"drift:runs:{identity}",
                        "proposal_payload_unrecoverable",
                    )
                )


def _inventory_jobs(path: Path, blocks: list[InventoryBlock]) -> None:
    if not path.is_file():
        return
    with closing(_connect_read_only(path)) as connection:
        if "job_outcomes" not in _tables(connection):
            blocks.append(InventoryBlock("jobs:job_outcomes", "schema_missing"))
            return
        rows = connection.execute(
            "SELECT plugin_id, job_name, invocation_id, state FROM job_outcomes "
            "ORDER BY invocation_id"
        ).fetchall()
    for row in rows:
        owner = str(row["plugin_id"])
        name = str(row["job_name"])
        state = str(row["state"])
        if owner not in _PROACTIVE_JOB_OWNERS or name not in _PROACTIVE_JOB_NAMES:
            continue
        if state not in _TERMINAL_JOB_STATES:
            blocks.append(
                InventoryBlock(
                    f"jobs:job_outcomes:{row['invocation_id']}",
                    "emotion_handoff_unavailable",
                )
            )


def _inventory_document_intents(path: Path, blocks: list[InventoryBlock]) -> None:
    intents = path / "intents"
    if not intents.is_dir():
        return
    for entry in sorted(intents.iterdir(), key=lambda item: item.name):
        if entry.name.startswith("."):
            blocks.append(
                InventoryBlock(
                    f"documents:intents:{entry.name}", "incomplete_intent_entry"
                )
            )
            continue
        blocks.append(
            InventoryBlock(
                f"documents:intents:{entry.name}",
                "paired_target_handoff_unavailable",
            )
        )


def _inventory_pending(path: Path, blocks: list[InventoryBlock]) -> None:
    if path.is_file() and path.stat().st_size > 0:
        blocks.append(
            InventoryBlock(
                "documents:proactive_pending.md", "emotion_handoff_unavailable"
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


def _row_bytes(row: sqlite3.Row) -> bytes:
    payload: dict[str, Any] = {key: row[key] for key in row.keys()}
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _required_owner(primary: object, fallback: object) -> str | None:
    return _required_identity(primary) or _required_identity(fallback)


def _required_identity(value: object) -> str | None:
    return (
        value if isinstance(value, str) and value and value.strip() == value else None
    )
