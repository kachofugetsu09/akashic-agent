"""Order target-owned proactive handoffs before append-only source lineage."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from agent.migrations.proactive_island.inventory import Inventory, LegacyFact


class HandoffStatus(StrEnum):
    READY = "ready"
    PLAN = "plan"
    APPLIED = "applied"
    BLOCK = "block"


class HandoffBlocked(RuntimeError):
    """Return one owner-classified preflight block to the maintenance caller."""


@dataclass(frozen=True, slots=True)
class TargetReceipt:
    """Identify one durable target fact without exposing its domain payload."""

    receipt_id: str
    receipt_digest: str
    target_identity: str


@dataclass(frozen=True, slots=True)
class AdapterPlan:
    """Report a target owner's read-only acceptance of one source fact."""

    target_identity: str


class HandoffAdapter(Protocol):
    """Let one target owner inspect, apply, and verify its own durable fact."""

    def accepts(self, fact: LegacyFact) -> bool: ...

    def plan(self, fact: LegacyFact) -> AdapterPlan: ...

    def apply(self, fact: LegacyFact, plan: AdapterPlan) -> TargetReceipt: ...

    def verify(self, fact: LegacyFact, receipt: TargetReceipt) -> bool: ...


@dataclass(frozen=True, slots=True)
class HandoffItem:
    locator: str
    source_digest: str
    target_identity: str | None
    receipt_id: str | None
    state: str
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class HandoffReport:
    status: HandoffStatus
    items: tuple[HandoffItem, ...]


_SCHEMA = """
CREATE TABLE attempts(
    attempt_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE TABLE lineage(
    locator TEXT NOT NULL,
    source_digest TEXT NOT NULL,
    receipt_id TEXT NOT NULL,
    receipt_digest TEXT NOT NULL,
    target_identity TEXT NOT NULL,
    attempt_id TEXT NOT NULL,
    marked_at TEXT NOT NULL,
    PRIMARY KEY(locator, source_digest),
    FOREIGN KEY(attempt_id) REFERENCES attempts(attempt_id)
);
CREATE INDEX lineage_locator_idx ON lineage(locator);
PRAGMA user_version = 1;
"""


def preflight_handoff(
    workspace: Path,
    inventory: Inventory,
    adapters: Sequence[HandoffAdapter],
) -> HandoffReport:
    """Plan missing targets and reverify every previously marked receipt."""

    if inventory.blocks:
        return HandoffReport(
            HandoffStatus.BLOCK,
            tuple(
                HandoffItem(
                    locator=block.locator,
                    source_digest=block.source_digest,
                    target_identity=None,
                    receipt_id=None,
                    state="blocked",
                    reason=block.reason,
                )
                for block in inventory.blocks
            ),
        )
    if not inventory.facts:
        return HandoffReport(HandoffStatus.READY, ())

    # 1. Read the optional central marker without creating its parent or database.
    lineage = _read_lineage(_lineage_path(workspace))
    items: list[HandoffItem] = []
    any_plan = False
    for fact in inventory.facts:
        matched = [adapter for adapter in adapters if adapter.accepts(fact)]
        if not matched:
            items.append(_blocked_fact(fact, "owner_adapter_unavailable"))
            continue
        if len(matched) > 1:
            items.append(_blocked_fact(fact, "owner_adapter_conflict"))
            continue
        adapter = matched.pop()
        try:
            plan = adapter.plan(fact)
        except HandoffBlocked as error:
            items.append(_blocked_fact(fact, str(error)))
            continue
        marker = lineage.get((fact.locator, fact.source_digest))
        if marker is None:
            items.append(
                HandoffItem(
                    fact.locator,
                    fact.source_digest,
                    plan.target_identity,
                    None,
                    "planned",
                )
            )
            any_plan = True
            continue
        receipt = TargetReceipt(marker[0], marker[1], marker[2])
        if receipt.target_identity != plan.target_identity:
            items.append(_blocked_fact(fact, "lineage_target_identity_drift"))
            continue
        if not adapter.verify(fact, receipt):
            items.append(_blocked_fact(fact, "target_receipt_unverified"))
            continue
        items.append(
            HandoffItem(
                fact.locator,
                fact.source_digest,
                receipt.target_identity,
                receipt.receipt_id,
                "applied",
            )
        )
    if any(item.state == "blocked" for item in items):
        status = HandoffStatus.BLOCK
    elif any_plan:
        status = HandoffStatus.PLAN
    else:
        status = HandoffStatus.APPLIED
    return HandoffReport(status, tuple(items))


def apply_handoff(
    workspace: Path,
    inventory: Inventory,
    adapters: Sequence[HandoffAdapter],
    *,
    planned: HandoffReport | None = None,
    after_target: Callable[[LegacyFact, TargetReceipt], None] | None = None,
) -> HandoffReport:
    """Apply each target idempotently, then append its exact source marker."""

    planned = planned or preflight_handoff(workspace, inventory, adapters)
    if planned.status in {HandoffStatus.READY, HandoffStatus.APPLIED}:
        return planned
    if planned.status is HandoffStatus.BLOCK:
        return planned

    # 1. Recheck every planned target before the first migration-side write.
    path = _lineage_path(workspace)
    marked = _read_lineage(path)
    expected = {
        (item.locator, item.source_digest): item.target_identity
        for item in planned.items
        if item.state == "planned"
    }
    plans: dict[tuple[str, str], AdapterPlan] = {}
    for fact in inventory.facts:
        key = (fact.locator, fact.source_digest)
        if key in marked:
            continue
        target_identity = expected.get(key)
        if target_identity is None:
            return HandoffReport(
                HandoffStatus.BLOCK,
                (_blocked_fact(fact, "expected_target_plan_missing"),),
            )
        adapter = _require_adapter(fact, adapters)
        try:
            current = adapter.plan(fact)
        except HandoffBlocked as error:
            return HandoffReport(
                HandoffStatus.BLOCK,
                (_blocked_fact(fact, str(error)),),
            )
        if current.target_identity != target_identity:
            return HandoffReport(
                HandoffStatus.BLOCK,
                (_blocked_fact(fact, "target_plan_drift_before_apply"),),
            )
        plans[key] = AdapterPlan(target_identity)

    # 2. Record one observable apply attempt only after plans remain exact.
    attempt_id = f"proactive-island-handoff:{uuid4().hex}"
    _begin_attempt(path, attempt_id)

    # 3. Commit and verify the exact target before publishing source lineage.
    for fact in inventory.facts:
        key = (fact.locator, fact.source_digest)
        if key in marked:
            continue
        adapter = _require_adapter(fact, adapters)
        plan = plans[key]
        receipt = adapter.apply(fact, plan)
        if receipt.target_identity != plan.target_identity:
            raise RuntimeError(f"target identity drift after plan: {fact.locator}")
        if not adapter.verify(fact, receipt):
            raise RuntimeError(f"target receipt verify failed: {fact.locator}")
        if after_target is not None:
            after_target(fact, receipt)
        _append_marker(path, attempt_id, fact, receipt)
    _complete_attempt(path, attempt_id)
    return preflight_handoff(workspace, inventory, adapters)


def _lineage_path(workspace: Path) -> Path:
    return workspace / "runtime" / "proactive-island-handoff" / "lineage.sqlite3"


def _adapter_for(
    fact: LegacyFact,
    adapters: Sequence[HandoffAdapter],
) -> HandoffAdapter | None:
    matched = [adapter for adapter in adapters if adapter.accepts(fact)]
    if len(matched) > 1:
        raise RuntimeError(f"multiple target owners accepted fact: {fact.locator}")
    return matched[0] if matched else None


def _require_adapter(
    fact: LegacyFact,
    adapters: Sequence[HandoffAdapter],
) -> HandoffAdapter:
    adapter = _adapter_for(fact, adapters)
    if adapter is None:
        raise RuntimeError(f"target owner unavailable: {fact.locator}")
    return adapter


def _blocked_fact(fact: LegacyFact, reason: str) -> HandoffItem:
    return HandoffItem(
        fact.locator,
        fact.source_digest,
        None,
        None,
        "blocked",
        reason,
    )


def _connect(path: Path, *, read_only: bool) -> sqlite3.Connection:
    if read_only:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        _ = connection.execute("PRAGMA query_only = ON")
    else:
        connection = sqlite3.connect(path)
        _ = connection.execute("PRAGMA foreign_keys = ON")
        _ = connection.execute("PRAGMA synchronous = FULL")
    connection.row_factory = sqlite3.Row
    return connection


def _read_lineage(path: Path) -> dict[tuple[str, str], tuple[str, str, str]]:
    if not path.is_file():
        return {}
    with closing(_connect(path, read_only=True)) as connection:
        _validate_lineage(connection)
        rows = connection.execute(
            "SELECT locator, source_digest, receipt_id, receipt_digest, "
            "target_identity FROM lineage"
        ).fetchall()
    return {
        (str(row["locator"]), str(row["source_digest"])): (
            str(row["receipt_id"]),
            str(row["receipt_digest"]),
            str(row["target_identity"]),
        )
        for row in rows
    }


def _initialize(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(path, read_only=False)) as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version == 0:
            _ = connection.executescript(_SCHEMA)
            connection.commit()
        else:
            _validate_lineage(connection)


def _validate_lineage(connection: sqlite3.Connection) -> None:
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version != 1:
        raise RuntimeError(f"unsupported proactive handoff lineage version: {version}")
    tables = tuple(
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    )
    if tables != ("attempts", "lineage"):
        raise RuntimeError("proactive handoff lineage table set mismatch")
    attempt_columns = tuple(
        str(row[1]) for row in connection.execute("PRAGMA table_info(attempts)")
    )
    if attempt_columns != ("attempt_id", "started_at", "completed_at"):
        raise RuntimeError("proactive handoff attempt columns mismatch")
    lineage_columns = tuple(
        str(row[1]) for row in connection.execute("PRAGMA table_info(lineage)")
    )
    if lineage_columns != (
        "locator",
        "source_digest",
        "receipt_id",
        "receipt_digest",
        "target_identity",
        "attempt_id",
        "marked_at",
    ):
        raise RuntimeError("proactive handoff lineage columns mismatch")
    check = [tuple(row) for row in connection.execute("PRAGMA quick_check")]
    if check != [("ok",)]:
        raise RuntimeError("proactive handoff lineage quick_check failed")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise RuntimeError("proactive handoff lineage foreign key check failed")


def _begin_attempt(path: Path, attempt_id: str) -> None:
    _initialize(path)
    with closing(_connect(path, read_only=False)) as connection:
        _ = connection.execute(
            "INSERT INTO attempts(attempt_id, started_at) VALUES(?, ?)",
            (attempt_id, datetime.now(UTC).isoformat()),
        )
        connection.commit()


def _append_marker(
    path: Path,
    attempt_id: str,
    fact: LegacyFact,
    receipt: TargetReceipt,
) -> None:
    with closing(_connect(path, read_only=False)) as connection:
        _ = connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT receipt_id, receipt_digest, target_identity FROM lineage "
            "WHERE locator=? AND source_digest=?",
            (fact.locator, fact.source_digest),
        ).fetchone()
        if existing is not None:
            if tuple(existing) != (
                receipt.receipt_id,
                receipt.receipt_digest,
                receipt.target_identity,
            ):
                raise RuntimeError(f"lineage receipt conflict: {fact.locator}")
            connection.commit()
            return
        _ = connection.execute(
            "INSERT INTO lineage(locator, source_digest, receipt_id, receipt_digest, "
            "target_identity, attempt_id, marked_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
            (
                fact.locator,
                fact.source_digest,
                receipt.receipt_id,
                receipt.receipt_digest,
                receipt.target_identity,
                attempt_id,
                datetime.now(UTC).isoformat(),
            ),
        )
        connection.commit()


def _complete_attempt(path: Path, attempt_id: str) -> None:
    with closing(_connect(path, read_only=False)) as connection:
        _ = connection.execute(
            "UPDATE attempts SET completed_at=? WHERE attempt_id=?",
            (datetime.now(UTC).isoformat(), attempt_id),
        )
        connection.commit()


def receipt_digest(payload: Mapping[str, object]) -> str:
    """Hash one target-owned receipt after its adapter has normalized it."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
