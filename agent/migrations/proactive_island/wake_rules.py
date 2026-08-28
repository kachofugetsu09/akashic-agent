"""Wake-private archive handoff for the retired proactive rules document."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import cast
from uuid import uuid4

from agent.migrations.proactive_island.handoff import (
    AdapterPlan,
    HandoffAdapter,
    HandoffBlocked,
    TargetReceipt,
    receipt_digest,
)
from agent.migrations.proactive_island.inventory import LegacyFact, LegacyFactKind
from agent.plugins.manifest import builtin_plugin_data_dir
RULES_DIRECTORY = "legacy-rules"
RULES_ARCHIVE = "PROACTIVE_CONTEXT.md"
RULES_RECEIPT = "receipt.json"

class WakeRulesArchiveAdapter(HandoffAdapter):
    """Archive exact rules bytes under Wake data and issue a verifiable receipt."""

    def __init__(self, workspace: Path) -> None:
        root = builtin_plugin_data_dir("wake", workspace)
        self._archive = root / RULES_DIRECTORY / RULES_ARCHIVE
        self._receipt = root / RULES_DIRECTORY / RULES_RECEIPT

    def accepts(self, fact: LegacyFact) -> bool:
        return fact.kind is LegacyFactKind.WAKE_RULES

    def plan(self, fact: LegacyFact) -> AdapterPlan:
        """Validate any existing target without creating its directory."""

        self._require_fact(fact)
        if self._archive.exists() and self._archive.read_bytes() != fact.opaque:
            raise HandoffBlocked("wake_rules_archive_conflict")
        if self._receipt.exists():
            receipt = self._read_receipt()
            if not self.verify(fact, receipt):
                raise HandoffBlocked("wake_rules_receipt_conflict")
        return AdapterPlan(f"wake-rules:{fact.source_digest}")

    def apply(self, fact: LegacyFact, plan: AdapterPlan) -> TargetReceipt:
        """Publish exact archive bytes before its target-owned receipt."""

        self._require_fact(fact)
        expected_identity = f"wake-rules:{fact.source_digest}"
        if plan.target_identity != expected_identity:
            raise RuntimeError("Wake rules target identity drift after plan")
        self._archive.parent.mkdir(parents=True, exist_ok=True)
        if self._archive.exists():
            if self._archive.read_bytes() != fact.opaque:
                raise RuntimeError("Wake rules archive content conflict")
        else:
            _write_atomic(self._archive, fact.opaque)
        payload = self._receipt_payload(fact)
        receipt = TargetReceipt(
            receipt_id=str(payload["receipt_id"]),
            receipt_digest=receipt_digest(payload),
            target_identity=expected_identity,
        )
        if self._receipt.exists():
            existing = self._read_receipt()
            if existing != receipt:
                raise RuntimeError("Wake rules archive receipt conflict")
        else:
            _write_atomic(
                self._receipt,
                (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8"),
            )
        return receipt

    def verify(self, fact: LegacyFact, receipt: TargetReceipt) -> bool:
        """Verify exact archive bytes and the durable receipt file read-only."""

        self._require_fact(fact)
        if not self._archive.is_file() or not self._receipt.is_file():
            return False
        if hashlib.sha256(self._archive.read_bytes()).hexdigest() != fact.source_digest:
            return False
        expected = self._receipt_payload(fact)
        return (
            receipt
            == TargetReceipt(
                receipt_id=str(expected["receipt_id"]),
                receipt_digest=receipt_digest(expected),
                target_identity=f"wake-rules:{fact.source_digest}",
            )
            and self._read_receipt() == receipt
        )

    @staticmethod
    def _require_fact(fact: LegacyFact) -> None:
        if fact.kind is not LegacyFactKind.WAKE_RULES:
            raise TypeError("Wake rules adapter received another fact kind")
        if hashlib.sha256(fact.opaque).hexdigest() != fact.source_digest:
            raise RuntimeError("Wake rules source digest mismatch")

    @staticmethod
    def _receipt_payload(fact: LegacyFact) -> dict[str, object]:
        return {
            "schema_version": 1,
            "receipt_id": f"wake-rules-archive:{fact.source_digest}",
            "archive": "PROACTIVE_CONTEXT.md",
            "archive_sha256": fact.source_digest,
            "source_locator": fact.locator,
        }

    def _read_receipt(self) -> TargetReceipt:
        decoded = json.loads(self._receipt.read_text(encoding="utf-8"))
        if not isinstance(decoded, dict):
            raise RuntimeError("Wake rules archive receipt must be an object")
        raw = cast(dict[str, object], decoded)
        return TargetReceipt(
            receipt_id=str(raw.get("receipt_id") or ""),
            receipt_digest=receipt_digest(raw),
            target_identity=f"wake-rules:{raw.get('archive_sha256') or ''}",
        )


def _write_atomic(path: Path, content: bytes) -> None:
    """Fsync bytes, atomically publish them, then fsync the owner directory."""

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


__all__ = ["WakeRulesArchiveAdapter"]
