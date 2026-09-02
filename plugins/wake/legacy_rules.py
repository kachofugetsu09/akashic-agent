"""Read the versioned Wake-private legacy rules archive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

RULES_DIRECTORY = "legacy-rules"
RULES_ARCHIVE = "PROACTIVE_CONTEXT.md"
RULES_RECEIPT = "receipt.json"


def read_archived_rules(data_root: Path) -> str | None:
    """Read and verify the versioned Wake-private rules archive without writing."""

    archive = data_root / RULES_DIRECTORY / RULES_ARCHIVE
    receipt_path = data_root / RULES_DIRECTORY / RULES_RECEIPT
    if not archive.exists() and not receipt_path.exists():
        return None
    if not archive.is_file() or not receipt_path.is_file():
        raise RuntimeError("Wake rules archive and receipt must exist together")

    # 1. Validate the target-owned receipt schema and archive identity.
    decoded = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise RuntimeError("Wake rules archive receipt must be an object")
    receipt = cast(dict[str, object], decoded)
    if receipt.get("schema_version") != 1:
        raise RuntimeError("unsupported Wake rules archive receipt")
    if receipt.get("archive") != RULES_ARCHIVE:
        raise RuntimeError("Wake rules archive receipt names another file")
    expected_digest = receipt.get("archive_sha256")
    if not isinstance(expected_digest, str) or not expected_digest:
        raise RuntimeError("Wake rules archive receipt lacks archive digest")

    # 2. Return the same stripped UTF-8 text consumed by the legacy runtime.
    content = archive.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_digest:
        raise RuntimeError("Wake rules archive digest mismatch")
    return content.decode("utf-8").strip()


__all__ = ["read_archived_rules"]
