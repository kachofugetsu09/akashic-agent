#!/usr/bin/env python3
"""Plan by default or explicitly apply the legacy proactive-island handoff."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from agent.migrations.proactive_island.cli import apply, plan, report_payload, retire
from agent.migrations.proactive_island.inventory import (
    inventory_digest,
    inventory_workspace,
)
from agent.migrations.proactive_island.history import LegacyProactiveHistory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--workspace", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group()
    _ = mode.add_argument("--apply", action="store_true")
    _ = mode.add_argument("--history", action="store_true")
    _ = mode.add_argument("--retire-blocks", action="store_true")
    _ = parser.add_argument("--backup-root", type=Path)
    _ = parser.add_argument("--expected-inventory-sha256")
    args = parser.parse_args()
    if args.history:
        print(
            json.dumps(
                LegacyProactiveHistory(args.workspace).snapshot(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.retire_blocks:
        if args.backup_root is None or args.expected_inventory_sha256 is None:
            parser.error(
                "--retire-blocks requires --backup-root and "
                "--expected-inventory-sha256"
            )
        report = retire(
            args.workspace,
            args.backup_root,
            args.expected_inventory_sha256,
        )
    elif args.apply:
        if args.backup_root is None:
            parser.error("--apply requires --backup-root")
        report = apply(args.workspace, args.backup_root)
    else:
        report = plan(args.workspace)
    payload = report_payload(report)
    payload["inventory_sha256"] = inventory_digest(inventory_workspace(args.workspace))
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if report.status.value != "block" else 2


if __name__ == "__main__":
    raise SystemExit(main())
