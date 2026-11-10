#!/usr/bin/env python3
"""对比重放前后的 Akasha 学习图；报告共同前缀长度和第一处差异。

重放只改变被排除/缺失的尾部，因此理论上历史前缀应当逐节点保持一致。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

_COLUMNS = (
    "node_id",
    "turn_id",
    "session_key",
    "user_seq",
    "user_message_id",
    "assistant_message_id",
    "started_at",
    "committed_at",
    "inter_gap_seconds",
)


def _nodes(path: Path) -> list[tuple[object, ...]]:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
        connection.execute("PRAGMA query_only = ON")
        return [
            tuple(row)
            for row in connection.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM turn_nodes ORDER BY node_id"
            )
        ]


def _applied(path: Path) -> list[dict[str, object]]:
    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True)) as connection:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='consumer_state_json'"
        ).fetchone()
    if row is None:
        return []
    payload = json.loads(str(row[0]))
    if not isinstance(payload, dict):
        return []
    applied = payload.get("applied")
    return [item for item in applied if isinstance(item, dict)] if isinstance(applied, list) else []


def _common_prefix(before: list[tuple[object, ...]], after: list[tuple[object, ...]]) -> int:
    limit = min(len(before), len(after))
    index = 0
    while index < limit and before[index] == after[index]:
        index += 1
    return index


def compare(before_path: Path, after_path: Path) -> dict[str, object]:
    """返回共同前缀、第一处差异和两端节点数量。"""

    before, after = _nodes(before_path), _nodes(after_path)
    prefix = _common_prefix(before, after)
    before_applied, after_applied = _applied(before_path), _applied(after_path)
    applied_prefix = _common_prefix(
        [tuple(sorted(item.items())) for item in before_applied],
        [tuple(sorted(item.items())) for item in after_applied],
    )
    first_difference: dict[str, object] | None = None
    if prefix < min(len(before), len(after)):
        first_difference = {
            "node_id": prefix,
            "before": dict(zip(_COLUMNS, before[prefix])),
            "after": dict(zip(_COLUMNS, after[prefix])),
        }
    return {
        "before": {"path": str(before_path), "turns": len(before), "applied": len(before_applied)},
        "after": {"path": str(after_path), "turns": len(after), "applied": len(after_applied)},
        "common_prefix_turns": prefix,
        "common_prefix_applied": applied_prefix,
        "first_difference": first_difference,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="对比重放前后的 Akasha 学习图")
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument(
        "--expect-prefix",
        type=int,
        default=None,
        help="要求共同前缀至少达到该节点数；不足即失败。",
    )
    parser.add_argument("--json", action="store_true", help="以 JSON 输出。")
    arguments = parser.parse_args()

    result = compare(arguments.before, arguments.after)
    if arguments.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            f"before={result['before']['turns']} turns  after={result['after']['turns']} turns  "
            f"common_prefix={result['common_prefix_turns']}"
        )
        print(f"applied: before={result['before']['applied']} after={result['after']['applied']} "
              f"common={result['common_prefix_applied']}")
        if result["first_difference"] is not None:
            print("first_difference:")
            print(json.dumps(result["first_difference"], ensure_ascii=False, indent=2))

    if arguments.expect_prefix is not None:
        actual = int(result["common_prefix_turns"])  # type: ignore[arg-type]
        if actual < arguments.expect_prefix:
            print(
                f"共同前缀不足: {actual} < {arguments.expect_prefix}",
                file=sys.stderr,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
