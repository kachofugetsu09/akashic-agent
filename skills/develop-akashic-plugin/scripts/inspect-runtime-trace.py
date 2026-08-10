#!/usr/bin/env python3
"""只读导出一个 programmatic turn 及其插件 reload 证据。"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


def _open_readonly(path: Path) -> sqlite3.Connection:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise FileNotFoundError(f"不是 SQLite 文件: {resolved}")
    connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _decode_json(raw: object, *, field: str) -> object:
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TypeError(f"{field} 必须是 JSON 字符串或 NULL")
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field} 包含无效 JSON") from exc


def _read_turn(connection: sqlite3.Connection, turn_id: str) -> dict[str, Any]:
    row = connection.execute("SELECT * FROM turns WHERE id = ?", (turn_id,)).fetchone()
    if row is None:
        raise LookupError(f"turn 不存在: {turn_id}")
    turn = dict(row)
    for source, target in (
        ("input_json", "input"),
        ("items_json", "items"),
        ("usage_json", "usage"),
        ("error_json", "error"),
    ):
        turn[target] = _decode_json(turn.pop(source), field=f"turns.{source}")
    return turn


def _read_session(
    connection: sqlite3.Connection,
    session_key: str,
) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT * FROM sessions WHERE key = ?",
        (session_key,),
    ).fetchone()
    if row is None:
        return None
    session = dict(row)
    session["metadata"] = _decode_json(
        session.get("metadata"),
        field="sessions.metadata",
    )
    return session


def _read_messages(
    connection: sqlite3.Connection,
    session_key: str,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT id, session_key, seq, role, content, tool_chain, extra, ts
        FROM messages
        WHERE session_key = ?
        ORDER BY seq
        """,
        (session_key,),
    ).fetchall()
    messages: list[dict[str, Any]] = []
    for row in rows:
        message = dict(row)
        message["tool_chain"] = _decode_json(
            message["tool_chain"],
            field=f"messages[{message['id']}].tool_chain",
        )
        message["extra"] = _decode_json(
            message["extra"],
            field=f"messages[{message['id']}].extra",
        )
        messages.append(message)
    return messages


def _read_reload(
    connection: sqlite3.Connection,
    plugin_id: str,
) -> dict[str, Any] | None:
    row = connection.execute(
        """
        SELECT *
        FROM reload_transactions
        WHERE plugin_id = ?
        ORDER BY started_at DESC
        LIMIT 1
        """,
        (plugin_id,),
    ).fetchone()
    if row is None:
        return None
    transaction = dict(row)
    events = connection.execute(
        """
        SELECT sequence, phase, details_json, created_at
        FROM reload_events
        WHERE tx_id = ?
        ORDER BY sequence
        """,
        (transaction["tx_id"],),
    ).fetchall()
    decoded_events: list[dict[str, Any]] = []
    for event in events:
        decoded_event: dict[str, Any] = dict(event)
        decoded_event["details"] = _decode_json(
            event["details_json"],
            field=f"reload_events[{event['sequence']}].details_json",
        )
        del decoded_event["details_json"]
        decoded_events.append(decoded_event)
    transaction["events"] = decoded_events
    return transaction


def _value_summary(value: object) -> dict[str, object]:
    """Describe stored content without exposing its values."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True)
    summary: dict[str, object] = {
        "type": type(value).__name__,
        "chars": len(encoded),
    }
    if isinstance(value, dict):
        summary["keys"] = sorted(str(key) for key in value)
    elif isinstance(value, list):
        summary["items"] = len(value)
    return summary


def _redact_report(report: dict[str, Any]) -> dict[str, Any]:
    """Keep diagnostic identity and timing while replacing content with shapes."""

    # 1. Preserve turn lifecycle fields and summarize model-visible values.
    turn = dict(report["turn"])
    for field in ("input", "items", "final_response", "error"):
        turn[f"{field}_summary"] = _value_summary(turn.pop(field, None))
    report["turn"] = turn

    # 2. Preserve session/message ordering without printing stored content.
    session = report.get("session")
    if isinstance(session, dict):
        session = dict(session)
        session["metadata_summary"] = _value_summary(session.pop("metadata", None))
        report["session"] = session
    messages: list[dict[str, Any]] = []
    for stored in report["messages"]:
        message = dict(stored)
        for field in ("content", "tool_chain", "extra"):
            message[f"{field}_summary"] = _value_summary(message.pop(field, None))
        messages.append(message)
    report["messages"] = messages

    # 3. Keep reload phases but summarize event details that may include paths/errors.
    reload = report.get("plugin_reload")
    if isinstance(reload, dict):
        reload = dict(reload)
        if "error" in reload:
            reload["error_summary"] = _value_summary(reload.pop("error"))
        events = []
        for stored in reload.get("events", []):
            event = dict(stored)
            event["details_summary"] = _value_summary(event.pop("details", None))
            events.append(event)
        reload["events"] = events
        report["plugin_reload"] = reload
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读导出 programmatic turn、SessionDB 消息与插件 reload 轨迹",
    )
    _ = parser.add_argument("--workspace", type=Path, required=True)
    _ = parser.add_argument("--turn-id", required=True)
    _ = parser.add_argument("--plugin-id")
    _ = parser.add_argument(
        "--include-content",
        action="store_true",
        help="输出完整会话正文、工具参数和 metadata；只用于可信本机终端",
    )
    return parser.parse_args()


def main() -> int:
    """读取权威诊断库并输出一份结构化报告。"""

    args = _parse_args()
    workspace = args.workspace.expanduser().resolve(strict=True)

    # 1. 从 SessionDB 读取目标 turn、所属 session 和完整消息轨迹。
    with closing(_open_readonly(workspace / "sessions.db")) as sessions:
        turn = _read_turn(sessions, args.turn_id)
        session_key = str(turn["session_key"])
        report: dict[str, Any] = {
            "workspace": str(workspace),
            "turn": turn,
            "session": _read_session(sessions, session_key),
            "messages": _read_messages(sessions, session_key),
        }

    # 2. 指定 plugin 时读取最新 reload transaction 及其完整事件链。
    if args.plugin_id:
        reload_path = workspace / "runtime" / "plugin-reloads.sqlite3"
        with closing(_open_readonly(reload_path)) as reloads:
            report["plugin_reload"] = _read_reload(reloads, args.plugin_id)

    # 3. 默认输出脱敏形状；完整内容必须由调用者显式选择。
    if not args.include_content:
        report = _redact_report(report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
