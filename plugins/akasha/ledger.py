"""只读列出已学习的逻辑 turn。

学习图是权威：每个节点恰好对应一个已学习的逻辑 turn。正文不在这里复制，
调用者按 message_id 从 canonical Message 读取。
"""
from __future__ import annotations

import json
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

_TURN_COLUMNS = (
    "turn.node_id",
    "turn.turn_id",
    "turn.session_key",
    "turn.user_seq",
    "turn.user_message_id",
    "turn.assistant_message_id",
    "turn.started_at",
    "turn.committed_at",
    "turn.inter_gap_seconds",
)

_RUN_COLUMNS = (
    "run.active_basin_count",
    "run.sharp_completion_count",
    "run.basin_completion_count",
    "run.relative_tail_count",
    "run.pushes",
    "run.residual_l1",
    "(SELECT COUNT(*) FROM recall_items AS item "
    "WHERE item.query_turn_node_id = turn.node_id) AS candidate_count",
)

_SELECT = f"""
    SELECT {', '.join(_TURN_COLUMNS + _RUN_COLUMNS)}
    FROM memory_events AS event
    JOIN turn_nodes AS turn ON turn.node_id = event.current_turn_node_id
    LEFT JOIN recall_runs AS run ON run.query_turn_node_id = turn.node_id
"""

_FIELDS = (
    "node_id", "turn_id", "session_key", "user_seq", "user_message_id",
    "assistant_message_id", "started_at", "committed_at", "inter_gap_seconds",
    "active_basin_count", "sharp_completion_count", "basin_completion_count",
    "relative_tail_count", "pushes", "residual_l1", "candidate_count",
)


def _connect(memory_path: Path) -> sqlite3.Connection:
    if not memory_path.is_file():
        raise ValueError(f"Akasha 学习图不存在: {memory_path}")
    connection = sqlite3.connect(f"file:{memory_path}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _row(row: sqlite3.Row) -> dict[str, object]:
    return {name: row[name] for name in _FIELDS}


def _consumption(memory_path: Path) -> dict[str, object]:
    with closing(_connect(memory_path)) as connection:
        row = connection.execute(
            "SELECT value FROM metadata WHERE key='consumer_state_json'"
        ).fetchone()
    if row is None:
        return {}
    payload = json.loads(str(row[0]))
    return payload if isinstance(payload, dict) else {}


def read_overview(memory_path: Path) -> dict[str, object]:
    """给出学习账本的总量与跳过分类。"""

    with closing(_connect(memory_path)) as connection:
        learned, sessions, first_at, last_at = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT session_key), MIN(started_at), MAX(started_at) "
            "FROM turn_nodes"
        ).fetchone()
    state = _consumption(memory_path)
    skipped = state.get("skipped", [])
    reasoned = Counter(
        str(item.get("reason", ""))
        for item in skipped
        if isinstance(item, dict)
    )
    return {
        "learned": int(learned or 0),
        "sessions": int(sessions or 0),
        "first_learned_at": first_at,
        "last_learned_at": last_at,
        "skipped": len(skipped) if isinstance(skipped, list) else 0,
        "skipped_reasons": dict(sorted(reasoned.items())),
    }


def list_turns(
    memory_path: Path, *, session_key: str = "", page: int = 1, page_size: int = 50,
) -> dict[str, object]:
    """按节点倒序分页；只做等值过滤，不做全表文本搜索。"""

    if page < 1 or not 1 <= page_size <= 200:
        raise ValueError("Akasha 账本分页参数无效")
    where = "WHERE turn.session_key = ?" if session_key else ""
    values: tuple[object, ...] = (session_key,) if session_key else ()
    with closing(_connect(memory_path)) as connection:
        total = int(connection.execute(
            f"SELECT COUNT(*) FROM turn_nodes AS turn {where}", values,
        ).fetchone()[0])
        rows = connection.execute(
            f"{_SELECT} {where} ORDER BY turn.node_id DESC LIMIT ? OFFSET ?",
            (*values, page_size, (page - 1) * page_size),
        ).fetchall()
    return {
        "items": [_row(row) for row in rows],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


def read_turn(memory_path: Path, node_id: int) -> dict[str, object] | None:
    """读一个节点的学习事实；正文由调用者按 message_id 提供。"""

    with closing(_connect(memory_path)) as connection:
        row = connection.execute(
            f"{_SELECT} WHERE turn.node_id = ?", (node_id,),
        ).fetchone()
    return None if row is None else _row(row)


def list_skipped(memory_path: Path) -> list[dict[str, object]]:
    """列出被明确跳过的闭段；它们不在学习图里。"""

    skipped = _consumption(memory_path).get("skipped", [])
    result: list[dict[str, object]] = []
    for item in skipped if isinstance(skipped, list) else []:
        if not isinstance(item, dict):
            continue
        ending = item.get("ending")
        seq = ending[0] if isinstance(ending, list) and ending else None
        message_id = ending[1] if isinstance(ending, list) and len(ending) > 1 else ""
        result.append({
            "session_key": str(item.get("session_id", "")),
            "seq": seq,
            "message_id": str(message_id),
            "reason": str(item.get("reason", "")),
        })
    return result
