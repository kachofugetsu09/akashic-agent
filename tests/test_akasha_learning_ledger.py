from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from plugins.akasha.ledger import list_skipped, list_turns, read_overview, read_turn


def _memory(path: Path, *, turns: int, skipped: int = 0) -> Path:
    """写一个最小学习图：每个节点一行 turn / event / recall run。"""

    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(
            """
            CREATE TABLE turn_nodes (
                node_id INTEGER PRIMARY KEY, turn_id TEXT NOT NULL, session_key TEXT NOT NULL,
                user_seq INTEGER NOT NULL, user_message_id TEXT NOT NULL,
                assistant_message_id TEXT NOT NULL, started_at TEXT NOT NULL,
                committed_at TEXT NOT NULL, inter_gap_seconds REAL
            );
            CREATE TABLE memory_events (
                event_id INTEGER PRIMARY KEY, current_turn_node_id INTEGER NOT NULL
            );
            CREATE TABLE recall_runs (
                query_turn_node_id INTEGER PRIMARY KEY, active_basin_count INTEGER NOT NULL,
                sharp_completion_count INTEGER NOT NULL, basin_direct_count INTEGER NOT NULL,
                basin_completion_count INTEGER NOT NULL, relative_tail_count INTEGER NOT NULL,
                pushes INTEGER NOT NULL, residual_l1 REAL NOT NULL
            );
            CREATE TABLE recall_items (
                query_turn_node_id INTEGER NOT NULL, candidate_turn_node_id INTEGER NOT NULL,
                rank INTEGER NOT NULL, score REAL NOT NULL, sources_json TEXT NOT NULL,
                basin_ids_json TEXT NOT NULL, is_pattern_only INTEGER NOT NULL,
                PRIMARY KEY (query_turn_node_id, candidate_turn_node_id)
            );
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """
        )
        for node in range(turns):
            session = "s" if node % 2 == 0 else "t"
            connection.execute(
                "INSERT INTO turn_nodes VALUES (?,?,?,?,?,?,?,?,?)",
                (node, f"{session}:{node}::{session}:{node}a", session, node,
                 f"{session}:{node}", f"{session}:{node}a",
                 f"2026-01-01T00:00:{node:02d}+00:00", f"2026-01-01T00:00:{node:02d}+00:00", 1.5),
            )
            connection.execute("INSERT INTO memory_events VALUES (?,?)", (node, node))
            connection.execute(
                "INSERT INTO recall_runs VALUES (?,?,?,?,?,?,?,?)",
                (node, 2, 3, 1, 4, 5, 10 * node, 1e-7),
            )
            for rank in range(2):
                connection.execute(
                    "INSERT INTO recall_items VALUES (?,?,?,?,?,?,?)",
                    (node, node * 10 + rank, rank, 1.0, "[]", "[]", 0),
                )
        state = {
            "version": 2,
            "cutover_heads": [],
            "applied": [],
            "skipped": [
                {
                    "session_id": "s", "ending": [index * 2 + 1, f"s:skip{index}"],
                    "reason": "missing-embedding",
                }
                for index in range(skipped)
            ],
        }
        connection.execute(
            "INSERT INTO metadata VALUES ('consumer_state_json', ?)",
            (json.dumps(state),),
        )
        connection.commit()
    return path


def test_overview_reports_learned_turns_and_skips(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "akasha.db", turns=6, skipped=2)
    overview = read_overview(memory)
    assert overview["learned"] == 6
    assert overview["sessions"] == 2
    assert overview["first_learned_at"] == "2026-01-01T00:00:00+00:00"
    assert overview["last_learned_at"] == "2026-01-01T00:00:05+00:00"
    assert overview["skipped"] == 2
    assert overview["skipped_reasons"] == {"missing-embedding": 2}


def test_list_turns_pages_newest_first_and_filters_by_session(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "akasha.db", turns=6)
    page = list_turns(memory, page=1, page_size=4)
    assert page["total"] == 6
    assert [row["node_id"] for row in page["items"]] == [5, 4, 3, 2]
    assert page["items"][0]["candidate_count"] == 2
    assert page["items"][0]["pushes"] == 50

    filtered = list_turns(memory, session_key="s", page=1, page_size=10)
    assert filtered["total"] == 3
    assert {row["session_key"] for row in filtered["items"]} == {"s"}


def test_list_turns_rejects_invalid_paging(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "akasha.db", turns=1)
    with pytest.raises(ValueError, match="分页参数无效"):
        _ = list_turns(memory, page=0)
    with pytest.raises(ValueError, match="分页参数无效"):
        _ = list_turns(memory, page_size=1000)


def test_read_turn_returns_none_for_missing_node(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "akasha.db", turns=2)
    assert read_turn(memory, 1) is not None
    assert read_turn(memory, 99) is None


def test_missing_memory_database_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="学习图不存在"):
        _ = read_overview(tmp_path / "absent.db")


def test_skipped_entries_expose_reason_without_touching_the_graph(tmp_path: Path) -> None:
    memory = _memory(tmp_path / "akasha.db", turns=3, skipped=2)
    skipped = list_skipped(memory)
    assert [item["reason"] for item in skipped] == ["missing-embedding"] * 2
    assert [item["seq"] for item in skipped] == [1, 3]
    assert read_overview(memory)["learned"] == 3
