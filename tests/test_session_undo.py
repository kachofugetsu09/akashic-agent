from __future__ import annotations

import asyncio
from pathlib import Path

from plugins.plugin_undo.plugin import _find_last_passive_turn, _undo_last_turn
from session.manager import SessionManager


_FRAME = '<system-reminder data-system-context-frame="true">内部</system-reminder>'


def _run(coro):
    return asyncio.run(coro)


def _add_turn(session, index: int, *, standalone_frame: bool = True) -> None:
    if standalone_frame:
        session.add_message("user", _FRAME)
    session.add_message("user", f"u{index}", llm_context_frame=None if standalone_frame else _FRAME)
    session.add_message("assistant", f"a{index}")


def _saved_manager(tmp_path: Path, *, turns: int, standalone_frame: bool = True) -> SessionManager:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("cli:1")
    for index in range(turns):
        _add_turn(session, index, standalone_frame=standalone_frame)
    manager.save(session)
    return manager


def _last_turn_message_ids(manager: SessionManager) -> list[str]:
    session = manager.get_or_create("cli:1")
    target = _find_last_passive_turn(session.messages)
    assert target is not None
    delete_indices, _, _ = target
    return [str(session.messages[i]["id"]) for i in delete_indices]


def test_undo_deletes_context_user_assistant_three_rows(tmp_path: Path):
    manager = _saved_manager(tmp_path, turns=2)
    session = manager.get_or_create("cli:1")
    session.last_consolidated = 6
    manager.save(session)
    message_ids = _last_turn_message_ids(manager)
    assert message_ids == ["cli:1:3", "cli:1:4", "cli:1:5"]

    result = _run(_undo_last_turn(manager, "cli:1", expected_message_ids=message_ids))

    assert result is not None
    assert result.deleted_ids == ["cli:1:3", "cli:1:4", "cli:1:5"]
    session = manager.get_or_create("cli:1")
    assert [m["content"] for m in session.messages] == [_FRAME, "u0", "a0"]
    assert session.last_consolidated == 3


def test_undo_uses_rollback_source_ids_for_consolidated_window_start(tmp_path: Path):
    manager = _saved_manager(tmp_path, turns=2)
    session = manager.get_or_create("cli:1")
    session.last_consolidated = 6
    manager.save(session)
    message_ids = _last_turn_message_ids(manager)

    result = _run(
        _undo_last_turn(
            manager,
            "cli:1",
            rollback_source_ids=["cli:1:0", "cli:1:1", "cli:1:2", "cli:1:3", "cli:1:4", "cli:1:5"],
            expected_message_ids=message_ids,
        )
    )

    assert result is not None
    assert result.last_consolidated_before == 6
    assert result.last_consolidated_after == 0


def test_undo_keeps_cursor_when_target_is_after_consolidated_prefix(tmp_path: Path):
    manager = _saved_manager(tmp_path, turns=3)
    session = manager.get_or_create("cli:1")
    session.last_consolidated = 6
    manager.save(session)
    message_ids = _last_turn_message_ids(manager)

    result = _run(_undo_last_turn(manager, "cli:1", expected_message_ids=message_ids))

    assert result is not None
    assert result.deleted_ids == ["cli:1:6", "cli:1:7", "cli:1:8"]
    assert manager.get_or_create("cli:1").last_consolidated == 6


def test_undo_deletes_user_assistant_when_frame_is_extra(tmp_path: Path):
    manager = _saved_manager(tmp_path, turns=1, standalone_frame=False)
    message_ids = _last_turn_message_ids(manager)
    assert message_ids == ["cli:1:0", "cli:1:1"]

    result = _run(_undo_last_turn(manager, "cli:1", expected_message_ids=message_ids))

    assert result is not None
    assert result.deleted_ids == ["cli:1:0", "cli:1:1"]
    assert manager.get_or_create("cli:1").messages == []


def test_undo_does_not_reuse_deleted_tail_message_ids(tmp_path: Path):
    manager = _saved_manager(tmp_path, turns=1)
    message_ids = _last_turn_message_ids(manager)

    result = _run(_undo_last_turn(manager, "cli:1", expected_message_ids=message_ids))
    assert result is not None
    assert result.deleted_ids == ["cli:1:0", "cli:1:1", "cli:1:2"]

    session = manager.get_or_create("cli:1")
    _add_turn(session, 2, standalone_frame=False)
    manager.save(session)

    assert [m["id"] for m in session.messages] == ["cli:1:3", "cli:1:4"]
