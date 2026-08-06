from __future__ import annotations

import pytest

from agent.model_runtime.session_selection import (
    SessionModelSelection,
    read_session_model_selection,
    write_session_model_selection,
)


def test_structured_selection_round_trips_model_and_effort() -> None:
    metadata: dict[str, object] = {}
    write_session_model_selection(
        metadata,
        SessionModelSelection("deepseek-main", "high"),
    )

    assert read_session_model_selection(metadata) == SessionModelSelection(
        "deepseek-main",
        "high",
    )


def test_legacy_override_remains_readable_until_next_explicit_change() -> None:
    metadata: dict[str, object] = {"model_runtime_override": "legacy-main"}
    assert read_session_model_selection(metadata) == SessionModelSelection(
        "legacy-main",
        "",
    )

    write_session_model_selection(metadata, SessionModelSelection("next", "low"))
    assert "model_runtime_override" not in metadata
    assert read_session_model_selection(metadata) == SessionModelSelection(
        "next",
        "low",
    )


def test_follow_default_removes_pinned_selection() -> None:
    metadata: dict[str, object] = {}
    write_session_model_selection(metadata, SessionModelSelection("pinned", "high"))

    write_session_model_selection(metadata, SessionModelSelection())

    assert metadata == {}


def test_effort_without_explicit_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="默认模型不能单独覆盖推理强度"):
        write_session_model_selection({}, SessionModelSelection("", "high"))
