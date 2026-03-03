from pathlib import Path
from unittest.mock import MagicMock

from agent.loop import AgentLoop


def _make_loop(tmp_path: Path) -> AgentLoop:
    return AgentLoop(
        bus=MagicMock(),
        provider=MagicMock(),
        tools=MagicMock(),
        session_manager=MagicMock(),
        workspace=tmp_path,
    )


def test_collect_skill_mentions_returns_unique_existing_names(tmp_path):
    loop = _make_loop(tmp_path)
    loop.context.skills.list_skills = MagicMock(
        return_value=[
            {"name": "feed-manage"},
            {"name": "refactor"},
        ]
    )

    got = loop._collect_skill_mentions("请用 $feed-manage 然后 $refactor 再来一次 $feed-manage")

    assert got == ["feed-manage", "refactor"]


def test_collect_skill_mentions_ignores_unknown_skill(tmp_path):
    loop = _make_loop(tmp_path)
    loop.context.skills.list_skills = MagicMock(return_value=[{"name": "known"}])

    got = loop._collect_skill_mentions("$known $unknown")

    assert got == ["known"]


def test_format_request_time_anchor_contains_iso_and_label():
    text = AgentLoop._format_request_time_anchor(None)
    assert text.startswith("request_time=")
    assert "(" in text and ")" in text
