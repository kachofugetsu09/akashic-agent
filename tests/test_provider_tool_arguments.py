from __future__ import annotations

import json

import pytest

from agent.provider import _parse_tool_arguments


def test_parse_tool_arguments_keeps_valid_json_object() -> None:
    assert _parse_tool_arguments('{"query":"维生素 B"}') == {"query": "维生素 B"}


def test_parse_tool_arguments_repairs_unterminated_json_object(caplog) -> None:
    assert _parse_tool_arguments('{"command":"echo ok') == {"command": "echo ok"}
    assert "repaired malformed tool arguments" in caplog.text


@pytest.mark.parametrize("payload", ['["not", "an", "object"]', "not json"])
def test_parse_tool_arguments_rejects_non_object_payload(payload: str) -> None:
    expected_error = TypeError if payload.startswith("[") else json.JSONDecodeError
    with pytest.raises(expected_error):
        _parse_tool_arguments(payload)
