from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from agent.core.passive_turn import _approval_runtime_from_context
from agent.policies.tool_approval_runtime import ToolApprovalRuntime
from agent.policies.tool_approval_store import ToolApprovalStore
from agent.provider import LLMResponse, ToolCall
from tests.test_tool_access_gateway_reasoner import (
    _Provider,
    _RecordingTool,
    _make_reasoner,
    _msg,
    _session,
)


UTC = timezone.utc


def test_default_reasoner_builds_workspace_approval_runtime(tmp_path: Path) -> None:
    class Context:
        workspace = tmp_path

    runtime = _approval_runtime_from_context(Context())

    assert runtime is not None
    assert runtime.store.db_path == ToolApprovalRuntime.approval_db_path_from_workspace(
        tmp_path
    )


@pytest.mark.asyncio
async def test_reasoner_deferred_write_uses_workspace_approval_store(
    tmp_path: Path,
) -> None:
    read_file = _RecordingTool("read_file")
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall("read-call", "read_file", {"path": "notes.md"})],
            ),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    reasoner = _make_reasoner(
        provider,
        read_file=read_file,
        read_file_risk="write",
    )
    reasoner._context.workspace = tmp_path

    result = await reasoner.run_turn(
        msg=_msg("write notes.md"),
        session=cast(Any, _session()),
    )

    db_path = ToolApprovalRuntime.approval_db_path_from_workspace(tmp_path)
    store = ToolApprovalStore(db_path)
    pending = store.list_pending_requests(session_key="cli:1", now=_now())
    payload = _last_tool_payload(provider)
    assert result.reply == "final"
    assert read_file.calls == []
    assert len(pending) == 1
    assert pending[0].request_id == "read-call"
    assert pending[0].tool_name == "read_file"
    assert payload["approval_request"]["approval_request_id"] == (
        pending[0].approval_request_id
    )
    assert payload["approval_request"]["expires_at"] == pending[0].expires_at


@pytest.mark.asyncio
async def test_reasoner_untrusted_model_approval_id_does_not_resume(
    tmp_path: Path,
) -> None:
    db_path = ToolApprovalRuntime.approval_db_path_from_workspace(tmp_path)
    store = ToolApprovalStore(db_path)
    original = store.create_or_get_pending_request(
        request_id="read-call",
        session_key="cli:1",
        channel="cli",
        chat_id="1",
        source="passive",
        tool_name="read_file",
        risk="write",
        approval_scope="tool_call",
        policy_reason="risk_strategy_write_requires_approval",
        arguments={"path": "notes.md"},
        now=_now(),
        ttl=timedelta(minutes=5),
    )
    store.approve_request(
        approval_request_id=original.approval_request_id,
        request_id=original.request_id,
        session_key=original.session_key,
        tool_name=original.tool_name,
        approval_scope=original.approval_scope,
        args_hash=original.args_hash,
        actor="user",
        now=_now(),
    )
    read_file = _RecordingTool("read_file")
    provider = _Provider(
        [
            LLMResponse(
                content="",
                tool_calls=[
                    ToolCall(
                        "read-call",
                        "read_file",
                        {
                            "path": "notes.md",
                            "approval_request_id": original.approval_request_id,
                        },
                    )
                ],
            ),
            LLMResponse(content="final", tool_calls=[]),
        ]
    )
    reasoner = _make_reasoner(
        provider,
        read_file=read_file,
        read_file_risk="write",
    )
    reasoner._context.workspace = tmp_path

    await reasoner.run_turn(
        msg=_msg("write notes.md with approval id"),
        session=cast(Any, _session()),
    )

    pending = store.list_pending_requests(session_key="cli:1", now=_now())
    assert read_file.calls == []
    assert store.get_request(original.approval_request_id).status == "approved"
    assert len(pending) == 1
    assert pending[0].approval_request_id != original.approval_request_id


def _last_tool_payload(provider: _Provider) -> dict[str, Any]:
    messages = provider.calls[-1]["messages"]
    tool_messages = [
        message for message in messages if message.get("role") == "tool"
    ]
    assert tool_messages
    return json.loads(tool_messages[-1]["content"])


def _now() -> datetime:
    return datetime.now(UTC)
