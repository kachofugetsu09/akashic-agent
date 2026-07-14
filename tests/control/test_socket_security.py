from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any, cast

import pytest

from agent.control.client import ControlClient, RemoteControlError
from agent.control.models import TurnRequest
from agent.control.runtime import ConversationRuntime
from agent.control.service import ControlService
from bootstrap.workspace_token import ensure_workspace_token
from infra.control.socket import SocketAppServer
from session.manager import SessionManager


async def _echo(request: TurnRequest) -> str:
    return request.input


@pytest.mark.asyncio
async def test_loopback_tcp_requires_workspace_token(tmp_path: Path) -> None:
    sessions = SessionManager(tmp_path)
    runtime = ConversationRuntime(sessions.control_store, _echo)
    token = ensure_workspace_token(tmp_path)
    service = ControlService(runtime, sessions, tmp_path, workspace_token=token)
    server = SocketAppServer("127.0.0.1:0", service)
    await server.start()
    endpoint = str(server.endpoint)
    try:
        with pytest.raises(RemoteControlError) as captured:
            _ = await ControlClient.connect(endpoint, workspace_token="wrong")
        assert captured.value.code == -32004
        async with await ControlClient.connect(endpoint, workspace_token=token) as client:
            status = await client.request("server/status", {})
            assert isinstance(status, dict)
            assert status["ready"] is True
    finally:
        await server.stop()
        await runtime.shutdown()
        sessions.close()


def test_tcp_rejects_non_loopback_and_token_is_private(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="只允许 loopback"):
        _ = SocketAppServer("0.0.0.0:2236", cast(Any, object()))
    _ = ensure_workspace_token(tmp_path)
    if os.name != "nt":
        assert stat.S_IMODE((tmp_path / ".app-server-token").stat().st_mode) == 0o600
