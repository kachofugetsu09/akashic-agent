from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from agent.config_models import Config
from bootstrap import app_server
from bootstrap.workspace_lock import WorkspaceInstanceLock


@pytest.mark.asyncio
async def test_stdio_runtime_clears_stale_admissions_only_after_lock(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """确认 stdio runtime 以 workspace owner 身份清理遗留租约。"""
    observed: dict[str, bool] = {}

    def fail_after_build_request(*args, **kwargs):
        observed["clear"] = kwargs["clear_stale_session_admissions"]
        raise RuntimeError("stop after owner routing check")

    monkeypatch.setattr(app_server, "build_core_runtime", fail_after_build_request)

    with pytest.raises(RuntimeError, match="owner routing"):
        await app_server.run_stdio_app_server(cast(Config, object()), tmp_path)

    assert observed == {"clear": True}
    lock = WorkspaceInstanceLock(tmp_path)
    lock.acquire()
    lock.release()


@pytest.mark.asyncio
async def test_stdio_runtime_binds_conversation_before_plugin_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []

    class FakeCore:
        def __init__(self) -> None:
            self.loop = object()
            self.event_bus = object()
            self.session_manager = SimpleNamespace(control_store=object())

        def bind_conversation_runtime(self, runtime: object) -> None:
            assert isinstance(runtime, FakeRuntime)
            events.append("bind")

        async def start(self) -> None:
            events.append("start")

        async def stop(self) -> None:
            events.append("stop")

    class FakeRuntime:
        def __init__(self, _store: object, _execute: object) -> None:
            events.append("runtime")

        async def shutdown(self) -> None:
            events.append("runtime.stop")

    class FakeService:
        def __init__(
            self, _runtime: object, _manager: object, _workspace: Path
        ) -> None:
            events.append("service")

        async def shutdown(self) -> None:
            events.append("service.stop")

    class FakeServer:
        def __init__(self, _service: object, *, max_message_bytes: int) -> None:
            assert max_message_bytes == 1024

        async def run(self) -> None:
            events.append("serve")

    core = FakeCore()
    config = SimpleNamespace(app_server=SimpleNamespace(max_message_bytes=1024))
    monkeypatch.setattr(app_server, "build_core_runtime", lambda *_a, **_kw: core)
    monkeypatch.setattr(app_server, "ConversationRuntime", FakeRuntime)
    monkeypatch.setattr(app_server, "ControlService", FakeService)
    monkeypatch.setattr(app_server, "StdioAppServer", FakeServer)

    await app_server.run_stdio_app_server(cast(Config, config), tmp_path)

    assert events[:5] == ["runtime", "bind", "start", "service", "serve"]
