"""Selected live Models code uses saved settings, credentials, and call records."""

from __future__ import annotations

from collections.abc import Mapping
import shutil
import socket
import sqlite3
from pathlib import Path

import pytest
from aiohttp import web

from agent.config_models import Config
from agent.plugin_composition import CHAT_MODELS, ModelRequest, ModelUnavailableError
from bootstrap import tools as bootstrap
from bootstrap.init_workspace import init_workspace
from core.net.http import SharedHttpResources
from plugins.models.settings import MODEL_SETTINGS
from plugins.models.store import ModelsStore
from tests.test_model_execution import _model_command, _mount_model_driver_graph


def _change_selected_code(source: Path, target: str) -> None:
    """Put an observable marker in the next selected local plugin artifact."""

    if target == "models":
        module = source / "models/state.py"
        old = "            call_id = self._store.resume_call("
        new = (
            '            request = replace(request, messages=(*request.messages, '
            '{"role": "user", "content": "selected models"}))\n'
            + old
        )
    else:
        module = source / "openai_compatible/driver.py"
        old = "        body = _chat_body(self._descriptor, connection, self._config, request)"
        new = (
            "        assert self.max_tool_schemas == 4\n"
            "        async with self._credential.exclusive():\n"
            '            await self._credential.refresh({"api_key": "refreshed-key"})\n'
            + old
        )
    text = module.read_text(encoding="utf-8")
    assert text.count(old) == 1
    updated = text.replace(old, new)
    if target == "openai_compatible":
        header = 'headers={"Authorization": f"Bearer {token}"}'
        assert header in updated
        updated = updated.replace(
            header,
            'headers={"Authorization": f"Bearer {token}", "X-Selected-Driver": "live"}',
        )
    module.write_text(updated, encoding="utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target,configured_default",
    [("models", True), ("openai_compatible", True), ("models", False)],
)
async def test_selected_code_uses_existing_settings_and_live_call_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    target: str, configured_default: bool,
) -> None:
    """A local update keeps registry facts and runs only the selected code."""

    requests: list[tuple[str, str | None, dict[str, object]]] = []

    async def models(_request: web.Request) -> web.Response:
        return web.json_response({"data": [{"id": "default-wire"}, {"id": "agent-wire"}]})

    async def completions(request: web.Request) -> web.Response:
        body = await request.json()
        assert isinstance(body, dict)
        requests.append((request.headers["Authorization"], request.headers.get("X-Selected-Driver"), body))
        return web.json_response({
            "choices": [{"message": {"content": "live result"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        })

    provider = web.Application()
    provider.router.add_get("/v1/models", models)
    provider.router.add_post("/v1/chat/completions", completions)
    runner = web.AppRunner(provider)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    await web.SockSite(runner, sock).start()

    workspace = tmp_path / "workspace"
    init_workspace(config_path=tmp_path / "config.toml", workspace=workspace)
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    sources = tmp_path / "plugins"
    for name in ("ui", "models", "openai_compatible"):
        shutil.copytree(
            Path(__file__).parents[1] / "plugins" / name,
            sources / name,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(), workspace, http, plugin_dirs=[sources])
    try:
        await core.start()
        await core.plugin_manager.start_runtime()
        await _model_command(core, {
            "type": "add_connection", "expected_revision": 0,
            "connection_id": "existing", "name": "Existing",
            "driver_id": "openai-compatible", "endpoint": f"http://127.0.0.1:{port}/v1",
            "auth_identity": "existing-account", "credential": {"api_key": "existing-key"},
            "driver_config": {"max_retries": 0, "read_timeout": 7},
        })
        for revision, model_id in ((1, "default-wire"), (2, "agent-wire")):
            await _model_command(core, {
                "type": "add_model", "expected_revision": revision,
                "model_id": model_id, "connection_id": "existing",
                "kind": "chat", "model": model_id,
                "capabilities": {
                    "context_window": 32000, "max_output_tokens": 4096,
                    "supports_tool_calls": True,
                    "supported_reasoning_efforts": ["high"],
                },
                "capability_sources": {}, "default_reasoning_effort": "high",
                "driver_config": {"max_tool_schemas": 4},
            })
        roles = (
            (("default", "default-wire"), ("agent", "agent-wire"))
            if configured_default else (("agent", "agent-wire"),)
        )
        for revision, (role, model_id) in enumerate(roles, 3):
            await _model_command(core, {
                "type": "set_default", "expected_revision": revision,
                "role": role, "model_id": model_id,
            })

        store = ModelsStore(workspace / "model-registry.sqlite3", tmp_path / "backups")
        before = store.read_snapshot()
        assert before is not None
        old_root = core.plugin_manager.live_root
        assert old_root is not None
        selected_before = core.plugin_manager._selection.read()
        _change_selected_code(sources, target)
        changed = await core.plugin_manager.reconcile_changed()
        assert any(item.get("publication_state") == "active" for item in changed), changed
        assert core.plugin_manager.live_root is old_root
        assert core.plugin_manager._selection.read() != selected_before
        assert store.read_snapshot() == before

        root = core.plugin_manager.live_root
        assert root is not None
        models_context, chat_models = root._service_provider(CHAT_MODELS)
        async with models_context.runtime_scope():
            if not configured_default:
                with pytest.raises(ModelUnavailableError, match="default"):
                    async with chat_models.execution():
                        raise AssertionError("missing default must reject before I/O")
            else:
                async with chat_models.execution() as execution:
                    chat = execution.chat("agent")
                    assert chat.descriptor.model_id == "agent-wire"
                    response = await chat.complete(ModelRequest(
                        messages=({"role": "user", "content": "use saved settings"},),
                    ))
                assert response.content == "live result"
                assert response.call_record_id is not None
                call = store.read_call(response.call_record_id)
                assert call["state"] == "success"
                assert call["binding"]["model_id"] == "agent-wire"
                assert call["binding"]["binding_id"] == chat.descriptor.binding_id
        if not configured_default:
            assert requests == []
            assert store.read_calls("", 100) == ()
        else:
            assert len(requests) == 1
            token, marker, body = requests[0]
            assert body["model"] == "agent-wire"
            assert body["reasoning_effort"] == "high"
            if target == "models":
                assert token == "Bearer existing-key"
                assert marker is None
                messages = body["messages"]
                assert isinstance(messages, list)
                assert any(
                    item.get("content") == "selected models"
                    for item in messages if isinstance(item, Mapping)
                )
            else:
                assert token == "Bearer refreshed-key"
                assert marker == "live"
                credential = await store.credential_handle("existing", "existing-account").read()
                assert credential["api_key"] == "refreshed-key"
        assert store.read_snapshot() == before
    finally:
        try:
            await core.bus.aclose()
        finally:
            try:
                await core.stop()
            finally:
                try:
                    await http.aclose()
                finally:
                    await runner.cleanup()


@pytest.mark.asyncio
async def test_live_settings_scope_and_missing_or_corrupt_registry(tmp_path: Path) -> None:
    """The live owner rejects bad model data without replacing its saved bytes."""
    from plugins.models.settings import SetDefaultModel

    graph = await _mount_model_driver_graph(tmp_path, with_live_models=True)
    try:
        state = graph["state"]
        context = graph["models_fiber"].context
        before = state.store.read_snapshot()
        async with context.runtime_scope():
            assert context.require(MODEL_SETTINGS) is state.settings
            with pytest.raises(ValueError, match="chat model is unavailable"):
                await state.settings.apply(SetDefaultModel(0, "default", "unconfigured"))
        assert state.store.read_snapshot() == before

        missing = tmp_path / "missing-models.sqlite3"
        with pytest.raises(FileNotFoundError):
            ModelsStore(missing, tmp_path / "backups", writable=False).initialize()
        assert not missing.exists()
        corrupt = tmp_path / "corrupt-models.sqlite3"
        corrupt.write_bytes(b"not a sqlite database")
        with pytest.raises(sqlite3.DatabaseError):
            ModelsStore(corrupt, tmp_path / "backups", writable=False).read_snapshot()
        assert corrupt.read_bytes() == b"not a sqlite database"
    finally:
        graph["release_consumer"].set()
        await graph["root"].dispose()
