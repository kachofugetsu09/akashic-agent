from __future__ import annotations

import ast
import asyncio
import base64
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
from contextlib import closing, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator, Mapping

import pytest

from agent.plugin_composition import (
    AddModel,
    CapabilitySources,
    CHAT_MODELS,
    FinishConnectionAuth,
    MODEL_CATALOG,
    MODEL_SETTINGS,
    ModelCapabilities,
    ModelContinuation,
    ModelKind,
    ModelAvailability,
    ModelRequest,
    ModelRole,
    ModelUnavailableError,
    SetDefaultModel,
    StartConnectionAuth,
    SyncModels,
    TransportError,
    UsageCoverage,
)
from agent.plugins.install import install_git_plugin, uninstall_plugin
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot
from bus.event_bus import EventBus


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int]) -> None:
        super().__init__(address, _Handler)
        self.requests: list[dict[str, Any]] = []
        self.refresh_count = 0
        self.response_count = 0
        self.poll_count = 0
        self.reject_barrier = threading.Barrier(2)
        self.use_reject_barrier = True
        self.refresh_status = 200
        self.catalog_default = "medium"
        self.catalog_efforts = ["medium", "high"]


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: object) -> None:
        _ = format, args

    def do_GET(self) -> None:
        self.server.requests.append(
            {
                "method": "GET",
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
            }
        )
        if self.path.startswith("/api/models"):
            self._json(
                200,
                {
                    "models": [
                        {
                            "slug": "gpt-codex-test",
                            "context_window": 120000,
                            "input_modalities": ["text", "image"],
                            "supported_reasoning_levels": [
                                {"effort": effort}
                                for effort in self.server.catalog_efforts
                            ],
                            "default_reasoning_level": self.server.catalog_default,
                            "supports_parallel_tool_calls": True,
                            "supports_reasoning_summary_parameter": True,
                        },
                        {
                            "slug": "hidden",
                            "context_window": 1,
                            "visibility": "hide",
                        },
                    ]
                },
            )
            return
        self._json(404, {"error": "missing"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) or b"{}"
        content_type = self.headers.get("Content-Type", "")
        body: object
        if "application/json" in content_type:
            body = json.loads(raw)
        else:
            body = raw.decode()
        self.server.requests.append(
            {
                "method": "POST",
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": body,
            }
        )
        if self.path == "/auth/api/accounts/deviceauth/usercode":
            self._json(
                200,
                {"device_auth_id": "device-1", "user_code": "ABCD-EFGH", "interval": 1},
            )
            return
        if self.path == "/auth/api/accounts/deviceauth/token":
            self.server.poll_count += 1
            if self.server.poll_count == 1:
                self._json(403, {"status": "pending"})
                return
            self._json(
                200,
                {"authorization_code": "code-1", "code_verifier": "verifier-1"},
            )
            return
        if self.path == "/auth/oauth/token":
            if isinstance(body, Mapping) and body.get("grant_type") == "refresh_token":
                self.server.refresh_count += 1
                if self.server.refresh_status != 200:
                    self._json(
                        self.server.refresh_status,
                        {"error": {"message": "temporary token failure"}},
                    )
                    return
                self._json(
                    200,
                    {
                        "access_token": "access-new",
                        "refresh_token": "refresh-new",
                        "expires_in": 3600,
                    },
                )
            else:
                self._json(
                    200,
                    {
                        "access_token": "access-old",
                        "refresh_token": "refresh-old",
                        "id_token": _id_token("account-1"),
                        "expires_in": 3600,
                    },
                )
            return
        if self.path != "/api/responses":
            self._json(404, {"error": "missing"})
            return
        if self.headers.get("Authorization") == "Bearer access-old":
            if self.server.use_reject_barrier:
                self.server.reject_barrier.wait(timeout=2)
            self._json(401, {"error": {"message": "expired access-old"}})
            return
        self.server.response_count += 1
        assert isinstance(body, dict)
        turn = self.server.response_count
        truncate = _input_contains(body.get("input"), "truncate")
        done_truncate = _input_contains(body.get("input"), "done-truncate")
        invalid_tool = _input_contains(body.get("input"), "invalid-tool")
        mismatch_done = _input_contains(body.get("input"), "mismatch-done")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        if truncate:
            self.send_header("Content-Length", "10000")
        self.end_headers()
        events: list[dict[str, object]] = [
            {
                "type": "response.reasoning_summary_text.delta",
                "delta": f"think-{turn}",
            },
            {
                "type": "response.output_text.delta",
                "delta": f"answer-{turn}",
            },
            {
                "type": "response.output_item.done",
                "item": {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": f"think-{turn}"}],
                    "encrypted_content": f"opaque-{turn}",
                    "id": "must-not-persist",
                },
            },
        ]
        if done_truncate:
            events = [{"type": "response.output_text.done", "text": "done-only"}]
        elif truncate:
            events = events[:1]
        if turn == 1 and not truncate:
            events.append(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "call-1",
                        "name": "lookup",
                        "arguments": '{"id":7}',
                    },
                }
            )
        if invalid_tool:
            events.append(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "invalid-call",
                        "name": "broken",
                        "arguments": "{",
                    },
                }
            )
        if mismatch_done:
            events.insert(
                1,
                {"type": "response.reasoning_summary_text.done", "text": "conflict"},
            )
        if not truncate:
            events.append(
                {
                    "type": "response.completed",
                    "response": {
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "input_tokens_details": {"cached_tokens": 3},
                            "output_tokens_details": {"reasoning_tokens": 2},
                        }
                    },
                }
            )
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
        self.wfile.flush()
        if truncate:
            self.close_connection = True

    def _json(self, status: int, payload: object) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _id_token(account_id: str) -> str:
    payload = base64.urlsafe_b64encode(
        json.dumps(
            {"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}
        ).encode()
    ).decode().rstrip("=")
    return f"header.{payload}.signature"


def _input_contains(raw: object, text: str) -> bool:
    return text in json.dumps(raw, ensure_ascii=False)


@contextmanager
def _provider() -> Iterator[tuple[_Server, str]]:
    server = _Server(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _commit(path: Path) -> None:
    for args in (
        ("init",),
        ("config", "user.name", "test"),
        ("config", "user.email", "test@example.com"),
        ("add", "."),
        ("commit", "-m", "initial"),
    ):
        result = subprocess.run(
            ("git", *args), cwd=path, capture_output=True, text=True, env=os.environ.copy()
        )
        assert result.returncode == 0, result.stderr


def _manager(tmp_path: Path) -> PluginManager:
    return PluginManager(
        plugin_dirs=[],
        event_bus=EventBus(),
        tool_registry=None,
        workspace=tmp_path / "workspace",
        installed_cache_root=tmp_path / "home" / "cache",
    )


@pytest.mark.asyncio
async def test_codex_is_an_ordinary_installed_plugin_with_login_refresh_and_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path("plugins/codex")
    for path in source.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not item.name.startswith("plugins.") for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("plugins.")
                if node.module.startswith("agent."):
                    assert node.module == "agent.plugin_composition"

    codex_repo = tmp_path / "codex-repo"
    models_repo = tmp_path / "models-repo"
    shutil.copytree(source, codex_repo)
    shutil.copytree(Path("plugins/models"), models_repo)
    shutil.rmtree(codex_repo / "__pycache__", ignore_errors=True)
    shutil.rmtree(models_repo / "__pycache__", ignore_errors=True)
    _commit(codex_repo)
    _commit(models_repo)
    models_install = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(models_repo),
        marketplace="ordinary-test",
        plugins_home=tmp_path / "home",
    )
    codex_install = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(codex_repo),
        marketplace="ordinary-test",
        plugins_home=tmp_path / "home",
    )

    blocked = ("plugins.models", "plugins.codex")
    for name in tuple(sys.modules):
        if name.startswith(blocked):
            monkeypatch.delitem(sys.modules, name)

    class _BlockRepositoryPlugins:
        def find_spec(
            self,
            fullname: str,
            path: object = None,
            target: object = None,
        ) -> None:
            _ = path, target
            if fullname.startswith(blocked):
                raise ModuleNotFoundError(fullname)
            return None

    monkeypatch.setattr(sys, "meta_path", [_BlockRepositoryPlugins(), *sys.meta_path])
    with _provider() as (server, endpoint):
        manager = _manager(tmp_path)
        await manager.load_all()
        for plugin_id, installed_path in (
            ("models@ordinary-test", models_install.installed_path),
            ("codex@ordinary-test", codex_install.installed_path),
        ):
            generation = manager.generation(plugin_id)
            assert generation is not None and generation.source_type == "installed"
            assert Path(generation.instance.module.__file__).resolve().is_relative_to(
                installed_path
            )

        snapshot = manager.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        root = snapshot.composition_root
        lease = await manager._snapshot_store.acquire()
        token = bind_runtime_snapshot(lease)
        try:
            settings = root.context.require(MODEL_SETTINGS)
            started = await settings.apply(
                StartConnectionAuth(
                    driver_id="codex",
                    connection_id="codex-main",
                    input={
                        "auth_base": f"{endpoint}/auth",
                        "api_base": f"{endpoint}/api",
                    },
                )
            )
            assert started.status == "pending" and started.attempt_id
            assert started.challenge == {
                "user_code": "ABCD-EFGH",
                "verification_uri": f"{endpoint}/auth/codex/device",
                "interval": 3,
            }
            finished = await settings.apply(
                FinishConnectionAuth(
                    expected_revision=0,
                    attempt_id=started.attempt_id,
                )
            )
            assert finished.status == "pending" and finished.revision == 0
            assert finished.challenge == started.challenge
            finished = await settings.apply(
                FinishConnectionAuth(
                    expected_revision=0,
                    attempt_id=started.attempt_id,
                )
            )
            assert finished.status == "committed" and finished.revision == 1
            registry_path = next((tmp_path / "workspace").rglob("model-registry.sqlite3"))
            with closing(sqlite3.connect(registry_path)) as connection:
                connection.execute(
                    "UPDATE model_connections SET catalog_provider_id = 'codex' "
                    "WHERE id = 'codex-main'"
                )
                connection.commit()
            revision = (
                await settings.apply(
                    SyncModels(
                        expected_revision=finished.revision,
                        connection_id="codex-main",
                    )
                )
            ).revision
            catalog = root.context.require(MODEL_CATALOG).snapshot()
            assert len(catalog.models) == 1
            model = catalog.models[0]
            assert model.model == "gpt-codex-test"
            assert model.capabilities.input_modalities == ("text", "image")
            server.catalog_default = "high"
            server.catalog_efforts = ["medium"]
            with pytest.raises(TransportError, match="不在支持列表"):
                await settings.apply(
                    SyncModels(
                        expected_revision=revision,
                        connection_id="codex-main",
                    )
                )
            assert root.context.require(MODEL_CATALOG).snapshot().revision == revision
            server.catalog_default = "medium"
            server.catalog_efforts = ["medium", "high"]
            with pytest.raises(ModelUnavailableError, match="does not provide embeddings"):
                await settings.apply(
                    AddModel(
                        expected_revision=revision,
                        model_id="forbidden-embedding",
                        connection_id="codex-main",
                        kind=ModelKind.EMBEDDING,
                        model="embed",
                        capabilities=ModelCapabilities(embedding_dimensions=3),
                        capability_sources=CapabilitySources(
                            embedding_dimensions="manual"
                        ),
                    )
                )
            revision = (
                await settings.apply(
                    SetDefaultModel(
                        expected_revision=revision,
                        role=ModelRole.DEFAULT,
                        model_id=model.model_id,
                    )
                )
            ).revision
            assert revision == 3
            async with root.context.require(CHAT_MODELS).execution() as execution:
                chat = execution.chat(ModelRole.DEFAULT)
                concurrent = await asyncio.gather(
                    chat.complete(ModelRequest(messages=({"role": "user", "content": "a"},))),
                    chat.complete(ModelRequest(messages=({"role": "user", "content": "b"},))),
                )
                assert {item.content for item in concurrent} == {"answer-1", "answer-2"}
                assert server.refresh_count == 1
                server.use_reject_barrier = False
                server.response_count = 0
                deltas: list[dict[str, str]] = []

                async def on_delta(delta: dict[str, str]) -> None:
                    deltas.append(delta)

                first = await chat.complete(
                    ModelRequest(
                        messages=({"role": "user", "content": "hello"},),
                        tools=({"type": "function", "function": {"name": "lookup"}},),
                        max_output_tokens=123,
                        on_delta=on_delta,
                    )
                )
                assert first.content == "answer-1"
                assert first.thinking == "think-1"
                assert first.tool_calls[0].arguments == {"id": 7}
                assert first.continuation is not None
                assert first.usage is not None
                assert first.usage.coverage is UsageCoverage.EXACT
                second = await chat.complete(
                    ModelRequest(
                        messages=(
                            {"role": "assistant", "tool_calls": [{
                                "id": "call-1",
                                "function": {"name": "lookup", "arguments": '{"id":7}'},
                            }]},
                            {"role": "tool", "tool_call_id": "call-1", "content": "found"},
                        ),
                        continuation=first.continuation,
                        disable_reasoning=True,
                    )
                )
                assert second.continuation is not None
                assert len(second.continuation.payload["items"]) == 2
                assert deltas == [
                    {"thinking_delta": "think-1"},
                    {"content_delta": "answer-1"},
                ]
                request_count = len(server.requests)
                with pytest.raises(ModelUnavailableError, match="binding"):
                    await chat.complete(
                        ModelRequest(
                            messages=(),
                            continuation=ModelContinuation(
                                binding_id="wrong-binding",
                                payload={"format_version": 1, "items": ()},
                            ),
                        )
                    )
                assert len(server.requests) == request_count
                interrupted_deltas: list[dict[str, str]] = []

                async def collect(delta: dict[str, str]) -> None:
                    interrupted_deltas.append(delta)

                with pytest.raises(TransportError, match="Codex Responses") as interrupted:
                    await chat.complete(
                        ModelRequest(
                            messages=({"role": "user", "content": "truncate"},),
                            on_delta=collect,
                        )
                    )
                assert interrupted_deltas
                assert interrupted.value.retryable is False
                done_deltas: list[dict[str, str]] = []

                async def collect_done(delta: dict[str, str]) -> None:
                    done_deltas.append(delta)

                with pytest.raises(TransportError) as done_interrupted:
                    await chat.complete(
                        ModelRequest(
                            messages=({"role": "user", "content": "done-truncate"},),
                            on_delta=collect_done,
                        )
                    )
                assert done_deltas == [{"content_delta": "done-only"}]
                assert done_interrupted.value.retryable is False
                with pytest.raises(TransportError, match="done text") as mismatch_done:
                    await chat.complete(
                        ModelRequest(
                            messages=({"role": "user", "content": "mismatch-done"},),
                            on_delta=lambda _delta: _async_none(),
                        )
                    )
                assert mismatch_done.value.retryable is False
                with pytest.raises(TransportError, match="arguments") as invalid_tool:
                    await chat.complete(
                        ModelRequest(
                            messages=({"role": "user", "content": "invalid-tool"},),
                            on_delta=lambda _delta: _async_none(),
                        )
                    )
                assert invalid_tool.value.retryable is False

                with closing(sqlite3.connect(registry_path)) as connection:
                    encoded = connection.execute(
                        "SELECT auth_payload FROM model_connections WHERE id = 'codex-main'"
                    ).fetchone()
                    assert encoded is not None
                    credential = json.loads(encoded[0])
                    credential["access_token"] = "access-old"
                    connection.execute(
                        "UPDATE model_connections SET auth_payload = ? WHERE id = 'codex-main'",
                        (json.dumps(credential),),
                    )
                    connection.commit()
                revision_before_refresh_failure = (
                    root.context.require(MODEL_CATALOG).snapshot().revision
                )
                server.refresh_status = 503
                with pytest.raises(TransportError, match="token.*503"):
                    await chat.complete(ModelRequest(messages=()))
                assert (
                    root.context.require(MODEL_CATALOG).snapshot().revision
                    == revision_before_refresh_failure
                )
        finally:
            reset_runtime_snapshot(token)
            await lease.release()
            await manager.terminate_all()

        assert server.refresh_count == 2
        response_bodies = [
            item["body"] for item in server.requests if item["path"] == "/api/responses"
        ]
        first_body = next(
            body for body in response_bodies if body.get("max_output_tokens") == 123
        )
        continuation_body = next(
            body
            for body in response_bodies
            if body.get("input")
            and isinstance(body["input"][0], dict)
            and body["input"][0].get("encrypted_content") == "opaque-1"
        )
        assert first_body["reasoning"]["effort"] == "medium"
        assert "reasoning" not in continuation_body
        assert "id" not in continuation_body["input"][0]
        assert continuation_body["input"][1:] == [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": '{"id":7}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "found",
            },
        ]

        _ = uninstall_plugin(
            "codex@ordinary-test",
            workspace=tmp_path / "workspace",
            plugins_home=tmp_path / "home",
        )
        without = _manager(tmp_path)
        await without.load_all()
        snapshot = without.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        assert all(
            model.availability is ModelAvailability.DRIVER_UNAVAILABLE
            for model in snapshot.composition_root.context.require(MODEL_CATALOG).snapshot().models
        )
        await without.terminate_all()

        restored_install = install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(codex_repo),
            marketplace="ordinary-test",
            plugins_home=tmp_path / "home",
        )
        restored = _manager(tmp_path)
        await restored.load_all()
        generation = restored.generation("codex@ordinary-test")
        assert generation is not None and generation.plugin_dir == restored_install.installed_path
        snapshot = restored.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        assert all(
            model.availability is ModelAvailability.AVAILABLE
            for model in snapshot.composition_root.context.require(MODEL_CATALOG).snapshot().models
        )
        server.refresh_status = 200
        restored_root = snapshot.composition_root
        restored_lease = await restored._snapshot_store.acquire()
        restored_token = bind_runtime_snapshot(restored_lease)
        try:
            async with restored_root.context.require(CHAT_MODELS).execution() as execution:
                result = await execution.chat(ModelRole.DEFAULT).complete(
                    ModelRequest(messages=({"role": "user", "content": "restored"},))
                )
                assert result.content is not None
        finally:
            reset_runtime_snapshot(restored_token)
            await restored_lease.release()
        await restored.terminate_all()

    assert not any(name.startswith(blocked) for name in sys.modules)


async def _async_none() -> None:
    return None
