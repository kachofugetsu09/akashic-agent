from __future__ import annotations

import asyncio
import ast
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import pytest

from agent.plugin_composition import (
    AddConnection,
    AuthenticationError,
    BoundModelDescriptor,
    CapabilitySources,
    CHAT_MODELS,
    ContextLengthError,
    DriverConnectionDescriptor,
    ModelCapabilities,
    ModelAvailability,
    MODEL_CATALOG,
    MODEL_SETTINGS,
    InvalidRequestError,
    ModelKind,
    ModelRequest,
    ModelRole,
    RateLimitError,
    QuotaError,
    SetDefaultModel,
    SyncModels,
    TransportError,
    UsageCoverage,
)
from agent.plugins.install import install_git_plugin, uninstall_plugin
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot
from bus.event_bus import EventBus
from plugins.opencode_go import driver as opencode_driver
from plugins.opencode_go.driver import _parse_cli_catalog, definition


class _Credential:
    def __init__(self, token: str = "secret") -> None:
        self.connection_id = "connection-1"
        self.auth_identity = "account-1"
        self.token = token

    async def read(self) -> Mapping[str, str]:
        return {"driver": "api_key", "access_token": self.token}

    async def refresh(self, payload: Mapping[str, str]) -> None:
        self.token = payload["access_token"]

    @asynccontextmanager
    async def exclusive(self) -> AsyncIterator[None]:
        yield


@pytest.fixture(autouse=True)
def _disable_host_opencode_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        opencode_driver,
        "_OPENCODE_EXECUTABLE",
        "/missing/opencode",
    )


def test_cli_catalog_parser_keeps_provider_owned_limits_and_variants() -> None:
    parsed = _parse_cli_catalog(
        """opencode-go/deepseek-v4-pro
{"limit":{"context":1000000,"output":384000},"variants":{"high":{},"max":{}}}
opencode-go/glm-5
{"limit":{"context":202752,"output":32768},"variants":{}}
"""
    )
    assert parsed["deepseek-v4-pro"]["limit"] == {
        "context": 1_000_000,
        "output": 384_000,
    }
    assert tuple(parsed["deepseek-v4-pro"]["variants"]) == ("high", "max")


@pytest.mark.asyncio
async def test_optional_cli_failures_degrade_and_reap_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProcess:
        def __init__(self, mode: str) -> None:
            self.mode = mode
            self.returncode = 1 if mode == "nonzero" else 0
            self.killed = False
            self.waited = False

        async def communicate(self) -> tuple[bytes, bytes]:
            if self.mode == "timeout":
                raise TimeoutError
            if self.mode == "cancel":
                raise asyncio.CancelledError
            if self.mode == "bad-utf8":
                return b"\xff", b""
            if self.mode == "bad-output":
                return b"not-a-catalog", b""
            return b"", b"command failed"

        def kill(self) -> None:
            self.killed = True

        async def wait(self) -> int:
            self.waited = True
            return 0

    current = FakeProcess("nonzero")

    async def create_process(*_args: object, **_kwargs: object) -> FakeProcess:
        return current

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_process)
    for mode in ("nonzero", "timeout", "bad-utf8", "bad-output"):
        current = FakeProcess(mode)
        assert await opencode_driver._load_cli_catalog() == {}
        if mode == "timeout":
            assert current.killed and current.waited

    current = FakeProcess("cancel")
    with pytest.raises(asyncio.CancelledError):
        await opencode_driver._load_cli_catalog()
    assert current.killed and current.waited


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], *, models_status: int = 200) -> None:
        super().__init__(address, _Handler)
        self.requests: list[dict[str, Any]] = []
        self.slow_started = threading.Event()
        self.models_status = models_status


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: object) -> None:
        _ = format, args

    def do_GET(self) -> None:
        self.server.requests.append(
            {"path": self.path, "authorization": self.headers.get("Authorization")}
        )
        if self.path == "/v1/models":
            self._json(
                self.server.models_status,
                (
                    {
                        "object": "list",
                        "data": [
                            {"id": "chat-a", "variants": {"high": {}, "max": {}}},
                            {"id": "qwen3.5-plus"},
                            {"id": "minimax-m2"},
                        ],
                    }
                    if self.server.models_status == 200
                    else {"error": {"message": "catalog unavailable"}}
                ),
            )
            return
        self._json(404, {"error": {"message": "missing"}})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "body": body,
            }
        )
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": {"message": "missing"}})
            return
        model = body.get("model")
        if model == "context-error":
            self._json(400, {"error": {"message": "maximum context length exceeded"}})
            return
        if model == "rate-error":
            self._json(429, {"error": {"message": "rate limited"}})
            return
        if model == "quota-error":
            self._json(429, {"error": {"message": "insufficient quota"}})
            return
        if model == "auth-error":
            self._json(401, {"error": {"message": "invalid token"}})
            return
        if model == "echo-secret-error":
            self._json(
                401,
                {"error": {"message": f"rejected {self.headers.get('Authorization')}"}},
            )
            return
        if model == "invalid-error":
            self._json(400, {"error": {"message": "unknown model"}})
            return
        if model == "slow-model":
            self.server.slow_started.set()
            time.sleep(2)
            self._json(200, _text_response("late"))
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = (
                {"choices": [{"delta": {"reasoning_content": "why "}}]},
                {"choices": [{"delta": {"reasoning": "because "}}]},
                {"choices": [{"delta": {"content": "hello "}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "function": {"name": "search", "arguments": '{"q":'},
                                    }
                                ]
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {"index": 0, "function": {"arguments": '"hi"}'}}
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
                {
                    "choices": [],
                    "usage": {
                        "prompt_cache_hit_tokens": 3,
                        "prompt_cache_miss_tokens": 7,
                        "completion_tokens": 4,
                    },
                },
            )
            if model == "invalid-tool-stream":
                chunks = (
                    {"choices": [{"delta": {"content": "visible "}}]},
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call-bad",
                                            "function": {
                                                "name": "search",
                                                "arguments": '{"q":',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                )
            for chunk in chunks:
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            if model != "truncated-stream":
                self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        reasoning_field = (
            "reasoning" if model == "reasoning-alias" else "reasoning_content"
        )
        self._json(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            reasoning_field: "checked",
                            "tool_calls": [
                                {
                                    "id": "call-2",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"id": 7}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "completion_tokens_details": {"reasoning_tokens": 1},
                },
            },
        )

    def _json(self, status: int, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        try:
            self.wfile.write(encoded)
        except BrokenPipeError:
            pass


def _text_response(content: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


@contextmanager
def _provider(*, models_status: int = 200) -> Iterator[tuple[_Server, str]]:
    server = _Server(("127.0.0.1", 0), models_status=models_status)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _connection(endpoint: str) -> DriverConnectionDescriptor:
    return DriverConnectionDescriptor(
        connection_id="connection-1",
        name="Test",
        driver_id="opencode-go",
        endpoint=endpoint,
        auth_identity="account-1",
        config={
            "format_version": 1,
            "max_retries": 0,
        },
    )


def _chat_descriptor(model: str = "chat-a") -> BoundModelDescriptor:
    return BoundModelDescriptor(
        binding_id=f"binding-{model}",
        plugin_snapshot_id="snapshot-1",
        model_revision=3,
        model_id=f"model-{model}",
        connection_id="connection-1",
        driver_id="opencode-go",
        driver_contract_version="1",
        auth_identity="account-1",
        model=model,
        role=ModelRole.DEFAULT,
        reasoning_effort="high",
        capabilities=ModelCapabilities(context_window=8192, supports_tool_calls=True),
        capability_sources=CapabilitySources(context_window="test"),
        capability_digest="digest-chat",
    )


@pytest.mark.asyncio
async def test_driver_discovers_and_runs_opencode_go_chat_contract() -> None:
    with _provider() as (server, endpoint):
        credential = _Credential()
        driver = definition()
        assert driver.probe is not None and driver.discover is not None
        await driver.probe(_connection(endpoint), credential)
        discovered = await driver.discover(_connection(endpoint), credential)
        assert [(item.model, item.kind) for item in discovered] == [
            ("chat-a", ModelKind.CHAT)
        ]
        assert discovered[0].capabilities.supported_reasoning_efforts == ()
        assert discovered[0].capabilities.supports_tool_calls is None
        assert discovered[0].capability_sources.tool_calls == "unknown"

        opened = await driver.open(_connection(endpoint), credential)
        chat = opened.bind_chat(_chat_descriptor(), {})
        nonstream = await chat.complete(
            ModelRequest(
                messages=({"role": "user", "content": "hello"},),
                tools=({"type": "function", "function": {"name": "lookup"}},),
                system_prompt="system",
            )
        )
        assert nonstream.content is None
        assert nonstream.thinking == "checked"
        assert [(call.name, call.arguments) for call in nonstream.tool_calls] == [
            ("lookup", {"id": 7})
        ]
        assert nonstream.usage is not None
        assert nonstream.usage.coverage is UsageCoverage.EXACT
        assert nonstream.usage.reasoning_output_tokens == 1
        assert chat.max_tool_schemas == 16

        alias_chat = opened.bind_chat(_chat_descriptor("reasoning-alias"), {})
        alias_response = await alias_chat.complete(
            ModelRequest(messages=({"role": "user", "content": "hello"},))
        )
        assert alias_response.thinking == "checked"

        deltas: list[dict[str, str]] = []

        async def on_delta(delta: dict[str, str]) -> None:
            deltas.append(delta)

        stream_chat = opened.bind_chat(_chat_descriptor("stream-model"), {})
        streamed = await stream_chat.complete(
            ModelRequest(
                messages=({"role": "user", "content": "stream"},),
                on_delta=on_delta,
            )
        )
        assert streamed.content == "hello"
        assert streamed.thinking == "why because"
        assert deltas == [
            {"thinking_delta": "why "},
            {"thinking_delta": "because "},
            {"content_delta": "hello "},
        ]
        assert [(call.id, call.name, call.arguments) for call in streamed.tool_calls] == [
            ("call-1", "search", {"q": "hi"})
        ]
        assert streamed.usage is not None
        assert streamed.usage.input_tokens == 10
        assert streamed.usage.cached_input_tokens == 3

        credential.token = "rotated"
        _ = await chat.complete(ModelRequest(messages=({"role": "user", "content": "again"},)))
        assert server.requests[-1]["authorization"] == "Bearer rotated"
        sent = next(
            request["body"]
            for request in server.requests
            if request.get("body", {}).get("model") == "chat-a"
        )
        assert sent["reasoning_effort"] == "high"
        assert sent["messages"][0] == {"role": "system", "content": "system"}


@pytest.mark.asyncio
async def test_five_wire_profiles_and_messages_models_are_owned_by_driver() -> None:
    with _provider() as (server, endpoint):
        opened = await definition().open(_connection(endpoint), _Credential())
        profiles = (
            ("future-chat", "high", 10),
            ("deepseek-v4-pro", "max", 10),
            ("glm-5", "high", 10),
            ("kimi-k3", "high", 10),
            ("mimo-v2.5-pro", "high", 131_072),
        )
        for model, effort, max_tokens in profiles:
            descriptor = replace(_chat_descriptor(model), reasoning_effort=effort)
            chat = opened.bind_chat(descriptor, {})
            await chat.complete(
                ModelRequest(
                    messages=(
                        {"role": "assistant", "content": "old"},
                        {"role": "user", "content": "again"},
                    ),
                    tools=(
                        {
                            "type": "function",
                            "function": {
                                "name": "probe",
                                "strict": True,
                                "parameters": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "value": {"type": ["string", "null"]}
                                    },
                                },
                            },
                        },
                    ),
                    max_output_tokens=200_000 if model.startswith("mimo-") else max_tokens,
                )
            )
        bodies = {item["body"]["model"]: item["body"] for item in server.requests if "body" in item}
        assert bodies["future-chat"]["reasoning_effort"] == "high"
        assert bodies["deepseek-v4-pro"]["thinking"] == {"type": "enabled"}
        assert bodies["deepseek-v4-pro"]["reasoning_effort"] == "max"
        assert bodies["deepseek-v4-pro"]["messages"][0]["reasoning_content"] == ""
        assert bodies["glm-5"]["reasoning_effort"] == "high"
        assert bodies["kimi-k3"]["reasoning_effort"] == "high"
        assert bodies["mimo-v2.5-pro"]["max_tokens"] == 131_072
        function = bodies["future-chat"]["tools"][0]["function"]
        assert "strict" not in function
        assert "additionalProperties" not in function["parameters"]
        assert function["parameters"]["properties"]["value"]["type"] == "string"

        for messages_model in ("qwen3.5-plus", "minimax-m2"):
            with pytest.raises(InvalidRequestError, match="Messages API"):
                opened.bind_chat(_chat_descriptor(messages_model), {})

        named = opened.bind_chat(
            replace(_chat_descriptor("deepseek-v4-pro"), reasoning_effort="max"),
            {},
        )
        await named.complete(
            ModelRequest(
                messages=(),
                tools=({"type": "function", "function": {"name": "probe"}},),
                tool_choice={"type": "function", "function": {"name": "probe"}},
            )
        )
        assert server.requests[-1]["body"]["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in server.requests[-1]["body"]


@pytest.mark.asyncio
async def test_auth_imports_api_key_sqlite_and_legacy_in_owner_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver = definition()
    assert driver.start_auth is not None and driver.finish_auth is not None
    direct = await driver.start_auth({"api_key": "direct", "endpoint": "https://example.test/v1"})
    completed = await driver.finish_auth(direct["state"])
    assert completed["credential"] == {"driver": "api_key", "access_token": "direct"}
    with pytest.raises(ValueError, match="unsupported auth fields"):
        await driver.start_auth({"database_path": "/tmp/secret"})
    with pytest.raises(ValueError, match="official endpoint"):
        await driver.start_auth({"endpoint": "https://attacker.test/v1"})

    database = tmp_path / "opencode.db"
    monkeypatch.setattr(opencode_driver, "_OPENCODE_DATA_DIR", tmp_path)
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE credential ("
        "id TEXT PRIMARY KEY, integration_id TEXT, value TEXT NOT NULL, "
        "active INTEGER, time_created INTEGER NOT NULL, time_updated INTEGER NOT NULL)"
    )
    connection.execute(
        "INSERT INTO credential VALUES (?, ?, ?, ?, ?, ?)",
        (
            "old",
            "opencode-go",
            json.dumps({"type": "key", "key": "old-key"}),
            1,
            1,
            1,
        ),
    )
    connection.execute(
        "INSERT INTO credential VALUES (?, ?, ?, ?, ?, ?)",
        (
            "inactive-newest",
            "opencode-go",
            json.dumps({"type": "key", "key": "inactive-key"}),
            0,
            3,
            3,
        ),
    )
    connection.execute(
        "INSERT INTO credential VALUES (?, ?, ?, ?, ?, ?)",
        (
            "new",
            "opencode-go",
            json.dumps({"type": "key", "key": "database-key"}),
            1,
            2,
            2,
        ),
    )
    connection.commit()
    connection.close()
    legacy = tmp_path / "auth.json"
    legacy.write_text(json.dumps({"opencode-go": {"key": "legacy-key"}}), encoding="utf-8")
    imported = await driver.start_auth({})
    result = await driver.finish_auth(imported["state"])
    assert result["credential"]["access_token"] == "database-key"

    connection = sqlite3.connect(database)
    connection.execute("UPDATE credential SET value = ? WHERE id = 'new'", ("not-json",))
    connection.commit()
    connection.close()
    with pytest.raises(AuthenticationError, match="database credential is invalid"):
        await driver.start_auth({})

    database.unlink()
    imported = await driver.start_auth({})
    result = await driver.finish_auth(imported["state"])
    assert result["credential"]["access_token"] == "legacy-key"


@pytest.mark.asyncio
async def test_driver_maps_errors_and_preserves_cancellation() -> None:
    with _provider() as (server, endpoint):
        opened = await definition().open(_connection(endpoint), _Credential())
        for model, error_type in (
            ("context-error", ContextLengthError),
            ("rate-error", RateLimitError),
            ("quota-error", QuotaError),
            ("auth-error", AuthenticationError),
            ("invalid-error", InvalidRequestError),
        ):
            chat = opened.bind_chat(_chat_descriptor(model), {})
            with pytest.raises(error_type):
                await chat.complete(ModelRequest(messages=()))

        slow = opened.bind_chat(_chat_descriptor("slow-model"), {})
        task = asyncio.create_task(slow.complete(ModelRequest(messages=())))
        assert await asyncio.to_thread(server.slow_started.wait, 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        class CallbackFailure(Exception):
            pass

        async def fail_callback(_delta: dict[str, str]) -> None:
            raise CallbackFailure("consumer failed")

        stream = opened.bind_chat(_chat_descriptor("stream-model"), {})
        with pytest.raises(CallbackFailure, match="consumer failed"):
            await stream.complete(ModelRequest(messages=(), on_delta=fail_callback))

        emitted: list[dict[str, str]] = []

        async def collect(delta: dict[str, str]) -> None:
            emitted.append(delta)

        truncated = opened.bind_chat(_chat_descriptor("truncated-stream"), {})
        with pytest.raises(TransportError, match="terminal marker") as truncated_error:
            await truncated.complete(ModelRequest(messages=(), on_delta=collect))
        assert emitted
        assert truncated_error.value.retryable is False

        retrying_connection = replace(
            _connection(endpoint),
            config={"format_version": 1, "max_retries": 2},
        )
        retrying_opened = await definition().open(retrying_connection, _Credential())
        invalid_tools = retrying_opened.bind_chat(
            _chat_descriptor("invalid-tool-stream"),
            {},
        )
        before = len(server.requests)
        with pytest.raises(TransportError) as invalid_tool_error:
            await invalid_tools.complete(ModelRequest(messages=(), on_delta=collect))
        matching = [
            item
            for item in server.requests[before:]
            if item.get("body", {}).get("model") == "invalid-tool-stream"
        ]
        assert len(matching) == 1
        assert invalid_tool_error.value.retryable is False

        leaked = opened.bind_chat(_chat_descriptor("echo-secret-error"), {})
        with pytest.raises(AuthenticationError) as caught:
            await leaked.complete(ModelRequest(messages=()))
        assert "secret" not in str(caught.value)
        assert "[REDACTED]" in str(caught.value)


@pytest.mark.asyncio
async def test_catalog_is_required() -> None:
    with _provider(models_status=404) as (_server, endpoint):
        with pytest.raises(InvalidRequestError, match="catalog unavailable"):
            await definition().probe(_connection(endpoint), _Credential())  # type: ignore[misc]

    with _provider() as (_server, closed_endpoint):
        closed = _connection(closed_endpoint)
    with pytest.raises(TransportError):
        await definition().probe(closed, _Credential())  # type: ignore[misc]


@pytest.mark.asyncio
async def test_persisted_config_rejects_unbounded_secret_surfaces() -> None:
    with _provider() as (_server, endpoint):
        for config in (
            {"headers": {"X-API-Key": "secret"}},
            {"extra_body": {"password": "secret"}},
            {"opencode_executable": "/tmp/evil"},
            {"catalog_provider_id": "openai"},
        ):
            with pytest.raises(ValueError):
                await definition().open(
                    DriverConnectionDescriptor(
                        connection_id="connection-1",
                        name="OpenCode Go",
                        driver_id="opencode-go",
                        endpoint=endpoint,
                        auth_identity="account-1",
                        config=config,
                    ),
                    _Credential(),
                )

        opened = await definition().open(_connection(endpoint), _Credential())
        for compatible in (
            {},
            {"format_version": 1},
            {"use_responses_lite": False, "reasoning_summary": "none"},
            {"use_responses_lite": 0, "reasoning_summary": ""},
            {"max_tool_schemas": 16},
            {"max_tool_schemas": 32},
            {"max_tool_schemas": None},
        ):
            assert opened.bind_chat(_chat_descriptor(), compatible).max_tool_schemas == 16
        for incompatible in (
            {"max_tool_schemas": 0},
            {"max_tool_schemas": True},
            {"max_tool_schemas": "16"},
            {"use_responses_lite": True},
            {"reasoning_summary": "auto"},
        ):
            with pytest.raises(ValueError):
                opened.bind_chat(_chat_descriptor(), incompatible)


@pytest.mark.asyncio
async def test_driver_is_an_installable_ordinary_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path("plugins/opencode_go")
    for path in source.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not item.name.startswith("plugins.") for item in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("plugins.")
                if node.module.startswith("agent."):
                    assert node.module == "agent.plugin_composition"

    repo = tmp_path / "driver-repo"
    shutil.copytree(source, repo)
    shutil.rmtree(repo / "__pycache__", ignore_errors=True)
    for args in (
        ("init",),
        ("config", "user.name", "test"),
        ("config", "user.email", "test@example.com"),
        ("add", "."),
        ("commit", "-m", "initial"),
    ):
        result = subprocess.run(
            ("git", *args),
            cwd=repo,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        assert result.returncode == 0, result.stderr
    installed = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(repo),
        marketplace="ordinary-test",
        plugins_home=tmp_path / "home",
    )
    assert installed.plugin_name == "opencode-go"
    assert installed.installed_path.is_relative_to(tmp_path / "home")
    assert installed.installed_path != source.resolve()

    models_repo = tmp_path / "models-repo"
    shutil.copytree(Path("plugins/models"), models_repo)
    shutil.rmtree(models_repo / "__pycache__", ignore_errors=True)
    for args in (
        ("init",),
        ("config", "user.name", "test"),
        ("config", "user.email", "test@example.com"),
        ("add", "."),
        ("commit", "-m", "initial"),
    ):
        result = subprocess.run(
            ("git", *args),
            cwd=models_repo,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        assert result.returncode == 0, result.stderr
    models_installed = install_git_plugin(
        workspace=tmp_path / "workspace",
        source=str(models_repo),
        marketplace="ordinary-test",
        plugins_home=tmp_path / "home",
    )

    blocked_prefixes = ("plugins.models", "plugins.opencode_go")
    for module_name in tuple(sys.modules):
        if module_name.startswith(blocked_prefixes):
            monkeypatch.delitem(sys.modules, module_name)

    class BlockRepositoryPlugins:
        def find_spec(
            self,
            fullname: str,
            path: object = None,
            target: object = None,
        ) -> None:
            _ = path, target
            if fullname.startswith(blocked_prefixes):
                raise ModuleNotFoundError(f"repository plugin import blocked: {fullname}")
            return None

    monkeypatch.setattr(sys, "meta_path", [BlockRepositoryPlugins(), *sys.meta_path])
    original_path = os.environ.get("PATH", "")
    empty_path = tmp_path / "empty-path"
    empty_path.mkdir()
    with _provider() as (_server, endpoint):
        manager = PluginManager(
            plugin_dirs=[],
            event_bus=EventBus(),
            tool_registry=None,
            workspace=tmp_path / "workspace",
            installed_cache_root=tmp_path / "home" / "cache",
        )
        await manager.load_all()
        for plugin_id, expected_path in (
            ("models@ordinary-test", models_installed.installed_path),
            ("opencode-go@ordinary-test", installed.installed_path),
        ):
            generation = manager.generation(plugin_id)
            assert generation is not None and generation.source_type == "installed"
            assert Path(generation.instance.module.__file__).resolve().is_relative_to(
                expected_path
            )

        snapshot = manager.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        root = snapshot.composition_root
        lease = await manager._snapshot_store.acquire()
        token = bind_runtime_snapshot(lease)
        try:
            monkeypatch.setenv("PATH", str(empty_path))
            settings = root.context.require(MODEL_SETTINGS)
            revision = (
                await settings.apply(
                    AddConnection(
                        expected_revision=0,
                        connection_id="opencode-go",
                        name="OpenCode Go",
                        driver_id="opencode-go",
                        endpoint=endpoint,
                        auth_identity="account",
                        credential={"driver": "api_key", "access_token": "secret"},
                        driver_config={
                            "max_retries": 0,
                            "catalog_provider_id": "opencode-go",
                        },
                    )
                )
            ).revision
            revision = (
                await settings.apply(
                    SyncModels(expected_revision=revision, connection_id="opencode-go")
                )
            ).revision
            monkeypatch.setenv("PATH", original_path)
            catalog = root.context.require(MODEL_CATALOG).snapshot()
            chat_id = next(model.model_id for model in catalog.models if model.model == "chat-a")
            revision = (
                await settings.apply(
                    SetDefaultModel(
                        expected_revision=revision,
                        role=ModelRole.DEFAULT,
                        model_id=chat_id,
                    )
                )
            ).revision
            async with root.context.require(CHAT_MODELS).execution() as execution:
                chat = execution.chat(ModelRole.DEFAULT)
                response = await chat.complete(
                    ModelRequest(messages=({"role": "user", "content": "hello"},))
                )
                assert response.thinking == "checked"
                assert response.tool_calls[0].name == "lookup"
                deltas: list[dict[str, str]] = []

                async def on_delta(delta: dict[str, str]) -> None:
                    deltas.append(delta)

                streamed = await chat.complete(
                    ModelRequest(
                        messages=({"role": "user", "content": "stream"},),
                        on_delta=on_delta,
                    )
                )
                assert streamed.thinking == "why because"
                assert streamed.tool_calls[0].name == "search"
                assert streamed.usage is not None
                assert streamed.usage.coverage is UsageCoverage.EXACT
                assert deltas == [
                    {"thinking_delta": "why "},
                    {"thinking_delta": "because "},
                    {"content_delta": "hello "},
                ]
        finally:
            reset_runtime_snapshot(token)
            await lease.release()
            await manager.terminate_all()

        registry_path = next((tmp_path / "workspace").rglob("model-registry.sqlite3"))

        def write_legacy_tool_limit(value: int) -> None:
            connection = sqlite3.connect(registry_path)
            try:
                row = connection.execute(
                    "SELECT capabilities_json FROM model_definitions WHERE model = ?",
                    ("chat-a",),
                ).fetchone()
                assert row is not None
                payload = json.loads(row[0])
                payload["driver_config"] = {
                    "format_version": 1,
                    "max_tool_schemas": value,
                    "use_responses_lite": False,
                    "reasoning_summary": "none",
                }
                connection.execute(
                    "UPDATE model_definitions SET capabilities_json = ? WHERE model = ?",
                    (json.dumps(payload), "chat-a"),
                )
                connection.commit()
            finally:
                connection.close()

        write_legacy_tool_limit(32)
        reloaded = PluginManager(
            plugin_dirs=[],
            event_bus=EventBus(),
            tool_registry=None,
            workspace=tmp_path / "workspace",
            installed_cache_root=tmp_path / "home" / "cache",
        )
        await reloaded.load_all()
        snapshot = reloaded.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        assert all(
            model.availability is ModelAvailability.AVAILABLE
            for model in snapshot.composition_root.context.require(MODEL_CATALOG).snapshot().models
        )
        lease = await reloaded._snapshot_store.acquire()
        token = bind_runtime_snapshot(lease)
        try:
            async with snapshot.composition_root.context.require(CHAT_MODELS).execution() as execution:
                chat = execution.chat(ModelRole.DEFAULT)
                assert chat.max_tool_schemas == 16
                response = await chat.complete(
                    ModelRequest(messages=({"role": "user", "content": "legacy 32"},))
                )
                assert response.tool_calls[0].name == "lookup"
        finally:
            reset_runtime_snapshot(token)
            await lease.release()
        await reloaded.terminate_all()

        write_legacy_tool_limit(16)
        _ = uninstall_plugin(
            "opencode-go@ordinary-test",
            workspace=tmp_path / "workspace",
            plugins_home=tmp_path / "home",
        )
        without_driver = PluginManager(
            plugin_dirs=[],
            event_bus=EventBus(),
            tool_registry=None,
            workspace=tmp_path / "workspace",
            installed_cache_root=tmp_path / "home" / "cache",
        )
        await without_driver.load_all()
        snapshot = without_driver.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        assert all(
            model.availability is ModelAvailability.DRIVER_UNAVAILABLE
            for model in snapshot.composition_root.context.require(MODEL_CATALOG).snapshot().models
        )
        await without_driver.terminate_all()

        restored_install = install_git_plugin(
            workspace=tmp_path / "workspace",
            source=str(repo),
            marketplace="ordinary-test",
            plugins_home=tmp_path / "home",
        )
        restored = PluginManager(
            plugin_dirs=[],
            event_bus=EventBus(),
            tool_registry=None,
            workspace=tmp_path / "workspace",
            installed_cache_root=tmp_path / "home" / "cache",
        )
        await restored.load_all()
        generation = restored.generation("opencode-go@ordinary-test")
        assert generation is not None
        assert generation.plugin_dir == restored_install.installed_path
        snapshot = restored.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        assert all(
            model.availability is ModelAvailability.AVAILABLE
            for model in snapshot.composition_root.context.require(MODEL_CATALOG).snapshot().models
        )
        lease = await restored._snapshot_store.acquire()
        token = bind_runtime_snapshot(lease)
        try:
            async with snapshot.composition_root.context.require(CHAT_MODELS).execution() as execution:
                chat = execution.chat(ModelRole.DEFAULT)
                assert chat.max_tool_schemas == 16
                response = await chat.complete(
                    ModelRequest(messages=({"role": "user", "content": "after reinstall"},))
                )
                assert response.tool_calls[0].name == "lookup"
                assert _server.requests[-1]["authorization"] == "Bearer secret"
        finally:
            reset_runtime_snapshot(token)
            await lease.release()
        await restored.terminate_all()

    assert not any(name.startswith(blocked_prefixes) for name in sys.modules)
