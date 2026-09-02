from __future__ import annotations

import asyncio
import ast
import gzip
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator

import pytest

from agent.plugin_composition import (
    AddConnection,
    AddModel,
    AuthenticationError,
    BoundModelDescriptor,
    CapabilitySources,
    CHAT_MODELS,
    ContextLengthError,
    DriverConnectionDescriptor,
    EMBEDDINGS,
    EmbeddingSpaceDescriptor,
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
from agent.plugins.install import (
    finalize_uninstall_plugin,
    install_git_plugin,
    set_installed_plugin_enabled,
)
from agent.plugins.manager import PluginManager
from agent.plugins.snapshot import bind_runtime_snapshot, reset_runtime_snapshot
from bus.event_bus import EventBus
from plugins.openai_compatible.driver import definition


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


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        *,
        models_status: int = 200,
        model_ids: list[str] | None = None,
        gzip_models: bool = False,
    ) -> None:
        super().__init__(address, _Handler)
        self.requests: list[dict[str, Any]] = []
        self.slow_started = threading.Event()
        self.release_plain_done = threading.Event()
        self.models_status = models_status
        self.model_ids = model_ids
        self.gzip_models = gzip_models


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
                            {"id": model_id}
                            for model_id in (self.server.model_ids or ["chat-a"])
                        ],
                    }
                    if self.server.models_status == 200
                    else {"error": {"message": "catalog unavailable"}}
                ),
                gzip_response=self.server.gzip_models,
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
        if self.path == "/v1/embeddings":
            inputs = body.get("input") if isinstance(body.get("input"), list) else []
            vectors = (
                ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
                if len(inputs) == 2
                else tuple(
                    [float(value.removeprefix("item-")), 0.0, 0.0]
                    if isinstance(value, str) and value.startswith("item-")
                    else [1.0, 0.0, 0.0]
                    for value in inputs
                )
            )
            self._json(
                200,
                {
                    "data": [
                        {"index": index, "embedding": vector}
                        for index, vector in reversed(tuple(enumerate(vectors)))
                    ],
                    "usage": {"prompt_tokens": 4},
                },
            )
            return
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
        if model == "think-tags" and not body.get("stream"):
            self._json(200, _text_response("<think>checked tags</think>answer"))
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            if model == "think-tags":
                chunks = (
                    {"choices": [{"delta": {"content": "<thi"}}]},
                    {"choices": [{"delta": {"content": "nk>checked "}}]},
                    {"choices": [{"delta": {"content": "tags</th"}}]},
                    {"choices": [{"delta": {"content": "ink>answer"}}]},
                )
            elif model == "think-tags-native-late":
                chunks = (
                    {
                        "choices": [
                            {"delta": {"content": "<think>legacy</think>answer"}}
                        ]
                    },
                    {"choices": [{"delta": {"reasoning_content": "native"}}]},
                )
            elif model == "plain-stream":
                chunks = (
                    {"choices": [{"delta": {"content": "hello "}}]},
                    {"choices": [{"delta": {"content": "world"}}]},
                )
            else:
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
                            "prompt_tokens": 10,
                            "completion_tokens": 4,
                            "prompt_tokens_details": {"cached_tokens": 3},
                        },
                    },
                )
            for index, chunk in enumerate(chunks):
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                if model == "plain-stream" and index == 0:
                    self.server.release_plain_done.wait(timeout=2)
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

    def _json(
        self,
        status: int,
        payload: Mapping[str, Any],
        *,
        gzip_response: bool = False,
    ) -> None:
        encoded = json.dumps(payload).encode()
        if gzip_response:
            encoded = gzip.compress(encoded)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        if gzip_response:
            self.send_header("Content-Encoding", "gzip")
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
def _provider(
    *,
    models_status: int = 200,
    model_ids: list[str] | None = None,
    gzip_models: bool = False,
) -> Iterator[tuple[_Server, str]]:
    server = _Server(
        ("127.0.0.1", 0),
        models_status=models_status,
        model_ids=model_ids,
        gzip_models=gzip_models,
    )
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
        driver_id="openai-compatible",
        endpoint=endpoint,
        auth_identity="account-1",
        config={"format_version": 1, "max_retries": 0},
    )


def _chat_descriptor(model: str = "chat-a") -> BoundModelDescriptor:
    return BoundModelDescriptor(
        binding_id=f"binding-{model}",
        plugin_snapshot_id="snapshot-1",
        model_revision=3,
        model_id=f"model-{model}",
        connection_id="connection-1",
        driver_id="openai-compatible",
        driver_contract_version="1",
        auth_identity="account-1",
        model=model,
        role=ModelRole.DEFAULT,
        reasoning_effort="high",
        capabilities=ModelCapabilities(context_window=8192, supports_tool_calls=True),
        capability_sources=CapabilitySources(context_window="test"),
        capability_digest="digest-chat",
    )


def _embedding_descriptor() -> EmbeddingSpaceDescriptor:
    return EmbeddingSpaceDescriptor(
        plugin_snapshot_id="snapshot-1",
        model_revision=3,
        model_id="embedding-1",
        connection_id="connection-1",
        driver_id="openai-compatible",
        driver_contract_version="1",
        auth_identity="account-1",
        connection_fingerprint="connection-digest",
        model="embed-a",
        dimensions=3,
        normalization="none",
        capability_digest="digest-embedding",
    )


@pytest.mark.asyncio
async def test_driver_discovers_chats_and_runs_nonstream_stream_and_embedding() -> None:
    with _provider() as (server, endpoint):
        credential = _Credential()
        driver = definition()
        assert driver.probe is not None and driver.discover is not None
        await driver.probe(_connection(endpoint), credential)
        discovered = await driver.discover(_connection(endpoint), credential)
        assert [(item.model, item.kind) for item in discovered] == [
            ("chat-a", ModelKind.CHAT)
        ]

        opened = await driver.open(_connection(endpoint), credential)
        chat = opened.bind_chat(_chat_descriptor(), {"max_tool_schemas": 32})
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
        assert chat.max_tool_schemas == 32

        alias_chat = opened.bind_chat(_chat_descriptor("reasoning-alias"), {})
        alias_response = await alias_chat.complete(
            ModelRequest(messages=({"role": "user", "content": "hello"},))
        )
        assert alias_response.thinking == "checked"

        tagged_chat = opened.bind_chat(_chat_descriptor("think-tags"), {})
        tagged = await tagged_chat.complete(ModelRequest(messages=()))
        assert tagged.content == "answer"
        assert tagged.thinking == "checked tags"

        tagged_deltas: list[dict[str, str]] = []

        async def on_tagged_delta(delta: dict[str, str]) -> None:
            tagged_deltas.append(delta)

        streamed_tagged = await tagged_chat.complete(
            ModelRequest(messages=(), on_delta=on_tagged_delta)
        )
        assert streamed_tagged.content == "answer"
        assert streamed_tagged.thinking == "checked tags"
        assert tagged_deltas == [
            {"thinking_delta": "checked tags"},
            {"content_delta": "answer"},
        ]

        late_deltas: list[dict[str, str]] = []

        async def on_late_delta(delta: dict[str, str]) -> None:
            late_deltas.append(delta)

        late_chat = opened.bind_chat(_chat_descriptor("think-tags-native-late"), {})
        late = await late_chat.complete(
            ModelRequest(messages=(), on_delta=on_late_delta)
        )
        assert late.content == "<think>legacy</think>answer"
        assert late.thinking == "native"
        assert late_deltas == [
            {"content_delta": "<think>legacy</think>answer"},
            {"thinking_delta": "native"},
        ]

        plain_deltas: list[dict[str, str]] = []
        plain_delta_received = asyncio.Event()

        async def on_plain_delta(delta: dict[str, str]) -> None:
            plain_deltas.append(delta)
            plain_delta_received.set()

        plain_chat = opened.bind_chat(_chat_descriptor("plain-stream"), {})
        plain_task = asyncio.create_task(
            plain_chat.complete(ModelRequest(messages=(), on_delta=on_plain_delta))
        )
        await asyncio.wait_for(plain_delta_received.wait(), 1)
        assert plain_deltas == [{"content_delta": "hello "}]
        server.release_plain_done.set()
        plain = await plain_task
        assert plain.content == "hello world"
        assert plain_deltas == [
            {"content_delta": "hello "},
            {"content_delta": "world"},
        ]

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
        assert [
            (call.id, call.name, call.arguments) for call in streamed.tool_calls
        ] == [("call-1", "search", {"q": "hi"})]
        assert streamed.usage is not None
        assert streamed.usage.input_tokens == 10
        assert streamed.usage.cached_input_tokens == 3

        embedding = opened.bind_embedding(_embedding_descriptor(), {})
        embedded = await embedding.embed(("first", "second"))
        assert embedded.vectors == (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        )
        assert embedded.usage is not None
        assert embedded.usage.coverage is UsageCoverage.PARTIAL

        credential.token = "rotated"
        _ = await chat.complete(
            ModelRequest(messages=({"role": "user", "content": "again"},))
        )
        assert server.requests[-1]["authorization"] == "Bearer rotated"
        sent = next(
            request["body"]
            for request in server.requests
            if request.get("body", {}).get("model") == "chat-a"
        )
        assert sent["reasoning_effort"] == "high"
        assert sent["messages"][0] == {"role": "system", "content": "system"}


@pytest.mark.asyncio
async def test_embedding_uses_default_batch_limit_and_preserves_order() -> None:
    with _provider() as (server, endpoint):
        opened = await definition().open(_connection(endpoint), _Credential())
        with pytest.raises(ValueError, match="embedding_batch_size"):
            opened.bind_embedding(
                _embedding_descriptor(),
                {"embedding_batch_size": 0},
            )
        embedding = opened.bind_embedding(_embedding_descriptor(), {})

        result = await embedding.embed(tuple(f"item-{index}" for index in range(23)))

        requests = [
            request["body"]["input"]
            for request in server.requests
            if request["path"] == "/v1/embeddings"
        ]
        assert [len(batch) for batch in requests] == [10, 10, 3]
        assert result.vectors == tuple((float(index), 0.0, 0.0) for index in range(23))
        assert result.usage is not None
        assert result.usage.input_tokens == 12
        assert result.usage.request_count == 3
        assert result.usage.coverage is UsageCoverage.PARTIAL


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

        leaked = opened.bind_chat(_chat_descriptor("echo-secret-error"), {})
        with pytest.raises(AuthenticationError) as caught:
            await leaked.complete(ModelRequest(messages=()))
        assert "secret" not in str(caught.value)
        assert "[REDACTED]" in str(caught.value)


@pytest.mark.asyncio
async def test_manual_gateway_does_not_require_models_catalog() -> None:
    with _provider(models_status=404) as (_server, endpoint):
        base = _connection(endpoint)
        descriptor = DriverConnectionDescriptor(
            connection_id=base.connection_id,
            name=base.name,
            driver_id=base.driver_id,
            endpoint=base.endpoint,
            auth_identity=base.auth_identity,
            config={
                "format_version": 1,
                "max_retries": 0,
                "allow_unverified_manual": True,
            },
        )
        credential = _Credential()
        await definition().probe(descriptor, credential)  # type: ignore[misc]
        opened = await definition().open(descriptor, credential)
        chat = opened.bind_chat(_chat_descriptor("manual-model"), {})
        response = await chat.complete(ModelRequest(messages=()))
        assert response.tool_calls[0].name == "lookup"
        with pytest.raises(InvalidRequestError, match="catalog unavailable"):
            await definition().discover(descriptor, credential)  # type: ignore[misc]

    with _provider() as (_server, closed_endpoint):
        closed = _connection(closed_endpoint)
    with pytest.raises(TransportError):
        await definition().probe(closed, _Credential())  # type: ignore[misc]


@pytest.mark.asyncio
async def test_discovery_has_fixed_retry_size_and_entry_limits() -> None:
    driver = definition()
    assert driver.discover is not None
    credential = _Credential()

    with _provider(model_ids=["compressed-model"], gzip_models=True) as (
        _server,
        endpoint,
    ):
        models = await driver.discover(_connection(endpoint), credential)
        assert [model.model for model in models] == ["compressed-model"]

    with _provider(models_status=429) as (server, endpoint):
        base = _connection(endpoint)
        unbounded_request = DriverConnectionDescriptor(
            connection_id=base.connection_id,
            name=base.name,
            driver_id=base.driver_id,
            endpoint=base.endpoint,
            auth_identity=base.auth_identity,
            config={
                "format_version": 1,
                "max_retries": 1_000_000,
                "connect_timeout": 1_000_000,
                "read_timeout": 1_000_000,
            },
        )
        with pytest.raises(RateLimitError):
            await driver.discover(unbounded_request, credential)
        assert [item["path"] for item in server.requests] == ["/v1/models"]

    with _provider(model_ids=[f"model-{index}" for index in range(10_001)]) as (
        _server,
        endpoint,
    ):
        with pytest.raises(TransportError, match="10000 entries"):
            await driver.discover(_connection(endpoint), credential)

    with _provider(model_ids=["x" * (4 * 1024 * 1024)]) as (_server, endpoint):
        with pytest.raises(TransportError, match="exceeds 4194304 bytes"):
            await driver.discover(_connection(endpoint), credential)

    with _provider(
        model_ids=["x" * (4 * 1024 * 1024)],
        gzip_models=True,
    ) as (_server, endpoint):
        with pytest.raises(TransportError, match="4194304 decoded bytes"):
            await driver.discover(_connection(endpoint), credential)

    for model_ids, message in (
        (["duplicate", "duplicate"], "duplicate id"),
        ([" outer-space"], "outer whitespace"),
        (["x" * 257], "longer than 256"),
    ):
        with _provider(model_ids=model_ids) as (_server, endpoint):
            with pytest.raises(TransportError, match=message):
                await driver.discover(_connection(endpoint), credential)


@pytest.mark.asyncio
async def test_persisted_config_rejects_unbounded_secret_surfaces() -> None:
    with _provider() as (_server, endpoint):
        for config in (
            {"headers": {"X-API-Key": "secret"}},
            {"extra_body": {"password": "secret"}},
        ):
            with pytest.raises(ValueError):
                await definition().open(
                    DriverConnectionDescriptor(
                        connection_id="connection-1",
                        name="OpenAI",
                        driver_id="openai-compatible",
                        endpoint=endpoint,
                        auth_identity="account-1",
                        config=config,
                    ),
                    _Credential(),
                )


@pytest.mark.asyncio
async def test_driver_is_an_installable_ordinary_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Path("plugins/openai_compatible")
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
    assert installed.plugin_name == "openai-compatible"
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

    blocked_prefixes = ("plugins.models", "plugins.openai_compatible")
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
                raise ModuleNotFoundError(
                    f"repository plugin import blocked: {fullname}"
                )
            return None

    monkeypatch.setattr(sys, "meta_path", [BlockRepositoryPlugins(), *sys.meta_path])
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
            ("openai-compatible@ordinary-test", installed.installed_path),
        ):
            generation = manager.generation(plugin_id)
            assert generation is not None and generation.source_type == "installed"
            assert (
                Path(generation.instance.module.__file__)
                .resolve()
                .is_relative_to(expected_path)
            )

        snapshot = manager.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        root = snapshot.composition_root
        lease = await manager._snapshot_store.acquire()
        token = bind_runtime_snapshot(lease)
        try:
            settings = root.context.require(MODEL_SETTINGS)
            revision = (
                await settings.apply(
                    AddConnection(
                        expected_revision=0,
                        connection_id="openai",
                        name="OpenAI",
                        driver_id="openai-compatible",
                        endpoint=endpoint,
                        auth_identity="account",
                        credential={"driver": "api_key", "access_token": "secret"},
                        driver_config={"max_retries": 0},
                    )
                )
            ).revision
            revision = (
                await settings.apply(
                    SyncModels(expected_revision=revision, connection_id="openai")
                )
            ).revision
            catalog = root.context.require(MODEL_CATALOG).snapshot()
            chat_id = next(
                model.model_id for model in catalog.models if model.model == "chat-a"
            )
            revision = (
                await settings.apply(
                    SetDefaultModel(
                        expected_revision=revision,
                        role=ModelRole.DEFAULT,
                        model_id=chat_id,
                    )
                )
            ).revision
            revision = (
                await settings.apply(
                    AddModel(
                        expected_revision=revision,
                        model_id="embedding",
                        connection_id="openai",
                        kind=ModelKind.EMBEDDING,
                        model="embedding-a",
                        capabilities=ModelCapabilities(embedding_dimensions=3),
                        capability_sources=CapabilitySources(
                            embedding_dimensions="manual"
                        ),
                    )
                )
            ).revision
            _ = await settings.apply(
                SetDefaultModel(
                    expected_revision=revision,
                    role=None,
                    model_id="embedding",
                )
            )

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
            async with root.context.require(EMBEDDINGS).bind() as embedding:
                result = await embedding.embed(("first", "second"))
                assert len(result.vectors) == 2
        finally:
            reset_runtime_snapshot(token)
            await lease.release()
            await manager.terminate_all()

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
            for model in snapshot.composition_root.context.require(MODEL_CATALOG)
            .snapshot()
            .models
        )
        await reloaded.terminate_all()

        set_installed_plugin_enabled(
            "openai-compatible@ordinary-test",
            enabled=False,
            plugins_home=tmp_path / "home",
        )
        _ = finalize_uninstall_plugin(
            "openai-compatible@ordinary-test",
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
            for model in snapshot.composition_root.context.require(MODEL_CATALOG)
            .snapshot()
            .models
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
        generation = restored.generation("openai-compatible@ordinary-test")
        assert generation is not None
        assert generation.plugin_dir == restored_install.installed_path
        snapshot = restored.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        assert all(
            model.availability is ModelAvailability.AVAILABLE
            for model in snapshot.composition_root.context.require(MODEL_CATALOG)
            .snapshot()
            .models
        )
        await restored.terminate_all()

    assert not any(name.startswith(blocked_prefixes) for name in sys.modules)
