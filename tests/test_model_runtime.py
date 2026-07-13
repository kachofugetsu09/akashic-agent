from __future__ import annotations

import base64
import inspect
import json
import os
from pathlib import Path

import pytest
import click

from agent.config import load_config
from agent.model_runtime.auth.codex import CodexAuthDriver
from agent.model_runtime.auth.store import Credential, CredentialStore
from agent.model_runtime.errors import (
    AuthenticationError,
    ContextWindowError,
    RateLimitError,
    TransportError,
)
from agent.model_runtime.fallback import ResilientLightProvider
from agent.model_runtime.catalog.codex import CodexModelCatalog
from agent.model_runtime.context_policy import (
    build_context_budget,
    recommended_memory_window,
)
from agent.model_runtime.transports.responses import (
    CodexResponsesTransport,
    _parse_usage,
    _responses_input,
    _responses_tools,
)
from agent.model_runtime.runtime import ChatRuntimeAdapter, ResponsesRuntime
from agent.model_runtime.types import (
    CapabilitySource,
    ModelCapabilities,
    ModelRequest,
    ModelUsage,
    UsageCoverage,
)
from agent.model_runtime.usage import aggregate_usage
from agent.config_models import Config, ModelRuntimeConfig
from bootstrap.providers import build_providers, build_vl_provider
from bootstrap.setup_wizard import (
    WizardAnswers,
    _persist_answer_credentials,
    _render_config,
    run_setup_wizard,
)
from session.manager import SessionManager
from session.store import _decode_message_extra


def test_recommended_memory_window_is_bounded() -> None:
    assert recommended_memory_window(32_000) == 20
    assert recommended_memory_window(64_000) == 40
    assert recommended_memory_window(128_000) == 80
    assert recommended_memory_window(272_000) == 160
    assert recommended_memory_window(400_000) == 160


def test_context_budget_reserves_output() -> None:
    capabilities = ModelCapabilities(
        context_window=100_000,
        max_output_tokens=10_000,
        effective_context_percent=0.9,
    )
    budget = build_context_budget(capabilities, 8_000)
    assert budget.effective_context == 90_000
    assert budget.input_budget == 82_000


def test_credential_store_writes_private_atomic_document(tmp_path: Path) -> None:
    path = tmp_path / "auth" / "auth.json"
    store = CredentialStore(path)
    credential = Credential(driver="codex", access_token="secret", refresh_token="rotation")

    store.put("codex_default", credential)

    assert store.get("codex_default") == credential
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["version"] == 1


def test_codex_account_id_comes_from_id_token() -> None:
    claims = {
        "https://api.openai.com/auth": {"chatgpt_account_id": "account-from-id"}
    }
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    id_token = f"header.{payload}.signature"

    credential = CodexAuthDriver._credential_from_token(
        {
            "id_token": id_token,
            "access_token": "opaque-access-token",
            "refresh_token": "refresh-token",
        }
    )

    assert credential.account_id == "account-from-id"


def test_codex_catalog_maps_reasoning_context_and_modalities() -> None:
    model = CodexModelCatalog._parse_model(
        {
            "slug": "gpt-test",
            "display_name": "GPT Test",
            "description": "test",
            "context_window": 272_000,
            "effective_context_window_percent": 90,
            "default_reasoning_level": "high",
            "supported_reasoning_levels": [
                {"effort": "medium", "description": ""},
                {"effort": "high", "description": ""},
            ],
            "input_modalities": ["text", "image"],
            "supports_parallel_tool_calls": True,
        }
    )

    assert model.capabilities.context_window == 272_000
    assert model.capabilities.input_modalities == ("text", "image")
    assert model.capabilities.supported_reasoning_efforts == ("medium", "high")
    assert model.capabilities.source is CapabilitySource.CATALOG
    assert model.input_modalities_known is True


def test_codex_catalog_uses_backend_compatible_client_version() -> None:
    class _Auth:
        pass

    catalog = CodexModelCatalog(_Auth())  # type: ignore[arg-type]

    assert catalog.client_version == "0.144.1"


@pytest.mark.asyncio
async def test_codex_catalog_only_returns_api_selectable_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Auth:
        def headers(self, *, force_refresh: bool = False) -> dict[str, str]:
            return {"Authorization": "Bearer test"}

    class _Response:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            base = {"context_window": 128_000}
            return {
                "models": [
                    {**base, "slug": "visible"},
                    {**base, "slug": "hidden", "visibility": "hide"},
                    {**base, "slug": "unsupported", "supported_in_api": False},
                ]
            }

    async def fake_get(self: object, *args: object, **kwargs: object) -> _Response:
        return _Response()

    monkeypatch.setattr("httpx.AsyncClient.get", fake_get)

    models = await CodexModelCatalog(_Auth()).list_models()  # type: ignore[arg-type]

    assert [model.slug for model in models] == ["visible"]


def test_responses_adapter_preserves_tools_and_usage() -> None:
    messages, instructions = _responses_input(
        [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
            {"role": "tool", "tool_call_id": "c1", "content": "done"},
        ],
        "identity",
    )
    tools = _responses_tools(
        [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
    )
    usage = _parse_usage(
        {
            "input_tokens": 100,
            "input_tokens_details": {"cached_tokens": 70},
            "output_tokens": 20,
            "output_tokens_details": {"reasoning_tokens": 8},
        }
    )

    assert instructions == "identity\n\nsystem"
    assert messages[-1] == {"type": "function_call_output", "call_id": "c1", "output": "done"}
    assert tools[0]["name"] == "lookup"
    assert usage is not None
    assert usage.cached_input_tokens == 70
    assert usage.reasoning_output_tokens == 8
    assert usage.total_tokens == 120


def test_codex_payload_matches_chatgpt_responses_contract() -> None:
    class _Auth:
        pass

    transport = CodexResponsesTransport(_Auth(), runtime_id="main")  # type: ignore[arg-type]
    payload = transport._build_payload(
        ModelRequest(
            messages=[{"role": "user", "content": "hello"}],
            tools=[],
            model="gpt-test",
            max_output_tokens=8192,
        )
    )

    assert "max_output_tokens" not in payload
    assert payload["tool_choice"] == "auto"
    assert payload["parallel_tool_calls"] is True


def test_codex_lite_payload_embeds_tools_and_instructions() -> None:
    class _Auth:
        pass

    transport = CodexResponsesTransport(
        _Auth(), runtime_id="main", use_responses_lite=True  # type: ignore[arg-type]
    )
    payload = transport._build_payload(
        ModelRequest(
            messages=[{"role": "user", "content": "hello"}],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
            model="gpt-test",
            max_output_tokens=8192,
            system_prompt="system",
            reasoning_effort="high",
        )
    )

    assert payload["instructions"] == ""
    assert "tools" not in payload
    assert payload["input"][0]["type"] == "additional_tools"
    assert payload["input"][1]["role"] == "developer"
    assert payload["parallel_tool_calls"] is False
    assert payload["reasoning"]["context"] == "all_turns"


def test_codex_lite_keeps_reasoning_context_without_explicit_effort() -> None:
    class _Auth:
        pass

    transport = CodexResponsesTransport(
        _Auth(),  # type: ignore[arg-type]
        runtime_id="main",
        use_responses_lite=True,
        reasoning_summary="auto",
    )
    payload = transport._build_payload(
        ModelRequest(messages=[], tools=[], model="gpt-test", max_output_tokens=8192)
    )

    assert payload["reasoning"] == {"summary": "auto", "context": "all_turns"}


def test_codex_named_tool_choice_filters_and_requires_tool() -> None:
    class _Auth:
        pass

    transport = CodexResponsesTransport(_Auth(), runtime_id="main")  # type: ignore[arg-type]
    payload = transport._build_payload(
        ModelRequest(
            messages=[],
            tools=[
                {"type": "function", "function": {"name": "first"}},
                {"type": "function", "function": {"name": "finish"}},
            ],
            model="gpt-test",
            max_output_tokens=100,
            tool_choice={"type": "function", "function": {"name": "finish"}},
        )
    )

    assert payload["tool_choice"] == "required"
    assert [tool["name"] for tool in payload["tools"]] == ["finish"]


def test_responses_transport_only_owns_network_timeouts() -> None:
    class _Auth:
        pass

    transport = CodexResponsesTransport(
        _Auth(),  # type: ignore[arg-type]
        runtime_id="main",
        connect_timeout_s=11,
        read_timeout_s=22,
        write_timeout_s=33,
        pool_timeout_s=44,
    )

    assert transport.network_timeout.connect == 11
    assert transport.network_timeout.read == 22
    assert transport.network_timeout.write == 33
    assert transport.network_timeout.pool == 44
    assert "wait_for" not in inspect.getsource(CodexResponsesTransport)


def test_new_runtime_config_loads_and_roles_can_reuse_main(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[llm]
main = "codex_main"
fast = "codex_main"
agent = "codex_main"
vl = "codex_main"

[llm.runtimes.codex_main]
provider = "codex"
auth = "codex_default"
model = "gpt-test"
reasoning_effort = "high"
context_window = 272000
max_output_tokens = 8192
input_modalities = ["text", "image"]

[agent]
system_prompt = "test"

[agent.context]
memory_window = 160
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.provider == "codex"
    assert config.runtime_id == "codex_main"
    assert config.auth_id == "codex_default"
    assert config.context_window == 272_000
    assert config.multimodal is True
    assert config.light_model == ""


def test_runtime_config_rejects_unknown_reasoning_summary(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[llm]
main = "codex_main"

[llm.runtimes.codex_main]
provider = "codex"
auth = "codex_default"
model = "gpt-test"
reasoning_summary = "verbose"
context_window = 128000

[agent]
system_prompt = "test"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reasoning_summary"):
        load_config(path)


def test_session_boundary_rejects_invalid_continuation_state() -> None:
    payload = json.dumps(
        {
            "model_state": {
                "schema_version": 2,
                "runtime_id": "main",
                "transport": "responses",
                "model": "gpt-test",
                "items": [],
            }
        }
    )
    with pytest.raises(ValueError, match="schema_version"):
        _decode_message_extra(payload, "session:1")


@pytest.mark.asyncio
async def test_responses_stream_builds_tool_usage_and_reasoning_state() -> None:
    class _Auth:
        pass

    async def events():
        yield {"type": "response.reasoning_summary_text.delta", "delta": "分析"}
        yield {"type": "response.output_text.delta", "delta": "结果"}
        yield {
            "type": "response.output_item.done",
            "item": {
                "type": "reasoning",
                "id": "r1",
                "status": "completed",
                "summary": [],
                "encrypted_content": "opaque",
            },
        }
        yield {
            "type": "response.output_item.done",
            "item": {
                "type": "function_call",
                "call_id": "c1",
                "name": "lookup",
                "arguments": '{"q":"x"}',
            },
        }
        yield {
            "type": "response.completed",
            "response": {
                "usage": {
                    "input_tokens": 50,
                    "input_tokens_details": {"cached_tokens": 20},
                    "output_tokens": 10,
                }
            },
        }

    transport = CodexResponsesTransport(_Auth(), runtime_id="main")  # type: ignore[arg-type]
    result = await transport._consume_stream(
        events(),
        ModelRequest(messages=[], tools=[], model="gpt-test", max_output_tokens=100),
    )

    assert result.content == "结果"
    assert result.thinking == "分析"
    assert result.tool_calls[0].arguments == {"q": "x"}
    assert result.cache_hit_tokens == 20
    assert result.continuation is not None
    assert result.continuation.items == (
        {"type": "reasoning", "summary": [], "encrypted_content": "opaque"},
    )


@pytest.mark.asyncio
async def test_responses_stream_rejects_eof_before_completed() -> None:
    class _Auth:
        pass

    async def events():
        yield {"type": "response.output_text.delta", "delta": "partial"}

    transport = CodexResponsesTransport(_Auth(), runtime_id="main")  # type: ignore[arg-type]
    with pytest.raises(TransportError, match="completed 事件前断流"):
        await transport._consume_stream(
            events(),
            ModelRequest(messages=[], tools=[], model="gpt-test", max_output_tokens=100),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "error_type"),
    [
        ("context_length_exceeded", ContextWindowError),
        ("rate_limit_exceeded", RateLimitError),
    ],
)
async def test_responses_stream_classifies_terminal_errors(
    code: str, error_type: type[Exception]
) -> None:
    class _Auth:
        pass

    async def events():
        yield {
            "type": "response.failed",
            "response": {"error": {"code": code, "message": "failed"}},
        }

    transport = CodexResponsesTransport(_Auth(), runtime_id="main")  # type: ignore[arg-type]
    with pytest.raises(error_type):
        await transport._consume_stream(
            events(),
            ModelRequest(messages=[], tools=[], model="gpt-test", max_output_tokens=100),
        )


def _state(runtime: str, model: str, item_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "runtime_id": runtime,
        "transport": "responses",
        "model": model,
        "items": [{"type": "reasoning", "id": item_id, "encrypted_content": "opaque"}],
    }


def test_session_round_trip_restores_tool_and_final_continuation(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    session = manager.get_or_create("chat:1")
    session.add_message("user", "hello")
    session.add_message(
        "assistant",
        "final",
        tool_chain=[
            {
                "text": None,
                "calls": [
                    {"call_id": "c1", "name": "lookup", "arguments": {}, "result": "ok"}
                ],
                "model_state": _state("main", "gpt-test", "tool-reasoning"),
            },
            {
                "text": "next",
                "calls": [
                    {"call_id": "c2", "name": "lookup", "arguments": {"n": 2}, "result": "ok2"}
                ],
                "model_state": _state("main", "gpt-test", "tool-reasoning-2"),
            },
        ],
        model_state=_state("main", "gpt-test", "final-reasoning"),
    )
    manager.save(session)
    manager.close()

    reloaded = SessionManager(tmp_path)
    history = reloaded.get_or_create("chat:1").get_history()
    reloaded.close()

    assistants = [item for item in history if item["role"] == "assistant"]
    assert assistants[0]["model_state"] == _state("main", "gpt-test", "tool-reasoning")
    assert assistants[1]["model_state"] == _state("main", "gpt-test", "tool-reasoning-2")
    assert assistants[-1]["model_state"] == _state("main", "gpt-test", "final-reasoning")


def test_responses_replays_only_matching_runtime_transport_and_model() -> None:
    matching = _state("main", "gpt-test", "keep")
    foreign = _state("other", "gpt-test", "drop")
    messages = [
        {"role": "assistant", "content": "one", "model_state": matching},
        {"role": "assistant", "content": "two", "model_state": foreign},
    ]

    converted, _ = _responses_input(messages, "", runtime_id="main", model="gpt-test")

    reasoning_items = [item for item in converted if item.get("type") == "reasoning"]
    assert reasoning_items == [{"type": "reasoning", "encrypted_content": "opaque"}]
    assert [item["content"] for item in converted if item.get("role") == "assistant"] == [
        "one",
        "two",
    ]


def test_responses_converts_chat_multimodal_blocks_by_role() -> None:
    converted, _ = _responses_input(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "看图"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA", "detail": "high"}},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "看到了"}]},
        ],
        "",
        runtime_id="main",
        model="gpt-test",
    )

    assert converted[0]["content"] == [
        {"type": "input_text", "text": "看图"},
        {"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "high"},
    ]
    assert converted[1]["content"] == [{"type": "output_text", "text": "看到了"}]


@pytest.mark.asyncio
async def test_chat_runtime_adapter_uses_canonical_request_contract() -> None:
    class _Chat:
        async def chat(self, **kwargs):
            self.kwargs = kwargs
            from agent.model_runtime.types import LLMResponse

            return LLMResponse(content="ok")

    implementation = _Chat()
    request = ModelRequest(
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        model="chat-model",
        max_output_tokens=100,
    )
    result = await ChatRuntimeAdapter(implementation).send(request)
    assert result.content == "ok"
    assert implementation.kwargs["model"] == "chat-model"


def test_distinct_named_roles_build_their_own_runtime() -> None:
    api_runtime = ModelRuntimeConfig(
        runtime_id="fast_api",
        provider="openai",
        model="fast-model",
        api_key="fast-key",
        base_url="https://example.test/v1",
        context_window=64_000,
    )
    codex_runtime = ModelRuntimeConfig(
        runtime_id="vl_codex",
        provider="codex",
        model="codex-vl",
        auth="codex_other",
        context_window=128_000,
        input_modalities=("text", "image"),
    )
    config = Config(
        provider="openai",
        model="main",
        api_key="main-key",
        system_prompt="system",
        light_model="fast-model",
        vl_model="codex-vl",
        multimodal=False,
        model_runtimes={"fast_api": api_runtime, "vl_codex": codex_runtime},
        fast_runtime_id="fast_api",
        vl_runtime_id="vl_codex",
    )

    _, light, _ = build_providers(config)
    vl = build_vl_provider(config)

    assert isinstance(light, ResilientLightProvider)
    assert isinstance(light.primary._runtime, ChatRuntimeAdapter)
    assert vl is not None and isinstance(vl._runtime, ResponsesRuntime)


def test_missing_named_role_reference_fails_loud(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[llm]
main = "main"
fast = "missing"
[llm.runtimes.main]
provider = "openai"
model = "main"
api_key = "key"
context_window = 64000
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="llm.fast 引用不存在"):
        load_config(path)


def test_setup_render_never_contains_new_api_key_secrets() -> None:
    answers = WizardAnswers(
        provider="openai",
        model="main",
        api_key="main-secret",
        auth_id="main_default",
        base_url="https://example.test/v1",
        context_window=64_000,
        fast_model="fast",
        fast_api_key="fast-secret",
        fast_auth_id="fast_default",
        fast_base_url="https://example.test/v1",
        vl_model="vision",
        vl_api_key="vl-secret",
        vl_auth_id="vl_default",
        vl_base_url="https://example.test/v1",
        embed_model="embed",
        embed_api_key="embed-secret",
        embed_auth_id="embedding_default",
        embed_base_url="https://example.test/v1",
    )
    rendered = _render_config(answers)
    for secret in ("main-secret", "fast-secret", "vl-secret", "embed-secret"):
        assert secret not in rendered


def test_setup_cancel_before_completion_does_not_persist_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import bootstrap.setup_wizard as wizard

    persisted = False

    def abort_collection():
        raise click.Abort()

    def mark_persisted(_: WizardAnswers) -> None:
        nonlocal persisted
        persisted = True

    monkeypatch.setattr(wizard, "_collect_answers", abort_collection)
    monkeypatch.setattr(wizard, "_persist_answer_credentials", mark_persisted)
    with pytest.raises(click.Abort):
        run_setup_wizard(tmp_path / "config.toml", tmp_path / "workspace")
    assert persisted is False


def test_credential_store_rejects_world_readable_token_file(tmp_path: Path) -> None:
    parent = tmp_path / "auth"
    parent.mkdir(mode=0o700)
    path = parent / "auth.json"
    path.write_text('{"version":1,"credentials":{}}', encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(AuthenticationError, match="权限过宽"):
        CredentialStore(path).get("missing")


def test_credential_store_wraps_corrupt_json_with_path(tmp_path: Path) -> None:
    parent = tmp_path / "auth"
    parent.mkdir(mode=0o700)
    path = parent / "auth.json"
    path.write_text("{broken", encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(AuthenticationError, match=rf"JSON 损坏: {path}"):
        CredentialStore(path).get("missing")


def test_named_runtime_rejects_unknown_provider(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[llm]
main = "main"
[llm.runtimes.main]
provider = "opneai"
model = "main"
auth = "main_default"
context_window = 64000
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="provider 不受支持.*opneai"):
        load_config(path)


def test_legacy_role_auth_references_load_api_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    CredentialStore().put_many(
        {
            "fast_default": Credential(driver="api_key", access_token="fast-secret"),
            "vl_default": Credential(driver="api_key", access_token="vl-secret"),
        }
    )
    path = tmp_path / "config.toml"
    path.write_text(
        """
[llm]
provider = "openai"
[llm.main]
model = "main"
api_key = "main-secret"
base_url = "https://main.example/v1"
context_window = 64000
multimodal = false
[llm.fast]
model = "fast"
auth = "fast_default"
base_url = "https://fast.example/v1"
[llm.vl]
model = "vision"
auth = "vl_default"
base_url = "https://vl.example/v1"
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.light_api_key == "fast-secret"
    assert config.vl_api_key == "vl-secret"


def test_setup_api_key_named_runtimes_load_and_build_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    answers = WizardAnswers(
        provider="openai",
        model="main-model",
        api_key="main-secret",
        auth_id="main_default",
        base_url="https://main.example/v1",
        context_window=64_000,
        fast_model="fast-model",
        fast_api_key="fast-secret",
        fast_auth_id="fast_default",
        fast_base_url="https://fast.example/v1",
        vl_model="vl-model",
        vl_api_key="vl-secret",
        vl_auth_id="vl_default",
        vl_base_url="https://vl.example/v1",
        embed_model="embed-model",
        embed_api_key="embed-secret",
        embed_auth_id="embedding_default",
        embed_base_url="https://embed.example/v1",
    )
    _persist_answer_credentials(answers)
    rendered = _render_config(answers)
    path = tmp_path / "config.toml"
    path.write_text(rendered, encoding="utf-8")

    config = load_config(path)
    main, fast, _ = build_providers(config)
    vl = build_vl_provider(config)

    assert config.runtime_id == "main"
    assert config.fast_runtime_id == "fast"
    assert config.vl_runtime_id == "vl"
    assert (
        config.model_runtimes["main"].provider,
        config.model_runtimes["main"].auth,
        config.model_runtimes["main"].api_key,
        config.model_runtimes["main"].base_url,
    ) == ("openai", "main_default", "main-secret", "https://main.example/v1")
    assert (
        config.model_runtimes["fast"].provider,
        config.model_runtimes["fast"].auth,
        config.model_runtimes["fast"].base_url,
    ) == ("openai", "fast_default", "https://fast.example/v1")
    assert (
        config.model_runtimes["vl"].provider,
        config.model_runtimes["vl"].auth,
        config.model_runtimes["vl"].base_url,
    ) == ("openai", "vl_default", "https://vl.example/v1")
    assert config.model_runtimes["vl"].input_modalities == ("text", "image")
    assert isinstance(main._runtime, ChatRuntimeAdapter)
    assert main._runtime.implementation._provider_name == "openai"
    assert isinstance(fast, ResilientLightProvider)
    assert isinstance(fast.primary._runtime, ChatRuntimeAdapter)
    assert vl is not None
    assert isinstance(vl._runtime, ChatRuntimeAdapter)
    assert fast.primary._runtime.implementation._base_url == "https://fast.example/v1"
    assert vl._runtime.implementation._base_url == "https://vl.example/v1"
    for secret in ("main-secret", "fast-secret", "vl-secret", "embed-secret"):
        assert secret not in rendered


def test_setup_codex_main_with_api_vl_loads_correct_transports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    CredentialStore().put(
        "codex_default",
        Credential(driver="codex", access_token="codex-secret"),
    )
    answers = WizardAnswers(
        provider="codex",
        model="gpt-codex",
        auth_id="codex_default",
        base_url="https://chatgpt.com/backend-api/codex",
        reasoning_effort="high",
        context_window=128_000,
        vl_model="vl-model",
        vl_api_key="vl-secret",
        vl_auth_id="vl_default",
        vl_base_url="https://vl.example/v1",
        embed_model="embed-model",
        embed_api_key="embed-secret",
        embed_auth_id="embedding_default",
        embed_base_url="https://embed.example/v1",
    )
    _persist_answer_credentials(answers)
    rendered = _render_config(answers)
    path = tmp_path / "config.toml"
    path.write_text(rendered, encoding="utf-8")

    config = load_config(path)
    main, fast, _ = build_providers(config)
    vl = build_vl_provider(config)

    assert (
        config.model_runtimes["main"].provider,
        config.model_runtimes["main"].auth,
        config.model_runtimes["main"].base_url,
    ) == (
        "codex",
        "codex_default",
        "https://chatgpt.com/backend-api/codex",
    )
    assert (
        config.model_runtimes["vl"].provider,
        config.model_runtimes["vl"].auth,
        config.model_runtimes["vl"].base_url,
    ) == ("openai", "vl_default", "https://vl.example/v1")
    assert config.model_runtimes["vl"].api_key == "vl-secret"
    assert isinstance(main._runtime, ResponsesRuntime)
    assert main._runtime.transport.base_url == "https://chatgpt.com/backend-api/codex"
    assert fast is None
    assert vl is not None
    assert isinstance(vl._runtime, ChatRuntimeAdapter)
    assert vl._runtime.implementation._base_url == "https://vl.example/v1"
    assert "vl-secret" not in rendered


def test_usage_aggregation_reports_partial_coverage_without_zero_filling() -> None:
    usage = aggregate_usage(
        [
            ModelUsage(
                input_tokens=100,
                output_tokens=20,
                request_count=1,
                covered_request_count=1,
                coverage=UsageCoverage.EXACT,
            ),
            ModelUsage(request_count=1),
        ]
    )
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.request_count == 2
    assert usage.covered_request_count == 1
    assert usage.coverage is UsageCoverage.PARTIAL
