from __future__ import annotations

import base64
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agent.config import load_config
from agent.config_models import ModelRuntimeConfig
from agent.model_runtime.auth.codex import CODEX_CLIENT_VERSION, CodexAuthDriver
from agent.model_runtime.auth.store import Credential, CredentialStore
from agent.model_runtime.catalog.codex import CodexModelCatalog
from agent.model_runtime.catalog.opencode_go import (
    OpenCodeGoModelCatalog,
    _parse_opencode_go_reasoning_efforts,
)
from agent.model_runtime.context_policy import (
    build_runtime_context_budget,
)
from agent.model_runtime.errors import (
    ContextWindowError,
    QuotaError,
    RateLimitError,
    RetryableTransportError,
    TransportError,
)
from agent.model_runtime.transports.responses import (
    CodexResponsesTransport,
    _parse_usage,
    _responses_input,
)
from agent.model_runtime.types import ModelRequest, ModelUsage, UsageCoverage
from agent.model_runtime.usage import aggregate_usage
from agent.provider import LLMProvider, _assemble_chat_messages
from bootstrap.setup_wizard import WizardAnswers, _persist_answer_credentials, _render_config
from session.store import _decode_message_extra


class _Auth:
    def headers(self, *, force_refresh: bool = False) -> dict[str, str]:
        return {"Authorization": "Bearer test", "ChatGPT-Account-ID": "account"}


def _transport(**kwargs: object) -> CodexResponsesTransport:
    return CodexResponsesTransport(_Auth(), runtime_id="main", **kwargs)  # type: ignore[arg-type]


def _request(**kwargs: object) -> ModelRequest:
    values = {
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [],
        "model": "gpt-test",
        "max_output_tokens": 8192,
    }
    values.update(kwargs)
    return ModelRequest(**values)  # type: ignore[arg-type]


def test_context_budget_and_runtime_config_share_the_same_boundary() -> None:
    budget = build_runtime_context_budget(100_000, 8_000)
    assert (budget.effective_context, budget.input_budget) == (100_000, 92_000)
    uncapped = build_runtime_context_budget(100_000, 0)
    assert (uncapped.input_budget, uncapped.reserved_output) == (100_000, 0)
    assert ModelRuntimeConfig(
        runtime_id="uncapped",
        provider="openai",
        model="model",
        context_window=10_000,
        max_output_tokens=0,
    ).max_output_tokens == 0
    assert ModelRuntimeConfig(
        runtime_id="default",
        provider="openai",
        model="model",
        context_window=10_000,
    ).max_output_tokens == 0
    with pytest.raises(ValueError, match="不能小于 0"):
        ModelRuntimeConfig(
            runtime_id="negative",
            provider="openai",
            model="model",
            context_window=10_000,
            max_output_tokens=-1,
        )
    with pytest.raises(ValueError, match="max_output_tokens 必须小于 context_window"):
        ModelRuntimeConfig(
            runtime_id="bad",
            provider="openai",
            model="model",
            context_window=10_000,
            max_output_tokens=10_000,
        )


def test_provider_estimator_matches_chat_system_message_assembly() -> None:
    provider = LLMProvider(
        api_key="test",
        system_prompt="runtime system " * 40,
    )
    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "请看图"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,AAAA", "detail": "low"},
            },
        ],
    }
    explicit_system = {"role": "system", "content": "caller system"}

    without_system = provider.estimate_context_tokens([user_message], tools)
    with_explicit_system = provider.estimate_context_tokens(
        [explicit_system, user_message],
        tools,
    )
    caller_owned = LLMProvider(api_key="test").estimate_context_tokens(
        [explicit_system, user_message],
        tools,
    )

    assert _assemble_chat_messages(
        "runtime system", [explicit_system, user_message]
    ) == [
        {"role": "system", "content": "caller system"},
        user_message,
    ]
    assert _assemble_chat_messages("runtime system", [user_message])[0] == {
        "role": "system",
        "content": "runtime system",
    }
    assert with_explicit_system == caller_owned
    assert without_system > with_explicit_system
    assert provider.estimate_context_tokens([user_message], []) < without_system


def test_opencode_go_profile_is_dynamic_and_rejects_wrong_wire() -> None:
    runtime = ModelRuntimeConfig(
        runtime_id="main",
        provider="opencode-go",
        model="glm-5.99",
        context_window=64_000,
    )
    assert runtime.model == "glm-5.99"

    with pytest.raises(ValueError, match="Messages API"):
        ModelRuntimeConfig(
            runtime_id="main",
            provider="opencode-go",
            model="qwen3.5-plus",
            context_window=64_000,
        )
    future = ModelRuntimeConfig(
        runtime_id="main",
        provider="opencode-go",
        model="future-model-1",
        context_window=64_000,
    )
    assert future.model == "future-model-1"
    with pytest.raises(ValueError, match="仅支持 input_modalities"):
        ModelRuntimeConfig(
            runtime_id="vl",
            provider="opencode-go",
            model="grok-4.5",
            context_window=64_000,
            input_modalities=("text", "image"),
        )


def test_opencode_go_reasoning_efforts_follow_verbose_catalog() -> None:
    output = """opencode-go/deepseek-v4-pro
{
  "id": "deepseek-v4-pro",
  "variants": {
    "high": {"reasoningEffort": "high"},
    "max": {"reasoningEffort": "max"}
  }
}
opencode-go/kimi-k3
{
  "id": "kimi-k3",
  "variants": {}
}
"""

    assert _parse_opencode_go_reasoning_efforts(output) == {
        "deepseek-v4-pro": ("high", "max"),
        "kimi-k3": (),
    }


@pytest.mark.asyncio
async def test_opencode_go_catalog_uses_http_boundary_and_opencode_variants(
    monkeypatch,
) -> None:
    requests: list[tuple[str, str | None]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append((self.path, self.headers.get("Authorization")))
            body = json.dumps(
                {
                    "object": "list",
                    "data": [
                        {"id": "glm-5.99", "object": "model"},
                        {"id": "kimi-k3", "object": "model"},
                        {"id": "qwen3.5-plus", "object": "model"},
                        {"id": "future-model-1", "object": "model"},
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    async def fake_reasoning_efforts(_executable: str):
        return {
            "glm-5.99": ("high", "max"),
            "future-model-1": ("low", "medium", "high"),
        }

    monkeypatch.setattr(
        "agent.model_runtime.catalog.opencode_go._load_opencode_go_reasoning_efforts",
        fake_reasoning_efforts,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host = server.server_address[0]
        port = server.server_address[1]
        models = await OpenCodeGoModelCatalog(
            "secret",
            base_url=f"http://{host}:{port}/v1",
        ).list_models()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    by_slug = {model.slug: model for model in models}
    assert [model.slug for model in models] == [
        "glm-5.99",
        "kimi-k3",
        "future-model-1",
    ]
    assert by_slug["glm-5.99"].supported_reasoning_efforts == ("high", "max")
    assert by_slug["kimi-k3"].supported_reasoning_efforts == ()
    assert by_slug["future-model-1"].supported_reasoning_efforts == (
        "low",
        "medium",
        "high",
    )
    assert requests == [("/v1/models", "Bearer secret")]


def test_credential_store_is_atomic_private_and_fail_loud(tmp_path: Path) -> None:
    path = tmp_path / "auth" / "auth.json"
    store = CredentialStore(path)
    credential = Credential(driver="codex", access_token="secret", refresh_token="rotation")

    store.put("codex_default", credential)
    assert store.get("codex_default") == credential
    assert os.stat(path).st_mode & 0o777 == 0o600

    path.write_text("{broken", encoding="utf-8")
    with pytest.raises(Exception, match="JSON 损坏"):
        store.get("codex_default")


def test_codex_token_and_catalog_metadata_are_resolved_once() -> None:
    claims = {"https://api.openai.com/auth": {"chatgpt_account_id": "account"}}
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    credential = CodexAuthDriver._credential_from_token({
        "id_token": f"header.{payload}.signature",
        "access_token": "access",
        "refresh_token": "refresh",
    })
    model = CodexModelCatalog._parse_model({
        "slug": "gpt-test",
        "context_window": 272_000,
        "max_context_window": 1_000_000,
        "effective_context_window_percent": 90,
        "default_reasoning_level": "high",
        "supported_reasoning_levels": [{"effort": "medium"}, {"effort": "high"}],
        "input_modalities": ["text", "image"],
        "supports_parallel_tool_calls": True,
        "supports_reasoning_summary_parameter": True,
    })

    assert credential.account_id == "account"
    assert model.capabilities.context_window == 272_000
    assert model.capabilities.max_context_window == 1_000_000
    assert model.capabilities.supported_reasoning_efforts == ("medium", "high")
    assert model.capabilities.input_modalities == ("text", "image")
    assert model.capabilities.supports_reasoning_summaries is True
    assert CodexModelCatalog(_Auth()).client_version == CODEX_CLIENT_VERSION  # type: ignore[arg-type]


def test_codex_refresh_uses_json_and_preserves_rotation_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CredentialStore(tmp_path / "auth.json")
    store.put("codex_default", Credential(
        driver="codex",
        access_token="old",
        refresh_token="rotation",
        account_id="account",
    ))
    captured: dict[str, object] = {}

    class Response:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {"access_token": "new", "expires_in": 3600}

    def post(url: str, **kwargs: object) -> Response:
        captured.update({"url": url, **kwargs})
        return Response()

    monkeypatch.setattr("agent.model_runtime.auth.codex.httpx.post", post)
    refreshed = CodexAuthDriver(store, "codex_default").refresh()

    assert captured["json"] == {
        "grant_type": "refresh_token",
        "refresh_token": "rotation",
        "client_id": "app_EMoamEEZ73f0CkXaXp7hrann",
    }
    assert refreshed.refresh_token == "rotation"


def test_responses_payload_variants_keep_one_internal_contract() -> None:
    standard = _transport(reasoning_summary="auto")
    payload = standard._build_payload(_request(
        tools=[
            {"type": "function", "function": {"name": "first"}},
            {"type": "function", "function": {"name": "finish"}},
        ],
        tool_choice={"type": "function", "function": {"name": "finish"}},
        reasoning_effort="xhigh",
    ))
    lite = _transport(use_responses_lite=True, reasoning_summary="auto")
    lite_payload = lite._build_payload(_request(
        tools=[{"type": "function", "function": {"name": "lookup"}}],
        system_prompt="system",
    ))

    assert "max_output_tokens" not in payload
    assert payload["tool_choice"] == "required"
    assert [tool["name"] for tool in payload["tools"]] == ["finish"]
    assert payload["reasoning"] == {"effort": "xhigh", "summary": "auto"}
    assert payload["extra_body"]["client_metadata"]["thread-id"] == standard.thread_id
    assert lite_payload["instructions"] == ""
    assert "tools" not in lite_payload
    assert lite_payload["input"][0]["type"] == "additional_tools"
    assert lite_payload["reasoning"] == {"summary": "auto", "context": "all_turns"}


def _state(runtime: str, model: str, item_id: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "runtime_id": runtime,
        "transport": "responses",
        "model": model,
        "items": [{"type": "reasoning", "id": item_id, "encrypted_content": "opaque"}],
    }


def test_responses_input_preserves_matching_state_tools_and_multimodal() -> None:
    converted, instructions = _responses_input([
        {"role": "system", "content": "rules"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "看图"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
            ],
        },
        {"role": "assistant", "content": "one", "model_state": _state("main", "gpt-test", "keep")},
        {"role": "assistant", "content": "two", "model_state": _state("other", "gpt-test", "drop")},
        {"role": "tool", "tool_call_id": "c1", "content": "done"},
    ], "identity", runtime_id="main", model="gpt-test")

    assert instructions == "identity\n\nrules"
    assert converted[0]["content"][1]["type"] == "input_image"
    assert [item for item in converted if item.get("type") == "reasoning"] == [
        {"type": "reasoning", "encrypted_content": "opaque"}
    ]
    assert converted[-1] == {
        "type": "function_call_output",
        "call_id": "c1",
        "output": "done",
    }


@pytest.mark.asyncio
async def test_responses_stream_builds_text_tools_usage_and_state() -> None:
    deltas: list[dict[str, str]] = []

    async def events():
        yield {"type": "response.reasoning_summary_text.delta", "delta": "分析"}
        yield {"type": "response.output_text.delta", "delta": "结"}
        yield {"type": "response.output_text.done", "text": "结果"}
        yield {"type": "response.output_item.done", "item": {
            "type": "reasoning", "status": "completed", "summary": [],
            "encrypted_content": "opaque",
        }}
        yield {"type": "response.output_item.done", "item": {
            "type": "function_call", "call_id": "c1", "name": "lookup",
            "arguments": '{"q":"x"}',
        }}
        yield {"type": "response.completed", "response": {"usage": {
            "input_tokens": 50,
            "input_tokens_details": {"cached_tokens": 20},
            "output_tokens": 10,
        }}}

    async def on_delta(delta: dict[str, str]) -> None:
        deltas.append(delta)

    result = await _transport()._consume_stream(events(), _request(on_delta=on_delta))

    assert (result.content, result.thinking) == ("结果", "分析")
    assert result.tool_calls[0].arguments == {"q": "x"}
    assert result.cache_hit_tokens == 20
    assert result.provider_fields["model_state"]["items"] == [
        {"type": "reasoning", "summary": [], "encrypted_content": "opaque"}
    ]
    assert deltas == [
        {"thinking_delta": "分析"},
        {"content_delta": "结"},
        {"content_delta": "果"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("context_length_exceeded", ContextWindowError),
        ("rate_limit_exceeded", RateLimitError),
        ("insufficient_quota", QuotaError),
        ("server_error", RetryableTransportError),
        ("invalid_prompt", TransportError),
    ],
)
async def test_responses_stream_classifies_terminal_errors(
    code: str, error: type[Exception]
) -> None:
    async def events():
        yield {"type": "response.failed", "response": {
            "error": {"code": code, "message": "failed"}
        }}

    with pytest.raises(error):
        await _transport()._consume_stream(events(), _request())


@pytest.mark.asyncio
async def test_responses_stream_fails_on_incomplete_eof() -> None:
    async def events():
        yield {"type": "response.output_text.delta", "delta": "partial"}

    with pytest.raises(RetryableTransportError, match="completed 事件前断流"):
        await _transport()._consume_stream(events(), _request())


def test_session_boundary_owns_model_state_validation() -> None:
    payload = json.dumps({"model_state": {
        "schema_version": 2,
        "runtime_id": "main",
        "transport": "responses",
        "model": "gpt-test",
        "items": [],
    }})
    with pytest.raises(ValueError, match="schema_version"):
        _decode_message_extra(payload, "session:1")


def test_named_runtime_config_and_setup_keep_secrets_out_of_toml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    answers = WizardAnswers(
        provider="deepseek",
        model="deepseek-chat",
        api_key="main-secret",
        auth_id="main_default",
        base_url="https://api.deepseek.com/v1",
        context_window=64_000,
        max_output_tokens=4096,
        embed_model="embedding",
        embed_api_key="embed-secret",
        embed_auth_id="embedding_default",
        embed_base_url="https://embed.example/v1",
    )
    _persist_answer_credentials(answers)
    rendered = _render_config(answers)
    path = tmp_path / "config.toml"
    path.write_text(rendered, encoding="utf-8")
    config = load_config(path, workspace=tmp_path)

    assert "main-secret" not in rendered
    assert config.model_runtimes["main"].provider == "deepseek"
    assert config.model_runtimes["main"].api_key == "main-secret"
    assert config.memory_window == 40
    assert config.context_compaction.keep_recent_tokens == 20_000


def test_config_multimodal_uses_main_runtime_and_keeps_other_runtime_isolated(
    tmp_path: Path,
) -> None:
    template = """
[llm]
main = "{main}"

[llm.runtimes.main]
provider = "openai"
model = "text-model"
api_key = "key"
context_window = 64000
max_output_tokens = 4096

[llm.runtimes.vl]
provider = "openai"
model = "vision-model"
api_key = "key"
context_window = 64000
max_output_tokens = 4096
input_modalities = ["text", "image"]
"""

    text_path = tmp_path / "text.toml"
    text_path.write_text(template.format(main="main"), encoding="utf-8")
    text_config = load_config(text_path, workspace=tmp_path)
    assert text_config.multimodal is False
    assert text_config.model_runtimes["vl"].input_modalities == ("text", "image")

    image_path = tmp_path / "image.toml"
    image_path.write_text(template.format(main="vl"), encoding="utf-8")
    image_config = load_config(image_path, workspace=tmp_path)
    assert image_config.multimodal is True
    assert image_config.provider == "openai"
    assert image_config.model == "vision-model"


def test_config_preserves_explicit_uncapped_output(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[llm]
main = "main"

[llm.runtimes.main]
provider = "deepseek"
model = "deepseek-v4-flash"
api_key = "key"
context_window = 1000000
max_output_tokens = 0
""",
        encoding="utf-8",
    )

    config = load_config(path, workspace=tmp_path)

    assert config.max_tokens == 0
    assert config.model_runtimes["main"].max_output_tokens == 0


def test_config_defaults_missing_output_limit_to_provider_boundary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[llm]
main = "main"

[llm.runtimes.main]
provider = "openai"
model = "model"
api_key = "key"
context_window = 64000
""",
        encoding="utf-8",
    )

    config = load_config(path, workspace=tmp_path)

    assert config.max_tokens == 0
    assert config.model_runtimes["main"].max_output_tokens == 0
    assert config.context_compaction.keep_recent_tokens == 20_000


def test_config_accepts_agent_compaction_policy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[llm]
main = "main"

[llm.runtimes.main]
provider = "openai"
model = "model"
api_key = "key"
context_window = 64000

[agent.context.compaction]
trigger_percent = 0.7
keep_recent_tokens = 21000
""",
        encoding="utf-8",
    )

    config = load_config(path, workspace=tmp_path)

    assert config.context_compaction.keep_recent_tokens == 21000


@pytest.mark.parametrize("trigger", [0, -0.1, 0.9, 1.0])
def test_config_ignores_legacy_compaction_trigger(
    tmp_path: Path,
    trigger: float,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        f"""
[llm]
main = "main"

[llm.runtimes.main]
provider = "openai"
model = "model"
api_key = "key"
context_window = 64000
effective_context_percent = 0.9
compaction_trigger_percent = {trigger}
""",
        encoding="utf-8",
    )

    config = load_config(path, workspace=tmp_path)

    assert config.context_compaction.keep_recent_tokens == 20_000


def test_config_preserves_explicit_legacy_output_limit_for_main_runtime(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[llm]
main = "main"

[llm.runtimes.main]
provider = "openai"
model = "model"
api_key = "key"
context_window = 64000

[agent]
max_tokens = 8192
""",
        encoding="utf-8",
    )

    config = load_config(path, workspace=tmp_path)

    assert config.max_tokens == 8192
    assert config.model_runtimes["main"].max_output_tokens == 8192


@pytest.mark.parametrize("modalities", ['"image"', "[1]"])
def test_config_rejects_invalid_runtime_modalities_before_config_build(
    tmp_path: Path,
    modalities: str,
) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text(
        f"""
[llm]
main = "main"

[llm.runtimes.main]
provider = "openai"
model = "model"
api_key = "key"
context_window = 64000
max_output_tokens = 4096
input_modalities = {modalities}
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="llm\\.runtimes\\.main\\.input_modalities 必须是字符串数组",
    ):
        load_config(path, workspace=tmp_path)


def test_config_accepts_api_compatible_provider(tmp_path: Path) -> None:
    template = """
[llm]
main = "main"
[llm.runtimes.main]
provider = "{provider}"
model = "model"
api_key = "key"
context_window = 64000
max_output_tokens = 4096
reasoning_summary = "{summary}"
"""
    path = tmp_path / "config.toml"
    path.write_text(template.format(provider="CustomAPI", summary="none"), encoding="utf-8")
    assert load_config(path, workspace=tmp_path).model_runtimes["main"].provider == "customapi"


def test_usage_keeps_partial_coverage_unknown() -> None:
    usage = aggregate_usage([
        ModelUsage(
            input_tokens=100,
            output_tokens=20,
            covered_request_count=1,
            coverage=UsageCoverage.EXACT,
        ),
        ModelUsage(),
    ])
    parsed = _parse_usage({
        "input_tokens": 100,
        "input_tokens_details": {"cached_tokens": 70},
        "output_tokens": 20,
        "output_tokens_details": {"reasoning_tokens": 8},
    })

    assert usage.coverage is UsageCoverage.PARTIAL
    assert (usage.input_tokens, usage.output_tokens) == (100, 20)
    assert parsed is not None
    assert (parsed.cached_input_tokens, parsed.reasoning_output_tokens) == (70, 8)
